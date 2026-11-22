"""Prototype device-selected [32,64,96,200] actor graphs for the fixed8 Policy.

Only the RNG-free learned network belongs to conditional CUDA branches. Every branch reads
one owned raw observation/mask buffer and copies to the same fixed-shape logits/value buffers.
Sampling runs afterwards through ordinary Torch CUDA or a separate Torch-owned sampling graph,
so PyTorch owns its graph-aware RNG bookkeeping and checkpoint state.

This module changes no simulator, trainer, policy implementation, entity cap or optimizer.
It requires the companion device_branch.DeviceBranchGraph CUDA 12.8 SWITCH implementation.
The conditional parent graph cannot become a child: outer rollout graph capture is rejected.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
import torch
from torch.distributions import Categorical

try:
    from .gpu_features import DeviceBatchEncoder
except ImportError:
    from gpu_features import DeviceBatchEncoder

BUCKETS = (32, 64, 96, 200)


class BucketActor:
    """Borrowed device outputs, valid until the next replay; policy parameters are shared.

    Construct once outside capture. forward(raw_obs,masks) returns shared (logits,value).
    sample(raw_obs,masks,greedy=False) returns device (action,logp,value). Inputs can carry
    leading environment/seat dimensions; their total seat count must equal batch_size.
    No observation contents or sampled actions are read by the CPU in either hot path.
    """

    def __init__(
        self,
        policy,
        batch_size: int,
        *,
        graph_sampler: bool = False,
        buckets: tuple[int, ...] = BUCKETS,
        outer_rollout_graph: bool = False,
    ):
        if outer_rollout_graph:
            raise ValueError("conditional bucket actor requires outer_rollout_graph=False")
        if tuple(buckets) != BUCKETS or policy.s.max_entities != 200:
            raise ValueError("fixed8 actor buckets must retain [32,64,96,200]")
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("positive static seat batch_size required")
        parameters = tuple(policy.parameters())
        if not parameters or parameters[0].device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("BucketActor requires actual CUDA policy parameters")
        if not getattr(policy, "device_indices", False):
            raise ValueError("enable Policy.device_indices before constructing BucketActor")
        if getattr(policy, "ragged_entities", False) or getattr(policy, "sparse_policy_head", False):
            raise ValueError("conditional network branches require the static Policy encoder/head")
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("construct BucketActor outside CUDA graph capture")
        try:
            from .device_branch import DeviceBranchGraph
        except ImportError:
            from device_branch import DeviceBranchGraph
        self.policy = policy
        self.schema = policy.s
        self.device = parameters[0].device
        self.batch_size = batch_size
        self.buckets = BUCKETS
        self.graph_sampler = graph_sampler
        self.closed = False
        self.count_index = int(self.schema.header["entity_count"][0])
        self.raw_obs = torch.zeros((batch_size, self.schema.seat_ints), dtype=torch.int32, device=self.device)
        self.masks = torch.zeros((batch_size, self.schema.actions), dtype=torch.uint8, device=self.device)
        self.masks[:, 0] = 1
        self.logits = torch.empty((batch_size, self.schema.actions), dtype=parameters[0].dtype, device=self.device)
        self.values = torch.empty((batch_size,), dtype=parameters[0].dtype, device=self.device)
        self._children = []
        self._child_outputs = []
        self._samplers = {}
        # Count0/wait-only inputs are safe warmups for every bucket. Capture never mutates live
        # native observation buffers, and no policy update occurs here.
        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(stream), torch.no_grad():
            for rows in self.buckets:
                encoder = DeviceBatchEncoder(self.schema, entity_rows=rows, copy=False)
                for _ in range(2):
                    policy(encoder(self.raw_obs, self.masks))
                child = torch.cuda.CUDAGraph(keep_graph=True)
                with torch.cuda.graph(child, stream=stream):
                    logits, values = policy(encoder(self.raw_obs, self.masks))
                    self.logits.copy_(logits)
                    self.values.copy_(values)
                self._children.append(child)
                self._child_outputs.append((encoder, logits, values))
        torch.cuda.current_stream(self.device).wait_stream(stream)
        torch.cuda.synchronize(self.device)
        self._branch = DeviceBranchGraph(self.raw_obs, self.count_index, list(self.buckets), self._children)
        # Prime the shared outputs with valid neutral inputs before sampler warmup.
        self._branch.replay()
        if self.graph_sampler:
            self._capture_sampler(False)
            self._capture_sampler(True)

    def _check_capture(self):
        if self.closed:
            raise RuntimeError("BucketActor is closed")
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("conditional bucket actor cannot run inside an outer rollout graph")

    def forward(self, raw_obs: torch.Tensor, masks: torch.Tensor):
        self._check_capture()
        if raw_obs.dtype != torch.int32 or raw_obs.device != self.device:
            raise ValueError("raw observations must already be int32 on the policy CUDA device")
        if masks.dtype not in (torch.uint8, torch.bool) or masks.device != self.device:
            raise ValueError("masks must already be uint8/bool on the policy CUDA device")
        if raw_obs.shape[-1] != self.schema.seat_ints or masks.shape[-1] != self.schema.actions:
            raise ValueError("native schema shape mismatch")
        if raw_obs.numel() != self.raw_obs.numel() or masks.numel() != self.masks.numel():
            raise ValueError("static actor seat batch size differs")
        self.raw_obs.copy_(raw_obs.reshape_as(self.raw_obs))
        self.masks.copy_(masks.reshape_as(self.masks))
        # Reject corrupt counts rather than allow a narrow bucket/encoder clamp to conceal them.
        # The SWITCH selector also guards invalid counts and retains full200 for the valid tail.
        counts = self.raw_obs[:, self.count_index]
        torch._assert_async(((counts >= 0) & (counts <= 200)).all(), "invalid public entity_count")
        self._branch.replay()
        return self.logits, self.values

    __call__ = forward

    def _sample(self, greedy: bool):
        distribution = Categorical(logits=self.logits, validate_args=False)
        action = distribution.logits.argmax(-1) if greedy else distribution.sample()
        return action, distribution.log_prob(action)

    def _capture_sampler(self, greedy: bool):
        # Warmup/capture must not consume a research trajectory's saved policy RNG stream.
        rng = torch.cuda.get_rng_state(self.device)
        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))
        try:
            with torch.cuda.stream(stream), torch.no_grad():
                for _ in range(2):
                    self._sample(greedy)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    action, logp = self._sample(greedy)
            torch.cuda.current_stream(self.device).wait_stream(stream)
            torch.cuda.synchronize(self.device)
            self._samplers[greedy] = (graph, action, logp)
        finally:
            torch.cuda.set_rng_state(rng, self.device)

    def sample(self, raw_obs: torch.Tensor, masks: torch.Tensor, greedy: bool = False):
        self.forward(raw_obs, masks)
        if self.graph_sampler:
            graph, action, logp = self._samplers[bool(greedy)]
            graph.replay()
        else:
            action, logp = self._sample(greedy)
        return action, logp, self.values

    def close(self):
        if self.closed:
            return
        torch.cuda.synchronize(self.device)
        if hasattr(self._branch, "close"):
            self._branch.close()
        self._branch = None
        for graph in self._children:
            graph.reset()
        for graph, _, _ in self._samplers.values():
            graph.reset()
        self._children.clear()
        self._child_outputs.clear()
        self._samplers.clear()
        self.closed = True


def _cpu_feature_qualification(fixture: dict[str, Any]):
    """Offline fixture boundary checks; never called by the device actor hot path."""
    from types import SimpleNamespace
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from model import Policy
    schema = SimpleNamespace(**fixture["schema"])
    with torch.random.fork_rng(devices=[]):
        policy = Policy(schema, fixture["vocab_size"], fixture["buff_count"], fixture["width"], fixture["embedding"])
    policy.load_state_dict(fixture["policy_state_dict"])
    policy.device_indices = True
    raw = fixture["observations"].reshape(-1, schema.seat_ints).clone()
    masks = fixture["masks"].reshape(-1, schema.actions).clone()
    dense = DeviceBatchEncoder(schema, copy=False)
    max_context_error = max_logit_error = max_value_error = 0.
    boundaries = [0, 1, 31, 32, 33, 63, 64, 65, 95, 96, 97, 199, 200]
    cases = [("real-public-observation", raw.clone())]
    for count in boundaries:
        state = raw.clone()
        state[:, schema.header["entity_count"][0]] = count
        cases.append((str(count), state))
    for name, state in cases:
        count = max(0, int(state[:, schema.header["entity_count"][0]].max()))
        rows = next(bucket for bucket in BUCKETS if count <= bucket)
        compact = DeviceBatchEncoder(schema, entity_rows=rows, copy=False)
        with torch.no_grad():
            expected_context, _ = policy.encode(dense(state, masks))
            context, _ = policy.encode(compact(state, masks))
            expected_logits, expected_values = policy(dense(state, masks))
            logits, values = policy(compact(state, masks))
        for actual, expected in ((context, expected_context), (logits, expected_logits), (values, expected_values)):
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6, msg=name)
        max_context_error = max(max_context_error, float((context-expected_context).abs().max()))
        max_logit_error = max(max_logit_error, float((logits-expected_logits).abs().max()))
        max_value_error = max(max_value_error, float((values-expected_values).abs().max()))
    return {"status": "passed", "cases": len(cases), "boundaries": boundaries,
            "cpu_max_absolute_error": {"context": max_context_error, "logits": max_logit_error, "value": max_value_error},
            "tolerance": {"rtol": 1e-5, "atol": 1e-6}, "cuda_executed": False}


def qualify_cuda_bucket_actor(policy, raw_obs: torch.Tensor, masks: torch.Tensor):
    """Device SWITCH boundary/feature and sampler-RNG-resume qualification, outside hot path."""
    schema = policy.s
    flat = raw_obs.reshape(-1, schema.seat_ints)
    legal = masks.reshape(-1, schema.actions)
    dense = DeviceBatchEncoder(schema, copy=False)
    count_index = schema.header["entity_count"][0]
    cases = [("real-public-observation", flat.clone())]
    for count in (0, 1, 31, 32, 33, 63, 64, 65, 95, 96, 97, 199, 200):
        state = flat.clone()
        state[:, count_index] = count
        cases.append((str(count), state))
    reports = []
    for graph_sampler in (False, True):
        before_capture = torch.cuda.get_rng_state(flat.device)
        actor = BucketActor(policy, len(flat), graph_sampler=graph_sampler)
        assert torch.equal(before_capture, torch.cuda.get_rng_state(flat.device)), "capture consumed policy RNG"
        max_logit_error = max_value_error = 0.
        try:
            for name, state in cases:
                with torch.no_grad():
                    expected_logits, expected_values = policy(dense(state, legal))
                    logits, values = actor(state, legal)
                    torch.testing.assert_close(logits, expected_logits, rtol=1e-5, atol=1e-6, msg=name)
                    torch.testing.assert_close(values, expected_values, rtol=1e-5, atol=1e-6, msg=name)
                    max_logit_error = max(max_logit_error, float((logits-expected_logits).abs().max()))
                    max_value_error = max(max_value_error, float((values-expected_values).abs().max()))
            saved_rng = torch.cuda.get_rng_state(flat.device)
            expected = []
            for _, state in cases:
                with torch.no_grad():
                    a, lp, value = actor.sample(state, legal)
                    expected.append((a.clone(), lp.clone(), value.clone()))
            final_rng = torch.cuda.get_rng_state(flat.device)
            actor.close()
            torch.cuda.set_rng_state(saved_rng, flat.device)
            resumed = BucketActor(policy, len(flat), graph_sampler=graph_sampler)
            try:
                for (_, state), triplet in zip(cases, expected):
                    with torch.no_grad():
                        actual = resumed.sample(state, legal)
                    for a, b in zip(actual, triplet):
                        torch.testing.assert_close(a, b, rtol=0, atol=0)
                assert torch.equal(final_rng, torch.cuda.get_rng_state(flat.device))
            finally:
                resumed.close()
            reports.append({"graph_sampler": graph_sampler, "cases": len(cases),
                            "max_logit_error": max_logit_error, "max_value_error": max_value_error,
                            "capture_preserves_rng": True, "recreated_actor_resume_exact": True})
        finally:
            actor.close()
    return {"status": "passed", "cuda_executed": True, "cuda_name": torch.cuda.get_device_name(flat.device),
            "buckets": list(BUCKETS), "outer_rollout_graph": False, "network_only_conditional_branches": True,
            "sampler_outside_conditional": True, "modes": reports}


if __name__ == "__main__":
    import argparse, hashlib, json, sys
    from types import SimpleNamespace
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cpu-only", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    fixture = torch.load(args.fixture, weights_only=True, map_location="cpu")
    torch.set_num_threads(1)
    result = {"cpu": _cpu_feature_qualification(fixture), "timing_claims": False,
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    if not args.cpu_only:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from model import Policy
        schema = SimpleNamespace(**fixture["schema"])
        policy = Policy(schema, fixture["vocab_size"], fixture["buff_count"], fixture["width"], fixture["embedding"]).cuda()
        policy.load_state_dict(fixture["policy_state_dict"])
        policy.device_indices = True
        torch.backends.cuda.matmul.allow_tf32 = False
        result["cuda"] = qualify_cuda_bucket_actor(policy, fixture["observations"].cuda(), fixture["masks"].cuda())
    (args.out/"result.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result), flush=True)
