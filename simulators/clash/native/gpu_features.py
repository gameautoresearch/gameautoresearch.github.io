"""Device-resident inputs for the existing Clash policy, with exact feature semantics.

The native environment writes int32 observations and uint8 masks on the policy's device.
``encode_device`` returns the same dictionary as ``model.Policy.pack``; every operation in
its hot path stays on that device. ``DeviceFeatureEncoder`` also exposes the existing learned
encoder, policy head, and actor without transferring observations or sampled actions to CPU.

The source of truth for feature order, categorical offsets/clamps, scales, flag bits, padded
sum/max reductions and embedding vocabulary is integrations/clash/model.py. This module uses
that encoder directly and therefore shares its weights and exact reduction/gradient behavior.

Data-dependent tensor SHAPES require a host synchronization in PyTorch. Consequently, absent
an externally known safe static entity-row bound, this interface uses all 200 schema rows.
The alive mask in Policy.encode still excludes every padded row. An explicitly supplied
``entity_rows`` reproduces Policy.pack's trim/bucket shape and is checked on device so that it
cannot drop a live entity. This is a storage shape, not a new world/observation entity cap.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn


def _field(schema: Any, name: str) -> Any:
    return schema[name] if isinstance(schema, Mapping) else getattr(schema, name)


def _shape(schema: Any, entity_rows: int | None, entity_bucket: int, trim: bool) -> tuple[int, int, int, int, int, int]:
    """Resolve host configuration only; never read a tensor's contents."""
    seat = int(_field(schema, "seat_ints"))
    header = int(_field(schema, "header_ints"))
    entity = int(_field(schema, "entity_ints"))
    maximum = int(_field(schema, "max_entities"))
    actions = int(_field(schema, "actions"))
    if seat != header + maximum * entity:
        raise ValueError("schema observation length does not match its entity layout")
    if entity_bucket < 0:
        raise ValueError("entity_bucket must be nonnegative")
    if entity_rows is not None and (isinstance(entity_rows, bool) or not isinstance(entity_rows, int)):
        raise TypeError("entity_rows must be a host integer known independently of device contents")
    # Mirror Policy.pack: at least one entity row, optional round-up, then schema cap.
    rows = maximum if entity_rows is None or not trim else max(1, entity_rows)
    if entity_bucket:
        rows = ((rows + entity_bucket - 1) // entity_bucket) * entity_bucket
    rows = min(rows, maximum)
    return seat, header, entity, maximum, actions, rows


def _pack_device(
    raw_obs: torch.Tensor,
    masks: torch.Tensor | None,
    shape: tuple[int, int, int, int, int, int],
    count_index: int,
    *,
    copy: bool,
    device: torch.device | str | None,
) -> dict[str, torch.Tensor]:
    """No NumPy, host scalar extraction, dynamic allocation shape, or device transfer."""
    if not isinstance(raw_obs, torch.Tensor) or raw_obs.dtype != torch.int32:
        raise TypeError("raw_obs must be an int32 torch tensor")
    if device is not None and raw_obs.device != torch.device(device):
        # This is validation, not an implicit host-to-device conversion of native state.
        requested = torch.device(device)
        if not (requested.type == raw_obs.device.type and requested.index is None):
            raise ValueError("native observations are not on the requested policy device")
    seat, header_ints, entity_ints, maximum, actions, rows = shape
    if raw_obs.ndim < 1 or raw_obs.shape[-1] != seat:
        raise ValueError(f"raw_obs must end in {seat} observation entries")
    observation = raw_obs.reshape(-1, seat)
    header = observation[:, :header_ints]
    entities = observation[:, header_ints:].reshape(-1, maximum, entity_ints)[:, :rows]
    if rows < maximum:
        # Policy.pack caps at maximum even if an overflow observation reports a larger count.
        # Negative counts yield an empty alive mask in the owning encoder, as before.
        needed = header[:, count_index].clamp(min=1, max=maximum)
        torch._assert_async(
            torch.all(needed <= rows),
            "static entity_rows would drop an exposed entity; use the full schema capacity",
        )
    batch = {
        "header": header.clone() if copy else header,
        "entities": entities.clone() if copy else entities,
    }
    if masks is not None:
        if not isinstance(masks, torch.Tensor) or masks.dtype not in (torch.uint8, torch.bool):
            raise TypeError("masks must be a uint8 or bool torch tensor")
        if masks.device != raw_obs.device:
            raise ValueError("observations and masks must already be on the same device")
        if masks.ndim < 1 or masks.shape[-1] != actions:
            raise ValueError(f"masks must end in {actions} action entries")
        legal = masks.reshape(-1, actions)
        if legal.shape[0] != observation.shape[0]:
            raise ValueError("observations and masks have different batch sizes")
        batch["mask"] = legal.to(dtype=torch.bool, copy=copy)
    return batch


def encode_device(
    raw_obs: torch.Tensor,
    masks: torch.Tensor | None,
    schema: Any,
    device: torch.device | str | None = None,
    *,
    entity_rows: int | None = None,
    entity_bucket: int = 0,
    trim: bool = True,
    copy: bool = True,
) -> dict[str, torch.Tensor]:
    """Pack native observations for ``Policy.forward``/``sample_batch`` on their device.

    Accepts either [N,7024] or leading environment/seat dimensions, exactly as Policy.pack.
    Returns int32 ``header``/``entities`` and bool ``mask``. ``copy=True`` owns a snapshot:
    a subsequent native step can safely overwrite the original buffers. ``copy=False`` lends
    views until the next native write and is suitable for an actor whose inference completes
    on the same ordered stream before the next step. The rollout store still needs a snapshot.

    Leave entity_rows unset for a completely device-resident dense actor. If a static upper
    bound is supplied, bucket round-up and the schema's 200-row cap match Policy.pack and an
    asynchronous device assertion rejects dropping an exposed row. No inference-time CPU
    count readback or hidden host trimming occurs.
    """
    shape = _shape(schema, entity_rows, entity_bucket, trim)
    count_index = int(_field(schema, "header")["entity_count"][0])
    return _pack_device(raw_obs, masks, shape, count_index, copy=copy, device=device)


class DeviceBatchEncoder(nn.Module):
    """Cache schema/configuration for static CUDA graph capture or torch.compile."""

    def __init__(
        self,
        schema: Any,
        *,
        entity_rows: int | None = None,
        entity_bucket: int = 0,
        trim: bool = True,
        copy: bool = True,
    ):
        super().__init__()
        self.layout = _shape(schema, entity_rows, entity_bucket, trim)
        self.count_index = int(_field(schema, "header")["entity_count"][0])
        self.copy = copy

    @property
    def entity_rows(self) -> int:
        return self.layout[-1]

    def forward(self, raw_obs: torch.Tensor, masks: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        return _pack_device(raw_obs, masks, self.layout, self.count_index, copy=self.copy, device=None)


class DeviceFeatureEncoder(nn.Module):
    """Share a Policy's exact learned features and heads; parameters are never copied.

    ``forward(raw_obs,masks)`` returns (logits,value), ``encode(raw_obs)`` returns the same
    (context,header) as Policy.encode, and ``sample_batch`` returns device action/logp/value.
    The policy must already be on the native environment's device. Optimizers/checkpoints may
    keep using the original Policy: this wrapper references the very same parameters.
    """

    def __init__(self, policy: nn.Module, **packing: Any):
        super().__init__()
        self.policy = policy
        self.batch_encoder = DeviceBatchEncoder(policy.s, **packing)

    def pack(self, raw_obs: torch.Tensor, masks: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        return self.batch_encoder(raw_obs, masks)

    def encode(self, raw_obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.policy.encode(self.pack(raw_obs))

    def forward(self, raw_obs: torch.Tensor, masks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.policy(self.pack(raw_obs, masks))

    def values(self, raw_obs: torch.Tensor) -> torch.Tensor:
        return self.policy.values(self.pack(raw_obs))

    def sample_batch(self, raw_obs: torch.Tensor, masks: torch.Tensor, greedy: bool = False):
        return self.policy.sample_batch(self.pack(raw_obs, masks), greedy)


def qualify_cuda_features(
    policy: nn.Module,
    raw_obs: torch.Tensor,
    masks: torch.Tensor,
    *,
    output_dir: str | None = None,
) -> dict[str, Any]:
    """Actual CUDA qualification; host copies here are deliberately outside the hot path.

    Accepts native device observations or the uploaded public Java observation fixture. Checks
    exact same-device pack/features/heads/gradients against the original Policy methods, CPU
    numerical parity, and graph replay after device input and parameter changes. It allocates no
    cloud worker and refuses CPU fallback. It performs correctness checks, never a timed comparison.
    """
    import json
    from pathlib import Path

    if not torch.cuda.is_available() or raw_obs.device.type != "cuda" or masks.device != raw_obs.device:
        raise RuntimeError("CUDA qualification requires actual CUDA observations and masks")
    parameters = tuple(policy.parameters())
    if not parameters or parameters[0].device != raw_obs.device:
        raise ValueError("policy and native observation device differ")
    # The native Trainer enables device indices too. Python index lists can construct host
    # indexing tensors during CUDA graph capture; Policy's registered device index buffers
    # preserve the same feature order while making this complete forward path capture-safe.
    policy.device_indices = True
    out = None if output_dir is None else Path(output_dir)
    if out is not None:
        out.mkdir(parents=True, exist_ok=False)
    # This is the independent legacy CPU packing entry point, including its real CUDA transfer.
    obs_cpu = raw_obs.detach().cpu()
    mask_cpu = masks.detach().cpu()
    legacy = policy.pack(obs_cpu.numpy(), mask_cpu.numpy(), device=raw_obs.device, trim=False)
    encoder = DeviceFeatureEncoder(policy, copy=True)
    packed = encoder.pack(raw_obs, masks)
    for key in legacy:
        torch.testing.assert_close(packed[key], legacy[key], rtol=0, atol=0)
    with torch.no_grad():
        expected_context, expected_header = policy.encode(legacy)
        context, header = encoder.encode(raw_obs)
        torch.testing.assert_close(context, expected_context, rtol=0, atol=0)
        torch.testing.assert_close(header, expected_header, rtol=0, atol=0)
        expected_logits, expected_values = policy(legacy)
        logits, values = encoder(raw_obs, masks)
        torch.testing.assert_close(logits, expected_logits, rtol=0, atol=0)
        torch.testing.assert_close(values, expected_values, rtol=0, atol=0)
    # Identical graph and embedding tensors must preserve every parameter gradient, including
    # first-index max subgradients. autograd.grad leaves the optimizer's .grad buffers untouched.
    logits_a, values_a = policy(legacy)
    grads_a = torch.autograd.grad(logits_a[:, 0].sum() + values_a.square().sum(), parameters, allow_unused=True)
    logits_b, values_b = encoder(raw_obs, masks)
    grads_b = torch.autograd.grad(logits_b[:, 0].sum() + values_b.square().sum(), parameters, allow_unused=True)
    for a, b in zip(grads_a, grads_b):
        if a is None:
            assert b is None
        else:
            torch.testing.assert_close(a, b, rtol=0, atol=0)
    # Reconstruct a CPU Policy instead of copying CUDA graphs or actor caches from a live model.
    with torch.random.fork_rng(devices=[]):
        cpu_policy = type(policy)(policy.s, policy.cards.num_embeddings - 1,
                                  policy.buffs.num_embeddings - 1, policy.width, policy.embedding)
    cpu_policy.load_state_dict({k: v.detach().cpu() for k, v in policy.state_dict().items()})
    cpu_policy.device_indices = bool(getattr(policy, "device_indices", False))
    with torch.no_grad():
        cpu_batch = cpu_policy.pack(obs_cpu.numpy(), mask_cpu.numpy(), trim=False)
        cpu_context, _ = cpu_policy.encode(cpu_batch)
        cpu_logits, cpu_values = cpu_policy(cpu_batch)
        # CPU and CUDA GEMM reductions need not be bitwise identical. Same-CUDA comparisons above
        # are exact; the cross-device tolerance is explicit and reported separately.
        torch.testing.assert_close(context.cpu(), cpu_context, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(logits.cpu(), cpu_logits, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(values.cpu(), cpu_values, rtol=1e-4, atol=1e-5)
        cpu_context_error = float((context.cpu() - cpu_context).abs().max())
        cpu_logit_error = float((logits.cpu() - cpu_logits).abs().max())
        cpu_value_error = float((values.cpu() - cpu_values).abs().max())
    # Capture the entire new packing and learned forward path. Replay observes mutation of the
    # native input buffer, then an in-place parameter update, without recapturing or stale inputs.
    graph_raw = raw_obs.clone()
    graph_masks = masks.clone()
    stream = torch.cuda.Stream(device=raw_obs.device)
    stream.wait_stream(torch.cuda.current_stream(raw_obs.device))
    with torch.cuda.stream(stream), torch.no_grad():
        for _ in range(2):
            encoder(graph_raw, graph_masks)
    torch.cuda.current_stream(raw_obs.device).wait_stream(stream)
    torch.cuda.synchronize(raw_obs.device)
    graph = torch.cuda.CUDAGraph()
    with torch.no_grad(), torch.cuda.graph(graph):
        graph_logits, graph_values = encoder(graph_raw, graph_masks)
    graph.replay()
    torch.testing.assert_close(graph_logits, expected_logits, rtol=0, atol=0)
    torch.testing.assert_close(graph_values, expected_values, rtol=0, atol=0)
    index = int(_field(policy.s, "header")["elixir"][0])
    graph_raw.reshape(-1, _field(policy.s, "seat_ints"))[:, index].add_(1000)
    graph.replay()
    with torch.no_grad():
        mutated_logits, mutated_values = encoder(graph_raw, graph_masks)
    torch.testing.assert_close(graph_logits, mutated_logits, rtol=0, atol=0)
    torch.testing.assert_close(graph_values, mutated_values, rtol=0, atol=0)
    old_bias = policy.value.bias.detach().clone()
    try:
        with torch.no_grad():
            policy.value.bias.add_(0.125)
        graph.replay()
        with torch.no_grad():
            updated_logits, updated_values = encoder(graph_raw, graph_masks)
        torch.testing.assert_close(graph_logits, updated_logits, rtol=0, atol=0)
        torch.testing.assert_close(graph_values, updated_values, rtol=0, atol=0)
    finally:
        with torch.no_grad():
            policy.value.bias.copy_(old_bias)
    result = {
        "status": "passed",
        "cuda_executed": True,
        "device": str(raw_obs.device),
        "cuda_name": torch.cuda.get_device_name(raw_obs.device),
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "seat_observations": raw_obs.numel() // _field(policy.s, "seat_ints"),
        "same_cuda_exact": ["packed_inputs", "context", "logits", "values", "parameter_gradients"],
        "graph_exact": ["initial_replay", "mutated_device_input", "in_place_parameter_update"],
        "cpu_cuda_max_absolute_error": {"context": cpu_context_error, "logits": cpu_logit_error, "value": cpu_value_error},
        "cpu_cuda_tolerance": {"rtol": 1e-4, "atol": 1e-5},
        "timing_comparison": False,
    }
    if out is not None:
        (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def _self_test(out_path: str) -> dict[str, Any]:
    """Real pinned Java observation parity; this entry point is outside the device hot path."""
    import hashlib
    import json
    import sys
    from pathlib import Path
    import numpy as np

    integration = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(integration))
    from game import Session, make_vector
    from runner import network, setup
    from crforge_env import ScriptedBot

    out = Path(out_path).resolve()
    out.mkdir(parents=True, exist_ok=False)
    cfg = json.loads((integration / "configs/smoke.json").read_text())
    cfg.update(envs=2, threads=1, torch_threads=1, device="cpu")
    setup(cfg)
    checked_batches = 0
    checked_seats = 0
    max_entities = 0
    with make_vector(cfg) as vector:
        Session(vector, cfg, np.random.default_rng(81108))
        policy = network(vector, cfg)
        encoder = DeviceFeatureEncoder(policy, copy=True)
        bot = ScriptedBot().bind(vector)
        rng = np.random.default_rng(81109)
        saved = None
        for decision in range(160):
            if decision % 4 == 0:
                raw = torch.from_numpy(vector.observations.copy())
                masks = torch.from_numpy(vector.masks.astype(np.uint8, copy=True))
                dense = policy.pack(vector.observations, vector.masks, trim=False)
                batch = encoder.pack(raw, masks)
                for key in dense:
                    torch.testing.assert_close(batch[key], dense[key], rtol=0, atol=0)
                with torch.no_grad():
                    actual_context, actual_h = encoder.encode(raw)
                    expected_context, expected_h = policy.encode(dense)
                    torch.testing.assert_close(actual_context, expected_context, rtol=0, atol=0)
                    torch.testing.assert_close(actual_h, expected_h, rtol=0, atol=0)
                    actual_logits, actual_values = encoder(raw, masks)
                    expected_logits, expected_values = policy(dense)
                    torch.testing.assert_close(actual_logits, expected_logits, rtol=0, atol=0)
                    torch.testing.assert_close(actual_values, expected_values, rtol=0, atol=0)
                # A host-derived trim bound is used only by this offline parity test.
                count = max(1, int(vector.observations[..., vector.schema.header["entity_count"][0]].max()))
                for bucket in (0, 32):
                    compact = policy.pack(vector.observations, vector.masks, trim=True, bucket_override=bucket)
                    packed = encode_device(raw, masks, vector.schema, entity_rows=count, entity_bucket=bucket)
                    for key in compact:
                        torch.testing.assert_close(packed[key], compact[key], rtol=0, atol=0)
                    with torch.no_grad():
                        actual = policy(packed)
                        expected = policy(compact)
                        for a, b in zip(actual, expected):
                            torch.testing.assert_close(a, b, rtol=0, atol=0)
                checked_batches += 1
                checked_seats += raw.reshape(-1, vector.schema.seat_ints).shape[0]
                max_entities = max(max_entities, count)
                saved = (raw, masks, dense)
            actions = np.array([[bot(o, m, rng) for o, m in zip(row, legal)] for row, legal in zip(vector.observations, vector.masks)], np.int32)
            vector.step(actions)
            for lane in np.flatnonzero(vector.dones):
                vector.reset_env(int(lane), int(rng.integers(1000000)))
        assert saved is not None
        raw, masks, dense = saved
        # This fixture contains public policy observations only, no engine RNG/world/private hand.
        # It lets a CUDA worker qualify the device feature path without launching a JVM there.
        schema_names = ("seat_ints", "header_ints", "entity_ints", "max_entities", "actions",
                        "cols", "rows", "hand_slots", "ability_slots", "first_ability", "buff_slots",
                        "flag_names", "state_names", "header", "entity")
        fixture = {"schema": {name: getattr(vector.schema, name) for name in schema_names},
                   "vocab_size": policy.cards.num_embeddings - 1,
                   "buff_count": policy.buffs.num_embeddings - 1,
                   "width": policy.width, "embedding": policy.embedding,
                   "policy_state_dict": policy.state_dict(),
                   "observations": raw.clone(), "masks": masks.clone(),
                   "provenance": "public player-visible observations from pinned Java fixed8 ScriptedBot correctness test"}
        torch.save(fixture, out / "java-public-observations.pt")
        owned = encode_device(raw, masks, vector.schema)
        raw.zero_()
        for key in dense:
            torch.testing.assert_close(owned[key], dense[key], rtol=0, atol=0)
        # Gradient equality checks more than forward equality of the learned feature encoder.
        policy.zero_grad(set_to_none=True)
        logits, value = policy(dense)
        (logits[:, 0].sum() + value.sum()).backward()
        expected_grad = {name: p.grad.clone() if p.grad is not None else None for name, p in policy.named_parameters()}
        policy.zero_grad(set_to_none=True)
        logits, value = policy(owned)
        (logits[:, 0].sum() + value.sum()).backward()
        for name, p in policy.named_parameters():
            expected = expected_grad[name]
            if expected is None:
                assert p.grad is None
            else:
                torch.testing.assert_close(p.grad, expected, rtol=0, atol=0)
    result = {
        "status": "passed",
        "real_java_observation_batches": checked_batches,
        "seat_observations": checked_seats,
        "max_exposed_entities": max_entities,
        "exact_comparisons": ["packed_header/entities/masks", "learned_context", "logits", "values", "trimmed_and_bucket32_inputs_outputs", "snapshot_ownership", "parameter_gradients"],
        "cuda_executed": False,
        "timing_comparison": False,
        "dynamic_trim_host_sync": False,
        "default_device_entity_rows": 200,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    import argparse
    import json
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--self-test", metavar="NEW_EVIDENCE_DIRECTORY")
    mode.add_argument("--qualify-cuda", metavar="PUBLIC_JAVA_FIXTURE_PT")
    parser.add_argument("--out", metavar="NEW_CUDA_EVIDENCE_DIRECTORY")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(_self_test(args.self_test), indent=2))
    else:
        if not args.out:
            parser.error("--qualify-cuda requires --out")
        import sys
        from pathlib import Path
        from types import SimpleNamespace
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from model import Policy
        if not torch.cuda.is_available():
            raise RuntimeError("actual CUDA feature qualification requires a CUDA device")
        fixture = torch.load(args.qualify_cuda, map_location="cpu", weights_only=True)
        schema = SimpleNamespace(**fixture["schema"])
        policy = Policy(schema, fixture["vocab_size"], fixture["buff_count"],
                        fixture["width"], fixture["embedding"]).cuda()
        policy.load_state_dict(fixture["policy_state_dict"])
        policy.device_indices = True
        print(json.dumps(qualify_cuda_features(policy, fixture["observations"].cuda(),
                                              fixture["masks"].cuda(), output_dir=args.out), indent=2))
