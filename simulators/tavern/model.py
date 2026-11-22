"""Position-aware card embeddings plus numeric features; masked actor/critic."""
import time
from collections import Counter
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
import taverngym as tg


class Policy(nn.Module):
    def __init__(self, hidden=128, embedding=8):
        super().__init__()
        self.arch = {"hidden": hidden, "embedding": embedding}
        self.cards = nn.Embedding(tg.N_CARDS, embedding, padding_idx=0)
        self.body = nn.Sequential(nn.Linear(tg.OBS_FLOATS + tg.OBS_CARDS * embedding, hidden),
                                  nn.LayerNorm(hidden), nn.Tanh(),
                                  nn.Linear(hidden, hidden), nn.Tanh())
        self.actor = nn.Linear(hidden, tg.N_ACTIONS)
        self.critic = nn.Linear(hidden, 1)
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.zeros_(self.actor.bias)

    @staticmethod
    def encode_numeric(features):
        return torch.sign(features) * torch.log1p(features.abs())

    def forward(self, features, cards, numeric_encoded=False):
        # No ordinal treatment of card IDs. Keep positions so hand/shop/action bindings survive.
        f = features if numeric_encoded else self.encode_numeric(features)
        x = self.body(torch.cat((f, self.cards(cards.long()).flatten(1)), dim=1))
        return self.actor(x), self.critic(x).squeeze(-1)

    def distribution(self, features, cards, mask, numeric_encoded=False):
        if not getattr(self, "fast_inference", False) and not bool(mask.any(-1).all()):
            raise ValueError("policy called for a non-ready/all-zero-mask seat")
        logits, value = self(features, cards, numeric_encoded=numeric_encoded)
        return Categorical(logits=logits.masked_fill(~mask.bool(), -1e9), validate_args=not getattr(self, "fast_inference", False)), value


def tensors(obs, indices, device):
    # Advanced indexing makes contiguous writable arrays even for immutable zero-copy views.
    return (torch.from_numpy(obs["features"][indices]).to(device),
            torch.from_numpy(obs["cards"][indices]).to(device=device, dtype=torch.long),
            torch.from_numpy(obs["action_mask"][indices]).to(device))


def graph_action(policy, inputs, deterministic):
    """Exact-shape CUDA graph; sampling keeps the caller's generator stream.

    Warmup/capture RNG is restored. Replay advances the live CUDA generator just as
    the eager Exp(1)/argmax path does. Qualified only by the explicit parity suite.
    Graphs are bounded by seat batch sizes and cleared with the owning policy.
    """
    if not getattr(policy, "fast_inference", False):
        raise ValueError("graph inference requires CPU-validated fast_inference")
    cache = getattr(policy, "_actor_graphs", None)
    if cache is None:
        cache = policy._actor_graphs = {}
    key = (len(inputs[0]), deterministic)
    if key not in cache and len(cache) >= getattr(policy, "actor_graph_max_shapes", 64):
        # Cohort tails can expose arbitrarily many ready counts in large batches.
        # Rare shapes fall back to eager rather than retaining unbounded graph pools.
        return None
    if key not in cache:
        started = time.perf_counter()
        static = tuple(torch.empty_like(x) for x in inputs)
        for dst, src in zip(static, inputs): dst.copy_(src)
        def compute():
            dist, value = policy.distribution(*static)
            if deterministic:
                actions = dist.logits.argmax(-1)
            else:
                probabilities = dist.probs
                q = torch.empty_like(probabilities).exponential_(1)
                torch.div(probabilities, q, out=q)
                actions = q.argmax(-1)
            return torch.stack((actions.float(), dist.log_prob(actions), value), dim=1)
        rng = torch.cuda.get_rng_state(inputs[0].device)
        stream = torch.cuda.Stream(device=inputs[0].device)
        stream.wait_stream(torch.cuda.current_stream(inputs[0].device))
        with torch.cuda.stream(stream):
            for _ in range(3): compute()
        torch.cuda.current_stream(inputs[0].device).wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph): output = compute()
        torch.cuda.set_rng_state(rng, inputs[0].device)
        cache[key] = static, graph, output
        policy.inference_metrics["graph_build_seconds"] += time.perf_counter()-started
    static, graph, output = cache[key]
    for dst, src in zip(static, inputs): dst.copy_(src)
    graph.replay()
    return output


@torch.no_grad()
def act(policy, obs, indices, device, deterministic=False):
    cuda = str(device).startswith("cuda")
    profile = getattr(policy, "inference_metrics", None)
    events = [torch.cuda.Event(enable_timing=True) for _ in range(3)] if cuda and profile is not None else None
    started = time.perf_counter()
    if events: events[0].record()
    inputs = tensors(obs, indices, device)
    if getattr(policy, "retain_actor_inputs", False):
        policy.last_actor_inputs = inputs
    if events: events[1].record()
    use_graph = cuda and getattr(policy, "graph_inference", False)
    if use_graph:
        graph_output = graph_action(policy, inputs, deterministic)
        use_graph = graph_output is not None
    if not use_graph:
        dist, value = policy.distribution(*inputs)
    if use_graph:
        pass
    elif deterministic:
        actions = dist.logits.argmax(-1)
    elif getattr(policy, "validated_sampler", False):
        # Same Exp(1)/argmax single-sample path as PyTorch 2.9.1 multinomial.
        # Masks were checked on CPU; finite outputs are checked after the mandatory D2H.
        probabilities = dist.probs
        q = torch.empty_like(probabilities).exponential_(1)
        torch.div(probabilities, q, out=q)
        actions = q.argmax(-1)
    else:
        actions = dist.sample()
    if not use_graph: logp = dist.log_prob(actions)
    if events: events[2].record()
    wait_started = time.perf_counter()
    if use_graph:
        packed = graph_output.cpu().numpy()
        result = packed[:,0].astype(np.int64), packed[:,1], packed[:,2]
    elif getattr(policy, "fast_inference", False):
        # One small D2H synchronization instead of three. All action IDs fit exactly in f32.
        packed = torch.stack((actions.float(), logp, value), dim=1).cpu().numpy()
        result = packed[:,0].astype(np.int64), packed[:,1], packed[:,2]
    else:
        result = actions.cpu().numpy(), logp.cpu().numpy(), value.cpu().numpy()
    if not np.isfinite(result[1]).all() or not np.isfinite(result[2]).all():
        raise RuntimeError("nonfinite policy/value output before simulator step")
    if profile is not None:
        profile["inference_calls"] += 1
        profile["inference_rows"] += len(indices)
        profile["inference_wall_seconds"] += time.perf_counter()-started
        profile["d2h_and_compute_wait_host_seconds"] += time.perf_counter()-wait_started
        profile["input_bytes"] += len(indices)*(tg.OBS_FLOATS*4+tg.OBS_CARDS*8+tg.N_ACTIONS)
        profile["output_bytes"] += len(indices)*(12 if getattr(policy,"fast_inference",False) else 16)
        if events:
            # The synchronous D2H has already completed these earlier stream events.
            profile["h2d_device_seconds"] += events[0].elapsed_time(events[1])/1000
            profile["inference_device_seconds"] += events[1].elapsed_time(events[2])/1000
    return result


def load_policy(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    with torch.random.fork_rng(devices=[]):
        policy = Policy(**ckpt["architecture"]).to(device)
    policy.load_state_dict(ckpt["policy"])
    policy.eval()
    return policy, ckpt


@torch.no_grad()
def act_many(groups, obs, device, deterministic=False, while_pending=None):
    """Queue original per-network math/RNG order, with one H2D batch and D2H.

    groups contains (policy, source_rows). Networks are not padded, mixed or
    vectorized numerically. Their categorical draws retain the original shapes
    and generator order. CPU-only scripts may run while queued CUDA work finishes.
    """
    if not groups:
        if while_pending is not None:while_pending()
        return []
    ids=np.concatenate([np.asarray(rows,np.int64) for _,rows in groups])
    if len(ids) and (ids.min()<0 or ids.max()>=len(obs['ready'])):
        raise ValueError('Invalid batched actor source rows')
    owner=groups[0][0];cuda=str(device).startswith('cuda');n=len(ids)
    if cuda:
        staging=getattr(owner,'_actor_staging',None)
        if staging is None or len(staging[0])<n:
            capacity=1 << (n-1).bit_length()
            staging=owner._actor_staging=(torch.empty((capacity,tg.OBS_FLOATS),pin_memory=True),
                torch.empty((capacity,tg.OBS_CARDS),dtype=torch.long,pin_memory=True),
                torch.empty((capacity,tg.N_ACTIONS),dtype=torch.bool,pin_memory=True))
        for dst,key in zip(staging,('features','cards','action_mask')):
            np.copyto(dst[:n].numpy(),obs[key][ids],casting='safe')
        combined=tuple(x[:n].to(device,non_blocking=True) for x in staging)
    else:
        combined=tensors(obs,ids,device)
    # Protect repeated calls to the same captured output buffer in one frame.
    repeats=Counter((id(p),len(rows),deterministic) for p,rows in groups)
    packed=[];boundaries=[];begin=0;retained=set()
    for policy,rows in groups:
        started=time.perf_counter();end=begin+len(rows);inputs=tuple(x[begin:end] for x in combined)
        if getattr(policy,'retain_actor_inputs',False) and id(policy) not in retained:
            policy.last_actor_inputs=inputs;retained.add(id(policy))
        output=graph_action(policy,inputs,deterministic) if cuda and getattr(policy,'graph_inference',False) else None
        if output is None:
            dist,value=policy.distribution(*inputs)
            if deterministic:actions=dist.logits.argmax(-1)
            elif getattr(policy,'validated_sampler',False):
                probabilities=dist.probs;q=torch.empty_like(probabilities).exponential_(1);torch.div(probabilities,q,out=q);actions=q.argmax(-1)
            else:actions=dist.sample()
            output=torch.stack((actions.float(),dist.log_prob(actions),value),dim=1)
        key=(id(policy),len(rows),deterministic);repeats[key]-=1
        if repeats[key]:output=output.clone()
        packed.append(output);boundaries.append((begin,end));begin=end
        profile=getattr(policy,'inference_metrics',None)
        if profile is not None:
            profile['inference_calls']+=1;profile['inference_rows']+=len(rows)
            profile['input_bytes']+=len(rows)*(tg.OBS_FLOATS*4+tg.OBS_CARDS*8+tg.N_ACTIONS)
            profile['output_bytes']+=len(rows)*12
            profile['batched_enqueue_seconds']+=time.perf_counter()-started
    if while_pending is not None:while_pending()
    cpu=torch.cat(packed,dim=0).cpu().numpy()
    if not np.isfinite(cpu[:,1:]).all():raise RuntimeError('Nonfinite batched policy/value before simulator step')
    return [(cpu[a:b,0].astype(np.int64),cpu[a:b,1],cpu[a:b,2]) for a,b in boundaries]
