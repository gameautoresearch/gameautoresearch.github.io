"""Complete-episode batched PPO; waiting seats never become policy samples."""
from __future__ import annotations
import copy
from collections import defaultdict
import time
import numpy as np
import torch
from torch import nn
from .network import Policy, batch, scripted, scripted_batch

class Timings:
    def __init__(self,cuda_events=False):
        self.cuda_events=cuda_events
        self.seconds = defaultdict(float)
        self.counts = defaultdict(int)
    def add(self, name, elapsed, count=1):
        self.seconds[name] += elapsed
        self.counts[name] += count
    def result(self):
        return {"seconds": dict(self.seconds), "counts": dict(self.counts)}

def sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)
    elif str(device) == "mps":
        torch.mps.synchronize()

class Inference:
    def __init__(self, device, timings, batched=True):
        self.device, self.timings, self.batched = device, timings, batched
        self.events=[torch.cuda.Event(enable_timing=True) for _ in range(4)] if str(device).startswith("cuda") and getattr(timings,"cuda_events",False) else None
    @torch.no_grad()
    def __call__(self, model, observations, deterministic=False):
        if not self.batched and len(observations)>1:
            parts = [self(model, [o], deterministic) for o in observations]
            return tuple(np.concatenate([p[i] for p in parts]) for i in range(3))
        start = time.perf_counter()
        if self.events:self.events[0].record()
        x = batch(observations, self.device)
        if self.events:self.events[1].record()
        sync(self.device)
        self.timings.add("pack_and_transfer", time.perf_counter()-start, len(observations))
        start = time.perf_counter()
        dist, values = model.distribution(x)
        actions = dist.logits.argmax(-1) if deterministic else dist.sample()
        logp = dist.log_prob(actions)
        if self.events:self.events[2].record()
        returning=time.perf_counter()
        result = actions.cpu().numpy(), logp.cpu().numpy(), values.cpu().numpy()
        if self.events:
            self.events[3].record();self.events[3].synchronize()
            self.timings.add('h2d_stream_elapsed',self.events[0].elapsed_time(self.events[1])/1000,len(observations))
            self.timings.add('policy_device_stream_elapsed',self.events[1].elapsed_time(self.events[2])/1000,len(observations))
        self.timings.add('d2h_and_host_synchronization',time.perf_counter()-returning,len(observations))
        sync(self.device)
        self.timings.add("policy_inference_and_return", time.perf_counter()-start, len(observations))
        return result

class Population:
    def __init__(self, config, model, device, states=None):
        self.config, self.device = config, device
        self.history = []
        for state in states or []:
            self.add(model, state)
    def add(self, model, state=None):
        frozen = Policy(model.hidden, model.embedding_size).to(self.device)
        frozen.load_state_dict(state or model.state_dict())
        frozen.eval()
        frozen.requires_grad_(False)
        self.history.append(frozen)
        self.history = self.history[-self.config.get("history_size", 4):]
    def states(self):
        return [{k:v.detach().cpu() for k,v in m.state_dict().items()} for m in self.history]
    def assignments(self, n, seed_offset, rng, model):
        options = self.config["opponents"]
        weights = np.asarray([x["weight"] for x in options], dtype=float)
        weights /= weights.sum()
        result = []
        for i in range(n):
            spec = options[int(rng.choice(len(options), p=weights))]
            kind = spec["kind"]
            if kind == "selfplay":
                result.append([model, model])
                continue
            if kind == "historical":
                opponent = self.history[int(rng.integers(len(self.history)))] if self.history else model
            elif kind in ("random", "power"):
                opponent = kind
            else:
                raise ValueError(f"Unknown opponent kind: {kind}")
            pair = [opponent, opponent]
            pair[(seed_offset+i)%2] = model
            result.append(pair)
        return result

def rollout(vector, model, assignments, seeds, cfg, rng, inference, timings, deadline=None,
            deterministic=False, swapped=None):
    if hasattr(inference,"begin_rollout"):inference.begin_rollout()
    start = time.perf_counter()
    states = vector.reset(seeds, cfg["max_decision_phases"], swapped)
    timings.add("reset_and_team_generation", time.perf_counter()-start, len(seeds))
    trajectories = [[[], []] for _ in seeds]
    opponent_rngs = [[np.random.default_rng(np.random.SeedSequence([int(seed), seat, 8675309])) for seat in range(2)] for seed in seeds]
    phases = 0
    total_ready_decisions=0;late_policy_samples=0
    worker_start = vector.seconds
    forced_stop = False
    while any(not (s["terminated"] or s["truncated"]) for s in states):
        if deadline is not None and time.perf_counter() >= deadline:
            forced_stop = True
            break
        actions = [[None, None] for _ in seeds]
        groups = defaultdict(list)
        scripts=[]
        for i, state in enumerate(states):
            for seat, ready in enumerate(state["ready"]):
                if not ready:
                    continue
                total_ready_decisions+=1
                policy = assignments[i][seat]
                obs = state["observations"][seat]
                if isinstance(policy, str):
                    if cfg.get('batched_scripted',False):scripts.append((i,seat,obs,policy))
                    else:actions[i][seat] = scripted(obs, opponent_rngs[i][seat], policy)
                else:
                    groups[policy].append((i, seat, obs))
        if scripts:
            acts=scripted_batch([r[2] for r in scripts],[opponent_rngs[r[0]][r[1]] for r in scripts],[r[3] for r in scripts])
            for (i,seat,_,_),action in zip(scripts,acts,strict=True):actions[i][seat]=action
        for policy, rows in groups.items():
            acts, logps, values = inference(policy, [r[2] for r in rows], deterministic)
            for (i, seat, obs), action, logp, value in zip(rows, acts, logps, values, strict=True):
                actions[i][seat] = int(action)
                if policy is model:
                    late_policy_samples+=int(obs['global'][0]>=.05)
                    trajectories[i][seat].append({"obs":obs, "action":int(action), "logp":float(logp),
                                                  "value":float(value), "reward":0.0})
        next_states = vector.step(actions)
        phases += 1
        for i, (old, new) in enumerate(zip(states, next_states, strict=True)):
            if old["terminated"] or old["truncated"]:
                continue
            for seat in range(2):
                if trajectories[i][seat]:
                    reward = cfg.get("faint_reward", 0.0) * (new["potential"][seat]-old["potential"][seat])
                    if new["terminated"]:
                        reward += new["returns"][seat]
                    trajectories[i][seat][-1]["reward"] += reward
        states = next_states
    timings.add("simulation_step_ipc_encode", vector.seconds-worker_start, phases)
    samples = []
    returns = []
    for i, state in enumerate(states):
        if forced_stop and not state["terminated"] and not state["truncated"]:
            state = dict(state, truncated=True, reason="allocation_limit")
            states[i] = state
        for seat, trajectory in enumerate(trajectories[i]):
            if not trajectory:
                continue
            if state["terminated"]:
                last_value = 0.0
                returns.append(state["returns"][seat])
            else:
                # A time/decision cutoff bootstraps the last player-visible observation;
                # it is never converted into a loss/tie or a natural terminal.
                with torch.no_grad():
                    _, values = model(batch([state["observations"][seat]], inference.device))
                    last_value = float(values[0].item())
            advantage = 0.0
            for t in reversed(trajectory):
                delta = t["reward"] + cfg["gamma"]*last_value - t["value"]
                advantage = delta + cfg["gamma"]*cfg["gae_lambda"]*advantage
                t["advantage"] = advantage
                t["target"] = advantage + t["value"]
                last_value = t["value"]
            samples.extend(trajectory)
    report = {"battles":len(states),"natural_terminals":sum(s["terminated"] for s in states),
              "truncations":sum(s["truncated"] for s in states),"phases":phases,
              "policy_samples":len(samples),"total_ready_player_decisions":total_ready_decisions,"late_policy_samples_turn50":late_policy_samples,"mean_training_terminal_return":float(np.mean(returns)) if returns else None,
              "rejected_visible_choices":sum(sum(s["rejections"]) for s in states),
              "current_policy_rejections":sum(s["rejections"][seat] for i,s in enumerate(states) for seat in range(2) if assignments[i][seat] is model),
              "results":[{k:v for k,v in s.items() if k not in ("observations","ready","potential")} for s in states]}
    return samples, report

def update(model, optimizer, samples, cfg, device, timings):
    if not samples:
        return {"samples":0}
    efficient=cfg.get("optimization","baseline") != "baseline"
    if efficient and any(not np.any(t["obs"]["mask"]) for t in samples):
        raise ValueError("Waiting/terminal sample in PPO")
    start = time.perf_counter()
    if cfg.get('optimization')=='shared':
        from .efficient import batch_flat
        x=batch_flat([t['obs'] for t in samples],device)
    else:x = batch([t["obs"] for t in samples], device)
    actions = torch.tensor([t["action"] for t in samples], device=device)
    old = torch.tensor([t["logp"] for t in samples], device=device)
    target = torch.tensor([t["target"] for t in samples], device=device)
    adv = torch.tensor([t["advantage"] for t in samples], device=device)
    adv = (adv-adv.mean()) / (adv.std(unbiased=False)+1e-8)
    sync(device)
    timings.add("update_pack_and_transfer", time.perf_counter()-start, len(samples))
    start = time.perf_counter()
    losses, kls, entropies = [], [], []
    device_stats=[]
    for _ in range(cfg["ppo_epochs"]):
        for indices in torch.randperm(len(samples), device=device).split(cfg["minibatch"]):
            mini={k:v[indices] for k,v in x.items()}
            if efficient:
                logits,values=model(mini)
                dist=torch.distributions.Categorical(logits=logits,validate_args=False)
            else:
                dist,values=model.distribution(mini)
            logp = dist.log_prob(actions[indices])
            ratio = (logp-old[indices]).exp()
            policy_loss = -torch.minimum(ratio*adv[indices], ratio.clamp(1-cfg["clip"],1+cfg["clip"])*adv[indices]).mean()
            value_loss = .5*(values-target[indices]).square().mean()
            entropy = dist.entropy().mean()
            loss = policy_loss + cfg["value_coef"]*value_loss - cfg["entropy_coef"]*entropy
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = nn.utils.clip_grad_norm_(model.parameters(), cfg["max_grad_norm"])
            if not efficient and not torch.isfinite(norm):
                raise FloatingPointError("Non-finite PPO gradients")
            optimizer.step()
            kl=((ratio-1)-(logp-old[indices])).mean()
            if efficient:
                device_stats.append(torch.stack((loss.detach(),kl.detach(),entropy.detach(),norm.detach())))
            else:
                losses.append(float(loss.detach().cpu()))
                kls.append(float(kl.detach().cpu()))
                entropies.append(float(entropy.detach().cpu()))
    if device_stats:
        statistics=torch.stack(device_stats).cpu().numpy().astype(np.float64)
        if not np.isfinite(statistics[:,3]).all():
            raise FloatingPointError("Non-finite PPO gradients")
        losses,kls,entropies=statistics[:,:3].T
    sync(device)
    timings.add("ppo_forward_backward_update", time.perf_counter()-start, len(samples))
    return {"samples":len(samples),"loss":float(np.mean(losses)),"approx_kl":float(np.mean(kls)),"entropy":float(np.mean(entropies)),
            "advantage_std":float(np.std([t["advantage"] for t in samples])),"minibatch_updates":len(losses)}


def inference_for(cfg,timings):
    if cfg.get('optimization','baseline')=='baseline':return Inference(cfg['hardware']['device'],timings)
    from .efficient import EfficientInference,HybridInference
    if cfg.get('optimization')=='hybrid':return HybridInference(cfg['hardware']['device'],timings)
    return EfficientInference(cfg['hardware']['device'],timings,graph=cfg.get('optimization')=='graph',flat_copy=cfg.get('optimization') in ('stream','shared'),bulk_stack=cfg.get('optimization')=='shared')


def rollout_queue(vector, model, assignments, seeds, cfg, rng, inference, timings,
                  deadline=None, deterministic=False, swapped=None):
    """Work-conserving complete-episode scheduler, with one learner barrier.

    A completed slot immediately receives the next scheduled seed. Policies stay
    fixed for the whole collection and GAE never crosses a reset. Changing ready
    batch shapes changes stochastic sampling order; this is an explicit recipe
    change, not bitwise-equivalent stochastic training. Results/samples are sorted
    in schedule order before PPO, independent of completion order.
    """
    if len(seeds)<vector.size or len(assignments)!=len(seeds):
        raise ValueError('Queue requires assignments/seeds >= vector size')
    if hasattr(inference,'begin_rollout'):inference.begin_rollout()
    swapped=swapped or [False]*len(seeds)
    start=time.perf_counter()
    states=vector.reset(seeds[:vector.size],cfg['max_decision_phases'],swapped[:vector.size])
    timings.add('reset_and_team_generation',time.perf_counter()-start,vector.size)
    trajectories=[[[],[]] for _ in seeds];finals=[None]*len(seeds)
    slots=list(range(vector.size));next_job=vector.size
    opponent_rngs=[[np.random.default_rng(np.random.SeedSequence([int(seed),seat,8675309])) for seat in range(2)] for seed in seeds]
    phases=ready_count=late_samples=0;simulation_seconds=0.;forced_stop=False
    while any(not(s['terminated'] or s['truncated']) for s in states):
        if deadline is not None and time.perf_counter()>=deadline:
            forced_stop=True;break
        actions=[[None,None] for _ in slots];groups=defaultdict(list)
        for slot,state in enumerate(states):
            job=slots[slot]
            for seat,ready in enumerate(state['ready']):
                if not ready:continue
                ready_count+=1;policy=assignments[job][seat];obs=state['observations'][seat]
                if isinstance(policy,str):actions[slot][seat]=scripted(obs,opponent_rngs[job][seat],policy)
                else:groups[policy].append((slot,job,seat,obs))
        for policy,rows in groups.items():
            acts,logps,values=inference(policy,[r[3] for r in rows],deterministic)
            for (slot,job,seat,obs),action,logp,value in zip(rows,acts,logps,values,strict=True):
                actions[slot][seat]=int(action)
                if policy is model:
                    late_samples+=int(obs['global'][0]>=.05)
                    trajectories[job][seat].append(dict(obs=obs,action=int(action),logp=float(logp),value=float(value),reward=0.))
        begin=vector.seconds;new_states=vector.step(actions);simulation_seconds+=vector.seconds-begin;phases+=1
        replacements=[None]*vector.size
        for slot,(old,new) in enumerate(zip(states,new_states,strict=True)):
            if old['terminated'] or old['truncated']:continue
            job=slots[slot]
            for seat in range(2):
                if trajectories[job][seat]:
                    reward=cfg.get('faint_reward',0.)*(new['potential'][seat]-old['potential'][seat])
                    if new['terminated']:reward+=new['returns'][seat]
                    trajectories[job][seat][-1]['reward']+=reward
            if new['terminated'] or new['truncated']:
                finals[job]=new
                if next_job<len(seeds) and (deadline is None or time.perf_counter()<deadline):
                    replacements[slot]=(seeds[next_job],swapped[next_job]);slots[slot]=next_job;next_job+=1
        states=new_states
        if any(u is not None for u in replacements):
            begin=time.perf_counter();states=vector.reset_slots(replacements,cfg['max_decision_phases'])
            timings.add('reset_and_team_generation',time.perf_counter()-begin,sum(u is not None for u in replacements))
    timings.add('simulation_step_ipc_encode',simulation_seconds,phases)
    for slot,state in enumerate(states):
        job=slots[slot]
        if finals[job] is None:
            finals[job]=dict(state,truncated=True,reason='allocation_limit') if forced_stop else state
    finals=finals[:next_job];samples=[];returns=[]
    for job,state in enumerate(finals):
        if state is None:raise RuntimeError('Queue lost a scheduled episode')
        for seat,trajectory in enumerate(trajectories[job]):
            if not trajectory:continue
            if state['terminated']:
                last_value=0.;returns.append(state['returns'][seat])
            else:
                with torch.no_grad():
                    _,values=model(batch([state['observations'][seat]],inference.device));last_value=float(values[0].item())
            advantage=0.
            for t in reversed(trajectory):
                delta=t['reward']+cfg['gamma']*last_value-t['value']
                advantage=delta+cfg['gamma']*cfg['gae_lambda']*advantage
                t['advantage']=advantage;t['target']=advantage+t['value'];last_value=t['value']
            samples.extend(trajectory)
    return samples,dict(battles=len(finals),scheduled_battles=len(seeds),unstarted_battles=len(seeds)-next_job,
        natural_terminals=sum(s['terminated'] for s in finals),truncations=sum(s['truncated'] for s in finals),
        phases=phases,policy_samples=len(samples),total_ready_player_decisions=ready_count,late_policy_samples_turn50=late_samples,
        mean_training_terminal_return=float(np.mean(returns)) if returns else None,
        rejected_visible_choices=sum(sum(s['rejections']) for s in finals),
        current_policy_rejections=sum(s['rejections'][seat] for job,s in enumerate(finals) for seat in range(2) if assignments[job][seat] is model),
        results=[{k:v for k,v in s.items() if k not in ('observations','ready','potential')} for s in finals])
