"""Episodic batched PPO collection: ready actions only, delayed rewards credited correctly.

Each cohort uses a fixed policy and runs each learner seat to completion. Cohort tails
are intentionally measured, not hidden. Fresh cohorts avoid stale partial trajectories.
Simulator faults discard all trajectories/results of the affected lobby in that cohort.
"""
import time
import json
import hashlib
from collections import Counter
from contextlib import nullcontext
import numpy as np
import taverngym as tg
from .bootstrap import GAME
from model import act
from adapter import STOP_REASONS, state_fault


class BudgetExpired(Exception):
    pass


class Timer:
    def __init__(self, device, synchronized=True):
        self.device = device
        self.synchronized = synchronized
        self.seconds = Counter()
        self.diagnostics = Counter()
        self.counters = Counter()
        self.context = []

    def sync(self):
        if not self.synchronized:
            return
        if str(self.device).startswith("cuda"):
            import torch
            torch.cuda.synchronize(self.device)
        elif str(self.device).startswith("mps"):
            import torch
            torch.mps.synchronize()

    def measure(self, name, diagnostic=False):
        timer = self
        class Span:
            def __enter__(self):
                timer.sync()
                self.start = time.perf_counter()
            def __exit__(self, *args):
                timer.sync()
                target = timer.diagnostics if diagnostic else timer.seconds
                target[name] += time.perf_counter() - self.start
        return Span()


def gae(trajectory, gamma, lam):
    """One entry per actual decision; rewards while waiting accumulate on its last action.

    All accepted trajectories end naturally, so the final bootstrap is exactly zero.
    Fault/external time-limit trajectories never reach this function.
    """
    advantage = np.zeros(len(trajectory), np.float32)
    carry, next_value = 0.0, 0.0
    for t in reversed(range(len(trajectory))):
        row = trajectory[t]
        delta = row["reward"] + gamma * next_value - row["value"]
        carry = delta + gamma * lam * carry
        advantage[t] = carry
        next_value = row["value"]
    return advantage, advantage + np.array([r["value"] for r in trajectory], np.float32)


def collect(policy, config, seed, device, timer, deadline, opponent=None,
            deterministic=False, copy_observations=False, assignment=None, population=()):
    collection_started = time.perf_counter()
    previous_stage_total = sum(timer.seconds.values())
    policy.fast_inference = config.get("fast_inference", False)
    policy.validated_sampler = config.get("validated_sampler", False)
    policy.graph_inference = config.get("graph_inference", False)
    policy.actor_graph_max_shapes = config.get("actor_graph_max_shapes", 64)
    policy.retain_actor_inputs = config.get("device_rollouts", False)
    policy.inference_metrics = Counter()
    for actor in population:
        if hasattr(actor, 'remote_act'):
            continue
        actor.fast_inference = config.get('fast_inference', False)
        actor.graph_inference = config.get('graph_inference', False)
        actor.actor_graph_max_shapes = config.get('actor_graph_max_shapes', 64)
        actor.validated_sampler = config.get('validated_sampler', False)
        actor.inference_metrics = Counter()
    if opponent is not None:
        opponent.fast_inference = config.get("fast_inference", False)
        opponent.validated_sampler = config.get("validated_sampler", False)
        opponent.graph_inference = config.get("graph_inference", False)
        opponent.actor_graph_max_shapes = config.get("actor_graph_max_shapes", 64)
        opponent.inference_metrics = Counter()
    seats = tuple(config["learning_seats"])
    controlled = seats if config["opponent"] == "scripted" else tuple(range(8))
    n_games = config["n_games"]
    with timer.measure("environment_init"):
        env = tg.BatchEnv(n_games, seed=int(seed), controlled=controlled,
                          config=config["rules"], threads=config["native_threads"])
    k, n = len(controlled), env.n_slots
    warm_info = {"target_turn": config.get("start_turn",1), "states": []}
    pre_finished = []
    if config.get("start_turn", 1) > 1:
        with timer.measure("environment_warmstart"):
            if config.get("native_warmstart", False):
                env.warm_start(config["start_turn"])
            for g in range(n_games):
                game = env.game(g)
                for guard in range(1000):
                    if game.turn >= config["start_turn"] or game.is_over(): break
                    for p in range(8):
                        if game.needs_action(p): game.bot_play(p)
                        if game.turn >= config["start_turn"] or game.is_over(): break
                else: raise RuntimeError("warm-start driver guard")
                state = json.loads(game.state_json())
                if state_fault(state): raise RuntimeError("fault while constructing busy workload")
                ended_before_profile = state["phase"] == "GameOver"
                if ended_before_profile or any(not state["players"][p]["alive"] for p in seats):
                    if not config.get("allow_profile_eliminated",False):
                        raise RuntimeError("busy-workload learning seat already eliminated; choose a valid fixed seed/turn")
                    pre_finished.extend(g*k+j for j,p in enumerate(controlled) if p in seats and
                        (ended_before_profile or not state["players"][p]["alive"]))
                if not config.get("native_warmstart", False):
                    env.load(g,json.dumps({"game":state,"index":0,"finished":[False]*k}))
                warm_info["states"].append({"game":g,"turn":game.turn,
                    "sha256":hashlib.sha256(game.state_json().encode()).hexdigest(),
                    "learning_alive":sum(state["players"][p]["alive"] for p in seats),
                    "board_sizes":[len(state["players"][p]["board"]) for p in seats]})
        timer.context.append(warm_info)
    learning = np.array([p in seats for _ in range(n_games) for p in controlled])
    if assignment is not None:
        if controlled != tuple(range(8)) or assignment.shape != (n_games,8):
            raise ValueError("population assignment requires eight controlled seats per lobby")
        learning = (assignment == -1).reshape(-1)
        if not learning.any() or not learning.reshape(n_games,8).any(1).all():
            raise ValueError("every lobby needs a current-policy learner")
    learning_columns = np.array([p in seats for p in controlled], dtype=bool)
    finished = np.zeros(n, dtype=bool)
    finished[pre_finished] = True
    if pre_finished:
        warm_info["excluded_preprofile_eliminated_seats"] = len(pre_finished)
    records, invalid, trajectories = [], set(), [[] for _ in range(n)]
    stats = Counter()
    chunks, device_rows = [], 0
    while not finished[learning].all():
        if time.perf_counter() >= deadline:
            raise BudgetExpired("allocated-worker deadline reached during rollout; partial cohort discarded")
        if stats["batch_steps"] >= config["max_batch_steps"]:
            raise RuntimeError("external rollout guard: cohort incomplete; no placements accepted")
        game_active = ((finished | ~learning).reshape(n_games,k).all(axis=1) == False
            if config.get("vector_bookkeeping", True) else
            np.array([not finished[g*k:(g+1)*k][learning[g*k:(g+1)*k]].all() for g in range(n_games)]))
        active_args = {"active_games": game_active} if config.get("skip_finished_lobbies", False) else {}
        with timer.measure("observation"):
            if config.get("sparse_observations", False):
                obs = env.observe_ready(copy=copy_observations, **active_args)
            else:
                obs = env.observe(copy=copy_observations, **active_args)
        if not np.array_equal(obs["ready"], obs["action_mask"].any(-1)):
            raise RuntimeError("ready/mask invariant failed")
        timer.counters["observation_serialized_rows"] += len(obs["ready"])
        timer.counters["observation_serialized_bytes"] += len(obs["ready"])*(tg.OBS_FLOATS*4+tg.OBS_CARDS*4+tg.N_ACTIONS+
            (8 if config.get("sparse_observations",False) else 1))
        active = np.repeat(game_active, k)
        if config.get("sparse_observations", False):
            slots = obs["slots"]
            if len(slots) and (slots[0]<0 or slots[-1]>=n or (np.diff(slots)<=0).any()):
                raise RuntimeError("compact observation slot mapping invariant failed")
            ready = np.zeros(n,dtype=bool)
            ready[slots] = obs["ready"]
            source_row = np.full(n,-1,dtype=np.int64)
            source_row[slots] = np.arange(len(slots))
        else:
            ready = obs["ready"].copy()
            source_row = np.arange(n)
        ready &= active & ~finished
        learner_indices = np.flatnonzero(ready & learning)
        learner_rows = source_row[learner_indices]
        # Legal fallback only for ignored/cohort-tail slots; never creates a learner sample.
        actions = (np.zeros(n, dtype=np.int64) if config.get("skip_finished_lobbies", False) and config.get("vector_bookkeeping", True)
                   else obs["action_mask"].argmax(-1).astype(np.int64))
        other = np.flatnonzero(ready & ~learning)
        if config.get('batched_policy_dispatch',False):
            if assignment is None:raise ValueError('Batched population dispatch requires explicit assignments')
            from model import act_many
            groups=[];destinations=[];scripts=[]
            if len(learner_indices):groups.append((policy,learner_rows));destinations.append(learner_indices)
            flat=assignment.reshape(-1)
            for identity in np.unique(flat[other]):
                selected=other[flat[other]==identity];actor=policy if identity==0 else population[int(identity)-1]
                if hasattr(actor,'remote_act'):scripts.append((actor,selected))
                else:groups.append((actor,source_row[selected]));destinations.append(selected)
            def run_scripts():
                with timer.measure('script_opponents',diagnostic=True):
                    for actor,selected in scripts:
                        actions[selected]=act(actor,obs,source_row[selected],device,deterministic)[0]
            with timer.measure('inference_transfer_sampling'):
                outputs=act_many(groups,obs,device,deterministic,while_pending=run_scripts)
                for selected,output in zip(destinations,outputs):actions[selected]=output[0]
                if len(learner_indices):
                    _,lp,val=outputs[0]
                    if config.get('device_rollouts',False):learner_inputs=policy.last_actor_inputs
            timer.diagnostics['batched_actor_frames']+=1
        with timer.measure("inference_transfer_sampling") if not config.get('batched_policy_dispatch',False) else nullcontext():
            if len(learner_indices) and not config.get('batched_policy_dispatch',False):
                with timer.measure("learner_actor", diagnostic=True):
                    a, lp, val = act(policy, obs, learner_rows, device, deterministic)
                actions[learner_indices] = a
                if config.get("device_rollouts", False):
                    learner_inputs = policy.last_actor_inputs
            if len(other) and not config.get('batched_policy_dispatch',False):
                if assignment is None:
                    a_other, _, _ = act(opponent or policy, obs, source_row[other], device, deterministic)
                    actions[other] = a_other
                else:
                    for identity in np.unique(assignment.reshape(-1)[other]):
                        selected=other[assignment.reshape(-1)[other] == identity]
                        actor=policy if identity == 0 else population[int(identity)-1]
                        actor.fast_inference=config.get("fast_inference",False)
                        # Each historical network/style receives one batch, retaining
                        # the original identity order and neural RNG consumption.
                        with timer.measure("script_opponents" if hasattr(actor,"remote_act") else "historical_opponents", diagnostic=True):
                            a_other,_,_=act(actor,obs,source_row[selected],device,deterministic)
                        actions[selected]=a_other
        with timer.measure("rollout_storage"):
            if len(learner_indices) and config.get("device_rollouts", False):
                chunks.append(learner_inputs)
            for j, slot in enumerate(learner_indices):
                obs_row = learner_rows[j]
                inputs = ({"_device_row": device_rows+j} if config.get("device_rollouts", False) else
                    {"features": obs["features"][obs_row].copy(), "cards": obs["cards"][obs_row].copy(),
                     "mask": obs["action_mask"][obs_row].copy()})
                trajectories[slot].append({**inputs,
                    "action": int(actions[slot]), "logp": float(lp[j]),
                    "value": float(val[j]), "reward": 0.0})
            device_rows += len(learner_indices)
        with timer.measure("environment_step"):
            reward, done, info = env.step(actions, **active_args)
        stats["cohort_tail_game_steps"] += int((~game_active).sum())
        stats["total_game_step_opportunities"] += n_games
        stats["active_learning_slot_steps"] += int((learning & active & ~finished).sum())
        stats["ready_turn_sum"] += float(obs["features"][learner_rows,0].sum()*20)
        stats["ready_board_count_sum"] += float(obs["features"][learner_rows,33].sum()*7)
        stats["ready_turn10_plus"] += int((obs["features"][learner_rows,0] >= .5).sum())
        stats["ready_five_minions_plus"] += int((obs["features"][learner_rows,33] >= 5/7-1e-6).sum())
        stats["batch_steps"] += 1
        stats["ready_learning_decisions"] += len(learner_indices)
        stats["ready_opponent_decisions"] += len(other)
        stats["rejected"] += int(info["rejected"][ready].sum())
        if info["rejected"][ready].any():
            raise RuntimeError("masked policy action rejected by simulator")
        # Earlier seats may create a new choice for a later seat in this joint
        # frame. Its stale response was not executed: remove the speculative row
        # before crediting rewards, then request a fresh visible observation.
        if "applied" not in info or "deferred" not in info:
            raise RuntimeError("native action-admission protocol unavailable")
        for slot in learner_indices[~info["applied"][learner_indices]]:
            trajectories[slot].pop()
        stats["applied_learning_decisions"] += int(info["applied"][learner_indices].sum())
        stats["unapplied_learning_decisions"] += int((~info["applied"][learner_indices]).sum())
        stats["deferred"] += int(info["deferred"][ready].sum())
        credited = learning & active & ~finished
        if config.get("vector_bookkeeping", True):credited &= reward != 0
        for slot in np.flatnonzero(credited):
            if trajectories[slot]:
                trajectories[slot][-1]["reward"] += float(reward[slot])
        for slot in np.flatnonzero(done & active & ~finished):
            finished[slot] = True
            reason = int(info["stop_reason"][slot])
            if reason != 1:
                invalid.add(int(slot // k))
                stats["fault_" + STOP_REASONS.get(reason, str(reason))] += 1
            if learning[slot]:
                records.append({"seed": int(seed), "game": int(slot // k),
                                "seat": int(controlled[slot % k]),
                                "placement": int(info["placement"][slot]),
                                "stop_reason": STOP_REASONS.get(reason, str(reason))})
    accepted, results = [], []
    for slot in np.flatnonzero(learning):
        if slot // k not in invalid and slot not in pre_finished:
            accepted.append(trajectories[slot])
    for record in records:
        record["valid"] = record["game"] not in invalid
        results.append(record)
    stats.update({"actor_"+key:value for key,value in policy.inference_metrics.items() if "seconds" not in key})
    timer.diagnostics.update({"actor_"+key:value for key,value in policy.inference_metrics.items() if "seconds" in key})
    if opponent is not None:
        stats.update({"opponent_"+key:value for key,value in opponent.inference_metrics.items() if "seconds" not in key})
        timer.diagnostics.update({"opponent_"+key:value for key,value in opponent.inference_metrics.items() if "seconds" in key})
    for actor in population:
        if hasattr(actor, 'remote_act'):
            continue
        stats.update({"population_"+key:value for key,value in actor.inference_metrics.items() if "seconds" not in key})
        timer.diagnostics.update({"population_"+key:value for key,value in actor.inference_metrics.items() if "seconds" in key})
    stats["invalid_lobbies"] = len(invalid)
    stats["accepted_learning_samples"] = sum(map(len, accepted))
    stats["completed_learning_episodes"] = len(accepted)
    with timer.measure("environment_save"):
        states = [env.save(g) for g in range(n_games)]
    if config.get("device_rollouts", False):
        from device_rollout import DeviceEpisodes
        accepted = DeviceEpisodes(accepted, chunks)
        if hasattr(policy, "last_actor_inputs"):
            del policy.last_actor_inputs
    measured = sum(timer.seconds.values()) - previous_stage_total
    timer.seconds["collection_python_overhead"] += max(0.0, time.perf_counter() - collection_started - measured)
    return accepted, results, dict(stats), states
