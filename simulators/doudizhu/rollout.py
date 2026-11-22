"""Batched on-policy DMC waves, frozen/history mixtures and measured loop components."""
from __future__ import annotations

from collections import defaultdict
import time

import numpy as np

from .artifacts import load_checkpoint
from .environment import ROLES, deal, digest, heuristic


class Policies:
    def __init__(self, learner, device, factored=True, max_candidates=4096, encoding_cache=True,
                 encoding_backend='direct', skip_forced=False, inference_backend='eager'):
        self.learner = learner
        if learner is not None:
            learner.eval()
        self.device = device
        self.factored = factored
        self.max_candidates = max_candidates
        self.encoding_cache = encoding_cache
        self.encoding_backend = encoding_backend
        self.skip_forced = skip_forced
        self.inference_backend = inference_backend
        self.native_fast = False
        self.actor_rng = 'numpy-pcg64-legacy'
        self.cache = {}

    def retain(self, descriptors):
        live = {(d['path'], d['sha256']) for d in descriptors if d['kind'] == 'checkpoint'}
        self.cache = {key: policy for key, policy in self.cache.items() if key in live}

    def resolve(self, descriptor):
        kind = descriptor['kind']
        if kind == 'learner':
            return self.learner
        if kind in ('random', 'heuristic'):
            return kind
        if kind != 'checkpoint':
            raise ValueError(f'Unknown policy kind {kind}')
        # Hash on first use per run; resume revalidates every immutable descriptor.
        key = (descriptor['path'], descriptor['sha256'])
        if key not in self.cache:
            self.cache[key] = load_checkpoint(key[0], self.device, key[1])[0]
        return self.cache[key]

    def choose(self, observations, descriptors, rng, epsilon):
        actions = {}
        groups = defaultdict(list)
        for i in sorted(observations):
            obs = observations[i]
            local_rng = rng[i] if isinstance(rng, dict) else rng
            descriptor = descriptors[i][obs.role]
            policy = self.resolve(descriptor)
            if policy == 'random':
                actions[i] = int(local_rng.integers(len(obs.legal)))
            elif policy == 'heuristic':
                actions[i] = heuristic(obs)
            else:
                groups[id(policy)].append((i, obs, policy, descriptor['kind'] == 'learner'))
        candidate_count = 0
        for group in groups.values():
            policy = group[0][2]
            scored = [row for row in group if not self.skip_forced or len(row[1].legal) > 1]
            if scored:
                values = policy.values_many([row[1] for row in scored], self.max_candidates, self.factored,
                                            self.encoding_cache, self.encoding_backend,
                                            self.inference_backend)
            else:
                values = []
            by_id = {row[0]:q for row,q in zip(scored,values)}
            # Keep the original group and RNG order, including forced learner turns.
            # Every selected forced action remains a terminal-return learner sample.
            for i, obs, _, trainable in group:
                local_rng = rng[i] if isinstance(rng, dict) else rng
                candidate_count += len(obs.legal)
                if trainable and epsilon and local_rng.random() < epsilon:
                    actions[i] = int(local_rng.integers(len(obs.legal)))
                else:
                    actions[i] = int(np.argmax(by_id[i])) if i in by_id else 0
        return actions, candidate_count


def assign_training(config, n, cursor, rng, historical):
    result = []
    settings = config['opponents']
    for i in range(n):
        if rng.random() < settings['selfplay_probability']:
            result.append({r: {'kind': 'learner'} for r in ROLES})
            continue
        role = ROLES[(cursor + i) % 3]
        population = settings.get('frozen', []) + historical
        if not population:
            raise ValueError('Opponent mixture needs at least one frozen or historical policy')
        # Choose one consistent three-role checkpoint for both opponent seats in a game.
        opponent = population[int(rng.integers(len(population)))]
        result.append({r: {'kind': 'learner'} if r == role else opponent for r in ROLES})
    return result


def collect(vector, policies, cards, assignments, rng, objective='wp', epsilon=0.,
            budget=None, record=False, training=False):
    if vector.cuda_mode:
        from .cuda.rollout import collect_device
        return collect_device(vector,policies,cards,assignments,rng,objective,epsilon,budget,record,training)
    if policies.native_fast and vector.native is not None and all(
            all(d['kind']=='learner' for d in a.values()) for a in assignments):
        from .native_rollout import collect_native
        return collect_native(vector,policies,cards,assignments,rng,objective,epsilon,budget,record,training)
    timing = defaultdict(float)
    started = time.perf_counter()
    t = time.perf_counter()
    active = {i: item[0] for i, item in vector.reset(cards, objective, record).items()}
    timing['reset_simulation_ipc_seconds'] += time.perf_counter() - t
    pending = {i: {r: [] for r in ROLES} for i in active}
    counter=policies.actor_rng=='splitmix-counter-v1'
    if counter:
        from .cuda.actor_rng import actor_seeds,CounterGameRNG
        seeds=actor_seeds(rng,len(cards))
    turn=0
    outcomes, traces = {}, {}
    decisions = candidates = 0
    role_decisions = {r: 0 for r in ROLES}
    max_legal = 0
    while active:
        if budget and budget.expired():
            return {'aborted': True, 'decisions': decisions, 'candidates': candidates,
                    'timing': dict(timing), 'samples': {r: [] for r in ROLES},
                    'outcomes': {}, 'traces': {}, 'partial_completed_games': len(outcomes)}
        t = time.perf_counter()
        actor_rng={i:CounterGameRNG(seeds[i],turn) for i in active} if counter else rng
        actions, count = policies.choose(active, assignments, actor_rng, epsilon)
        timing['encoding_transfer_inference_action_seconds'] += time.perf_counter() - t
        candidates += count
        t = time.perf_counter()
        for i, obs in active.items():
            role_decisions[obs.role] += 1
            max_legal = max(max_legal, len(obs.legal))
            if training and assignments[i][obs.role]['kind'] == 'learner':
                x, z, action = obs.encode(policies.encoding_cache, policies.encoding_backend)
                pending[i][obs.role].append((np.concatenate((x, action[actions[i]])), z))
        timing['training_sample_encoding_seconds'] += time.perf_counter() - t
        t = time.perf_counter()
        replies = vector.step(actions)
        timing['simulation_observation_ipc_seconds'] += time.perf_counter() - t
        decisions += len(active)
        turn+=1
        for i, (obs, outcome, trace) in replies.items():
            if outcome is not None:
                outcomes[i] = outcome
                if trace is not None:
                    traces[i] = trace
                del active[i]
            else:
                active[i] = obs
    samples = {r: [] for r in ROLES}
    # Worker replies arrive in shard order. Training sample order must be independent
    # of process count so the learner's minibatch RNG selects the same experiences.
    for i in sorted(pending):
        for role in ROLES:
            samples[role].extend((x, z, outcomes[i]['returns'][role]) for x, z in pending[i][role])
    timing['rollout_total_seconds'] = time.perf_counter() - started
    return {'aborted': False, 'samples': samples, 'outcomes': outcomes, 'traces': traces,
            'decisions': decisions, 'candidates': candidates, 'role_decisions': role_decisions,
            'max_legal_candidates': max_legal, 'timing': dict(timing),
            'deal_sha256': [digest(c) for c in cards]}
