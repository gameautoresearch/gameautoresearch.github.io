"""Game-local train/evaluate/profile/replay commands. All paths stay game-local."""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from .artifacts import (Budget, HERE, checkpoint_descriptor, load_checkpoint,
                        manifest, save_checkpoint, sha256, verify_source, write_json, contract_hash)
from .environment import ROOT, ROLES, RULESET, Game, deal, digest, replay, seed_value
from .model import Policy, update
from .rollout import Policies, assign_training, collect
from .workers import VectorGames


def resolve(path):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def validate(config):
    if config['ruleset'] != RULESET:
        raise ValueError('Unknown frozen ruleset')
    if config['device'] not in ('cpu', 'cuda:0'):
        raise ValueError('Use cpu or cuda:0 on one allocated GPU')
    if config['allocation']['gpus'] not in (0, 1):
        raise ValueError('This benchmark allocates at most one GPU per worker')
    if config['device'] == 'cuda:0':
        if config['allocation']['gpus'] != 1 or not torch.cuda.is_available():
            raise ValueError('CUDA recipe requires one real allocated GPU')
    for key in ('vector_games', 'torch_threads', 'max_seconds', 'max_candidates'):
        if config[key] <= 0:
            raise ValueError(f'{key} must be positive')
    if not 0 <= config['workers'] <= config['vector_games']:
        raise ValueError('workers must be between zero and vector_games')
    if config['objective'] not in ('wp', 'adp', 'logadp'):
        raise ValueError('Unknown return objective')
    if config.get('engine_backend', 'view') not in ('view', 'upstream', 'native','cuda'):
        raise ValueError('Unknown engine backend')
    if config.get('encoding_backend', 'direct') not in ('direct', 'upstream'):
        raise ValueError('Unknown public encoder')
    if config.get('engine_backend')=='native' and config.get('encoding_backend','direct')!='direct':
        raise ValueError('Native engine requires direct public features')
    if config.get('native_fast',False) and (config.get('engine_backend')!='native' or
            config.get('inference_backend','eager')!='eager' or not config.get('factored_history',True)):
        raise ValueError('Bulk native rollout requires native engine and factored eager inference')
    if config.get('engine_backend')=='cuda':
        if config['device']!='cuda:0' or config['workers']!=0 or config.get('inference_backend','eager')!='eager':
            raise ValueError('Resident simulator requires cuda:0, zero CPU workers and resident eager inference')
        config.setdefault('actor_rng','splitmix-counter-v1')
        if config['actor_rng']!='splitmix-counter-v1':
            raise ValueError('Unknown resident actor RNG protocol')
    elif config.get('actor_rng','numpy-pcg64-legacy')!='numpy-pcg64-legacy':
        if config['actor_rng']!='splitmix-counter-v1':
            raise ValueError('Unknown CPU actor RNG protocol')
    if config.get('learner_shuffle','numpy') not in ('numpy','torch'):
        raise ValueError('Unknown learner shuffle protocol')
    if config.get('matmul_precision','ieee') not in ('ieee','tf32'):
        raise ValueError('Unknown head matmul precision')
    if config.get('compute_dtype','float32') not in ('float32','bfloat16'):
        raise ValueError('Unknown neural compute dtype')
    if config.get('compute_dtype')=='bfloat16' and (config['device']!='cuda:0' or config.get('inference_backend','eager')=='graph'):
        raise ValueError('BF16 compute requires CUDA eager/resident inference')
    if config.get('inference_backend', 'eager') not in ('eager', 'graph'):
        raise ValueError('Unknown inference backend')
    if config.get('inference_backend', 'eager') == 'graph' and (
            config['device'] != 'cuda:0' or not config.get('factored_history',True)):
        raise ValueError('Graph inference requires factored history on CUDA')
    if config.get('compact_workers',False) and (not config['workers'] or config.get('encoding_backend','direct')!='direct'):
        raise ValueError('Compact transport requires direct encoding and process workers')
    for key in ('seed', 'deal_seed'):
        if config[key] < 0:
            raise ValueError(f'{key} must be nonnegative')
    if 'opponents' in config:
        if not 0 <= config['opponents']['selfplay_probability'] <= 1:
            raise ValueError('Invalid self-play probability')
        for descriptor in config['opponents'].get('frozen', []):
            validate_descriptor(descriptor)
        if config['opponents'].get('historical_every_waves',0) and config['opponents'].get('historical_size',0) < 1:
            raise ValueError('Historical population needs positive capacity')
    if not 0 <= config.get('epsilon',0) <= 1:
        raise ValueError('Exploration probability must be between zero and one')
    if 'architecture' in config and any(v < 1 for v in config['architecture'].values()):
        raise ValueError('Architecture dimensions must be positive')
    return config


def validate_descriptor(descriptor):
    if descriptor['kind'] not in ('random', 'heuristic', 'checkpoint'):
        raise ValueError('Unsupported opponent')
    if descriptor['kind'] == 'checkpoint':
        if len(descriptor.get('sha256', '')) != 64:
            raise ValueError('Frozen checkpoint needs an explicit SHA256')
        descriptor['path'] = str(resolve(descriptor['path']))
        if sha256(descriptor['path']) != descriptor['sha256']:
            raise ValueError('Frozen checkpoint does not match declared hash')


def initialize(config):
    torch.set_num_threads(config['torch_threads'])
    torch.use_deterministic_algorithms(True)
    if config['device'].startswith('cuda'):
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = config.get('matmul_precision','ieee')=='tf32'
    torch.manual_seed(config['seed'])
    return np.random.default_rng(config['seed'])


def make_optimizers(policy, config):
    return {r: torch.optim.RMSprop(policy.roles[r].parameters(),
                                   lr=config['learning_rate'], alpha=.99, eps=1e-5)
            for r in ROLES}


def log_line(out, filename, item):
    with (out / filename).open('a') as f:
        f.write(json.dumps(item, sort_keys=True, allow_nan=False) + '\n')


def train(config, args, started_perf, started_unix):
    rng = initialize(config)
    out = resolve(args.out)
    if (out / 'manifest.json').exists() and not args.resume:
        raise ValueError('Output run already exists; use a new directory or explicit --resume')
    out.mkdir(parents=True, exist_ok=True)
    state = {'episodes': 0, 'next_deal': 0, 'waves': 0, 'decisions': 0,
             'candidate_evaluations': 0, 'updates': {r: 0 for r in ROLES},
             'samples': {r: 0 for r in ROLES}, 'historical': [], 'timing': {},
             'allocated_worker_seconds': 0., 'aborted_waves': 0, 'used_deal_hashes': []}
    if args.resume:
        policy, data = load_checkpoint(resolve(args.resume), config['device'])
        if config.get('actor_rng','numpy-pcg64-legacy')!=data['config'].get('actor_rng','numpy-pcg64-legacy'):
            raise ValueError('Resume changes actor RNG protocol; start a fresh experiment')
        immutable = [k for k in config if k not in {
            'max_seconds', 'max_episodes', 'budget_gpu_hours', 'device', 'allocation',
            'workers', 'torch_threads', 'development_evaluation','engine_backend','encoding_backend',
            'native_fast','record_games','compact_workers','inference_backend'}]
        if any(config[k] != data['config'].get(k) for k in immutable):
            raise ValueError('Resume changes training semantics; start a fresh run instead')
        state = data['state']
        reservation=resolve(args.resume).parent/'deal-reservation.json'
        if reservation.exists():
            reserved=json.loads(reservation.read_text())
            if (reserved['ruleset']!=RULESET or reserved['contract_sha256']!=contract_hash()
                    or reserved['deal_seed']!=config['deal_seed']):
                raise ValueError('Resume deal reservation contract mismatch')
            if reserved['next_deal']>state['next_deal']:
                state['next_deal']=reserved['next_deal']
                state['aborted_waves']+=1
        rng.bit_generator.state = state['numpy_rng']
        torch.set_rng_state(data['torch_rng'].cpu())
        if data['cuda_rng'] and config['device'].startswith('cuda'):
            torch.cuda.set_rng_state_all([x.cpu() for x in data['cuda_rng']])
        optimizers = make_optimizers(policy, config)
        for r in ROLES:
            optimizers[r].load_state_dict(data['optimizers'][r])
        for descriptor in state['historical']:
            validate_descriptor(descriptor)
    else:
        policy = Policy(config['architecture']).to(config['device'])
        optimizers = make_optimizers(policy, config)
    policy.compute_dtype=config.get('compute_dtype','float32')
    budget = Budget(config, started_perf, started_unix,
                    max(state['allocated_worker_seconds'], args.prior_worker_seconds), args.allocated_start_unix)
    policies = Policies(policy, config['device'], config.get('factored_history', True), config['max_candidates'],
                        encoding_backend=config.get('encoding_backend', 'direct'),
                        skip_forced=config.get('skip_forced',False),
                        inference_backend=config.get('inference_backend','eager'))
    policies.native_fast=config.get('native_fast',False)
    policies.actor_rng=config.get('actor_rng','numpy-pcg64-legacy')
    # Verify frozen populations before starting episodes, including when self-play happens
    # to be sampled for the first wave. Do not silently substitute random opponents.
    for descriptor in config['opponents'].get('frozen', []) + state['historical']:
        policies.resolve(descriptor)
    write_json(out / 'manifest.json', manifest(config))
    initial_parameters = {k: p.detach().cpu().clone() for k, p in policy.state_dict().items()}

    def checkpoint(path):
        state['numpy_rng'] = rng.bit_generator.state
        state['allocated_worker_seconds'] = budget.elapsed()
        save_checkpoint(path, policy, optimizers, state, config)

    if not args.resume:
        checkpoint(out / 'initial.pt')
    last_metrics = None
    with VectorGames(config['workers'], engine_backend=config.get('engine_backend', 'view'),
                     compact=config.get('compact_workers',False),record_limit=config.get('record_games')) as vector:
        while state['episodes'] < config['max_episodes'] and not budget.expired():
            n = min(config['vector_games'], config['max_episodes'] - state['episodes'])
            cursor = state['next_deal']
            cards = [deal('train', config['deal_seed'], cursor + i) for i in range(n)]
            assignments = assign_training(config, n, cursor, rng, state['historical'])
            # Reserve the entire wave; an interrupted/aborted wave never reuses deals.
            state['next_deal'] += n
            # Persist before the first simulator call. A hard kill or native fault
            # can leave the last checkpoint behind this cursor, but resume must not
            # reuse the attempted deals or erase that failed work.
            write_json(out/'deal-reservation.json',{'ruleset':RULESET,'contract_sha256':contract_hash(),
                'deal_seed':config['deal_seed'],'next_deal':state['next_deal'],'wave':state['waves']})
            try:
                result = collect(vector, policies, cards, assignments, rng, config['objective'],
                                 config['epsilon'], budget, record=state['waves'] == 0, training=True)
            except BaseException as exc:
                log_line(out,'events.jsonl',{'event':'unscored_failed_wave','reserved_cursor':cursor,
                    'reserved_deals':n,'error_type':type(exc).__name__})
                raise
            state['decisions'] += result['decisions']
            state['candidate_evaluations'] += result['candidates']
            timing = defaultdict(float, state['timing'])
            for k, v in result['timing'].items():
                timing[k] += v
            if result['aborted']:
                state['aborted_waves'] += 1
                state['timing'] = dict(timing)
                log_line(out, 'events.jsonl', {'event': 'budget_abort_wave', 'cursor': cursor,
                                             'reserved_deals': n, 'decisions': result['decisions']})
                break
            t = time.perf_counter()
            policy.train()
            last_metrics = update(policy, optimizers, result['samples'], config['batch_size'],
                                  config['epochs'], rng, config['max_grad_norm'],
                                  shuffle_backend=config.get('learner_shuffle','numpy'))
            policy.eval()
            timing['learner_update_seconds'] += time.perf_counter() - t
            state['timing'] = dict(timing)
            state['episodes'] += n
            state['waves'] += 1
            state['used_deal_hashes'].extend(result['deal_sha256'])
            for r in ROLES:
                state['updates'][r] += last_metrics['updates'][r]
                state['samples'][r] += last_metrics['samples'][r]
            row = {'event': 'trained_wave', 'episodes': state['episodes'],
                   'wave': state['waves'], 'decisions': state['decisions'],
                   'budget': budget.report(), 'metrics': last_metrics,
                   'landlord_win_rate': float(np.mean([o['landlord_won'] for o in result['outcomes'].values()])),
                   'assignments_sha256': digest(assignments), 'deal_sha256': result['deal_sha256']}
            log_line(out, 'events.jsonl', row)
            if result['traces']:
                write_json(out / 'private-first-wave-trajectories.json', list(result['traces'].values()))
            every = config['opponents'].get('historical_every_waves', 0)
            if every and state['waves'] % every == 0:
                path = out / f'historical-{state["waves"]:06d}.pt'
                checkpoint(path)
                state['historical'].append(checkpoint_descriptor(path))
                state['historical'] = state['historical'][-config['opponents']['historical_size']:]
                policies.retain(config['opponents'].get('frozen', []) + state['historical'])
            t = time.perf_counter()
            checkpoint(out / 'latest.pt')
            state['timing']['checkpoint_io_seconds'] = state['timing'].get('checkpoint_io_seconds', 0.) + time.perf_counter() - t
            if config.get('development_evaluation') and state['waves'] % config['development_evaluation']['every_waves'] == 0:
                settings = config['development_evaluation']
                eval_config = copy.deepcopy(config)
                eval_config.update(split='practice', deals=settings['deals'],
                                   opponents_panel=settings.get('opponents', [{'kind': 'heuristic'}]))
                t = time.perf_counter()
                rows, _ = evaluate_policy(eval_config, policy, budget)
                log_line(out, 'development-evaluation.jsonl', {'wave': state['waves'], 'rows': rows})
                state['timing']['development_evaluation_seconds'] = state['timing'].get('development_evaluation_seconds', 0.) + time.perf_counter() - t
            print(json.dumps({'episodes': state['episodes'], 'mse': last_metrics['mse'],
                              'allocated_worker_seconds': budget.elapsed()}), flush=True)
    delta = sum(float((p.detach().cpu() - initial_parameters[k]).square().sum()) for k, p in policy.state_dict().items()) ** .5
    checkpoint(out / 'latest.pt')
    summary = {'status': 'budget_stopped' if budget.expired() else 'episode_cap_reached',
               'state': state, 'last_metrics': last_metrics, 'parameter_l2_change_this_invocation': delta,
               'checkpoint': checkpoint_descriptor(out / 'latest.pt'), 'budget': budget.report(),
               'errors': {'illegal_actions': 0, 'simulator_faults': 0, 'nonfinite': 0},
               'claim': 'local functional learning smoke only; no competitive-strength claim'}
    write_json(out / 'result.json', summary)
    print(json.dumps({'result': str(out / 'result.json'), 'budget': budget.report()}), flush=True)


def summarize_evaluation(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row['opponent_id'], row['candidate_role'])].append(row)
    result = []
    for (opponent, role), group in sorted(groups.items()):
        n = len(group)
        wins = sum(row['won'] for row in group)
        p = wins / n
        # Wilson interval is descriptive per role. Paired comparisons bootstrap deal IDs.
        z = 1.959963984540054
        center = (p + z*z/(2*n)) / (1 + z*z/n)
        half = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n)
        result.append({'opponent_id': opponent, 'candidate_role': role, 'games': n,
                       'wins': wins, 'win_rate': p, 'win_rate_wilson95': [center-half, center+half],
                       'average_economic_score': float(np.mean([r['score'] for r in group])),
                       'average_training_objective_return': float(np.mean([r['return'] for r in group]))})
    return result


def evaluate_policy(config, policy, budget):
    policies = Policies(policy, config['device'], config.get('factored_history', True), config['max_candidates'],
                        encoding_backend=config.get('encoding_backend', 'direct'),
                        skip_forced=config.get('skip_forced',False),
                        inference_backend=config.get('inference_backend','eager'))
    policies.native_fast=config.get('native_fast',False)
    policies.actor_rng=config.get('actor_rng','numpy-pcg64-legacy')
    jobs = []
    for opponent_id, descriptor in enumerate(config['opponents_panel']):
        validate_descriptor(descriptor)
        policies.resolve(descriptor)
        for index in range(config['deals']):
            cards = deal(config['split'], config['deal_seed'], index)
            for role in ROLES:
                jobs.append((opponent_id, index, role, cards, descriptor))
    rows, traces = [], []
    with VectorGames(config['workers'], engine_backend=config.get('engine_backend', 'view'),
                     compact=config.get('compact_workers',False),record_limit=config.get('record_games')) as vector:
        for start in range(0, len(jobs), config['vector_games']):
            if budget.expired():
                break
            batch = jobs[start:start+config['vector_games']]
            assignments = [{r: {'kind': 'learner'} if r == role else desc for r in ROLES}
                           for _, _, role, _, desc in batch]
            rngs = {i: np.random.default_rng(seed_value(config['split'], config['seed'], index) % 2**64
                                            ^ int(digest([opponent_id, role])[:16], 16))
                    for i, (opponent_id, index, role, _, _) in enumerate(batch)}
            result = collect(vector, policies, [job[3] for job in batch], assignments, rngs,
                             config['objective'], budget=budget, record=start == 0)
            if result['aborted']:
                break
            for i, (opponent_id, index, role, cards, _) in enumerate(batch):
                outcome = result['outcomes'][i]
                rows.append({'opponent_id': opponent_id, 'deal_index': index,
                             'deal_sha256': digest(cards), 'candidate_role': role,
                             'won': outcome['landlord_won'] == (role == 'landlord'),
                             'score': outcome['scores'][role], 'return': outcome['returns'][role],
                             'bombs': outcome['bombs'], 'decisions': outcome['decisions']})
            traces.extend(result['traces'].values())
    return rows, traces


def evaluate(config, args, started_perf, started_unix):
    if config['split'] not in ('practice', 'heldout'):
        raise ValueError('Evaluation requires practice or heldout domain, never training deals')
    initialize(config)
    out = resolve(args.out)
    if (out / 'result.json').exists():
        raise ValueError('Evaluation output already exists')
    out.mkdir(parents=True, exist_ok=True)
    policy, checkpoint = load_checkpoint(resolve(args.checkpoint), config['device'])
    policy.compute_dtype=config.get('compute_dtype','float32')
    budget = Budget(config, started_perf, started_unix, allocated_start_unix=args.allocated_start_unix)
    write_json(out / 'manifest.json', manifest(config))
    rows, traces = evaluate_policy(config, policy, budget)
    used = set(checkpoint['state'].get('used_deal_hashes', []))
    if used.intersection(row['deal_sha256'] for row in rows):
        raise RuntimeError('Actual evaluation/training deal overlap')
    write_json(out / 'games.json', rows)
    write_json(out / 'private-trajectories.json', traces)
    expected = len(config['opponents_panel']) * config['deals'] * 3
    write_json(out / 'result.json', {
        'status': 'complete' if len(rows) == expected else 'incomplete_budget',
        'games': len(rows), 'expected_games': expected,
        'checkpoint': checkpoint_descriptor(resolve(args.checkpoint)),
        'summary': summarize_evaluation(rows), 'budget': budget.report(),
        'paired_deal_schedule_sha256': digest([[r['opponent_id'], r['deal_sha256'], r['candidate_role']] for r in rows]),
        'aggregation': 'report each role separately; peasant win is shared team win; '
                       'economic landlord score is twice each peasant score; do not pool raw scores across seats',
        'errors': {'illegal_actions': 0, 'simulator_faults': 0, 'truncations': 0}})
    print(json.dumps({'result': str(out / 'result.json'), 'games': len(rows), 'budget': budget.report()}))


def profile(config, args, started_perf, started_unix):
    # Matched full training loops: same weights, seeds, deals, optimizer and RNG.
    # Timed region includes worker startup, rollout, updates, checkpoint I/O and teardown.
    # Run alternating order to expose warmup/ordering noise. Import/setup charged separately.
    initialize(config)
    budget = Budget(config, started_perf, started_unix, allocated_start_unix=args.allocated_start_unix)
    out = resolve(args.out)
    if (out / 'result.json').exists():
        raise ValueError('Profile output already exists')
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / 'manifest.json', manifest(config))
    trials = []
    for repeat in range(config['profile_repeats']):
        variants = [(0, False, False), (0, True, False), (0, True, True), (config.get('profile_workers', 2), True, True)]
        if config.get('engine_backend')=='cuda':
            variants=[(0,True,True)]
        if repeat % 2:
            variants.reverse()
        for workers, factored, cached in variants:
            if budget.expired():
                break
            started = time.perf_counter()
            rng = initialize(config)
            policy = Policy(config['architecture']).to(config['device'])
            optimizer = make_optimizers(policy, config)
            policies = Policies(policy, config['device'], factored, config['max_candidates'], encoding_cache=cached)
            n = config['vector_games']
            cards = [deal('train', config['deal_seed'], i) for i in range(n)]
            assignments = [{r: {'kind': 'learner'} for r in ROLES} for _ in range(n)]
            with VectorGames(workers, cache_observations=cached,
                             engine_backend=config.get('engine_backend','view'),
                             record_limit=config.get('record_games')) as vector:
                result = collect(vector, policies, cards, assignments, rng, config['objective'],
                                 config['epsilon'], budget, record=True, training=True)
                if result['aborted']:
                    break
                update_start = time.perf_counter()
                metrics = update(policy, optimizer, result['samples'], config['batch_size'], config['epochs'], rng,
                                 shuffle_backend=config.get('learner_shuffle','numpy'))
                update_seconds = time.perf_counter() - update_start
                path = out / f'profile-{repeat}-{workers}-{int(factored)}-{int(cached)}.pt'
                save_checkpoint(path, policy, optimizer, {'episodes': n}, config)
            elapsed = time.perf_counter() - started
            trial = {'repeat': repeat, 'workers': workers, 'factored_history': factored,
                     'engine_backend':config.get('engine_backend','view'),
                     'public_encoding_and_view_cache': cached,
                     'full_loop_seconds': elapsed, 'decisions_per_second': result['decisions']/elapsed,
                     'episodes': n, 'decisions': result['decisions'], 'candidates': result['candidates'],
                     'timing': result['timing'], 'learner_update_seconds': update_seconds,
                     'metrics': metrics, 'max_legal_candidates': result['max_legal_candidates'],
                     'trajectory_sha256': digest(result['traces']),
                     'trained_model_sha256': digest({k: v.detach().cpu().tolist() for k, v in policy.state_dict().items()})}
            trials.append(trial)
            print(json.dumps({k: trial[k] for k in ('repeat', 'workers', 'factored_history', 'full_loop_seconds')}), flush=True)
    write_json(out / 'result.json', {'trials': trials, 'budget': budget.report(),
                                    'correctness': 'trajectory hashes must match across variants; '
                                                   'floating-point parameter rounding can differ after factoring',
                                    'simulation': 'CPU Python; CUDA, when selected, is policy inference and learner only'})


def main(started_perf, started_unix):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['train', 'evaluate', 'profile', 'replay', 'manifest'])
    parser.add_argument('--recipe', default='integrations/doudizhu/recipes/smoke.json')
    parser.add_argument('--out', default='evidence/doudizhu-implementation/run')
    parser.add_argument('--resume')
    parser.add_argument('--checkpoint')
    parser.add_argument('--trajectory')
    parser.add_argument('--seed-file', help='Evaluator-only private JSON seed manifest; evaluation only')
    parser.add_argument('--allocated-start-unix', type=float,
                        help='start of an already allocated lease; includes prelaunch setup/transfer time')
    parser.add_argument('--prior-worker-seconds', type=float, default=0., help=argparse.SUPPRESS)
    args = parser.parse_args()
    config = validate(json.loads(resolve(args.recipe).read_text()))
    if args.seed_file:
        if args.command != 'evaluate':
            parser.error('--seed-file is restricted to evaluation')
        private_path = resolve(args.seed_file)
        seeds = json.loads(private_path.read_text())
        if seeds['split'] != 'heldout':
            raise ValueError('Private seed manifests must use heldout domain')
        config.update(split='heldout',deal_seed=seeds['deal_seed'],seed=seeds['policy_seed'],deals=seeds['deals'],
                      private_seed_manifest_sha256=sha256(private_path))
        validate(config)
    verify_source()
    out = resolve(args.out)
    try:
        if args.command == 'train':
            train(config, args, started_perf, started_unix)
        elif args.command == 'evaluate':
            if not args.checkpoint:
                parser.error('evaluate requires --checkpoint')
            evaluate(config, args, started_perf, started_unix)
        elif args.command == 'profile':
            profile(config, args, started_perf, started_unix)
        elif args.command == 'manifest':
            initialize(config)
            write_json(out / 'manifest.json', manifest(config))
        elif args.command == 'replay':
            if not args.trajectory:
                parser.error('replay requires --trajectory (JSON object or list)')
            data = json.loads(resolve(args.trajectory).read_text())
            items = data if isinstance(data, list) else [data]
            for trajectory in items:
                replay(trajectory)
            print(json.dumps({'replayed': len(items), 'status': 'exact_public_hashes_and_returns'}))
    except Exception as exc:
        # A failed simulator/policy run is not reported as losses or successful training.
        write_json(out / 'failure.json', {'status': 'failed', 'error_type': type(exc).__name__,
                                         'message': str(exc), 'wall_seconds': time.monotonic()-started_perf})
        raise
