"""Bounded, matched full-population training profiles; never a strength campaign.

All arms import the same explicitly verified trained parent. Baseline/optimized
pairs use identical cohorts, population schedules, RNG, PPO and checkpoint cadence.
Larger cohorts are separately labeled scaling experiments, not exact ablations.
"""
import argparse
import copy
import fcntl
import json
import platform
import time
from pathlib import Path
import torch
from .common import HERE, ROOT, binding, write
from .train import run
from ..validate import equal

CONTROLS = dict(device_rollouts=True, batched_scripts=True, preprocess_numeric=True)
PARITY_KEYS = ('policy', 'optimizer', 'population', 'rng', 'history', 'env_states',
               'next_cohort_seed', 'update', 'lineage')


def compare(left, right):
    a = torch.load(left, map_location='cpu', weights_only=False)
    b = torch.load(right, map_location='cpu', weights_only=False)
    checks = {k: equal(a[k], b[k]) for k in PARITY_KEYS}
    if not all(checks.values()):
        raise ValueError('Full-training parity failed: ' + str(checks))
    return checks


def execute(inputs, out, device='cpu', trials=2, updates=2, sizes=(48, 96, 192, 256),
            seconds=900, proof=False, graphs=False):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    c = json.loads((HERE/'curriculum-quarter.json').read_text())
    c.update(device=device, updates=updates, profile_synchronized=True)
    controls = dict(CONTROLS)
    if graphs:
        if not device.startswith('cuda'):
            raise ValueError('CUDA graph qualification requires an actual CUDA device')
        controls.update(graph_inference=True, actor_graph_max_shapes=4)
    write(out/'manifest.json', dict(binding=binding(), inputs=str(Path(inputs).resolve()),
          torch=torch.__version__, platform=platform.platform(), device=device,
          trials=trials, updates=updates, sizes=list(sizes), seconds=seconds,
          optimized_controls=controls, full_from_start=True,
          opponent_mixture='Four current-policy learners, three frozen visible styles, one original retained/history selection per lobby'))
    results = []

    def arm(config, name, resume=None, limit=None):
        remaining = seconds-(time.perf_counter()-started)
        if remaining < 10:
            raise TimeoutError('Finite performance-suite deadline exhausted')
        write(out/(name+'-config.json'), config)
        checkpoint = run(config, out/name, inputs, remaining, resume=resume, limit_updates=limit)
        result = json.loads((checkpoint.parent/'result.json').read_text())
        profile = json.loads((checkpoint.parent/'profile.json').read_text())
        if result['termination'] != 'update_limit':
            raise TimeoutError('Partial suite arm; retained but not a completed profile')
        ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
        counts = [h['counts'] for h in ck['history'] if h['update'] > result['initial_update']]
        row = dict(name=name, n_games=config['n_games'], result=result, profile=profile,
                   accepted_samples=result['new_samples'],
                   trainer_session_samples_per_second=result['new_samples']/result['elapsed_seconds'],
                   natural_episodes=sum(x['completed_learning_episodes'] for x in counts),
                   faults=sum(x['invalid_lobbies'] for x in counts),
                   rejected=sum(x['rejected'] for x in counts),
                   late_decisions=sum(x['ready_turn10_plus'] for x in counts),
                   busy_decisions=sum(x['ready_five_minions_plus'] for x in counts))
        results.append(row)
        write(out/'results.json', results)
        del ck
        if row['faults'] or row['rejected']:
            raise ValueError('Faulted full-game profile; preserve diagnostics and stop')
        return checkpoint

    if proof:
        q = dict(c, **controls, n_games=2, population_snapshot_every=1, updates=4)
        full = arm(q, 'proof-full')
        half = arm(q, 'proof-half', limit=2)
        resumed = arm(q, 'proof-resumed', resume=half, limit=2)
        write(out/'resume-parity.json', compare(full, resumed))
        bad = torch.load(half, map_location='cpu', weights_only=False)
        bad['binding']['native_source_hash'] = 'invalid-source'
        torch.save(bad, out/'must-reject.pt')
        try:
            run(q, out/'must-reject', inputs, 10, resume=out/'must-reject.pt', limit_updates=1)
        except ValueError as exc:
            if 'Exact source/config/identity' not in str(exc):
                raise
            write(out/'source-rejection.json', dict(rejected=True, reason=str(exc)))
        else:
            raise ValueError('Source drift was admitted')
    for trial in range(trials):
        configs = dict(baseline=dict(c, n_games=48), optimized=dict(c, **controls, n_games=48))
        paths = {}
        for name in (('baseline', 'optimized') if trial%2 == 0 else ('optimized', 'baseline')):
            paths[name] = arm(configs[name], f'{name}-48-trial{trial}')
        write(out/f'parity-trial{trial}.json', compare(paths['baseline'], paths['optimized']))
    for size in sizes:
        if size != 48:
            arm(dict(c, **controls, n_games=size), f'optimized-{size}')
    write(out/'completed.json', dict(elapsed_seconds=time.perf_counter()-started,
          completed_arms=len(results), all_complete=True,
          full_allocation_scope='Outer provider allocation, including interpreter startup and teardown, is recorded separately.'))
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--device', default='cpu')
    p.add_argument('--trials', type=int, default=2)
    p.add_argument('--updates', type=int, default=2)
    p.add_argument('--sizes', type=int, nargs='+', default=[48, 96, 192, 256])
    p.add_argument('--seconds', type=float, default=900)
    p.add_argument('--proof', action='store_true')
    p.add_argument('--graphs', action='store_true')
    a = p.parse_args()
    if a.trials < 1 or a.updates < 1 or any(x not in range(1,257) for x in a.sizes):
        p.error('Positive bounded trials/updates and cohorts of at most256 required')
    with (ROOT/'evidence/local-performance.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        execute(a.inputs, a.out, a.device, a.trials, a.updates, a.sizes, a.seconds, a.proof, a.graphs)


if __name__ == '__main__':
    main()
