#!/usr/bin/env python3
"""Controlled sequential copy/thread ablation including real neural updates.

Run after all other local workloads finish. Each trial uses the same seeds and
checks bitwise weights, accepted samples and placements against the copying path.
This is a local CPU measurement, not a GPU calibration.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--trials', type=int, default=3)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    here = Path(__file__).resolve().parent
    runs = []
    # Interleave modes to reduce a monotonic thermal/order effect.
    for trial in range(args.trials):
        modes = [(1, 1), (1, 0), (4, 1), (4, 0)]
        if trial % 2:
            modes.reverse()
        for threads, copying in modes:
            run = args.out / f'trial{trial}-t{threads}-copy{copying}'
            config = here / f'recipes/profile-t{threads}-copy{copying}.json'
            with (args.out / f'{run.name}.log').open('w') as log:
                subprocess.run([sys.executable, str(here/'run.py'), 'profile', '--config', str(config), '--out', str(run)], stdout=log, stderr=subprocess.STDOUT, check=True)
            profile = json.loads((run/'profile.json').read_text())
            accounting = json.loads((run/'accounting.json').read_text())
            metrics = [json.loads(line) for line in (run/'metrics.jsonl').read_text().splitlines()]
            counts = json.loads((run/'training.json').read_text())['counts']
            loop = sum(profile['stage_seconds'].values())
            runs.append({'trial': trial, 'threads': threads, 'copy': bool(copying),
                'run': str(run), 'stage_seconds': profile['stage_seconds'],
                'loop_seconds': loop, 'session_seconds': accounting['session_allocated_worker_seconds'],
                'accepted_samples': counts['accepted_learning_samples'],
                'ready_decisions': counts['ready_learning_decisions'],
                'ready_decisions_per_loop_second': counts['ready_learning_decisions']/loop,
                'episodes': [row['episodes'] for row in metrics]})
    import torch
    checks = []
    for trial in range(args.trials):
        trial_runs = [r for r in runs if r['trial'] == trial]
        base = next(r for r in trial_runs if r['threads'] == 1 and r['copy'])
        weights = torch.load(Path(base['run'])/'latest.pt', weights_only=False, map_location='cpu')['policy']
        for run in trial_runs:
            other = torch.load(Path(run['run'])/'latest.pt', weights_only=False, map_location='cpu')['policy']
            same = all(torch.equal(weights[k], other[k]) for k in weights)
            check = {'trial': trial, 'threads': run['threads'], 'copy': run['copy'],
                     'identical_weights': same, 'identical_episodes': base['episodes'] == run['episodes'],
                     'identical_sample_count': base['accepted_samples'] == run['accepted_samples']}
            checks.append(check)
            if not all(check[k] for k in ['identical_weights','identical_episodes','identical_sample_count']):
                raise RuntimeError(f'ablation changed behavior: {check}')
    import statistics
    summary = {}
    for threads in [1,4]:
        for copying in [True,False]:
            rows = [r for r in runs if r['threads']==threads and r['copy']==copying]
            summary[f't{threads}-copy{int(copying)}'] = {
                'median_loop_seconds': statistics.median(r['loop_seconds'] for r in rows),
                'median_ready_decisions_per_loop_second': statistics.median(r['ready_decisions_per_loop_second'] for r in rows),
                'median_observation_seconds': statistics.median(r['stage_seconds']['observation'] for r in rows),
                'min_loop_seconds': min(r['loop_seconds'] for r in rows),
                'max_loop_seconds': max(r['loop_seconds'] for r in rows)}
    (args.out/'summary.json').write_text(json.dumps({'runs': runs, 'summary': summary,
        'correctness': checks, 'outer_allocated_worker_seconds': time.perf_counter()-started,
        'measurement_kind':'local_cpu_validation','cuda_measured':False},indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    import fcntl
    lock_path = Path(__file__).resolve().parents[2] / 'evidence/local-performance.lock'
    with lock_path.open('a+') as lock:
        print('waiting for exclusive local performance lock', flush=True)
        fcntl.flock(lock, fcntl.LOCK_EX)
        main()
