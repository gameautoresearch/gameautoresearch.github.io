"""Independent subprocess train/evaluate comparison including cold worker costs.

Run under the shared local timing lock on CPU. GPU runs use a dedicated recorded
single-GPU allocation. Historical source archive is extracted into a new isolated
fixture directory and never used to bypass a current checkpoint's source checks.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--output', required=True)
    p.add_argument('--baseline-archive', required=True)
    p.add_argument('--device', choices=('cpu', 'gpu'), default='cpu')
    p.add_argument('--candidate', default='integrations/generals/recipes/cpu-fast.json')
    p.add_argument('--seeds', default='42,43,44')
    p.add_argument('--steps', type=int, default=128)
    p.add_argument('--resume-check', action='store_true')
    p.add_argument('--cli-check', action='store_true')
    a = p.parse_args()
    root = Path(__file__).resolve().parents[2]
    out = Path(a.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    with tarfile.open(out/'executed-source.tgz', 'w:gz') as tar:
        for file in (root/'integrations/generals').rglob('*'):
            if "autoresearch" in file.relative_to((root/'integrations/generals')).parts:
                continue
            if file.is_file() and file.suffix in ('.py', '.json', '.lock'):
                tar.add(file, arcname=str(file.relative_to(root)))
    old = out/'baseline-source'
    old.mkdir()
    with tarfile.open(a.baseline_archive) as tar:
        tar.extractall(old, filter='data')
    (old/'candidates').mkdir()
    (old/'candidates/generals-bots').symlink_to(root/'candidates/generals-bots', target_is_directory=True)
    candidate = json.loads(Path(a.candidate).read_text())
    base = json.loads((old/'integrations/generals/recipes/smoke.json').read_text())
    for key in base.keys() & candidate.keys():
        base[key] = candidate[key]
    for cfg in (base, candidate):
        cfg.update(num_envs=candidate['num_envs'], max_updates=a.steps, allocated_worker_seconds=3600,
                   hardware=candidate['hardware'])
    base['execution'] = 'optimized' if a.device == 'gpu' else 'baseline'
    rows = []
    if a.device == 'cpu':
        lock = (root/'evidence/local-performance.lock').open('a+')
        print('Waiting for evidence/local-performance.lock', flush=True)
        fcntl.flock(lock, fcntl.LOCK_EX)
    start = time.monotonic()
    for i, seed in enumerate(map(int, a.seeds.split(','))):
        for name in (('baseline', 'candidate') if i % 2 == 0 else ('candidate', 'baseline')):
            cfg = dict(base if name == 'baseline' else candidate, seed=seed)
            recipe = out/f'{seed}-{name}.json'
            recipe.write_text(json.dumps(cfg, indent=2)+'\n')
            folder = out/f'{seed}-{name}'
            cwd = old if name == 'baseline' else root
            env = dict(os.environ, PYTHONPATH=str(cwd)+':'+str(root/'candidates/generals-bots'),
                       JAX_COMPILATION_CACHE_DIR=str(out/f'cache-{name}'))
            command = [sys.executable, '-m', 'integrations.generals', 'train',
                       '--device', a.device, '--recipe', str(recipe), '--output', str(folder)]
            t0 = time.monotonic()
            with (out/f'{seed}-{name}.log').open('w') as log:
                process = subprocess.run(command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=600)
            elapsed = time.monotonic()-t0
            if process.returncode:
                raise RuntimeError(f'{name} seed {seed} failed; retained log at {out}')
            result = json.loads((folder/'result.json').read_text())
            logs = [json.loads(line) for line in (folder/'train.jsonl').read_text().splitlines()]
            row = dict(seed=seed, variant=name, config=cfg, subprocess_seconds=elapsed,
                       learner_samples=result['learner_samples_total'], result=result,
                       captures_won=sum(x['captures_won'] for x in logs), captures_lost=sum(x['captures_lost'] for x in logs),
                       caps=sum(x['truncations'] for x in logs), invalid=sum(x['invalid_actions'] for x in logs),
                       evaluation=json.loads((folder/'development-evaluation.json').read_text()),
                       samples_per_full_process_second=result['learner_samples_total']/elapsed,
                       metrics_finite=all(all(__import__('math').isfinite(x[k]) for k in ('loss', 'gradient_norm')) for x in logs))
            rows.append(row)
            (out/'comparison.json').write_text(json.dumps(dict(results=rows, elapsed_seconds=time.monotonic()-start), indent=2)+'\n')
            print(json.dumps(dict(seed=seed, variant=name, seconds=elapsed, samples=row['learner_samples'],
                                  invalid=row['invalid'], captures_won=row['captures_won'], captures_lost=row['captures_lost'], caps=row['caps'])), flush=True)
    if a.resume_check:
        # A fresh-process resume must match an uninterrupted candidate, including
        # source-bound checkpoint validation, optimizer, RNG and history caches.
        first = next(r for r in rows if r['variant'] == 'candidate')
        seed = first['seed']
        recipe = out/f'{seed}-candidate.json'
        folder = out/'resume-check'
        env = dict(os.environ, PYTHONPATH=str(root)+':'+str(root/'candidates/generals-bots'),
                   JAX_COMPILATION_CACHE_DIR=str(out/'cache-candidate'))
        for tag, extra in [('partial', ['--max-updates', str(a.steps//2)]),
                           ('resumed', ['--resume', str(folder/'latest.msgpack')])]:
            command = [sys.executable, '-m', 'integrations.generals', 'train', '--device', a.device,
                       '--recipe', str(recipe), '--output', str(folder)]+extra
            with (out/f'{tag}.log').open('w') as log:
                subprocess.run(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=600)
        # Read raw payloads so source binding remains exactly the one used by the
        # normal CLI above. CPU should be bitwise; CUDA qualification is numeric.
        from flax import serialization
        import numpy as np
        import jax
        x = serialization.msgpack_restore((folder/'latest.msgpack').read_bytes())
        y = serialization.msgpack_restore((out/f'{seed}-candidate/latest.msgpack').read_bytes())
        errors, exact = [], True
        for u, v in zip(jax.tree.leaves(x), jax.tree.leaves(y)):
            exact &= bool(np.array_equal(u, v))
            if np.issubdtype(np.asarray(u).dtype, np.inexact):
                errors.append(float(np.max(np.abs(u-v))))
                np.testing.assert_allclose(u, v, atol=5e-5, rtol=5e-5)
            else:
                np.testing.assert_array_equal(u, v)
        (out/'resume-verification.json').write_text(json.dumps(dict(all_payload_arrays_exact=exact, max_abs_error=max(errors)), indent=2)+'\n')
    if a.cli_check:
        folder = out/'cli'
        recipe = out/f'{rows[0]["seed"]}-candidate.json'
        checkpoint = folder/'train/latest.msgpack'
        commands = [
            ['train', '--max-updates', '16', '--output', str(folder/'train')],
            ['train', '--max-updates', '24', '--resume', str(checkpoint), '--output', str(folder/'train')],
            ['evaluate', '--checkpoint', str(checkpoint), '--split', 'practice',
             '--opponents', 'random,expansion', '--output', str(folder/'evaluation')],
            ['profile', '--output', str(folder/'profile')],
        ]
        folder.mkdir()
        env = dict(os.environ, PYTHONPATH=str(root)+':'+str(root/'candidates/generals-bots'),
                   JAX_COMPILATION_CACHE_DIR=str(out/'cache-candidate'))
        checks = []
        for i, cmd in enumerate(commands):
            t0 = time.monotonic()
            command = [sys.executable, '-m', 'integrations.generals', cmd[0],
                       '--recipe', str(recipe), '--device', a.device]+cmd[1:]
            with (folder/f'command-{i}.log').open('w') as log:
                result = subprocess.run(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=180)
            checks.append(dict(command=command, seconds=time.monotonic()-t0, exit_code=result.returncode))
            (folder/'commands.json').write_text(json.dumps(checks, indent=2)+'\n')
            if result.returncode:
                raise RuntimeError('CLI validation failed; previous training retained')
        command = [sys.executable, '-m', 'integrations.generals.replay', str(folder/'evaluation/replay.npz'),
                   '--output', str(folder/'replay-verification.json')]
        with (folder/'replay.log').open('w') as log:
            subprocess.run(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=180)


if __name__ == '__main__':
    main()
