"""Current-source packed GPU learner, exact saved arrays and numeric continuation."""
import time
START = time.monotonic()
import argparse
import json
from pathlib import Path
import tarfile
import os
import subprocess
import sys
import jax
import jax.numpy as jnp
import numpy as np
from .runtime import HERE, sources, write_json, Ledger, save_checkpoint, read_checkpoint, hardware
from .learning import create_train_state, initial_carry
from .environment import build_pool
from .model import Policy
from .efficiency import make_iteration, make_collect_optimized
from .structural_experiment import neural_action_simulator_profile


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--output', required=True)
    p.add_argument('--recipe', default='integrations/generals/recipes/l4-frontier.json')
    a = p.parse_args()
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    cfg = json.loads(Path(a.recipe).read_text())
    # Stress the larger qualified cohort even when the L4 throughput recipe uses
    # the smaller, equally fast 2,048-board working set.
    cfg.update(num_envs=8192, minibatches=32, max_updates=16, allocated_worker_seconds=600)
    binding = sources()
    write_json(out/'source-manifest.json', binding)
    with tarfile.open(out/'executed-source.tgz', 'w:gz') as tar:
        for file in HERE.rglob('*'):
            if "autoresearch" in file.relative_to(HERE).parts:
                continue
            if file.is_file() and file.suffix in ('.py', '.json', '.lock'):
                tar.add(file, arcname='integrations/generals/'+str(file.relative_to(HERE)))
    hw = hardware(cfg, 'gpu' if jax.default_backend() == 'gpu' else 'cpu')
    model = Policy(cfg['network_width'], convolution=cfg.get('convolution', 'direct'), compute_dtype=cfg.get('compute_dtype', 'float32'))
    key, ik = jax.random.split(jax.random.PRNGKey(cfg['seed']))
    train = create_train_state(model, ik, cfg)
    from .autoresearch.contract import seed_manifest
    seed_data = seed_manifest()
    pool = build_pool(seed_data['train'])
    bank = jax.tree.map(lambda p: jnp.repeat(p[None], cfg['history_size'], 0), train.params)
    carry = initial_carry(pool, key, cfg, bank)
    iteration = make_iteration(model, cfg, 'optimized')
    logs, cursor = [], 0
    for u in range(1, 17):
        t0 = time.monotonic()
        train, carry, metrics = iteration(train, bank, carry, pool)
        jax.block_until_ready((train, carry, metrics))
        m = np.asarray(metrics)
        if not np.isfinite(m).all() or m[9]:
            raise AssertionError('Invalid/nonfinite current-source training')
        logs.append(dict(update=u, seconds=time.monotonic()-t0, metrics=m.tolist()))
        if u % cfg['history_interval'] == 0:
            bank = jax.tree.map(lambda b, p: b.at[cursor % cfg['history_size']].set(p), bank, train.params)
            cursor += 1
    checkpoint = out/'latest.msgpack'
    meta = save_checkpoint(checkpoint, train, bank, carry, pool, cfg, binding, 16, Ledger(START, 600), cursor)
    _, restored = read_checkpoint(checkpoint, train, cfg, binding)
    expected = train, bank, carry, pool
    arrays_exact = all(bool(jnp.array_equal(x, y)) for x, y in zip(jax.tree.leaves(expected), jax.tree.leaves(restored)))
    if not arrays_exact:
        raise AssertionError('Indexed checkpoint restore changed arrays')
    rt, rb, rc, rp = restored
    next_expected = iteration(train, bank, carry, pool)
    next_actual = iteration(rt, rb, rc, rp)
    jax.block_until_ready((next_expected, next_actual))
    simulator_exact = all(bool(jnp.array_equal(x, y)) for x, y in zip(jax.tree.leaves(next_expected[1]), jax.tree.leaves(next_actual[1])))
    if not simulator_exact:
        raise AssertionError('Restored simulator/RNG/opponent continuation differs')
    max_error = 0.
    for x, y in zip(jax.tree.leaves(next_expected), jax.tree.leaves(next_actual)):
        if jnp.issubdtype(x.dtype, jnp.inexact):
            max_error = max(max_error, float(jnp.max(jnp.abs(x-y))))
            if not bool(jnp.allclose(x, y, atol=5e-5, rtol=5e-5)):
                raise AssertionError('Resumed floating update exceeds tolerance')
        elif not bool(jnp.array_equal(x, y)):
            raise AssertionError('Resumed discrete state differs')
    traced = dict(cfg, trace_actions=True)
    final, data = make_collect_optimized(model, traced)(train.params, bank, carry, pool)
    phases = neural_action_simulator_profile(carry, final, pool, data, traced, 7)
    result = dict(config=cfg, hardware=hw, source=binding,
        logs=logs, checkpoint_format=meta['format'], checkpoint_bytes=checkpoint.stat().st_size,
        unique_history_rows=meta['unique_history_rows'], checkpoint_arrays_exact=arrays_exact,
        resumed_simulator_rng_opponent_cache_exact=simulator_exact, resumed_update_max_abs_error=max_error,
        invalid_actions=0, neural_action_simulator=phases,
        allocated_worker_seconds=time.monotonic()-START,
        device_memory_stats=jax.devices()[0].memory_stats())
    write_json(out/'validation.json', result)
    if jax.default_backend() == 'gpu':
        cpu_cfg = dict(cfg, num_envs=128, minibatches=4, max_updates=4,
                       simulation_device='cpu', compute_dtype='float32', evaluate_before_after=False,
                       historical_convolution='slice', checkpoint_interval=2, allocated_worker_seconds=600)
        recipe = out/'cpu-producer-recipe.json'
        write_json(recipe, cpu_cfg)
        folder = out/'cpu-cli'
        folder.mkdir()
        ckpt = folder/'train/latest.msgpack'
        commands = [
            ['train', '--device', 'gpu', '--max-updates', '2', '--output', str(folder/'train')],
            ['train', '--device', 'gpu', '--max-updates', '4', '--resume', str(ckpt), '--output', str(folder/'train')],
            ['evaluate', '--device', 'gpu', '--checkpoint', str(ckpt), '--split', 'practice',
             '--opponents', 'random,expansion', '--output', str(folder/'gpu-evaluation')],
            ['evaluate', '--device', 'cpu', '--checkpoint', str(ckpt), '--split', 'practice',
             '--opponents', 'random,expansion', '--output', str(folder/'cpu-evaluation')],
            ['profile', '--device', 'gpu', '--output', str(folder/'profile')],
        ]
        checks = []
        for i, command in enumerate(commands):
            t0 = time.monotonic()
            with (folder/f'command-{i}.log').open('w') as log:
                completed = subprocess.run([sys.executable, '-m', 'integrations.generals', command[0],
                    '--recipe', str(recipe)]+command[1:], stdout=log, stderr=subprocess.STDOUT, timeout=180)
            checks.append(dict(command=command, seconds=time.monotonic()-t0, exit_code=completed.returncode))
            write_json(folder/'commands.json', checks)
            if completed.returncode:
                raise RuntimeError('CPU-producer public CLI check failed; resident validation already retained')
        for device in ('cpu', 'gpu'):
            env = dict(os.environ, JAX_PLATFORMS='cpu' if device == 'cpu' else 'cuda')
            with (folder/f'{device}-replay.log').open('w') as log:
                subprocess.run([sys.executable, '-m', 'integrations.generals.replay',
                    str(folder/f'{device}-evaluation/replay.npz'), '--output', str(folder/f'{device}-replay.json')],
                    env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=120)
        result['cpu_producer_cli'] = checks
        result['allocated_worker_seconds'] = time.monotonic()-START
        write_json(out/'validation.json', result)
    print(json.dumps(dict(checkpoint_arrays_exact=arrays_exact, resumed_simulator_exact=simulator_exact,
                          max_error=max_error, seconds=time.monotonic()-START)), flush=True)


if __name__ == '__main__':
    main()
