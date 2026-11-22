"""Matched compiled CPU→GPU and fully GPU-resident neural learning loops."""
import os
os.environ['JAX_PLATFORMS'] = 'cuda,cpu'
import argparse
import json
from pathlib import Path
import time
import jax
import jax.numpy as jnp
import numpy as np
from .cuda_experiment import cfg_for, trained_state, timed
from .efficiency import make_collect_optimized, make_iteration
from .learning import make_update, advantages
from .model import Policy
from .runtime import sources, write_json, HERE
from .split_device import make_collect_split, make_iteration_split


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--output', required=True)
    p.add_argument('--snapshot', required=True)
    p.add_argument('--batches', default='128,2048,8192')
    p.add_argument('--dtype', choices=('float32', 'bfloat16'), default='float32')
    a = p.parse_args()
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    cpu, gpu = jax.devices('cpu')[0], jax.devices('cuda')[0]
    write_json(out/'manifest.json', dict(source=sources(), cpu=str(cpu), gpu=str(gpu),
        gpu_kind=gpu.device_kind, jax=jax.__version__, platforms=os.environ['JAX_PLATFORMS'],
        contract=dict(gpu_count=1, cpu_physical_cores=4, ram_gib=16)))
    import tarfile
    with tarfile.open(out/'source.tgz', 'w:gz') as tar:
        for file in HERE.rglob('*'):
            if "autoresearch" in file.relative_to(HERE).parts:
                continue
            if file.is_file() and file.suffix in ('.py', '.json', '.lock'):
                tar.add(file, arcname='integrations/generals/'+str(file.relative_to(HERE)))
    results = []
    for b in map(int, a.batches.split(',')):
        cfg = cfg_for(b)
        cfg.update(packed_rollout=True, compact_opponent_inference=True,
                   historical_convolution='slice', sparse_reset_capacity=min(128, b),
                   compute_dtype=a.dtype, minibatches=max(1, b*32//8192))
        model = Policy(cfg['network_width'], compute_dtype=a.dtype)
        train, bank, c, pool, provenance = trained_state(cfg, model, a.snapshot)
        cfg['train_pool_size'] = pool.time.shape[0]
        resident = make_collect_optimized(model, cfg)
        for _ in range(3):
            c, _ = resident(train.params, bank, c, pool)
        jax.block_until_ready(c)
        t0 = time.monotonic()
        c_cpu = c._replace(states=jax.tree.map(lambda x: jax.device_put(x, cpu), c.states))
        pool_cpu = jax.tree.map(lambda x: jax.device_put(x, cpu), pool)
        jax.block_until_ready((c_cpu, pool_cpu))
        setup_copy = time.monotonic()-t0
        collectors = {'resident': resident, 'cpu_producer': make_collect_split(model, cfg)}
        inputs = {'resident': (c, pool), 'cpu_producer': (c_cpu, pool_cpu)}
        data, cold_collect = {}, {}
        for name, fn in collectors.items():
            (nc, batch), seconds = timed(fn, train.params, bank, *inputs[name])
            data[name] = nc, batch
            cold_collect[name] = seconds
        ca, da = data['resident']; cb, db = data['cpu_producer']
        exact_state = all(np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(jax.tree.leaves(ca), jax.tree.leaves(cb)))
        if not exact_state:
            raise AssertionError('CPU/GPU neural trajectory or RNG/cache parity failed')
        for key in ('action', 'packed_mask', 'packed_armies', 'packed_bits', 'packed_scalars',
                    'terminated', 'truncated', 'wins', 'losses', 'invalid', 'own_land'):
            np.testing.assert_array_equal(da[key], db[key])
        errors = {}
        for key in ('value', 'old_logp', 'next_value', 'reward'):
            aa, bb = np.asarray(da[key]), np.asarray(db[key])
            np.testing.assert_allclose(aa, bb, atol=5e-5, rtol=5e-5)
            errors[key] = float(np.max(np.abs(aa-bb)))
        for aa, bb in zip(advantages(da, cfg['gamma'], cfg['gae_lambda']), advantages(db, cfg['gamma'], cfg['gae_lambda'])):
            np.testing.assert_allclose(aa, bb, atol=5e-5, rtol=5e-5)
        fns = {'resident': make_iteration(model, cfg, 'optimized'), 'cpu_producer': make_iteration_split(model, cfg)}
        row = dict(batch=b, config=cfg, provenance=provenance, exact_trajectory_rng_history_cache=True,
                   max_abs_errors=errors, initial_cpu_state_pool_copy_seconds=setup_copy,
                   cold_collect_seconds=cold_collect, variants={name: dict(cold_iteration_seconds=timed(fn, train, bank, *inputs[name])[1], trials=[])
                       for name, fn in fns.items()})
        for repeat in range(7):
            for name in (('resident', 'cpu_producer') if repeat % 2 == 0 else ('cpu_producer', 'resident')):
                (trained, nc, metrics), seconds = timed(fns[name], train, bank, *inputs[name])
                values = np.asarray(metrics)
                if not np.isfinite(values).all() or values[9] or values[5] <= 0:
                    raise AssertionError('Nonfinite/illegal or zero-gradient architecture trial')
                row['variants'][name]['trials'].append(dict(seconds=seconds, metrics=values.tolist()))
        for value in row['variants'].values():
            value['median_seconds'] = float(np.median([x['seconds'] for x in value['trials']]))
            value['learner_samples_per_second'] = b*32/value['median_seconds']
        row['resident_speedup'] = row['variants']['cpu_producer']['median_seconds']/row['variants']['resident']['median_seconds']
        results.append(row)
        write_json(out/'comparison.json', dict(results=results,
            note='Same physical GPU and fixed CPU/RAM. CPU producer includes packed H2D observations, D2H actions/control, CPU step/reset/observation and GPU sampling/GAE/PPO. Seven alternating matched trained-policy trials. No per-game Python workers.'))
        print(json.dumps(dict(batch=b, variants={k:v['median_seconds'] for k,v in row['variants'].items()},
                              resident_speedup=row['resident_speedup'], parity=exact_state)), flush=True)


if __name__ == '__main__':
    main()
