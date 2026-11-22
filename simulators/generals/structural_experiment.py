"""Local structural comparison, serialized against every other game's timings.

This performs no cloud launch. A retained checkpoint is an explicit migration
fixture (same frozen engine/rules), never a source-bound resume. Output provenance
includes its payload SHA, current source, config and exact starting state arrays.
"""
import argparse
import fcntl
import json
import os
import platform
import resource
import time
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_DEFAULT_MATMUL_PRECISION", "highest")
import jax
import jax.numpy as jnp
import numpy as np
from .cuda_experiment import cfg_for, trained_state, timed, phase_profile
from .learning import make_collect, make_update
from .efficiency import make_collect_optimized, make_iteration
from .model import Policy
from .runtime import ROOT, sources, write_json, digest, compact_history


def simulator_profile(carry, pool, cfg, repeats):
    from .environment import observations, action_mask, decode_action, transition, reset_done
    from .learning import sample
    b, t = cfg['num_envs'], cfg['rollout_steps']
    @jax.jit
    def run(states, key, maps):
        def tick(c, _):
            states, key = c
            key, ka, kr = jax.random.split(key, 3)
            masks = action_mask(observations(states))
            ids = sample(ka, jnp.where(masks, 0., -1e9))
            boundary = transition(states, decode_action(ids))
            done = boundary.terminated | boundary.truncated
            states = reset_done(boundary.final_state, done, maps, jax.random.randint(kr, (b,), 0, maps.time.shape[0]))
            invalid = ~jnp.take_along_axis(masks, ids[..., None], -1)[..., 0]
            return (states, key), jnp.array([invalid.sum(), boundary.terminated.sum(), boundary.truncated.sum()])
        return jax.lax.scan(tick, (states, key), None, length=t)
    result, cold = timed(run, carry.states, carry.key, pool)
    trials = []
    for _ in range(repeats):
        result, seconds = timed(run, carry.states, carry.key, pool)
        if int(result[1][:, 0].sum()):
            raise AssertionError('Simulator profile emitted an illegal action')
        trials.append(seconds)
    median = float(np.median(trials))
    return dict(cold_seconds=cold, warm_seconds=trials, median_seconds=median,
                game_ticks_per_second=b*t/median, invalid_actions=0,
                captures=int(result[1][:, 1].sum()), caps=int(result[1][:, 2].sum()),
                workload='Real trained-policy starting boards; random legal joint actions, simulation/reset/visible mask construction; no network or PPO.')


def neural_action_simulator_profile(carry, expected, pool, data, cfg, repeats):
    """Time the actual recorded neural action/reset stream, then verify final boards."""
    from .environment import observations, encode, action_mask, decode_action, transition, reset_done
    b, t = cfg['num_envs'], cfg['rollout_steps']
    results = {}
    for visible in (False, True):
        @jax.jit
        def run(states, actions, reset_ids, maps):
            def tick(states, pair):
                ids, indices = pair
                signature = jnp.float32(0)
                if visible:
                    obs = observations(states)
                    x, mask = encode(obs), action_mask(obs)
                    # Force every actor-visible plane and legal bit to be built;
                    # the scalar is diagnostic and never enters a network.
                    signature = x.sum()+mask.sum()
                boundary = transition(states, decode_action(ids))
                states = reset_done(boundary.final_state, boundary.terminated|boundary.truncated, maps, indices)
                return states, signature
            return jax.lax.scan(tick, states, (actions, reset_ids))
        args = carry.states, data['joint_action'], data['reset_map_ids'], pool
        (final, signature), cold = timed(run, *args)
        for x, y in zip(jax.tree.leaves(final), jax.tree.leaves(expected.states)):
            if not bool(jnp.array_equal(x, y)):
                raise AssertionError('Neural-action simulator replay differs from collection')
        trials = [timed(run, *args)[1] for _ in range(repeats)]
        median = float(np.median(trials))
        results['transition_reset_visible_encoding_masks' if visible else 'transition_reset'] = dict(
            cold_seconds=cold, warm_seconds=trials, median_seconds=median,
            game_ticks_per_second=b*t/median, exact_final_states=True)
    return results


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--output', required=True)
    p.add_argument('--snapshot', default='evidence/generals-implementation/validated-smoke/latest.msgpack')
    p.add_argument('--batches', default='16,64')
    p.add_argument('--repeats', type=int, default=7)
    p.add_argument('--variants', default='direct-baseline,direct-optimized,patch-baseline,patch-optimized')
    p.add_argument('--phases', action='store_true')
    p.add_argument('--minibatch-size', type=int, default=0,
                   help='0 retains four minibatches; positive value fixes samples per Adam step')
    p.add_argument('--massive', action='store_true', help='Use packed/GEMM inference for shared preconditioning at large batch sizes')
    a = p.parse_args()
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    variants = a.variants.split(',')
    source = sources()
    write_json(out/'manifest.json', dict(source=source, snapshot=a.snapshot,
        snapshot_sha256=digest(Path(a.snapshot).read_bytes()), args=vars(a),
        platform=platform.platform(), machine=platform.machine(), cpu_count=os.cpu_count(),
        devices=[str(d) for d in jax.devices()], jax=jax.__version__,
        timing_lock=str(ROOT/'evidence/local-performance.lock'),
        xla_flags=os.environ.get('XLA_FLAGS', ''),
        note='CPU is local development. CUDA must run inside a recorded fixed allocation.'))
    # Save all executed integration inputs: source digests alone are insufficient
    # to reproduce an uncommitted, shared-checkout experiment later.
    import tarfile
    with tarfile.open(out/'source.tgz', 'w:gz') as tar:
        for file in (ROOT/'integrations/generals').rglob('*'):
            if "autoresearch" in file.relative_to((ROOT/'integrations/generals')).parts:
                continue
            if file.is_file() and file.suffix in ('.py', '.json', '.lock'):
                tar.add(file, arcname=str(file.relative_to(ROOT)))
    results = []
    (ROOT/'evidence').mkdir(exist_ok=True)
    print('Waiting for evidence/local-performance.lock', flush=True)
    with (ROOT/'evidence/local-performance.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        start = time.monotonic()
        print('Acquired timing lock', flush=True)
        for b in map(int, a.batches.split(',')):
            cfg = cfg_for(b)
            if a.minibatch_size:
                n = b*cfg['rollout_steps']
                if n % a.minibatch_size:
                    raise ValueError('minibatch-size must divide rollout samples')
                cfg['minibatches'] = n//a.minibatch_size
            train, bank, c, pool, provenance = trained_state(cfg, Policy(cfg['network_width']), a.snapshot)
            # A real trained-policy trajectory populates late-game boards and
            # cached immutable historical weights, with normal captures/resets.
            advance_cfg = dict(cfg, packed_rollout=True, historical_convolution='slice', compact_opponent_inference=True) if a.massive else cfg
            advance = make_collect_optimized(Policy(cfg['network_width']), advance_cfg) if a.massive else make_collect(Policy(cfg['network_width']), cfg)
            for _ in range(3):
                c, _ = advance(train.params, bank, c, pool)
            jax.block_until_ready(c)
            unique_history, history_rows = compact_history(c.history_params)
            saved_c = c._replace(history_params=unique_history)
            arrays = {str(i): np.asarray(x) for i, x in enumerate(jax.tree.leaves((train.params, bank, saved_c, pool)))}
            if history_rows is not None:
                arrays['history_row_ids'] = np.asarray(history_rows)
            np.savez_compressed(out/f'start-{b}.npz', **arrays)
            write_json(out/f'start-{b}.json', dict(
                format='Flattened (params,bank,carry,pool) tree; expand carry.history_params with history_row_ids when present',
                tree=str(jax.tree.structure((train.params, bank, saved_c, pool))),
                unique_history_rows=int(jax.tree.leaves(unique_history)[0].shape[0]),
                lossless_reconstruction_verified=history_rows is not None))
            row = dict(batch=b, config=cfg, preconditioning_config=advance_cfg, provenance=provenance,
                times=np.asarray(c.states.time).tolist(),
                army=np.asarray(c.states.armies.sum((1, 2))).tolist(), variants={})
            row['simulator'] = simulator_profile(c, pool, cfg, a.repeats)
            expected = None
            fns = {}
            for name in variants:
                conv, execution = name.split('-')
                vcfg = dict(cfg)
                if conv == 'bf16':
                    conv = 'direct'
                    vcfg['compute_dtype'] = 'bfloat16'
                if execution in ('packed', 'histgemm', 'scaled', 'frontier', 'frontiertrace'):
                    vcfg['packed_rollout'] = True
                    if execution in ('histgemm', 'scaled', 'frontier', 'frontiertrace'):
                        vcfg.update(historical_convolution='slice', compact_opponent_inference=True)
                    if execution in ('frontier', 'frontiertrace'):
                        vcfg['sparse_reset_capacity'] = min(128, b)
                    if execution == 'frontiertrace':
                        vcfg['trace_actions'] = True
                    if execution == 'scaled':
                        vcfg['sampler'] = 'inverse_cdf'
                    execution = 'optimized'
                if execution in ('guarded', 'compact', 'hybrid', 'cdf', 'fast', 'unroll2', 'unroll4'):
                    vcfg['guard_resets'] = True
                    vcfg['compact_opponent_inference'] = execution not in ('guarded', 'hybrid', 'cdf')
                    if execution in ('cdf', 'fast'):
                        vcfg['sampler'] = 'inverse_cdf'
                    if execution == 'hybrid':
                        vcfg['learner_convolution'] = 'patch'
                    vcfg['rollout_unroll'] = {'unroll2': 2, 'unroll4': 4}.get(execution, 1)
                    execution = 'optimized'
                model = Policy(cfg['network_width'], convolution=conv, compute_dtype=vcfg.get('compute_dtype', 'float32'))
                collect = make_collect_optimized(model, vcfg) if execution == 'optimized' else make_collect(model, vcfg)
                (nc, data), cold_collect = timed(collect, train.params, bank, c, pool)
                if vcfg.get('trace_actions'):
                    row['neural_action_simulator'] = neural_action_simulator_profile(c, nc, pool, data, vcfg, a.repeats)
                comparable = data
                if vcfg.get('packed_rollout'):
                    from .packed import unpack_observation
                    px, pm = unpack_observation(data)
                    comparable = dict(data, x=px, mask=pm)
                if expected is None:
                    expected = nc, comparable
                errors = {}
                reference = expected
                changed_trajectory = (vcfg.get('sampler') == 'inverse_cdf' or vcfg.get('compute_dtype') == 'bfloat16'
                                      or vcfg.get('historical_convolution', conv) != conv)
                if changed_trajectory:
                    # This is a distribution-equivalent sampling algorithm, not
                    # an RNG-identical fast path. Compare to its own rerun only.
                    reference = timed(collect, train.params, bank, c, pool)[0]
                    if vcfg.get('packed_rollout'):
                        px, pm = unpack_observation(reference[1])
                        reference = reference[0], dict(reference[1], x=px, mask=pm)
                # Integer state/RNG/actions must match exactly for this fixed
                # one-rollout fixture. Do not claim long-run float identity.
                for x, y in zip(jax.tree.leaves(reference[0]), jax.tree.leaves(nc)):
                    if not bool(jnp.array_equal(x, y)):
                        raise AssertionError('State/RNG comparison failed')
                for key in ('action', 'mask', 'reward', 'terminated', 'truncated', 'invalid'):
                    if not bool(jnp.array_equal(reference[1][key], comparable[key])):
                        raise AssertionError(f'{key} comparison failed')
                for key in ('x', 'value', 'old_logp', 'next_value'):
                    x, y = reference[1][key], comparable[key]
                    if key == 'next_value':
                        x = jnp.where(data['terminated'], 0., x)
                        y = jnp.where(data['terminated'], 0., y)
                    if not bool(jnp.allclose(x, y, atol=5e-5, rtol=5e-5)):
                        raise AssertionError(f'{key} numeric comparison failed')
                    errors[key] = float(jnp.max(jnp.abs(x-y)))
                update = make_update(model, vcfg)
                _, cold_update = timed(update, train, data, jax.random.PRNGKey(80))
                fn = make_iteration(model, vcfg, execution)
                result, cold_iteration = timed(fn, train, bank, c, pool)
                fns[name] = fn, collect, update, data
                row['variants'][name] = dict(config=vcfg, cold_collect_seconds=cold_collect,
                    cold_update_seconds=cold_update, cold_iteration_seconds=cold_iteration,
                    max_abs_errors=errors, exact_state_rng_actions=True,
                    trajectory_comparison='own deterministic rerun' if changed_trajectory else 'baseline', trials=[])
                row['variants'][name]['rollout_storage_bytes'] = sum(x.nbytes for x in jax.tree.leaves(data))
            for repeat in range(a.repeats):
                for name in (variants if repeat % 2 == 0 else list(reversed(variants))):
                    fn, collect, update, data = fns[name]
                    (tr, nc, metrics), seconds = timed(fn, train, bank, c, pool)
                    m = np.asarray(metrics)
                    if not np.isfinite(m).all() or m[9]:
                        raise AssertionError('Nonfinite or invalid action')
                    _, ct = timed(collect, train.params, bank, c, pool)
                    _, ut = timed(update, train, data, jax.random.PRNGKey(80))
                    row['variants'][name]['trials'].append(dict(iteration_seconds=seconds,
                        collect_seconds=ct, update_seconds=ut, metrics=m.tolist()))
            for result in row['variants'].values():
                result['median_seconds'] = {phase: float(np.median([t[phase] for t in result['trials']]))
                    for phase in ('iteration_seconds', 'collect_seconds', 'update_seconds')}
                result['warm_samples_per_second'] = b*cfg['rollout_steps']/result['median_seconds']['iteration_seconds']
            if a.phases:
                model = Policy(cfg['network_width'])
                row['phases'] = phase_profile(model, train, bank, c, pool, cfg, make_collect(model, cfg), make_update(model, cfg))
            results.append(row)
            write_json(out/'comparison.json', dict(results=results, elapsed_locked_seconds=time.monotonic()-start,
                peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if platform.system() == 'Darwin' else 1024),
                device_memory_stats=jax.devices()[0].memory_stats()))
            print(json.dumps(dict(batch=b, medians={k:v['median_seconds'] for k,v in row['variants'].items()})), flush=True)


if __name__ == '__main__':
    main()
