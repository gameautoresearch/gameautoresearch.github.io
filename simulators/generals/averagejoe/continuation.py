"""Autonomous finite continuation; unchanged upstream model, PPO and curriculum."""
from pathlib import Path
import argparse
import json
import os
import signal
import subprocess
import time
import traceback

from .bootstrap import activate, binding, sha
activate()
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from .framework import recipe, initial, make_env, memories, make_kernels, next_stage
from .assessment import make_assessor
from .lineage import verify_parent
from .prepare import save_arrays
from .train import load_bank, write


def persist(root):
    """v2 Volume sync commits files and metadata; local test directories use fsync."""
    if os.environ.get('AVERAGEJOE_VOLUME_V2') == '1':
        subprocess.run(['sync', str(root)], check=True, timeout=45)


def equal_tensors(a, b):
    return all(np.array_equal(np.asarray(x), np.asarray(y))
               for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)) if eqx.is_array(x))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--parent', type=Path, required=True)
    p.add_argument('--parent-source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seconds', type=float, required=True)
    a = p.parse_args()
    started = time.monotonic()
    a.output.mkdir(parents=True, exist_ok=False)
    cfg = recipe()
    lineage = verify_parent(a.parent, a.parent_source)
    write(a.output / 'lineage.json', lineage)
    source = binding()
    write(a.output / 'source.json', source)
    write(a.output / 'config.json', cfg.to_dict())
    meta = json.loads((a.inputs / 'maps.json').read_text())
    if meta['config'] != cfg.to_dict():
        raise ValueError('Map recipe drift')
    for n, h in meta['files'].items():
        if sha(a.inputs / n) != h:
            raise ValueError('Input map checksum drift: ' + n)
    if jax.default_backend() != 'gpu' or len(jax.devices()) != 1:
        raise RuntimeError('Exactly one GPU required')
    jax.config.update('jax_default_matmul_precision', 'tensorfloat32')
    stop = False
    def stop_handler(*_):
        nonlocal stop
        stop = True
    signal.signal(signal.SIGTERM, stop_handler)
    signal.signal(signal.SIGINT, stop_handler)
    def event(kind, **values):
        row = {'kind': kind, 'elapsed_seconds': time.monotonic() - started, **values}
        with (a.output / 'events.jsonl').open('a') as f:
            f.write(json.dumps(row, allow_nan=False) + '\n')
        write(a.output / 'status.json', row)
        print(json.dumps(row), flush=True)
        return row

    event('initializing', lineage=lineage)
    net, opt, opt_state, key = initial(cfg)
    m0 = memories(cfg); m1 = memories(cfg)
    states = load_bank(a.inputs / 'stage0-start.npz')
    net, opt_state, ema, key, states, m0, m1 = eqx.tree_deserialise_leaves(
        a.parent / 'final-state.eqx', (net, opt_state, net, key, states, m0, m1))
    parent_meta = json.loads((a.parent / 'final-state.json').read_text())
    parent_net = eqx.tree_deserialise_leaves(a.parent / 'final.eqx', net)
    if not equal_tensors(net, parent_net):
        raise ValueError('Restored model does not match parent standalone weights')
    probe = np.asarray(net.policy_head.weight).copy()
    iteration = parent_meta['iteration']; initial_iteration = iteration; stage = 0
    pool = load_bank(a.inputs / 'stage0-pool.npz')
    if parent_meta['pool_sha256'] != sha(a.inputs / 'stage0-pool.npz'):
        raise ValueError('Parent pool checksum mismatch')
    starts = load_bank(a.inputs / 'stage0-eval.npz')
    env = make_env(cfg, stage); env.pool_size = len(pool.time)
    rollout, update, ema_step = make_kernels(cfg, env, net)
    evaluate = make_assessor(cfg, general_trade=env.general_trade)
    pools = a.output / 'pools'; pools.mkdir()
    checkpoints = a.output / 'checkpoints'; checkpoints.mkdir()
    pool_file = str(a.inputs / 'stage0-pool.npz')
    pool_hash = parent_meta['pool_sha256']
    last_eval = None
    totals = {'captures': 0, 'caps': 0, 'nonzero_rewards': 0, 'illegal_actions': 0}

    def checkpoint(label):
        folder = checkpoints / label; folder.mkdir()
        target = (net, opt_state, ema, key, states, m0, m1)
        path = folder / 'state.eqx'
        eqx.tree_serialise_leaves(path.with_suffix('.tmp'), target)
        path.with_suffix('.tmp').replace(path)
        restored = eqx.tree_deserialise_leaves(path, target)
        if not equal_tensors(target, restored):
            raise AssertionError('Checkpoint readback mismatch')
        eqx.tree_serialise_leaves(folder / 'model.eqx', net)
        eqx.tree_serialise_leaves(folder / 'ema.eqx', ema)
        save_arrays(folder / 'evaluation-starts.npz', starts._asdict())
        state_meta = {'iteration': iteration, 'stage': stage, 'source_sha256': source['sha256'],
                      'parent': lineage, 'pool_path': pool_file, 'pool_sha256': pool_hash,
                      'files': {f.name: sha(f) for f in folder.iterdir() if f.is_file()},
                      'exact_tensor_readback': True, 'last_evaluation': last_eval,
                      'totals_this_run': totals}
        write(folder / 'state.json', state_meta)
        persist(a.output)
        write(a.output / 'latest.json', {'checkpoint': str(folder), **state_meta})
        persist(a.output)
        event('checkpoint_committed', iteration=iteration, stage=stage, checkpoint=str(folder))
        return state_meta

    def assess(label):
        # Fixed public assessment RNG never consumes the training PRNG stream.
        result, _ = evaluate(net, starts, jax.random.PRNGKey(2026110900 + stage))
        result = {k: int(v) for k, v in jax.device_get(result).items()}
        if result['illegal_actions'] or result['finished'] != result['games']:
            raise AssertionError('Incomplete or illegal assessment')
        event(label, iteration=iteration, stage=stage, general_trade=env.general_trade, **result)
        persist(a.output)
        return result

    event('parent_restored', iteration=iteration, stage=stage,
          parameters=sum(x.size for x in jax.tree.leaves(eqx.filter(net, eqx.is_array))),
          restore_exact_model=True)
    baseline = assess('baseline_evaluation'); last_eval = baseline
    # Durable restored state exists even if first pool compilation fails.
    checkpoint('restored-000020')
    iteration_cost = 65.
    first_update = True
    train_deadline = a.seconds - 240
    while not stop and time.monotonic() - started + max(iteration_cost, 20) < train_deadline:
        tick = time.monotonic()
        if iteration and iteration % cfg.reset_pool_every == 0:
            event('pool_refresh_started', iteration=iteration, stage=stage)
            key, pk = jax.random.split(key)
            pool, _ = env.reset(pk); jax.block_until_ready(pool)
            if len(pool.time) != meta['actual_pool_size']:
                raise AssertionError('Refreshed pool cardinality changed')
            path = pools / f'stage-{stage}-iteration-{iteration:06d}.npz'
            save_arrays(path, pool._asdict())
            pool_file, pool_hash = str(path), sha(path)
            write(path.with_suffix('.json'), {'stage': stage, 'iteration': iteration,
                  'generation_key': np.asarray(pk).tolist(), 'size': len(pool.time),
                  'sha256': pool_hash, 'engine_files': source['sources']['engine_files']})
            persist(a.output)
            event('pool_refresh_complete', iteration=iteration, stage=stage,
                  pool_size=len(pool.time), pool_sha256=pool_hash)
        states, data, key, m0, m1 = rollout(net, states, key, m0, m1, pool)
        net, opt_state, key, metrics = update(net, opt_state, key, data, jnp.int32(iteration))
        ema = ema_step(ema, net); jax.block_until_ready(metrics)
        values = {k: float(v) for k, v in jax.device_get(metrics).items()}
        if not all(np.isfinite(list(values.values()))) or values['invalid_actions']:
            raise RuntimeError('Nonfinite metrics or illegal sampled action')
        iteration += 1
        totals['captures'] += int(values['terminated']); totals['caps'] += int(values['caps'])
        totals['nonzero_rewards'] += int(values['nonzero_rewards'])
        iteration_cost = time.monotonic() - tick
        row = {'iteration': iteration, 'stage': stage,
               'stage_distance': [env.min_generals_distance, env.max_generals_distance],
               'iterations_this_run': iteration - initial_iteration,
               'player_transitions_this_run': (iteration - initial_iteration) * 2 * cfg.num_envs * cfg.num_steps,
               'iteration_seconds': iteration_cost, **values, 'totals': dict(totals)}
        with (a.output / 'progress.jsonl').open('a') as f:
            f.write(json.dumps(row, allow_nan=False) + '\n')
        event('update_complete', **row)
        del data
        if iteration % cfg.eval_every == 0:
            last_eval = assess('curriculum_evaluation')
            ns = next_stage(cfg, stage, last_eval['wins'] / last_eval['games'], evaluation_complete=True)
            # Save BEFORE changing stage so interruption leaves a usable checkpoint.
            checkpoint(f'iteration-{iteration:06d}-stage-{stage}')
            if ns != stage:
                stage = ns; env = make_env(cfg, stage)
                event('curriculum_advancing', iteration=iteration, stage=stage)
                key, pk = jax.random.split(key); pool, _ = env.reset(pk); jax.block_until_ready(pool)
                path = pools / f'stage-{stage}-entry-{iteration:06d}.npz'
                save_arrays(path, pool._asdict()); pool_file, pool_hash = str(path), sha(path)
                key, ik = jax.random.split(key)
                states = jax.jit(jax.vmap(env.init_state))(jax.random.split(ik, cfg.num_envs))
                starts = jax.tree.map(lambda x: x[:cfg.eval_games // 2], states)
                m0 = memories(cfg); m1 = memories(cfg)
                rollout, update, ema_step = make_kernels(cfg, env, net)
                checkpoint(f'stage-{stage}-entry-{iteration:06d}')
                event('curriculum_advanced', iteration=iteration, stage=stage,
                      distance=[env.min_generals_distance, env.max_generals_distance])
        elif first_update or iteration % 25 == 0:
            checkpoint(f'iteration-{iteration:06d}-stage-{stage}')
        first_update = False

    saved = checkpoint('final')
    final = assess('final_evaluation')
    recording_result = None
    try:
        if time.monotonic() - started < a.seconds - 130:
            from .recordings import export_recordings
            event('recordings_started', iteration=iteration, stage=stage)
            recording_result = export_recordings(cfg, env, net, a.output / 'replays', saved['files']['model.eqx'], stage)
            event('recordings_complete', **recording_result)
    except Exception:
        (a.output / 'recording-error.txt').write_text(traceback.format_exc())
        event('recordings_failed', error=traceback.format_exc()[-1200:])
    result = {'status': 'complete', 'initial_iteration': initial_iteration, 'iteration': iteration,
              'updates_this_run': iteration - initial_iteration, 'stage': stage,
              'baseline': baseline, 'final': final, 'totals_this_run': totals,
              'assessment_rules': {'general_trade': env.general_trade, 'matches_self_play': True},
              'legacy_smoke_assessment': json.loads((a.parent/'result.json').read_text())['final'],
              'legacy_smoke_assessment_general_trade': True,
              'player_transitions_this_run': (iteration - initial_iteration) * 2 * cfg.num_envs * cfg.num_steps,
              'policy_head_l2_change_from_parent': float(np.linalg.norm(np.asarray(net.policy_head.weight) - probe)),
              'checkpoint': saved, 'recordings': recording_result,
              'elapsed_seconds': time.monotonic() - started, 'source_sha256': source['sha256'],
              'strength': 'Unqualified; random-opponent curriculum performance alone is not expert calibration'}
    write(a.output / 'result.json', result)
    event('completed', iteration=iteration, stage=stage, updates_this_run=iteration-initial_iteration)
    persist(a.output)


if __name__ == '__main__':
    main()
