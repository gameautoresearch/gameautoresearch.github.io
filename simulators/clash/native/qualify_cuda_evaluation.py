"""Bounded actual-CUDA native evaluation and CPU replay qualification.

Run only inside an already allocated single-GPU worker. This command never
provisions compute. It creates a fresh same-source native-cuda PPO checkpoint,
uses only the frozen public practice seeds, and caps every evaluation episode
at eight decisions. The earlier full CPU natural-outcome evidence is retained;
these diagnostic truncations have official_score=null. No Java startup or
VectorBattle transition is allowed, including while binding ScriptedBot.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import copy
import hashlib
import json
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))


def qualify(out, *, launch_threads=1, render_mode=4):
    import jpype
    import numpy as np
    import torch
    from train import Trainer, binding
    from evaluate import (DATA, PRACTICE, StaticSchema, evaluate_checkpoints,
                          load_checkpoint, no_jvm, public_digest, public_outputs,
                          pure_opponent_modules, replay_evaluation)
    from runtime import NativeCPU, NativeCUDA
    from snapshots import decode_engine_state, encode_engine_state, snapshot_sizes
    no_jvm()
    if torch.cuda.device_count() != 1:
        raise ValueError('CUDA evaluation qualification requires exactly one visible GPU')
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    original_binding = binding(DATA)
    pure_opponent_modules()
    # Fail before any Java setup/lookup/transition could execute. Importing the
    # pinned dataclasses and policy is permitted and does not start a JVM.
    import crforge_env._jvm as jvm
    import crforge_env.vector as vector
    saved_java_entrypoints = jvm.start, jvm.java_class, vector.VectorBattle.__init__
    blocked_calls = []
    def forbid_java(*args, **kwargs):
        blocked_calls.append('attempted Java entry point')
        raise AssertionError('CUDA native evaluation attempted a Java entry point')
    jvm.start = forbid_java
    jvm.java_class = forbid_java
    vector.VectorBattle.__init__ = forbid_java
    trainer = None
    gpu = cpu = None
    try:
        config = dict(backend='native-cuda', seed=81108, envs=2, torch_threads=1,
                      width=16, embedding=4, learning_rate=.0003, rollout_steps=2,
                      epochs=1, minibatch=4, gamma=.99, gae_lambda=.95, clip=.2,
                      value_coef=.5, entropy_coef=.001, max_grad_norm=.5,
                      target_kl=0., active_policy_loss=True, graph_sampling=False,
                      updates=0, budget_seconds=1, max_ticks=0, checkpoint_codec='zlib1',
                      cuda_launch_threads=launch_threads, cuda_render_mode=render_mode,
                      threads=1)
        trainer = Trainer(config, DATA)
        # This checkpoint is a qualification artifact, never training: replace
        # initialization's untouched training-seed worlds with public practice
        # worlds before saving or executing any transition.
        practice_seeds = json.loads(PRACTICE.read_text())['engine_seeds'][:2]
        trainer.env.reset(practice_seeds)
        trainer.refresh_actor()
        checkpoint = out / 'fresh-native-cuda-checkpoint.pt'
        trainer.checkpoint(checkpoint, 0.)
        original_build = copy.deepcopy(trainer.backend_build)
        original_runtime = copy.deepcopy(trainer.execution_runtime)
        trainer.close(); trainer = None
        _, data, weight_receipt = load_checkpoint(checkpoint, device='cuda')
        assert weight_receipt['samples'] == 0 and weight_receipt['update'] == 0
        assert weight_receipt['original_backend_build'] == original_build
        assert weight_receipt['original_execution_runtime'] == original_runtime
        schedule = json.loads(PRACTICE.read_text())
        schedule['engine_seeds'] = schedule['engine_seeds'][:2]
        schedule_path = out / 'public-practice-subset.json'
        schedule_path.write_text(json.dumps(schedule, indent=2) + '\n')
        first = evaluate_checkpoints([checkpoint], out / 'evaluation', backend='native-cuda',
                                     schedule=schedule_path, max_decisions=8,
                                     cuda_launch_threads=launch_threads, cuda_render_mode=render_mode)
        expected_games = 2 * len(schedule['engine_seeds']) * len(schedule['opponents'])
        assert first['summary']['games'] == expected_games == 12
        assert first['summary']['truncations'] == 12 and first['summary']['natural'] == 0
        assert first['summary']['official_score'] is None
        for key in ('illegal_actions', 'illegal_attempts', 'not_modelled', 'simulator_faults', 'exceptions', 'entity_overflows'):
            assert first['summary'][key] == 0, (key, first['summary'])
        replay = replay_evaluation(out / 'evaluation/manifest.json', out / 'native-cpu-replay')
        assert replay['games'] == 12 and replay['exact_public_transitions'] == 96
        schedule['engine_seeds'].reverse(); schedule['opponents'].reverse()
        reversed_path = out / 'public-practice-reordered.json'
        reversed_path.write_text(json.dumps(schedule, indent=2) + '\n')
        second = evaluate_checkpoints([checkpoint], out / 'reordered', backend='native-cuda',
                                      schedule=reversed_path, max_decisions=8,
                                      cuda_launch_threads=launch_threads, cuda_render_mode=render_mode)
        def tapes(manifest, directory):
            return {(game['engine_seed'], game['opponent'], game['learner_side']):
                    json.loads((directory / game['tape']).read_text()) for game in manifest['games']}
        assert tapes(first, out / 'evaluation') == tapes(second, out / 'reordered'), 'CUDA evaluation policy RNG depends on schedule order'
        # Raw CUDA save/load is a separate state-restoration proof. A busy
        # native CPU world controls all public transitions before/after restore.
        seeds = sorted(schedule['engine_seeds'])
        gpu = NativeCUDA(seeds, launch_threads=launch_threads, render_mode=render_mode)
        cpu = NativeCPU(seeds)
        rng = np.random.default_rng(8110819)
        s = StaticSchema(data)
        maximum_entities = 0
        def assert_public(label):
            a, b = public_outputs(gpu), public_outputs(cpu)
            for key in a:
                np.testing.assert_array_equal(a[key], b[key], err_msg=label + ' ' + key)
            return public_digest(a)
        def actions():
            return [[int(rng.choice(np.flatnonzero(cpu.masks[lane, side]))) for side in (0, 1)] for lane in range(len(seeds))]
        assert_public('initial')
        for index in range(96):
            chosen = actions(); gpu.step(chosen); cpu.step(chosen)
            assert_public('busy ' + str(index))
            maximum_entities = max(maximum_entities, int(cpu.observations[..., s.header['entity_count'][0]].max()))
        assert maximum_entities > 6
        saved_public_digest = assert_public('before raw save')
        # Save ALL bytes directly from the live allocation, without a second
        # full CUDA allocation. Decode preserves every byte, including padding.
        raw = gpu.states.detach().cpu()
        encoded = encode_engine_state(raw, workers=1)
        decoded = decode_engine_state(encoded, as_tensor=True, expected_count=len(seeds),
                                      expected_world_bytes=gpu.states.shape[1], workers=1,
                                      max_total_bytes=raw.numel())
        assert torch.equal(raw, decoded)
        for index in range(8):
            gpu.step(np.zeros((len(seeds), 2), np.int32))
        gpu.load(decoded)
        assert torch.equal(gpu.states.cpu(), raw), 'same-allocation CUDA raw restore changed bytes'
        gpu.read()
        assert assert_public('after raw restore') == saved_public_digest
        for index in range(16):
            chosen = actions(); gpu.step(chosen); cpu.step(chosen)
            assert_public('restored continuation ' + str(index))
        restored_final_digest = assert_public('restored final')
        gpu.close(); gpu = None
        cpu.close(); cpu = None
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        saved['binding']['source']['native/evaluate.py'] = '0' * 64
        bad_checkpoint = out / 'rejected-source-checkpoint.pt'
        torch.save(saved, bad_checkpoint)
        try:
            load_checkpoint(bad_checkpoint, device='cuda')
        except ValueError as exc:
            assert 'binding' in str(exc)
        else:
            raise AssertionError('CUDA evaluator accepted a different-source checkpoint')
        tape_path = out / 'evaluation' / first['games'][0]['tape']
        payload = tape_path.read_bytes()
        tape_path.write_bytes(payload + b' ')
        try:
            replay_evaluation(out / 'evaluation/manifest.json', out / 'rejected-tape-replay')
        except ValueError as exc:
            assert 'checksum' in str(exc)
        else:
            raise AssertionError('native CPU replayer accepted a changed CUDA action tape')
        finally:
            tape_path.write_bytes(payload)
        no_jvm()
        assert not blocked_calls and not jpype.isJVMStarted()
        assert binding(DATA) == original_binding, 'native source changed during CUDA evaluation qualification'
        result = {'status': 'passed', 'fresh_same_source_native_cuda_checkpoint': True,
                  'jvm_started': False, 'java_entrypoint_attempts': blocked_calls,
                  'evaluation_summary': first['summary'], 'games': expected_games,
                  'exact_native_cpu_replay_transitions': replay['exact_public_transitions'],
                  'per_game_seat_rng_schedule_independent': True,
                  'source_checkpoint_rejected': True, 'tape_checksum_rejected': True,
                  'raw_cuda_restore': {'complete_bytes_exact': True, 'busy_decisions': 96,
                                       'maximum_entities': maximum_entities, 'lanes': len(seeds),
                                       'exact_continuation_decisions': 16,
                                       'final_public_digest': restored_final_digest,
                                       'compressed_snapshot': snapshot_sizes(encoded)},
                  'engine_policy': first['runtime']['engine_policy'], 'runtime': first['runtime'],
                  'cuda_backend_build': original_build, 'cpu_replay_backend_build': replay['backend_build'],
                  'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                  'binding': original_binding,
                  'natural_outcome_scope': 'Full native CPU scripted-vs-wait natural evidence is retained separately; bounded CUDA diagnostic games intentionally have no official score.'}
        (out / 'summary.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
        return result
    finally:
        if trainer is not None:
            trainer.close()
        if gpu is not None:
            gpu.close()
        if cpu is not None:
            cpu.close()
        jvm.start, jvm.java_class, vector.VectorBattle.__init__ = saved_java_entrypoints


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--launch-threads', type=int, default=1)
    parser.add_argument('--render-mode', type=int, default=4)
    args = parser.parse_args()
    result = qualify(args.out, launch_threads=args.launch_threads, render_mode=args.render_mode)
    print(json.dumps({key: result[key] for key in ('status', 'games', 'exact_native_cpu_replay_transitions', 'jvm_started')}, indent=2))
