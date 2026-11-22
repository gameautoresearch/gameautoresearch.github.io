"""Fresh same-source native checkpoint/evaluator/replay CPU qualification.

Run .venv-clash/bin/python integrations/clash/native/tests/test_evaluate.py
    --out NEW_EVIDENCE_DIRECTORY
No Java startup, previous-source checkpoint bypass, timing benchmark or GPU job.
"""
from pathlib import Path
import argparse
import copy
import json
import sys

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))


def qualify(out):
    import jpype
    import numpy as np
    import torch
    from train import Trainer, binding
    from evaluate import (DATA, PRACTICE, PublicActor, StaticSchema, classify,
                          evaluate_checkpoints, frozen_facts, load_checkpoint,
                          no_jvm, public_digest, public_outputs, replay_evaluation,
                          run_episode, summarize)
    from runtime import NativeCPU
    no_jvm()
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    initial_binding = binding(DATA)
    # The evaluator/test files already exist before this fresh checkpoint's
    # binding is computed. No previous source or old checkpoint is accepted.
    config = dict(backend='native-cpu', seed=81108, envs=1, torch_threads=1,
                  width=16, embedding=4, learning_rate=.0003, rollout_steps=2,
                  epochs=1, minibatch=2, gamma=.99, gae_lambda=.95, clip=.2,
                  value_coef=.5, entropy_coef=.001, max_grad_norm=.5,
                  target_kl=0., active_policy_loss=True, graph_sampling=False,
                  updates=0, budget_seconds=1, max_ticks=0)
    trainer = Trainer(config, DATA)
    checkpoint = out / 'fresh-checkpoint.pt'
    try:
        trainer.checkpoint(checkpoint, 0.)
    finally:
        trainer.close()
    model, data, receipt = load_checkpoint(checkpoint)
    assert receipt['samples'] == 0 and receipt['update'] == 0
    facts = frozen_facts(data)
    # Frozen exported level-11 values, independently checked against Java's
    # table readers by existing export qualifications; the formulas are tested
    # here on every supported fixed8 unit and spell, not only a bot's actions.
    by_name = {name: i for i, name in enumerate(data['vocabulary'])}
    expected_spells = {'Fireball': (2500, 688, 159), 'Arrows': (3500, 366, 75), 'Zap': (2500, 192, 48)}
    for name, expected in expected_spells.items():
        fact = facts[1][by_name[name]]
        assert (fact.radius, fact.damage, fact.tower_damage) == expected, (name, fact)
    assert len(facts[0]) == 8 and len(facts[1]) == 3 and len(facts[2]) == 5
    assert facts[2][by_name['Giant']].buildings_only
    s = StaticSchema(data)
    practice = json.loads(PRACTICE.read_text())
    practice['engine_seeds'] = practice['engine_seeds'][:2]
    practice['opponents'] = ['random']
    schedule = out / 'practice-subset.json'
    schedule.write_text(json.dumps(practice, indent=2) + '\n')
    first = evaluate_checkpoints([checkpoint], out / 'evaluation', schedule=schedule, max_decisions=8)
    replay = replay_evaluation(out / 'evaluation/manifest.json', out / 'replay')
    assert replay['games'] == 4 and replay['exact_public_transitions'] == 32
    assert first['summary']['truncations'] == 4 and first['summary']['official_score'] is None
    # Per-game/seat RNGs make results independent of scheduling order.
    practice['engine_seeds'].reverse()
    reverse_schedule = out / 'practice-reversed.json'
    reverse_schedule.write_text(json.dumps(practice, indent=2) + '\n')
    second = evaluate_checkpoints([checkpoint], out / 'reordered', schedule=reverse_schedule, max_decisions=8)
    def episode_map(manifest, directory):
        return {(g['engine_seed'], g['learner_side']): json.loads((directory / g['tape']).read_text()) for g in manifest['games']}
    assert episode_map(first, out / 'evaluation') == episode_map(second, out / 'reordered')
    # A complete natural public scripted-vs-wait episode exercises the actual
    # frozen facts, policy schema adapter, observation loop and native outcomes.
    env = NativeCPU([1_000_000])
    try:
        natural = run_episode(env, [PublicActor('scripted', 8128, data=data), PublicActor('wait', 8129)], s)
    finally:
        env.close()
    assert natural['outcome']['natural'] and natural['outcome']['valid'] and natural['outcome']['winner'] == 0, natural['outcome']
    (out / 'scripted-vs-wait-natural.json').write_text(json.dumps(natural, indent=2, sort_keys=True) + '\n')
    # Preserve actual simulator illegal-action counters; never mask or replace
    # a closed action when diagnosing a broken policy.
    class IllegalActor:
        def __call__(self, observation, mask): return 2307
        def state(self): return {'kind': 'illegal-test'}
    env = NativeCPU([1_000_000])
    try:
        illegal = run_episode(env, [IllegalActor(), PublicActor('wait', 1)], s, max_decisions=1)
    finally:
        env.close()
    assert illegal['outcome']['illegal'] == 1 and illegal['outcome']['illegal_attempts'] == 1 and not illegal['outcome']['valid']
    env = NativeCPU([1_000_000])
    try:
        truncated = run_episode(env, [PublicActor('wait', 1), PublicActor('wait', 2)], s, max_ticks=1)
    finally:
        env.close()
    assert truncated['outcome']['engine_truncated'] and not truncated['outcome']['natural']
    # Explicit classification controls retain natural draws, invalid natural
    # results, simulator faults and unsupported commands in separate counters.
    info = np.zeros(s.info_ints, np.int32)
    info[s.info['winner']] = 2
    draw = classify(info, s)
    assert draw['draw'] and draw['valid']
    info[s.info['fault']] = 2
    fault = classify(info, s)
    assert fault['simulator_fault'] == 2 and not fault['natural'] and not fault['valid']
    info[s.info['fault']] = 0; info[s.info['not_modelled']] = 1
    unsupported = classify(info, s)
    assert unsupported['natural'] and unsupported['not_modelled'] and not unsupported['valid']
    assert summarize([{'outcome': fault, 'learner_side': 0}])['official_score'] is None
    assert summarize([{'outcome': draw, 'learner_side': 0}])['official_score'] == .5
    # Test strict source/data/rules and architecture rejection with current
    # freshly-created weights. No bypass flag exists.
    original = torch.load(checkpoint, map_location='cpu', weights_only=False)
    changes = {
        'source': lambda v: v['binding']['source'].__setitem__('model.py', '0' * 64),
        'data': lambda v: v['binding'].__setitem__('task_data_sha256', '0' * 64),
        'rules': lambda v: v['binding'].__setitem__('rules_lock_sha256', '0' * 64),
        'architecture': lambda v: v['config'].__setitem__('width', v['config']['width'] + 1),
        'build': lambda v: v['backend_build'].__setitem__('identity_sha256', '0' * 64),
        'frozen-buffer': lambda v: v['model']['global_scale'].add_(1),
        'nonfinite': lambda v: v['model']['tiles'].fill_(float('nan')),
    }
    rejected = []
    for name, change in changes.items():
        value = copy.deepcopy(original); change(value)
        path = out / ('tampered-' + name + '.pt'); torch.save(value, path)
        try:
            load_checkpoint(path)
        except (ValueError, RuntimeError):
            rejected.append(name)
        else:
            raise AssertionError('tampered checkpoint accepted: ' + name)
    # A byte changed in a tape must fail before native replay executes it.
    manifest_path = out / 'evaluation/manifest.json'
    tape_path = out / 'evaluation' / first['games'][0]['tape']
    payload = tape_path.read_bytes()
    tape_path.write_bytes(payload + b' ')
    try:
        replay_evaluation(manifest_path, out / 'tampered-replay')
    except ValueError as exc:
        assert 'checksum' in str(exc)
    else:
        raise AssertionError('tampered tape accepted')
    finally:
        tape_path.write_bytes(payload)
    assert binding(DATA) == initial_binding, 'native source changed during same-code test'
    assert not jpype.isJVMStarted()
    result = {'status': 'passed', 'jvm_started': False, 'fresh_untrained_checkpoint': True,
              'bounded_games': 4, 'exact_cpu_replay_transitions': 32, 'episode_order_independent': True,
              'natural_scripted_vs_wait': {'decisions': len(natural['steps']), 'outcome': natural['outcome']},
              'illegal_control': illegal['outcome'], 'engine_truncation_control': truncated['outcome'],
              'checkpoint_tamper_rejections': rejected, 'tape_tamper_rejected': True,
              'facts': facts[3], 'binding': initial_binding}
    (out / 'summary.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = qualify(args.out)
    print(json.dumps({k: result[k] for k in ('status', 'jvm_started', 'exact_cpu_replay_transitions')}, indent=2))
