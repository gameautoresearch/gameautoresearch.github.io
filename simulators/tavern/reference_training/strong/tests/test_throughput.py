"""Full-game script and retained-rollout parity, including trained busy boards."""
import copy
import json
import time
from pathlib import Path
import numpy as np
import pytest
import torch
from simulators.tavern.reference_training.strong.common import ROOT
from simulators.tavern.reference_training.strong.lineage import import_parent, module
from simulators.tavern.reference_training.strong.population import schedule
from simulators.tavern.reference_training.strong.scripts import BatchedVisibleScript
from simulators.tavern.reference_training.strong.train import rng_state, restore_rng
from simulators.tavern.reference_training.strong.learner import update
from simulators.tavern.reference_training.validate import equal
from simulators.tavern.reference_training import collection


def test_cuda_acceleration_rejects_unqualified_runtime_before_allocation(monkeypatch):
    from simulators.tavern.reference_training.strong.common import validate_config
    c = json.loads((Path(__file__).parents[1]/'curriculum-throughput.json').read_text())
    c.update(device='cuda', machine=dict(gpu='H100', cpu_cores=4, memory_gib=16),
             batched_policy_dispatch=True, ppo_backward_graph=True)
    monkeypatch.setattr(torch, '__version__', '2.10.0+cu130')
    with pytest.raises(ValueError, match='qualified PyTorch2.9.1'):
        validate_config(c)
    monkeypatch.setattr(torch, '__version__', '2.9.1+cu128')
    validate_config(c)


def test_public_full_game_scripts_match_scalar_oracle():
    import taverngym as tg
    from simulators.tavern.evaluation_readiness import opponents as scripts
    oracle = [scripts.VisibleScript(s) for s in scripts.STYLES]
    batched = [BatchedVisibleScript(scripts, s) for s in scripts.STYLES]
    c = json.loads((Path(__file__).parents[1]/'curriculum-quarter.json').read_text())
    env = tg.BatchEnv(3, seed=5531907, controlled=range(8), config=c['rules'], threads=1)
    finished = np.zeros(24, bool)
    rows = busy = choices = 0
    for _ in range(2000):
        active = ~finished.reshape(3,8).all(1)
        obs = env.observe_ready(active_games=active)
        ids = np.arange(len(obs['ready']))
        actions = np.zeros(24, np.int64)
        for old, new in zip(oracle, batched):
            expected, got = old.remote_act(obs, ids)[0], new.remote_act(obs, ids)[0]
            assert np.array_equal(expected, got)
        for i, slot in enumerate(obs['slots']):
            actions[slot] = batched[int(slot)%3].remote_act(obs, np.array([i]))[0][0]
        rows += len(ids)
        busy += int((obs['features'][:,0] >= .5).sum())
        choices += int(obs['action_mask'][:,30:46].any(1).sum())
        _, done, info = env.step(actions, active_games=active)
        assert not info['rejected'].any()
        assert not (done & (info['stop_reason'] != 1)).any()
        finished |= done
        if finished.all():
            break
    assert finished.all() and rows > 1000 and busy > 0 and choices > 0


def test_trained_full_cohort_batched_scripts_resident_ppo():
    inputs = ROOT/'evidence/tavern-population-throughput-20261109/inputs-quarter157'
    if not inputs.exists():
        pytest.skip('Private trained-policy fixture is local')
    torch.set_num_threads(1)
    c = json.loads((Path(__file__).parents[1]/'curriculum-quarter.json').read_text())
    c.update(device='cpu', n_games=8, ppo_epochs=1)
    model, parent, pop, _ = import_parent(inputs, torch.device('cpu'))
    scripts = module(inputs/'scripts/opponents.py', 'throughput_script_oracle')
    actors = pop.actors(torch.device('cpu'))
    assignment = schedule(c['seed'], 0, c['n_games'], len(pop.entries))
    torch.manual_seed(5531902)
    rng = rng_state()
    a = collection.collect(model, c, c['seed'], torch.device('cpu'), collection.Timer('cpu'),
                           time.perf_counter()+120, assignment=assignment,
                           population=actors+[scripts.VisibleScript(s) for s in scripts.STYLES])
    restore_rng(rng)
    checked = dict(rows=0, busy=0, choices=0)
    check_actors = {s: BatchedVisibleScript(scripts, s) for s in scripts.STYLES}

    class Checked(BatchedVisibleScript):
        def remote_act(self, obs, ids):
            got = super().remote_act(obs, ids)
            # Check every style on every encountered public row, including
            # pending choices and late boards, against the immutable scalar code.
            for style in scripts.STYLES:
                expected = scripts.VisibleScript(style).remote_act(obs, ids)[0]
                actual = check_actors[style].remote_act(obs, ids)[0]
                assert np.array_equal(expected, actual), (style, ids[np.flatnonzero(expected != actual)])
            checked['rows'] += len(ids)
            checked['busy'] += int((obs['features'][ids, 0] >= .5).sum())
            checked['choices'] += int(obs['action_mask'][ids, 30:46].any(1).sum())
            return got

    c['device_rollouts'] = True
    c['batched_policy_dispatch'] = True
    b = collection.collect(model, c, c['seed'], torch.device('cpu'), collection.Timer('cpu'),
                           time.perf_counter()+120, assignment=assignment,
                           population=actors+[Checked(scripts, s) for s in scripts.STYLES])
    assert equal(a[1], b[1]) and equal(a[3], b[3])
    assert a[2]['accepted_learning_samples'] == b[2]['accepted_learning_samples']
    assert b[2]['invalid_lobbies'] == b[2]['rejected'] == 0
    assert checked['rows'] > 1000 and checked['busy'] > 0 and checked['choices'] > 0
    flat = [row for ep in b[0] for row in ep]
    packed = b[0].pack(flat, torch.device('cpu'))
    original = [row for ep in a[0] for row in ep]
    for column, name in enumerate(('features', 'cards', 'mask')):
        assert np.array_equal(packed[column].numpy(), np.stack([r[name] for r in original]))
    left, right = copy.deepcopy(model), copy.deepcopy(model)
    lo = torch.optim.Adam(left.parameters(), lr=c['learning_rate'], eps=1e-5)
    ro = torch.optim.Adam(right.parameters(), lr=c['learning_rate'], eps=1e-5)
    lo.load_state_dict(copy.deepcopy(parent['optimizer'])); ro.load_state_dict(copy.deepcopy(parent['optimizer']))
    rng = rng_state()
    ma = update(left, lo, a[0], c, torch.device('cpu'))
    restore_rng(rng)
    c['preprocess_numeric'] = True
    mb = update(right, ro, b[0], c, torch.device('cpu'))
    assert equal(ma, mb) and equal(left.state_dict(), right.state_dict())
    assert equal(lo.state_dict(), ro.state_dict())


def test_timer_diagnostic_does_not_double_count_stage():
    timer = collection.Timer('cpu', synchronized=False)
    with timer.measure('total'):
        with timer.measure('subset', diagnostic=True):
            pass
    assert set(timer.seconds) == {'total'} and set(timer.diagnostics) == {'subset'}
