import json
import sys
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch
import taverngym as tg

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adapter import TavernEnv
from model import Policy
from rollout import Timer, collect, gae


def test_guard_codes_survive_reset_and_never_become_placement():
    for field, value, code in [("stopped_reason", "driver_guard", 3),
                                ("turn_limit_end", True, 4),
                                ("truncated_combats", 1, 5),
                                ("truncated_recruit_contexts", 1, 6)]:
        env = tg.BatchEnv(1, seed=91, controlled=list(range(8)))
        state = json.loads(env.save(0))
        state["game"][field] = value
        env.load(0, json.dumps(state))
        assert not env.observe()["ready"].any()
        rewards, dones, info = env.step(np.zeros(8, dtype=np.int64))
        assert dones.all() and info["truncated"].all() and not info["terminated"].any()
        assert (info["stop_reason"] == code).all()
        assert not info["placement"].any() and not rewards.any()
        assert env.episode_index(0) == 1
        assert env.observe()["ready"].all()


def test_zero_copy_is_readonly_owned_and_identical():
    env = tg.BatchEnv(2, seed=17, controlled=range(8))
    views, copies = env.observe(copy=False), env.observe(copy=True)
    for key in views:
        assert np.array_equal(views[key], copies[key])
        assert not views[key].flags.writeable
        assert copies[key].flags.writeable
    saved = {key: a.copy() for key, a in views.items()}
    env.step(views["action_mask"].argmax(-1))
    env.observe()
    assert all(np.array_equal(views[k], saved[k]) for k in views)


def test_engine_rng_restore_across_reset_and_thread_counts():
    a = tg.BatchEnv(3, seed=913, controlled=[0], threads=1)
    b = tg.BatchEnv(3, seed=913, controlled=[0], threads=3)
    for _ in range(180):
        obs = a.observe()
        actions = obs["action_mask"].argmax(-1)
        ar, ad, ai = a.step(actions)
        br, bd, bi = b.step(actions)
        assert np.array_equal(ar, br) and np.array_equal(ad, bd)
        assert all(np.array_equal(ai[k], bi[k]) for k in ai)
    for g in range(3):
        assert a.save(g) == b.save(g)
        b.load(g, a.save(g))
    assert max(a.episode_index(g) for g in range(3)) > 0
    for _ in range(10):
        actions = a.observe()["action_mask"].argmax(-1)
        a.step(actions); b.step(actions)
    assert all(a.save(g) == b.save(g) for g in range(3))


def test_gym_adapter_reports_turn_cap_as_truncation():
    env = TavernEnv(config={"max_turns": 1, "turn_action_caps": [2]})
    assert isinstance(env, gym.Env)
    obs, _ = env.reset(seed=3)
    assert env.observation_space.contains(obs)
    for _ in range(20):
        obs, reward, terminated, truncated, info = env.step(int(obs["action_mask"].argmax()))
        if terminated or truncated:
            assert truncated and not terminated and reward == 0 and info["placement"] == 0
            assert info["stop_reason"] == "turn_limit"
            break
    else:
        pytest.fail("turn-cap truncation not reached")
    with pytest.raises(RuntimeError):
        env.step(0)


def test_masked_policy_rejects_waiting_and_never_samples_illegal():
    torch.set_num_threads(1)
    model = Policy(hidden=16, embedding=2)
    f, c = torch.zeros(4, tg.OBS_FLOATS), torch.zeros(4, tg.OBS_CARDS, dtype=torch.long)
    mask = torch.zeros(4, tg.N_ACTIONS, dtype=torch.bool)
    mask[:, 31] = True
    dist, _ = model.distribution(f, c, mask)
    assert (dist.sample() == 31).all()
    with pytest.raises(ValueError):
        model.distribution(f, c, torch.zeros_like(mask))


def test_gae_terminal_and_actual_learning_samples():
    adv, returns = gae([{"reward": 0., "value": .2}, {"reward": 1., "value": .3}], 1., 1.)
    assert np.allclose(adv, [.8, .7]) and np.allclose(returns, [1., 1.])
    config = json.loads((Path(__file__).resolve().parents[1] / 'recipes/starter-local.json').read_text())
    config.update(n_games=1, learning_seats=[0, 3], opponent="selfplay")
    torch.manual_seed(6)
    eps, records, stats, states = collect(Policy(hidden=16, embedding=2), config, 87,
        torch.device('cpu'), Timer(torch.device('cpu')), time.perf_counter()+30)
    assert stats['ready_learning_decisions'] >= sum(map(len, eps)) > 0
    assert sum(map(len, eps)) == stats['accepted_learning_samples']
    assert len(records) == 2
    assert stats['rejected'] == 0
    assert all(row['mask'][row['action']] for ep in eps for row in ep)
    # Terminal placement is attached to the last real action even when elimination happened waiting.
    for ep, seat in zip(eps, config["learning_seats"]):
        record = next(r for r in records if r["seat"] == seat)
        assert np.isclose(sum(row['reward'] for row in ep), 1 - 2*(record['placement']-1)/7)

@pytest.mark.parametrize('fault', [False, True])
def test_reward_while_waiting_is_attached_to_last_ready_action(monkeypatch, fault):
    class DelayedEnv:
        n_slots = 1
        def __init__(self, *args, **kwargs):
            self.t = 0
        def observe(self, copy=True):
            mask = np.zeros((1, tg.N_ACTIONS), bool)
            mask[0, 31] = self.t == 0
            return {'features': np.zeros((1, tg.OBS_FLOATS), np.float32),
                    'cards': np.zeros((1, tg.OBS_CARDS), np.int32),
                    'ready': np.array([self.t == 0]), 'action_mask': mask}
        def step(self, actions):
            self.t += 1
            done = self.t == 2
            reason = (5 if fault else 1) if done else 0
            return np.array([-1. if done and not fault else 0.]), np.array([done]), {
                'stop_reason': np.array([reason]), 'placement': np.array([8 if done and not fault else 0]),
                'rejected': np.array([False]),
                'applied': np.array([self.t == 1]), 'deferred': np.array([False])}
        def save(self, g):
            return '{}'
    monkeypatch.setattr(tg, 'BatchEnv', DelayedEnv)
    config = json.loads((Path(__file__).resolve().parents[1] / 'recipes/starter-local.json').read_text())
    config['n_games'] = 1
    eps, rows, stats, _ = collect(Policy(hidden=16, embedding=2), config, 1,
        torch.device('cpu'), Timer(torch.device('cpu')), time.perf_counter()+30)
    assert stats['ready_learning_decisions'] == 1 and stats['batch_steps'] == 2
    if fault:
        assert eps == [] and not rows[0]['valid'] and stats['accepted_learning_samples'] == 0
    else:
        assert len(eps[0]) == 1 and eps[0][0]['reward'] == -1.
        assert rows[0]['valid'] and stats['accepted_learning_samples'] == 1


def test_persistent_and_scoped_workers_match_through_resets(monkeypatch):
    monkeypatch.setenv('TAVERNGYM_WORKERS','scoped')
    a=tg.BatchEnv(3,seed=9223,controlled=[0,3],threads=3)
    monkeypatch.setenv('TAVERNGYM_WORKERS','pooled')
    b=tg.BatchEnv(3,seed=9223,controlled=[0,3],threads=3)
    rng=np.random.default_rng(12)
    for _ in range(500):
        x,y=a.observe(),b.observe()
        assert all(np.array_equal(x[k],y[k]) for k in x)
        acts=x['action_mask'].argmax(-1)
        for slot in np.flatnonzero(x['ready']):
            legal=np.flatnonzero(x['action_mask'][slot])
            acts[slot]=0 if 0 in legal and rng.random()<.4 else rng.choice(legal)
        ar,ad,ai=a.step(acts);br,bd,bi=b.step(acts)
        assert np.array_equal(ar,br) and np.array_equal(ad,bd)
        assert all(np.array_equal(ai[k],bi[k]) for k in ai)
    assert all(a.save(g)==b.save(g) for g in range(3))
    assert max(a.episode_index(g) for g in range(3)) > 0


def test_inactive_lobby_freezes_without_affecting_enabled_lobby():
    a=tg.BatchEnv(2,seed=54,controlled=[0,3],threads=2)
    b=tg.BatchEnv(2,seed=54,controlled=[0,3],threads=2)
    old=a.save(0)
    for _ in range(160):
        obs=a.observe(active_games=[False,True])
        assert not obs['ready'][:2].any() and not obs['features'][:2].any()
        actions=b.observe()['action_mask'].argmax(-1)
        ar,ad,ai=a.step(actions,active_games=[False,True]);br,bd,bi=b.step(actions)
        assert not ad[:2].any() and not ar[:2].any()
        assert np.array_equal(ar[2:],br[2:]) and np.array_equal(ad[2:],bd[2:])
        assert a.save(0)==old and a.save(1)==b.save(1)
    with pytest.raises(ValueError):a.observe(active_games=[True])


def test_optimized_rollout_and_update_preserve_actions_and_weights(monkeypatch):
    from run import ppo_update
    torch.set_num_threads(1)
    config=json.loads((Path(__file__).resolve().parents[1]/'recipes/starter-local.json').read_text())
    config.update(n_games=4,native_threads=2)
    outputs=[]
    for fast in [False,True]:
        monkeypatch.setenv('TAVERNGYM_WORKERS','pooled' if fast else 'scoped')
        torch.manual_seed(44)
        model=Policy(hidden=16,embedding=2)
        optimizer=torch.optim.Adam(model.parameters(),lr=config['learning_rate'],eps=1e-5)
        c=dict(config,fast_inference=fast,skip_finished_lobbies=fast,defer_learner_metrics=fast)
        eps,rows,counts,states=collect(model,c,89,torch.device('cpu'),Timer(torch.device('cpu')),time.perf_counter()+30)
        losses=ppo_update(model,optimizer,eps,c,torch.device('cpu'))
        outputs.append((eps,rows,model.state_dict()))
    assert outputs[0][1]==outputs[1][1]
    assert [[r['action'] for r in ep] for ep in outputs[0][0]]==[[r['action'] for r in ep] for ep in outputs[1][0]]
    assert all(torch.equal(outputs[0][2][k],outputs[1][2][k]) for k in outputs[0][2])


def test_validated_sampler_matches_torch_sampling_and_rng():
    from model import act
    env=tg.BatchEnv(4,seed=991,controlled=[0],threads=2)
    obs=env.observe();indices=np.flatnonzero(obs['ready'])
    torch.manual_seed(31);policy=Policy(hidden=16,embedding=2)
    outputs=[]
    for fast in [False,True]:
        torch.manual_seed(681)
        policy.fast_inference=True;policy.validated_sampler=fast
        result=act(policy,obs,indices,torch.device('cpu'))
        outputs.append((result,torch.get_rng_state()))
    assert all(np.array_equal(a,b) for a,b in zip(outputs[0][0],outputs[1][0]))
    assert torch.equal(outputs[0][1],outputs[1][1])
