"""Contract tests target fog isolation, boundaries, legality, learning and exact resume."""
import json
import time
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from generals.core import game
from generals.core.action import compute_valid_move_mask
from integrations.generals.environment import (RULESET, make_env, build_pool, encode, action_mask, encode_action,
                                               decode_action, observations, transition, reset_done)
from integrations.generals.learning import (advantages, create_train_state, initial_carry, make_collect, make_update)
from integrations.generals.model import Policy
from integrations.generals.runtime import (Ledger, HERE, read_checkpoint, save_checkpoint, sources)


def equal(a, b):
    aa, ta = jax.tree.flatten(a)
    bb, tb = jax.tree.flatten(b)
    return ta == tb and all(np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(aa, bb))


def state():
    grid = jnp.zeros((8, 8), jnp.int32).at[0, 0].set(1).at[7, 7].set(2)
    return game.create_initial_state(grid)


def batch(s):
    return jax.tree.map(lambda x: x[None], s)


def test_joint_codec_exhaustive_roundtrip_and_split_mask():
    ids = jnp.arange(513)
    assert np.array_equal(np.asarray(encode_action(decode_action(ids,width=8),width=8)), np.asarray(ids))
    s = state()._replace(armies=state().armies.at[0, 0].set(2))
    obs = game.get_observation(s, 0)
    mask = action_mask(obs)
    assert mask.shape == (513,) and mask[0]
    joint = mask[1:].reshape((8, 8, 4, 2))
    expected = compute_valid_move_mask(obs.armies, obs.owned_cells, obs.mountains)
    assert np.array_equal(joint[..., 0], expected) and np.array_equal(joint[..., 1], expected)
    # Source row 0 cannot go up or left; two-army source can send half or all-but-one.
    assert not joint[0, 0, 0].any() and not joint[0, 0, 2].any()
    assert joint[0, 0, 1].all() and joint[0, 0, 3].all()
    for action in np.asarray(decode_action(jnp.where(mask)[0][1:],width=8)):
        moved = game.execute_action(s, 0, jnp.asarray(action))
        assert np.array_equal(np.sum(moved.ownership, axis=0)+moved.ownership_neutral, np.ones((8,8)))
        assert not equal(moved.armies, s.armies)


def test_visible_mask_has_no_hidden_mountain_or_general_oracle():
    s = state()
    # Change hidden neutral armies; public scoreboard and hidden structure silhouettes unchanged.
    other = s._replace(armies=s.armies.at[6, 6].set(99))
    # Move opponent general within fog, preserving public opponent totals.
    other = other._replace(ownership=other.ownership.at[1, 7, 7].set(False).at[1, 7, 6].set(True),
                           ownership_neutral=other.ownership_neutral.at[7, 7].set(True).at[7, 6].set(False),
                           generals=other.generals.at[7, 7].set(False).at[7, 6].set(True),
                           armies=other.armies.at[7, 7].set(0).at[7, 6].set(1))
    o, p = game.get_observation(s, 0), game.get_observation(other, 0)
    assert equal(o, p) and equal(encode(o), encode(p)) and equal(action_mask(o), action_mask(p))
    m = Policy(8)
    x, mask = encode(o)[None], action_mask(o)[None]
    params = m.init(jax.random.PRNGKey(0), x, mask)
    assert equal(m.apply(params, x, mask), m.apply(params, encode(p)[None], action_mask(p)[None]))
    # Global enemy army counts are intentionally public; they are not a hidden spatial map.
    changed_total = s._replace(armies=s.armies.at[7, 7].set(10))
    assert not equal(encode(o), encode(game.get_observation(changed_total, 0)))


def test_terminal_capture_cap_priority_final_observation_and_reset():
    s = state()
    s = s._replace(armies=s.armies.at[6, 7].set(10), ownership=s.ownership.at[0, 6, 7].set(True),
                   ownership_neutral=s.ownership_neutral.at[6, 7].set(False), time=jnp.int32(RULESET['truncation']-1))
    actions = jnp.array([[[0, 6, 7, 1, 0], [1, 0, 0, 0, 0]]])
    result = jax.jit(transition)(batch(s), actions)
    assert result.terminated[0] and not result.truncated[0]
    assert np.array_equal(result.reward, [[1., -1.]])
    assert int(result.final_state.time[0]) == RULESET['truncation']
    # A timeout has no capture reward or fabricated winner.
    cap = transition(batch(state()._replace(time=jnp.int32(RULESET['truncation']-1))), jnp.ones((1, 2, 5), jnp.int32))
    assert cap.truncated[0] and not cap.terminated[0] and not cap.reward.any()
    assert cap.final_state.winner[0] == -1
    reset = reset_done(cap.final_state, cap.truncated, batch(state()), jnp.array([0]))
    assert reset.time[0] == 0 and cap.final_state.time[0] == RULESET['truncation']
    # Match upstream non-reset results under the same frozen flags.
    env = make_env()
    timestep, reset_state = env.step(s, actions[0], batch(state()))
    assert equal(timestep.last_state._replace(pool_idx=jnp.int32(0)), jax.tree.map(lambda x: x[0], result.final_state))
    assert reset_state.time == 0


def test_gae_bootstraps_truncation_but_stops_across_reset():
    d = dict(reward=jnp.array([[1.], [100.]]), value=jnp.array([[2.], [3.]]),
             next_value=jnp.array([[5.], [7.]]), terminated=jnp.array([[False], [True]]),
             truncated=jnp.array([[True], [False]]))
    adv, ret = advantages(d, .9, .95)
    np.testing.assert_allclose(adv[:, 0], [3.5, 97.])
    np.testing.assert_allclose(ret[:, 0], [5.5, 100.])


def test_seed_reset_determinism_and_vmap_equivalence():
    p = build_pool([101, 102, 103, 104])
    q = build_pool([101, 102, 103, 104])
    assert equal(p, q)
    assert not equal(jax.tree.map(lambda x: x[0], p), jax.tree.map(lambda x: x[1], p))
    obs = observations(p)
    mask = action_mask(obs)
    ids = jax.random.categorical(jax.random.PRNGKey(0), jnp.where(mask, 0., -1e9))
    actions = decode_action(ids)
    result = transition(p, actions)
    singles = [game.step(jax.tree.map(lambda x: x[i], p), actions[i], general_trade=True)[0] for i in range(4)]
    stacked = jax.tree.map(lambda *a: jnp.stack(a), *singles)
    assert equal(stacked, result.final_state)


def test_ppo_finite_nonzero_update_and_exact_checkpoint_resume(tmp_path):
    cfg = json.loads((HERE/'recipes/full-smoke.json').read_text())
    cfg.update(num_envs=4, rollout_steps=8, epochs=1, network_width=8)
    model = Policy(8)
    tr = create_train_state(model, jax.random.PRNGKey(1), cfg)
    pool = build_pool([101, 102, 103, 104])
    bank = jax.tree.map(lambda x: jnp.repeat(x[None], cfg['history_size'], 0), tr.params)
    c = initial_carry(pool, jax.random.PRNGKey(42), cfg, bank)
    collect, update = make_collect(model, cfg), make_update(model, cfg)
    c, data = collect(tr.params, bank, c, pool)
    assert not data['invalid'].any()
    tr2, metrics = update(tr, data, jax.random.PRNGKey(7))
    assert np.isfinite(metrics).all() and metrics[-1] > 0
    assert not equal(tr.params, tr2.params)
    binding = sources()
    f = tmp_path/'state.msgpack'
    save_checkpoint(f, tr2, bank, c, pool, cfg, binding, 1, Ledger(time.monotonic(), 60, 3), 0)
    meta, (restored, rb, rc, rp) = read_checkpoint(f, tr, cfg, binding)
    assert equal(tr2.params, restored.params) and equal(tr2.opt_state, restored.opt_state)
    assert equal(c, rc) and equal(bank, rb) and equal(pool, rp)
    uninterrupted = collect(tr2.params, bank, c, pool)
    resumed = collect(restored.params, rb, rc, rp)
    assert equal(uninterrupted, resumed)
    t1, m1 = update(tr2, uninterrupted[1], jax.random.PRNGKey(8))
    t2, m2 = update(restored, resumed[1], jax.random.PRNGKey(8))
    assert equal(t1.params, t2.params) and equal(t1.opt_state, t2.opt_state) and equal(m1, m2)
    assert meta['budget']['allocated_worker_seconds'] >= 3
    wrong = dict(binding, rules_sha256='changed')
    with pytest.raises(ValueError, match='rules_sha256'):
        read_checkpoint(f, binding=wrong)
    f.write_bytes(f.read_bytes()+b'bad')
    with pytest.raises(ValueError, match='checksum'):
        read_checkpoint(f)


def test_budget_charges_idle_wall_and_prior_allocation():
    ledger = Ledger(time.monotonic()-.1, .05, previous_seconds=2)
    r = ledger.report()
    assert r['allocated_worker_seconds'] >= 2.1 and ledger.expired()
    assert r['overrun_seconds'] >= 2.05


def test_scripted_actions_always_follow_visible_joint_mask():
    from integrations.generals.environment import scripted_logits
    s = build_pool([91, 92, 93, 94])
    # Exercise sources that can move, rather than only initial forced passes.
    s = s._replace(armies=s.armies + (s.ownership.any(axis=1)*8))
    obs = observations(s)
    mask = action_mask(obs)
    for kind in (0, 1):
        logits = scripted_logits(obs, kind)
        ids = jax.random.categorical(jax.random.PRNGKey(kind), logits)
        assert jnp.take_along_axis(mask, ids[..., None], -1).all()
        assert np.isfinite(jax.nn.softmax(logits)).all()


def test_allocation_supervisor_counts_cpu_idle_and_enforces_deadline(tmp_path):
    import subprocess
    import sys
    cfg = {'allocated_worker_seconds': .3, 'hardware': {'gpu_model':'test-only'}}
    recipe, ledger = tmp_path/'recipe.json', tmp_path/'ledger.json'
    recipe.write_text(json.dumps(cfg))
    command = [sys.executable, '-m', 'integrations.generals.lease', '--recipe', str(recipe),
               '--ledger', str(ledger), '--', sys.executable, '-c', 'import time; time.sleep(10)']
    result = subprocess.run(command, timeout=5, capture_output=True)
    assert result.returncode == 124
    report = json.loads(ledger.read_text())
    assert report['allocated_worker_seconds'] >= .3
    assert report['leases'][0]['stop_reason'] == 'allocated_worker_deadline'
    # Same ledger cannot acquire another worker after its budget is exhausted.
    again = subprocess.run(command, timeout=5, capture_output=True)
    assert again.returncode != 0 and 'exhausted' in again.stderr.decode()
    assert len(json.loads(ledger.read_text())['leases']) == 1


def test_paired_evaluation_and_policy_independent_recorded_replay(tmp_path):
    from integrations.generals.evaluation import evaluate
    from integrations.generals.replay import verify
    model = Policy(8)
    params = model.init(jax.random.PRNGKey(99), jnp.zeros((1,8,8,14)), jnp.ones((1,513),bool))['params']
    path = tmp_path/'replay.npz'
    result = evaluate(model, params, [90201, 90202], ('random',), replay_path=path)
    assert len(result['records']) == 4
    assert {(r['seed'],r['seat']) for r in result['records']} == {(s,p) for s in (90201,90202) for p in (0,1)}
    assert all(r['wins']+r['losses']+r['draws'] == 1 and not r['active'] and r['invalid'] == 0 for r in result['records'])
    replay = verify(path)
    assert replay['exact_final_state'] and replay['invalid_actions'] == replay['active_boundary_mismatches'] == 0


def test_history_snapshot_is_immutable_until_episode_reset():
    cfg = json.loads((HERE/'recipes/full-smoke.json').read_text())
    cfg.update(num_envs=4, rollout_steps=2, epochs=1, network_width=8)
    model=Policy(8)
    tr=create_train_state(model,jax.random.PRNGKey(4),cfg)
    bank=jax.tree.map(lambda p:jnp.repeat(p[None],4,0),tr.params)
    pool=build_pool([801,802,803,804])
    c=initial_carry(pool,jax.random.PRNGKey(10),cfg,bank)
    replaced=jax.tree.map(lambda p:p+.01,bank)
    collect=make_collect(model,cfg)
    continued,_=collect(tr.params,replaced,c,pool)
    assert equal(c.history_params,continued.history_params)
    at_cap=c._replace(states=c.states._replace(time=jnp.full((4,),RULESET['truncation']-1,jnp.int32)))
    reset,_=collect(tr.params,replaced,at_cap,pool)
    expected=jax.tree.map(lambda p:p[reset.history_ids],replaced)
    assert equal(reset.history_params,expected)
    assert not equal(reset.history_params,c.history_params)


def test_simulator_source_binding_excludes_nested_research_harness():
    # Moving the harness under its game must not package hidden research recipes
    # as simulator runtime inputs or change the frozen engine's field contract.
    binding = sources()['integration_sha256']
    assert 'environment.py' in binding and 'runtime.py' in binding
    assert not any(name.startswith('autoresearch/') for name in binding)
