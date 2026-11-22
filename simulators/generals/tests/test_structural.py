"""Structural fast paths preserve visible inputs, boundaries and PPO objectives."""
import json
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from integrations.generals.runtime import HERE
from integrations.generals.model import Policy
from integrations.generals.learning import create_train_state, initial_carry, make_collect, make_update
from integrations.generals.environment import build_pool,RULESET
from integrations.generals.efficiency import make_collect_optimized


@pytest.mark.parametrize('shape', [(8,8),(24,24),(19,23)])
def test_implicit_policy_vmapped_parameters_match_bf16_flax(shape):
    from integrations.generals.implicit import apply
    model = Policy(16, convolution='slice', compute_dtype='bfloat16')
    keys = jax.random.split(jax.random.PRNGKey(341), 4)
    x = jax.random.normal(keys[0], (2, *shape, 14))
    masks = jax.random.bernoulli(keys[1], .5, (2, 1+shape[0]*shape[1]*8)).at[:, 0].set(True)
    params = jax.vmap(lambda k: model.init(k, x[:1], masks[:1])['params'])(keys[2:])
    reference = jax.vmap(lambda p, a, m: model.apply({'params': p}, a[None], m[None]))(params, x, masks)
    actual = jax.vmap(lambda p, a, m: apply(p, a[None], m[None], interpret=True))(params, x, masks)
    for a, b in zip(reference, actual):
        np.testing.assert_array_equal(a, b)


def test_implicit_inference_rejects_unsupported_precision():
    from integrations.generals.model import historical_policy
    with pytest.raises(ValueError, match='width16 BF16'):
        historical_policy(Policy(16), {'historical_convolution': 'implicit'})


def test_inverse_cdf_distribution_legality_and_seed_reproducibility():
    from integrations.generals.learning import sample
    # Include zero mass before/between/after legal actions, and a heavily
    # dominant action: cumulative rounding must never select a masked tail.
    probs = np.array([0., .1, .2, 0., .3, .4, 0.], np.float32)
    logits = jnp.broadcast_to(jnp.log(jnp.asarray(probs)), (100000, 7))
    key = jax.random.PRNGKey(71)
    draw = jax.jit(lambda k, l: sample(k, l, 'inverse_cdf'))
    ids = np.asarray(draw(key, logits))
    np.testing.assert_array_equal(ids, draw(key, logits))
    observed = np.bincount(ids, minlength=7)/len(ids)
    np.testing.assert_allclose(observed, probs, atol=.005, rtol=0)
    assert not observed[[0, 3, 6]].any()
    mask = jax.random.bernoulli(key, .1, (20000, 513)).at[:, 0].set(True)
    sparse = jnp.where(mask, jax.random.normal(key, (20000, 513))*10, -1e9)
    chosen = draw(key, sparse)
    assert jnp.take_along_axis(mask, chosen[:, None], -1).all()


def test_compiled_evaluation_and_pool_factories_reuse_functions_not_state():
    from integrations.generals.environment import _pool_builder
    from integrations.generals.evaluation import make_evaluator
    assert _pool_builder() is _pool_builder()
    assert make_evaluator(Policy(8), ('random',)) is make_evaluator(Policy(8), ('random',))
    a = build_pool([10001, 10002])
    b = build_pool([10002, 10001])
    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)):
        np.testing.assert_array_equal(x, y[::-1])


@pytest.mark.parametrize('kernel', ['patch', 'slice'])
def test_convolution_parameters_forward_and_gradients(kernel):
    # Nonuniform random inputs expose padding/channel-layout mistakes that all
    # zero fixture inputs cannot; identical named parameters allow old weights.
    x = jax.random.normal(jax.random.PRNGKey(13), (3, 8, 8, 14))
    mask = jax.random.bernoulli(jax.random.PRNGKey(14), .4, (3, 513)).at[:, 0].set(True)
    baseline, candidate = Policy(8), Policy(8, convolution=kernel)
    params = baseline.init(jax.random.PRNGKey(15), x, mask)['params']
    def output(model, p):
        return model.apply({'params': p}, x, mask)
    for a, b in zip(output(baseline, params), output(candidate, params)):
        np.testing.assert_allclose(a, b, atol=3e-6, rtol=3e-5)
    def loss(model, p):
        logits, values = output(model, p)
        return jax.nn.log_softmax(logits)[:, 0].mean()+jnp.mean(values**2)
    a = jax.grad(lambda p: loss(baseline, p))(params)
    b = jax.grad(lambda p: loss(candidate, p))(params)
    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)):
        np.testing.assert_allclose(x, y, atol=3e-6, rtol=3e-5)


@pytest.mark.parametrize('time_value', [0, RULESET['truncation']-10, RULESET['truncation']-1])
def test_compact_inference_and_guarded_resets_preserve_frozen_history(time_value):
    cfg = json.loads((HERE/'recipes/full-smoke.json').read_text())
    cfg.update(num_envs=4, rollout_steps=16, network_width=8)
    model = Policy(8)
    train = create_train_state(model, jax.random.PRNGKey(7), cfg)
    # All four opponent kinds plus genuinely different cached weights; replacing
    # a bank slot must not change an already running historical episode.
    bank = jax.tree.map(lambda p: jnp.stack([p+i*.01 for i in range(4)]), train.params)
    pool = build_pool([10001, 10002, 10003, 10004])
    carry = initial_carry(pool, jax.random.PRNGKey(8), cfg, bank)
    carry = carry._replace(opponents=jnp.arange(4), states=carry.states._replace(
        time=jnp.full((4,), time_value, jnp.int32)))
    replaced = jax.tree.map(lambda p: p+.005, bank)
    a, da = make_collect(model, cfg)(train.params, replaced, carry, pool)
    fast = dict(cfg, guard_resets=True, compact_opponent_inference=True)
    b, db = make_collect_optimized(model, fast)(train.params, replaced, carry, pool)
    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)):
        np.testing.assert_array_equal(x, y)
    for key in ('action', 'mask', 'reward', 'terminated', 'truncated', 'invalid'):
        np.testing.assert_array_equal(da[key], db[key])
    for key in ('value', 'old_logp', 'next_value'):
        x, y = np.asarray(da[key]), np.asarray(db[key])
        if key == 'next_value':
            x = np.where(da['terminated'], 0., x)
            y = np.where(db['terminated'], 0., y)
        np.testing.assert_allclose(x, y, atol=5e-5, rtol=5e-5)
    if time_value >= RULESET['truncation']-10:
        assert int(db['truncated'].sum()) == 4
    else:
        for x, y in zip(jax.tree.leaves(carry.history_params), jax.tree.leaves(b.history_params)):
            np.testing.assert_array_equal(x, y)


def test_hybrid_learner_matches_ppo_update():
    cfg = json.loads((HERE/'recipes/full-smoke.json').read_text())
    cfg.update(num_envs=4, rollout_steps=8, network_width=8, epochs=1)
    model = Policy(8)
    train = create_train_state(model, jax.random.PRNGKey(61), cfg)
    bank = jax.tree.map(lambda p: jnp.repeat(p[None], 4, 0), train.params)
    pool = build_pool([101, 102, 103, 104])
    carry = initial_carry(pool, jax.random.PRNGKey(62), cfg, bank)
    _, data = make_collect(model, cfg)(train.params, bank, carry, pool)
    a = make_update(model, cfg)(train, data, jax.random.PRNGKey(63))
    b = make_update(model, dict(cfg, learner_convolution='patch'))(train, data, jax.random.PRNGKey(63))
    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)):
        np.testing.assert_allclose(x, y, atol=5e-5, rtol=5e-5)


def test_bfloat16_compute_keeps_float32_parameters_outputs_and_legal_learning():
    cfg = json.loads((HERE/'recipes/full-smoke.json').read_text())
    cfg.update(num_envs=4, rollout_steps=8, network_width=8, epochs=1,
               guard_resets=True, compact_opponent_inference=True, sampler='inverse_cdf')
    model = Policy(8, compute_dtype='bfloat16')
    train = create_train_state(model, jax.random.PRNGKey(81), cfg)
    assert all(p.dtype == jnp.float32 for p in jax.tree.leaves(train.params))
    bank = jax.tree.map(lambda p: jnp.repeat(p[None], 4, 0), train.params)
    pool = build_pool([101, 102, 103, 104])
    carry = initial_carry(pool, jax.random.PRNGKey(82), cfg, bank)
    _, data = make_collect_optimized(model, cfg)(train.params, bank, carry, pool)
    assert not data['invalid'].any() and data['value'].dtype == jnp.float32
    changed, metrics = make_update(model, cfg)(train, data, jax.random.PRNGKey(83))
    assert np.isfinite(metrics).all() and float(metrics[-1]) > 0
    assert all(p.dtype == jnp.float32 for p in jax.tree.leaves(changed.params))


def test_packed_visible_storage_and_ppo_are_lossless():
    from integrations.generals.environment import observations, encode, action_mask
    from integrations.generals.packed import pack_observation, unpack_observation
    pool = build_pool([101, 102, 103, 104])
    obs = observations(pool)
    mask = action_mask(obs)
    packed = jax.jit(pack_observation)(obs, mask)
    x, m = jax.jit(unpack_observation)(packed)
    np.testing.assert_array_equal(x, encode(obs))
    np.testing.assert_array_equal(m, mask)
    before = encode(obs).nbytes+mask.nbytes
    after = sum(v.nbytes for v in packed.values())
    assert before/after > 8
    cfg = json.loads((HERE/'recipes/full-smoke.json').read_text())
    cfg.update(num_envs=4, rollout_steps=8, network_width=8, epochs=1)
    model = Policy(8)
    train = create_train_state(model, jax.random.PRNGKey(91), cfg)
    bank = jax.tree.map(lambda p: jnp.repeat(p[None], 4, 0), train.params)
    carry = initial_carry(pool, jax.random.PRNGKey(92), cfg, bank)
    a, da = make_collect_optimized(model, cfg)(train.params, bank, carry, pool)
    pcfg = dict(cfg, packed_rollout=True)
    b, db = make_collect_optimized(model, pcfg)(train.params, bank, carry, pool)
    for u, v in zip(jax.tree.leaves(a), jax.tree.leaves(b)):
        np.testing.assert_array_equal(u, v)
    x, m = unpack_observation(db)
    np.testing.assert_array_equal(x, da['x'])
    np.testing.assert_array_equal(m, da['mask'])
    ta = make_update(model, cfg)(train, da, jax.random.PRNGKey(93))
    tb = make_update(model, pcfg)(train, db, jax.random.PRNGKey(93))
    for u, v in zip(jax.tree.leaves(ta), jax.tree.leaves(tb)):
        np.testing.assert_allclose(u, v, atol=5e-6, rtol=5e-5)


@pytest.mark.parametrize('count', [0, 1, 2, 3, 4])
def test_sparse_refresh_matches_dense_including_cohort_fallback(count):
    from integrations.generals.resetting import refresh, opponent_weights
    cfg = json.loads((HERE/'recipes/full-smoke.json').read_text())
    cfg.update(num_envs=4, network_width=8)
    model = Policy(8)
    train = create_train_state(model, jax.random.PRNGKey(101), cfg)
    bank = jax.tree.map(lambda p: jnp.stack([p+i*.01 for i in range(4)]), train.params)
    old = jax.tree.map(lambda p: p[::-1], bank)
    pool = build_pool([101, 102, 103, 104])
    final = jax.tree.map(lambda x: x[::-1], pool)
    done = jnp.arange(4) < count
    kinds = jnp.array([0, 1, 2, 3])
    cached = opponent_weights(train.params, old, kinds)
    args = (final, done, pool, jnp.arange(4), bank, jnp.arange(4), old, train.params, kinds, cached)
    dense = jax.jit(lambda *a: refresh(*a, capacity=0))(*args)
    sparse = jax.jit(lambda *a: refresh(*a, capacity=2))(*args)
    for x, y in zip(jax.tree.leaves(dense), jax.tree.leaves(sparse)):
        np.testing.assert_array_equal(x, y)


def test_indexed_checkpoint_roundtrip_and_collision_fallback(tmp_path, monkeypatch):
    import time
    from integrations.generals import runtime
    cfg = json.loads((HERE/'recipes/full-smoke.json').read_text())
    cfg.update(num_envs=4, network_width=8, compact_checkpoint=True)
    model = Policy(8)
    train = create_train_state(model, jax.random.PRNGKey(111), cfg)
    bank = jax.tree.map(lambda p: jnp.repeat(p[None], 4, 0), train.params)
    # Distinct head-only weights cannot be merged based on a trunk signature.
    bank['Dense_0']['bias'] = bank['Dense_0']['bias'].at[1].add(.25)
    pool = build_pool([101, 102, 103, 104])
    carry = initial_carry(pool, jax.random.PRNGKey(112), cfg, bank)
    carry = carry._replace(history_params=jax.tree.map(lambda p: p[jnp.array([0, 1, 0, 1])], bank))
    binding = runtime.sources()
    f = tmp_path/'indexed.msgpack'
    meta = runtime.save_checkpoint(f, train, bank, carry, pool, cfg, binding, 0,
                                   runtime.Ledger(time.monotonic(), 60), 0)
    assert meta['format'] == 2 and meta['unique_history_rows'] == 2
    _, restored = runtime.read_checkpoint(f, train, cfg, binding)
    for x, y in zip(jax.tree.leaves((train, bank, carry, pool)), jax.tree.leaves(restored)):
        np.testing.assert_array_equal(x, y)
    monkeypatch.setattr(runtime, '_history_fingerprints', lambda h: jnp.zeros((4, 2), jnp.uint32))
    selected, ids = runtime.compact_history(carry.history_params)
    assert ids is None  # An intentional collision must fall back, never corrupt.
