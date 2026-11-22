"""Compute per move is measured from the lowered forward pass and capped; parameters only have a storage ceiling."""
import jax
import jax.numpy as jnp
import pytest
from flax import linen as nn
from integrations.generals.environment import BOARD_SHAPE, ACTION_COUNT
from integrations.generals_autoresearch import flops
from integrations.generals_autoresearch.contract import DEFAULT, HERE


def init(model):
    return model.init(jax.random.PRNGKey(0), jnp.zeros((1, *BOARD_SHAPE, 14)), jnp.ones((1, ACTION_COUNT), bool))['params']


def starter(width=16):
    import importlib.util
    spec = importlib.util.spec_from_file_location('starter_policy_flops', HERE / 'starter/policy.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module.create({'width': width})


def test_the_starter_is_far_inside_the_cap_and_the_count_grows_with_width():
    small, wide = starter(16), starter(48)
    a, b = flops.check(small, init(small)), flops.check(wide, init(wide))
    assert 2e6 < a < 5e6 and b > 5 * a and b < DEFAULT['policy_flops_cap']


def test_a_policy_over_the_cap_is_refused():
    model = starter(16)
    with pytest.raises(ValueError, match='Policy compute cap exceeded'):
        flops.check(model, init(model), cap=1e6)


def test_a_while_loop_cannot_be_counted_and_is_refused():
    class Looping(nn.Module):
        @nn.compact
        def __call__(self, x, mask):
            h = nn.Conv(8, (3, 3))(x)
            h = jax.lax.while_loop(lambda c: c[0] < x.shape[1], lambda c: (c[0] + 1, jnp.tanh(c[1])), (0, h))[1]
            moves = nn.Conv(8, (1, 1))(h).reshape((x.shape[0], -1))
            return jnp.where(mask, jnp.concatenate([jnp.zeros((x.shape[0], 1)), moves], -1), -1e9), moves[:, 0]
    model = Looping()
    with pytest.raises(ValueError, match='while loop'):
        flops.measure(model, init(model))


def test_the_contract_declares_the_cap_and_a_storage_ceiling_not_a_parameter_cap():
    assert DEFAULT['policy_flops_cap'] == 1_000_000_000 and DEFAULT['policy_tensor_cap'] == 32_000_000
    assert DEFAULT['artifact_bytes_cap'] >= 4 * DEFAULT['policy_tensor_cap']


def test_dilated_convolutions_and_scans_are_counted():
    class Dilated(nn.Module):
        @nn.compact
        def __call__(self, x, mask):
            for d in (1, 2, 4):
                x = nn.relu(nn.Conv(16, (3, 3), kernel_dilation=(d, d))(x))
            moves = nn.Conv(8, (1, 1))(x).reshape((x.shape[0], -1))
            return jnp.where(mask, jnp.concatenate([jnp.zeros((x.shape[0], 1)), moves], -1), -1e9), moves[:, 0]
    model = Dilated()
    plain = starter(16)
    assert flops.measure(model, init(model)) > 0.8 * flops.measure(plain, init(plain))
    def body(c, _): return jnp.tanh(c), None
    def f(x): return jax.lax.scan(body, x, None, length=5)[0]
    assert flops.count(jax.make_jaxpr(f)(jnp.zeros((4, 4))).jaxpr) == 5 * 16
