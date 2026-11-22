"""Compute per move: the FLOPs of one policy forward pass for one board, counted from its jaxpr.

The policy worker measures every submitted policy with this before it plays a game, and refuses one over
`DEFAULT['policy_flops_cap']`. Researchers can run the same measurement on their own files:

    python -m integrations.generals_autoresearch.flops --dir DIR    (DIR holds policy.py, model.json, weights.npz)

The count is taken from the traced computation (the jaxpr) of one forward pass at batch 1 on the padded board,
from shapes alone: 2 FLOPs per multiply-add in convolutions and matrix products, 1 per output element of other
arithmetic, 1 per input element of a reduction. It needs no device or compiler, so every worker and grader gets the
same number. A while loop has no countable trip count and is refused; a scan counts its body once per step.
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path


def forward(model, params):
    import jax
    import jax.numpy as jnp

    def act(x, mask):
        def one(a, m): return model.apply({'params': params}, a[None], m[None])[0][0]
        logits = jax.vmap(one)(x, mask)
        return logits, jnp.argmax(jnp.where(mask, logits, -jnp.inf), axis=-1)
    return act


# Primitives that cost nothing to count: data movement, shape changes, comparisons with no arithmetic result.
FREE = {'reshape', 'broadcast_in_dim', 'squeeze', 'expand_dims', 'transpose', 'slice', 'dynamic_slice',
        'dynamic_update_slice', 'concatenate', 'pad', 'gather', 'scatter', 'select_n', 'convert_element_type',
        'iota', 'copy', 'copy_p', 'rev', 'stop_gradient', 'device_put', 'bitcast_convert_type', 'argmax', 'argmin',
        'eq', 'ne', 'lt', 'le', 'gt', 'ge', 'and', 'or', 'not', 'xor', 'is_finite', 'sharding_constraint',
        'optimization_barrier'}
SUB = ('jaxpr', 'call_jaxpr', 'fun_jaxpr')


def _size(var):
    shape = getattr(var.aval, 'shape', ())
    n = 1
    for d in shape:
        n *= int(d)
    return n


def count(jaxpr):
    """FLOPs of a jaxpr: 2 per multiply-add in convolutions and matrix products, 1 per output element of other
    arithmetic, 1 per input element of a reduction. Nested calls are counted; a scan counts its body `length`
    times; a cond counts its most expensive branch; a while loop has no countable trip count and is refused."""
    total = 0
    for eqn in jaxpr.eqns:
        name, params = eqn.primitive.name, eqn.params
        if name == 'while':
            raise ValueError('The forward pass contains a while loop, whose cost cannot be counted; unroll it or '
                             'use a fixed number of layers')
        if name == 'dot_general':
            (lc, rc), (lb, rb) = params['dimension_numbers']
            lhs, rhs = eqn.invars[0].aval.shape, eqn.invars[1].aval.shape
            contract = 1
            for d in lc:
                contract *= int(lhs[d])
            total += 2 * _size(eqn.outvars[0]) * contract
        elif name == 'conv_general_dilated':
            rhs = eqn.invars[1].aval.shape
            spec = params['dimension_numbers'].rhs_spec  # (out feature, in feature, *spatial)
            per_output = int(rhs[spec[1]])
            for d in spec[2:]:
                per_output *= int(rhs[d])
            total += 2 * _size(eqn.outvars[0]) * per_output
        elif name == 'scan':
            total += int(params['length']) * count(params['jaxpr'].jaxpr)
        elif name == 'cond':
            total += max(count(branch.jaxpr) for branch in params['branches'])
        elif any(key in params for key in SUB):
            inner = next(params[key] for key in SUB if key in params)
            total += count(getattr(inner, 'jaxpr', inner))
        elif name.startswith('reduce_') or name in ('cumsum', 'cumprod', 'cummax', 'cumlogsumexp'):
            total += sum(_size(v) for v in eqn.invars if hasattr(v, 'aval'))
        elif name not in FREE:
            total += sum(_size(v) for v in eqn.outvars)
    return total


def measure(model, params):
    """FLOPs per board of `model` with `params`, as the policy worker runs it (one board, greedy action).

    Counted from the traced computation (the jaxpr) of one forward pass at batch 1 on the padded board, from shapes
    alone: no device, no compiler, so a GPU worker, a CPU grader and this machine get the same number.
    """
    import jax
    import jax.numpy as jnp
    from integrations.generals.environment import BOARD_SHAPE, ACTION_COUNT

    def act(p, x, mask):
        def one(a, m): return model.apply({'params': p}, a[None], m[None])[0][0]
        logits = jax.vmap(one)(x, mask)
        return logits, jnp.argmax(jnp.where(mask, logits, -jnp.inf), axis=-1)
    shapes = jax.tree.map(lambda a: jax.ShapeDtypeStruct(jnp.shape(a), jnp.result_type(a)), params)
    closed = jax.make_jaxpr(act)(shapes, jax.ShapeDtypeStruct((1, *BOARD_SHAPE, 14), jnp.float32),
                                 jax.ShapeDtypeStruct((1, ACTION_COUNT), jnp.bool_))
    flops = float(count(closed.jaxpr))
    if not flops > 0:
        raise ValueError('The forward pass does no counted arithmetic')
    return flops


def check(model, params, cap=None):
    from .contract import DEFAULT
    cap = DEFAULT['policy_flops_cap'] if cap is None else cap
    flops = measure(model, params)
    if flops > cap:
        raise ValueError(f'Policy compute cap exceeded: {flops:,.0f} FLOPs per board, the cap is {cap:,.0f}')
    return flops


def load(directory):
    root = Path(directory)
    sys.path.insert(0, str(root))
    spec = importlib.util.spec_from_file_location('entrant_policy_flops', root / 'policy.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = json.loads((root / 'model.json').read_text())
    from .bundle import load_weights
    return module.create(config), load_weights(root / 'weights.npz')


def main():
    from .contract import DEFAULT
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('--dir', required=True, help='folder with policy.py, model.json and weights.npz')
    a = p.parse_args()
    model, params = load(a.dir)
    flops = measure(model, params)
    cap = DEFAULT['policy_flops_cap']
    print(json.dumps({'flops_per_board': flops, 'cap': cap, 'within_cap': flops <= cap,
                      'share_of_cap': round(flops / cap, 4)}))


if __name__ == '__main__':
    main()
