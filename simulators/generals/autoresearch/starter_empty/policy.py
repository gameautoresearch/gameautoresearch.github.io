"""Placeholder network: one linear map from the visible board to the action scores. Replace it.

What the evaluator fixes is the interface only. `create(config)` returns a Flax module with
`init(key, x, mask)` and `apply({'params': p}, x, mask)`, where `x` is the visible
`[batch, 16, 16, 14]` board and `mask` the `[batch, 2049]` legal-action mask (action 0 is pass,
then 8 per cell: 4 directions x whole/half stack). `apply` returns a tuple whose first element is
the `[batch, 2049]` action scores; the evaluator plays the highest-scoring legal action and ignores
everything after the first element. `config` is the dictionary `train.py` writes as `model.json`.
"""
import jax.numpy as jnp
from flax import linen as nn


class Policy(nn.Module):
    @nn.compact
    def __call__(self, x, mask):
        moves = nn.Conv(8, (1, 1))(x).reshape((x.shape[0], -1))
        passed = nn.Dense(1)(jnp.mean(x, axis=(1, 2)))
        scores = jnp.concatenate([passed, moves], -1)
        return jnp.where(mask, scores, -1e9), jnp.zeros(x.shape[0])


def create(config):
    return Policy()
