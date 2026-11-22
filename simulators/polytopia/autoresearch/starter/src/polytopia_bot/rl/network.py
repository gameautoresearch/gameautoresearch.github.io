"""The starter network: a plain residual CNN over the board, a global vector, a scalar value and autoregressive masked
action heads (family -> source tile -> target). Pure JAX, so the rollout and PPO run on the same GPU.

Interface used by the trainer, the evaluator and the harness (keep these signatures if you change the network; a
checkpoint plays with the network.py saved next to it, and its spec is passed back to `init` as keyword arguments):

    init(key, board_channels, global_channels, *, size, width, blocks, architecture, ...) -> params
    trunk(params, features) -> (tiles [size*size, width], g [width], value [])
    evaluate(params, features, choice, masks) -> (logp, value, entropy)
    sample(params, features, masks, key, *, size, deterministic=False) -> (choice, path masks, logp, value)

The trainer uses one network for the policy and a second one with the same structure as the critic (its value).
"""
import math
import os

import jax
import jax.numpy as jnp

from .actions import Choice, PathMasks, family_mask, source_mask, target_mask
from .mode import SIZE
from .observation import fit_board, fit_global

# Non-spatial target families (unit, structure, tech, ...) choose among at most this many parameter ids; only spatial
# families (move, attack) target tiles, through the conv features.
PARAMETER_TARGETS = 128
# Numerics (set by the training CLI before JAX is imported; the evaluator runs fp32 on CPU):
#   POLYTOPIA_MATMUL_PRECISION  dense / conv precision, HIGHEST (fp32, default) or DEFAULT (TF32 on the GPU)
#   POLYTOPIA_HEAD_PRECISION    the heads' matmuls (defaults to the above)
#   POLYTOPIA_TRUNK_DTYPE       the convolutions' operand dtype (float32 default, bfloat16 on the GPU learner)
PRECISION = getattr(jax.lax.Precision, os.environ.get('POLYTOPIA_MATMUL_PRECISION', 'HIGHEST'))
HEAD_PRECISION = getattr(jax.lax.Precision, os.environ.get('POLYTOPIA_HEAD_PRECISION', PRECISION.name))
TRUNK_DTYPE = jnp.dtype(os.environ.get('POLYTOPIA_TRUNK_DTYPE', 'float32'))


def init(key, board_channels, global_channels, *, size=SIZE, width=32, blocks=2, architecture='normalized-v2'):
    """Fresh weights. Every key of the architecture JSON arrives here as a keyword: add parameters for new options."""
    if architecture != 'normalized-v2':
        raise ValueError('the starter network is normalized-v2 only')

    def weights(shape, scale=1.):
        nonlocal key
        key, draw = jax.random.split(key)
        return jax.random.normal(draw, shape) * (scale / math.sqrt(math.prod(shape[:-1])))

    def linear(a, b, scale=1.):
        return {'w': weights((a, b), scale), 'b': jnp.zeros(b)}

    return {
        'stem': weights((3, 3, board_channels, width), math.sqrt(2)),
        'blocks': tuple((weights((3, 3, width, width), math.sqrt(2)), weights((3, 3, width, width))) for _ in range(blocks)),
        'global': linear(global_channels + 2 * width, width),
        'norm': {'scale': jnp.ones(width), 'bias': jnp.zeros(width)},
        'critic_hidden': linear(global_channels + 2 * width, width),
        'family': linear(width, 17, .01),
        'source': linear(width, 17, .01),
        'query': linear(2 * width, width),
        'family_embedding': weights((17, width), .1),
        'parameter_embedding': weights((17, max(size * size, PARAMETER_TARGETS), width), .1),
        'value': linear(width, 1, .1),
    }


def dense(x, layer):
    return jnp.matmul(x, layer['w'], precision=HEAD_PRECISION) + layer['b']


def layer_norm(x):
    centered = x - x.mean(axis=-1, keepdims=True)
    return centered * jax.lax.rsqrt(jnp.mean(centered ** 2, axis=-1, keepdims=True) + 1e-5)


def trunk(params, features):
    """Tile features [tiles, width], the global vector g [width] and the value (from the critic_hidden branch)."""
    def conv(x, kernel):
        return jax.lax.conv_general_dilated(x[None].astype(TRUNK_DTYPE), kernel.astype(TRUNK_DTYPE), (1, 1), 'SAME',
                                            dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
                                            precision=PRECISION)[0].astype(jnp.float32)

    activate = lambda z: jax.nn.silu(layer_norm(z))
    x = activate(conv(fit_board(features.board, params['stem'].shape[2]), params['stem']))
    for first, second in params['blocks']:
        x = activate(x + conv(activate(conv(x, first)), second))
    tiles = x.reshape(-1, x.shape[-1])
    width = tiles.shape[-1]
    glob = fit_global(features.global_state, params['global']['w'].shape[0] - 2 * width)
    pooled = jnp.concatenate((tiles.mean(0), tiles.max(0), glob))
    g = jax.nn.silu(layer_norm(dense(pooled, params['global'])) * params['norm']['scale'] + params['norm']['bias'])
    critic = jax.nn.silu(layer_norm(dense(pooled, params['critic_hidden'])))
    return tiles, g, dense(critic, params['value'])[0]


def target_logits(params, tiles, g, family, source):
    query = layer_norm(dense(jnp.concatenate((tiles[source], g)), params['query']) + params['family_embedding'][family])
    # rows past PARAMETER_TARGETS are never legal for non-spatial families; zero keys keep every legal logit identical
    rows = max(tiles.shape[0], PARAMETER_TARGETS)
    spatial = jnp.pad(tiles, ((0, rows - tiles.shape[0]), (0, 0)))
    embedding = jnp.pad(params['parameter_embedding'][family][:PARAMETER_TARGETS], ((0, rows - PARAMETER_TARGETS), (0, 0)))
    keys = jnp.where(family < 2, spatial, embedding)
    return jnp.matmul(keys, query, precision=HEAD_PRECISION) / jnp.sqrt(tiles.shape[-1])


def log_distribution(logits, mask):
    # a lane with no legal move (a finished game) gets a sentinel distribution
    safe = mask.at[0].set(mask[0] | ~jnp.any(mask))
    return jax.nn.log_softmax(jnp.where(safe, logits, -1e9))


def logprob_entropy(logits, mask, selected):
    logp = log_distribution(logits, mask)
    entropy = -jnp.sum(jnp.where(mask, jnp.exp(logp) * logp, 0.))
    return logp[selected], entropy


def evaluate(params, features, choice, masks):
    """log pi(choice), the value and the entropy summed over the three heads (on the taken path)."""
    tiles, g, value = trunk(params, features)
    logits = (dense(g, params['family']), dense(tiles, params['source'])[:, choice.family],
              target_logits(params, tiles, g, choice.family, choice.source))
    pairs = [logprob_entropy(l, m, a) for l, m, a in zip(logits, masks, choice, strict=True)]
    return sum(p[0] for p in pairs), value, sum(p[1] for p in pairs)


def sample(params, features, masks, key, *, size, deterministic=False):
    tiles, g, value = trunk(params, features)
    keys = jax.random.split(key, 3)

    def draw(logits, mask, key):
        lp = log_distribution(logits, mask)
        return jnp.argmax(lp).astype(jnp.int32) if deterministic else jax.random.categorical(key, lp).astype(jnp.int32)
    fm = family_mask(masks)
    f = draw(dense(g, params['family']), fm, keys[0])
    sm = source_mask(masks, f, size)
    s = draw(dense(tiles, params['source'])[:, f], sm, keys[1])
    tm = target_mask(masks, f, s, size)
    t = draw(target_logits(params, tiles, g, f, s), tm, keys[2])
    choice, path = Choice(f, s, t), PathMasks(fm, sm, tm)
    logits = (dense(g, params['family']), dense(tiles, params['source'])[:, f], target_logits(params, tiles, g, f, s))
    logp = sum(logprob_entropy(l, m, a)[0] for l, m, a in zip(logits, path, choice, strict=True))
    return choice, path, logp, value
