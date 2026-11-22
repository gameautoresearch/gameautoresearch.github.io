"""Placeholder network: one linear layer per tile and a linear layer over the pooled tiles and the global vector,
with linear action heads. It is untrained and deliberately weak. Replace it with your own architecture.

Interface used by the trainer, the evaluator (/opt/v1/policy.py, shown under simulator/) and the harness. Keep these
signatures if you change the network: a checkpoint plays with the network.py saved next to it, and its architecture
spec (the checkpoint's `size`, `width`, `blocks`, `architecture` and every key of configs/architecture.json) is passed
back to `init` as keyword arguments.

    init(key, board_channels, global_channels, *, size, width, blocks, architecture, ...) -> params
    trunk(params, features) -> (tiles [size*size, width], g [width], value [])
    evaluate(params, features, choice, masks) -> (logp, value, entropy)     on the taken path (for your learner)
    sample(params, features, masks, key, *, size, deterministic=False) -> (choice, path masks, logp, value)

`features` is one position as rl/observation.py encodes the mover's own view (`Features(board [size, size, C],
global_state [G])`); `masks` are the legal-action masks of that position (polytopia_bot.gpu.actions.Masks). An action
is chosen in three masked steps (rl/actions.py): a family (17: move, attack, ..., end turn, resign), a source tile,
then a target: a tile for the two spatial families (move = 0, attack = 1), otherwise a parameter id (unit type,
technology, ...) below PARAMETER_TARGETS. The evaluator plays `sample(..., deterministic=True)`.
"""
import math

import jax
import jax.numpy as jnp

from .actions import Choice, PathMasks, family_mask, source_mask, target_mask
from .mode import SIZE
from .observation import fit_board, fit_global

FAMILIES = 17
PARAMETER_TARGETS = 128


def init(key, board_channels, global_channels, *, size=SIZE, width=32, blocks=0, architecture='linear', **options):
    """Fresh weights. `blocks`, `architecture` and any other option are recorded with each checkpoint and ignored by
    this placeholder."""
    keys = iter(jax.random.split(key, 8))

    def linear(a, b, scale=1.):
        return {'w': jax.random.normal(next(keys), (a, b)) * (scale / math.sqrt(a)), 'b': jnp.zeros(b)}

    return {
        'tile': linear(board_channels, width),
        'global': linear(global_channels + width, width),
        'family': linear(width, FAMILIES, .01),
        'source': linear(width, FAMILIES, .01),
        'target': linear(width, FAMILIES, .01),
        'parameter': jnp.zeros((FAMILIES, PARAMETER_TARGETS)),
        'value': linear(width, 1, .1),
    }


def dense(x, layer):
    return x @ layer['w'] + layer['b']


def trunk(params, features):
    board = fit_board(features.board, params['tile']['w'].shape[0])
    tiles = jax.nn.relu(dense(board.reshape(-1, board.shape[-1]), params['tile']))
    width = tiles.shape[-1]
    glob = fit_global(features.global_state, params['global']['w'].shape[0] - width)
    g = jax.nn.relu(dense(jnp.concatenate((tiles.mean(0), glob)), params['global']))
    return tiles, g, dense(g, params['value'])[0]


def target_logits(params, tiles, family):
    rows = max(tiles.shape[0], PARAMETER_TARGETS)
    spatial = jnp.pad(dense(tiles, params['target'])[:, family], (0, rows - tiles.shape[0]))
    parameter = jnp.pad(params['parameter'][family], (0, rows - PARAMETER_TARGETS))
    return jnp.where(family < 2, spatial, parameter)


def log_distribution(logits, mask):
    # a position with no legal action (a finished game) gets a sentinel distribution
    safe = mask.at[0].set(mask[0] | ~jnp.any(mask))
    return jax.nn.log_softmax(jnp.where(safe, logits, -1e9))


def logprob_entropy(logits, mask, selected):
    logp = log_distribution(logits, mask)
    return logp[selected], -jnp.sum(jnp.where(mask, jnp.exp(logp) * logp, 0.))


def evaluate(params, features, choice, masks):
    """log pi(choice), the value and the entropy summed over the three heads, on the taken path (`masks` are its
    PathMasks)."""
    tiles, g, value = trunk(params, features)
    logits = (dense(g, params['family']), dense(tiles, params['source'])[:, choice.family],
              target_logits(params, tiles, choice.family))
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
    t = draw(target_logits(params, tiles, f), tm, keys[2])
    choice, path = Choice(f, s, t), PathMasks(fm, sm, tm)
    logits = (dense(g, params['family']), dense(tiles, params['source'])[:, f], target_logits(params, tiles, f))
    logp = sum(logprob_entropy(l, m, a)[0] for l, m, a in zip(logits, path, choice, strict=True))
    return choice, path, logp, value
