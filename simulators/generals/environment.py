"""Mid-size 1v1 rules (sides 12..15, padded to 16), player-visible codec and explicit episode boundaries.

No GameState is accepted by encode/mask: spatial hidden state cannot enter the network.
The scoreboard's global opponent army/land totals ARE public under the frozen rules.
"""
from typing import NamedTuple
from functools import lru_cache
import numpy as np
import jax
import jax.numpy as jnp
from generals import GeneralsEnv, compute_valid_move_mask
from generals.core import game
from generals.core.action import DIRECTIONS

PAD_TO = 16
MIN_SIDE, MAX_SIDE = 12, 15
BOARD_SHAPE = (PAD_TO, PAD_TO)
ACTION_COUNT = 1+PAD_TO*PAD_TO*8
RULESET = {
    "id": "generals-research-1v1-medium-v2", "min_grid_size": MIN_SIDE,
    "max_grid_size": MAX_SIDE, "pad_to": PAD_TO,
    "dimension_sampling": "Independent uniform integer sides, inclusive 12..15, fold_in(seed,0x53495a45); board key=PRNGKey(seed)",
    "mountain_density_range": [0.20, 0.20], "num_castles_range": [4, 6],
    "castle_val_range": [10, 16], "min_generals_distance": 10,
    "max_generals_distance": None, "truncation": 1000, "num_players": 2,
    "perfect_info": False, "build_castles": False, "deathtouch_turn": None,
    "general_trade": True,
    "timeout_score": "draw; zero terminal reward",
    "reward": "capture +/-1; nothing else from the environment",
    "action_codec": "0 pass; 1+(((row*16+col)*4+direction)*2+split); padded sources/destinations masked",
    "range_semantics": "Map sides and castle counts inclusive; castle army upper bound exclusive, as in pinned upstream",
    "observation_codec": "9 spatial + 5 public scalar broadcast planes, normalized; no memory",
}
SPATIAL = ("armies", "generals", "castles", "mountains", "neutral_cells", "owned_cells",
           "opponent_cells", "fog_cells", "structures_in_fog")
SCALARS = ("owned_land_count", "owned_army_count", "opponent_land_count", "opponent_army_count", "timestep")


def make_env():
    keys = ("mountain_density_range", "num_castles_range", "castle_val_range")
    kw = {k: tuple(RULESET[k]) for k in keys}
    kw.update({k: RULESET[k] for k in ("min_generals_distance", "max_generals_distance", "truncation",
                                      "num_players", "perfect_info", "build_castles", "deathtouch_turn", "general_trade")})
    kw.update({k: RULESET[k] for k in ('min_grid_size', 'max_grid_size', 'pad_to')})
    return GeneralsEnv(**kw, pool_size=(MAX_SIDE-MIN_SIDE+1)**2)


def encode(obs):
    """Works with arbitrary batch axes; shape (..., H, W, 14)."""
    shape = obs.armies.shape
    planes = [jnp.log1p(obs.armies.astype(jnp.float32)) / 6.0]
    planes += [getattr(obs, name).astype(jnp.float32) for name in SPATIAL[1:]]
    for i, name in enumerate(SCALARS):
        x = getattr(obs, name).astype(jnp.float32)
        x = x / (shape[-2]*shape[-1]) if i in (0, 2) else (x / RULESET['truncation'] if i == 4 else jnp.log1p(x) / 8)
        planes.append(jnp.broadcast_to(x[..., None, None], shape))
    return jnp.stack(planes, axis=-1)


def action_mask(obs):
    shape = obs.armies.shape
    h, w = shape[-2:]
    flat = jax.tree.map(lambda x: x.reshape((-1,) + x.shape[len(shape)-2:]), obs)
    moves = jax.vmap(compute_valid_move_mask)(flat.armies, flat.owned_cells, flat.mountains)
    # All immediate destinations of owned cells are visible (3x3 sight). No fog oracle.
    # Both split choices send >0 for armies>1; no source/direction/split marginal sampling.
    joint = jnp.repeat(moves[..., None], 2, axis=-1).reshape(shape[:-2] + (h*w*8,))
    return jnp.concatenate([jnp.ones(shape[:-2] + (1,), bool), joint], axis=-1)


def decode_action(ids, width=PAD_TO):
    idx = jnp.maximum(ids - 1, 0)
    return jnp.stack([(ids == 0).astype(jnp.int32), idx // (width*8), (idx // 8) % width,
                      (idx // 2) % 4, idx % 2], axis=-1).astype(jnp.int32)


def encode_action(actions, width=PAD_TO):
    a = actions
    return jnp.where(a[..., 0] == 1, 0, 1 + ((a[..., 1]*width+a[..., 2])*4+a[..., 3])*2+a[..., 4])


def potential(obs):
    """Public own land potential; bounded, no privileged critic or dense hidden reward."""
    return obs.owned_land_count.astype(jnp.float32) / np.prod(obs.armies.shape[-2:])


class Boundary(NamedTuple):
    final_state: game.GameState
    terminated: jax.Array
    truncated: jax.Array
    reward: jax.Array


def transition(states, actions):
    """No autoreset; final_state is the true post-action board."""
    final, info = jax.vmap(lambda s, a: game.step(s, a, general_trade=True))(states, actions)
    terminated = info.is_done
    truncated = (final.time >= RULESET["truncation"]) & ~terminated
    reward = jnp.where(terminated[:, None], jnp.where(jnp.arange(2)[None] == info.winner[:, None], 1., -1.), 0.)
    return Boundary(final, terminated, truncated, reward)


def reset_done(final, done, pool, indices):
    reset = jax.tree.map(lambda x: x[indices], pool)
    return jax.tree.map(lambda r, s: jnp.where(done.reshape(done.shape + (1,)*(s.ndim-1)), r, s), reset, final)


def observations(states):
    return jax.vmap(game.get_observations)(states)


@lru_cache(maxsize=(MAX_SIDE-MIN_SIDE+1)**2)
def _pool_builder(height=MAX_SIDE, width=MAX_SIDE):
    # Cache the compiled callable, not mutable maps or seed contents. Recreating
    # a jitted bound method for every evaluation defeats JAX's in-memory cache.
    env = make_env()
    # Upstream init_state() always uses max_grid_size. Select the actual static
    # shape explicitly so a variable-size declaration cannot silently become the largest size.
    return jax.jit(jax.vmap(lambda key: env._make_single_state_fixed(key, height, width)))


@jax.jit
def _dimensions(seeds):
    return jax.vmap(lambda s: jax.random.randint(
        jax.random.fold_in(jax.random.PRNGKey(s), 0x53495a45), (2,), MIN_SIDE, MAX_SIDE+1))(seeds)


def map_dimensions(seeds):
    return np.asarray(jax.device_get(_dimensions(jnp.asarray(seeds, jnp.uint32))))


def build_pool(seeds, dimensions=None):
    """Stable seed-to-map mapping over all 16 rectangular sizes, padded to 16.

    Grouping is setup work only. Fixed eight-key chunks reuse each size's JIT
    across train/evaluation/replay; dummy keys never enter the returned pool.
    Explicit dimensions are for differential fixtures, not a hidden curriculum.
    """
    if not len(seeds):
        raise ValueError('Map seed list must be nonempty')
    dims = map_dimensions(seeds) if dimensions is None else np.asarray(dimensions)
    if dims.shape != (len(seeds), 2) or not np.isin(dims, np.arange(MIN_SIDE, MAX_SIDE+1)).all():
        raise ValueError('Mid-size maps require two inclusive 12..15 dimensions per seed')
    seeds = np.asarray(seeds, np.uint32)
    batches, order = [], []
    for h, w in sorted(set(map(tuple, dims.tolist()))):
        indices = np.flatnonzero((dims == (h,w)).all(-1))
        for start in range(0,len(indices),8):
            chosen = indices[start:start+8]
            keys = jax.vmap(jax.random.PRNGKey)(jnp.asarray(np.pad(
                seeds[chosen], (0,8-len(chosen)), mode='edge')))
            batch = _pool_builder(int(h),int(w))(keys)
            batches.append(jax.tree.map(lambda x:x[:len(chosen)],batch))
            order.extend(chosen.tolist())
    inverse = jnp.asarray(np.argsort(order))
    return jax.tree.map(lambda *xs:jnp.concatenate(xs)[inverse],*batches)


def scripted_logits(obs, kind):
    """Dense vectorized random / expansion policy. Always uses the same joint codec.

    kind=0 uniform joint legal, kind=1 public expansion heuristic. Distinct from
    upstream ExpanderAgent; independent fixed implementation avoids argwhere sorting.
    """
    mask = action_mask(obs)
    h, w = obs.armies.shape[-2:]
    rows = jnp.arange(h)[:, None, None] + DIRECTIONS[None, None, :, 0]
    cols = jnp.arange(w)[None, :, None] + DIRECTIONS[None, None, :, 1]
    rows, cols = jnp.clip(rows, 0, h-1), jnp.clip(cols, 0, w-1)
    target = obs.armies[..., rows, cols]
    mine = obs.owned_cells[..., rows, cols]
    enemy = obs.opponent_cells[..., rows, cols]
    general = obs.generals[..., rows, cols]
    source = obs.armies[..., None]
    moved = jnp.stack([source-1, source//2], -1)
    capture = moved > target[..., None]
    score = 3.*capture*(~mine[..., None]) + 2.*enemy[..., None]*capture
    score += 12.*general[..., None]*enemy[..., None]*capture
    score += 0.15*jnp.log1p(jnp.maximum(moved, 0))
    logits = jnp.concatenate([jnp.full(obs.armies.shape[:-2]+(1,), -2.), score.reshape(obs.armies.shape[:-2]+(-1,))], -1)
    logits = jnp.where(jnp.asarray(kind)[..., None] == 0, 0., logits)
    return jnp.where(mask, logits, -1e9)
