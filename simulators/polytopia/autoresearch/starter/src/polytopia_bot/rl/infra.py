"""Frozen plumbing shared by the trainer, the evaluator and the harness (not research code; not editable).

  END_FAMILY, VALUE_CHUNK            action-family index of "end turn"; rows per chunk of a value pass
  feature_shapes(players, obs)       (board channels, global features) of the observation encoder
  build_network(network, spec, key)  `network.init` with a checkpoint / architecture spec (every spec key passed through)
  network_spec(path)                 a checkpoint's architecture: its JSON config, overlaid with ar_arch.json next to it
  checkpoint_template(path, network) the parameter tree to load a checkpoint into
  MapTable, MAP_TABLE_DIRS           the precomputed training maps
  game_outcomes(...)                 a finished game's result for both seats: the one rule training and evaluation use
  env_setup()                        single-threaded CPU JAX for the evaluator processes
"""
import json
import os
from pathlib import Path

import numpy as np

from .mode import SIZE

END_FAMILY = 15                     # Masks field order: ..., diplomatic, end, resign
VALUE_CHUNK = 8192                  # rows per chunk of a value pass
MAP_TABLE_DIRS = ('/results/maps', os.path.expanduser('~/.cache/polytopia/maps'))
SPEC_SKIP = ('size', 'players', 'obs_version')   # spec keys that are not network.init options


def feature_shapes(players=2, obs_version=1):
    """(board channels, global features) of observation.encode_view as a network of `obs_version` reads them."""
    import jax
    import jax.numpy as jnp
    from .cpu_engine import unpack_view, view_layout
    from .observation import encode_view
    layout = view_layout(SIZE, players)
    width = sum(int(np.prod(s)) for _, s in layout)
    shapes = jax.eval_shape(lambda: encode_view(unpack_view(jnp.zeros(width, jnp.int16).at[0].set(1), layout), SIZE,
                                                obs_version))
    return shapes.board.shape[-1], shapes.global_state.shape[-1]


def build_network(network, spec, key=None, *, groups=1):
    """network.init(key, board, glob, size=..., **every other spec key). groups > 1 multiplies the input widths
    (several views concatenated)."""
    import jax
    board, glob = feature_shapes(spec.get('players', 2), spec.get('obs_version', 1))
    options = {k: v for k, v in spec.items() if k not in SPEC_SKIP and not k.startswith('_')}
    return network.init(jax.random.key(0) if key is None else key, board * groups, glob * groups,
                        size=spec.get('size', SIZE), **options)


def network_spec(path):
    """A checkpoint's architecture spec: size / width / blocks / architecture / obs_version from its JSON config, then
    every key of the ar_arch.json saved next to it (the agent's architecture file), so options a modified network.py
    adds round-trip to the evaluator and the grader."""
    config = json.loads(Path(path).with_suffix('.json').read_text())['config']
    spec = dict(size=config['size'], blocks=config['blocks'], width=config['width'], architecture=config['architecture'])
    for key in ('extra_blocks', 'global_pool', 'block_pool', 'attention'):   # older checkpoints (the hidden panel's)
        if config.get(key):
            spec[key] = config[key]
    if config.get('players', 2) != 2:
        spec['players'] = config['players']
    if config.get('obs_version', 1) != 1:
        spec['obs_version'] = config['obs_version']
    arch = Path(path).with_name('ar_arch.json')
    if arch.exists():
        spec.update({k: v for k, v in json.loads(arch.read_text()).items() if not k.startswith('_')})
    return spec


def checkpoint_template(path, network=None):
    if network is None:
        from . import network
    return build_network(network, network_spec(path))


def game_outcomes(terminated, truncated, engine_rewards, scores):
    """The result of finished two-seat games for both seats, [k, 2] float32: +1 win, -1 loss, 0 draw.

    This is the task's outcome rule, and the trainer and the evaluator (v1/grade.py) both call it. A game that ended
    with a player eliminated has the engine's own terminal reward (+1 / -1). A game cut at the round limit
    (`truncated`) goes to the higher in-game score, and equal scores draw. `scores` [k, 2] is `players.score` of the
    final state, which both engines report; `engine_rewards` [k, 2] is their terminal reward."""
    scores = np.asarray(scores, np.int64).reshape(-1, 2)
    lead = np.sign(scores[:, 0] - scores[:, 1]).astype(np.float32)
    capped = np.stack([lead, -lead], 1)
    truncated = np.asarray(truncated, bool).reshape(-1, 1)
    return np.where(truncated, capped, np.asarray(engine_rewards, np.float32).reshape(-1, 2)).astype(np.float32)


def env_setup():
    os.environ.setdefault('JAX_PLATFORMS', 'cpu')
    os.environ['XLA_FLAGS'] = (os.environ.get('XLA_FLAGS', '') +
                               ' --xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1 ' +
                               os.environ.get('EXTRA_XLA_FLAGS', '')).strip()


class MapTable:
    """Every board of a map mix for a seed range, precomputed: the layouts maps.stream.MapSource.layout(seed) returns."""

    def __init__(self, path):
        with np.load(path) as z:
            self.lo, self.hi = int(z['lo']), int(z['hi'])
            self.types = [str(t) for t in z['types']]
            self.kind = z['kind']
            self.arrays = tuple(np.ascontiguousarray(z[k], np.int32) for k in ('terrain', 'structure', 'resource', 'capitals'))

    @staticmethod
    def find(mix, size, players, lo, hi):
        name = f'table-{mix}-{size}-{players}p-{lo}-{hi}.npz'
        for d in MAP_TABLE_DIRS:
            if os.path.exists(os.path.join(d, name)):
                return os.path.join(d, name)
        return None
