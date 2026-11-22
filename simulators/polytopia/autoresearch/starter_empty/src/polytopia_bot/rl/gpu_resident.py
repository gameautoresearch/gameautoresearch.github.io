"""The boilerplate trainer: self-play rollouts on one GPU with the CUDA engine stepping thousands of games in place,
and no learning.

What it does:

  * self-play: one policy plays both seats of every game, sampling every decision on the GPU;
  * the games' results by the rule the evaluator scores with (infra.game_outcomes): +1 win / -1 loss when a player is
    eliminated; a game cut at the round limit goes to the higher in-game score (+1 / -1) and equal scores draw (0);
  * every `batch` decisions it calls `_update()`, which changes nothing: there is no loss, no optimizer and no update
    (training/learner.py is empty). Trained as it is, the network keeps its random initialization.

The learning algorithm is yours to write: what to store from the rollout, returns or advantages, the loss and the
optimizer. The places to put it are `_record` (each step's decisions), `_finished` (each finished game's result) and
`_update` (the update itself). What each step gives you, as device arrays (torch on the GPU; `self._to_jax` and
`self._to_torch` convert without copies):

  view [Gs, view_width] int16       the mover's own packed view of the position (rl/cpu_engine.py view_layout);
                                    `rl.cpu_engine.view_features` encodes it for the network (rl/observation.py)
  opp_view [Gs, view_width] int16   the same position from the other seat's view
  masks [Gs, mask_bytes] uint8      the position's packed legal-action masks (unpacked by `dense_masks` below)
  choice [Gs, 3] int32              the (family, source, target) the policy chose
  path [Gs, ...] uint8              the masks on the chosen path, packed (family 3 bytes, source, target)
  logp [Gs] float32                 log pi(choice) under the policy that chose it
  mover [Gs] int32                  the moving seat, 1 or 2

A set's games that finish in a step restart at once on fresh maps, so a game index is a lane, not one game; the
results of the step reach `_finished` one step later (the engine runs ahead).

Rollout mechanics (tuned for throughput; change them if you need to): the games are split into `sets` engine batches
stepped in a round robin, so the engine of one set runs while the GPU samples another; each set's encode + sampling is
one fused jit.

Checkpoints: runner.save_checkpoint, as latest / latest-critic / snapshots/update-N / final (+ -critic) in the run
folder, with protocol.json and training.jsonl. `critic` is a second network of the same structure, written as the
-critic files that the host's packaging expects; nothing here trains or uses it. Runs always start from scratch.

Exports: after the first `_update()` and then every `export_seconds`, the current policy and critic are also written
as `exports/update-N/` (final.npz, final.json, final-critic.npz, final-critic.json, export.json). A folder appears
under that name only once it is whole (it is written under another name and renamed), and the two newest are kept. If
the run is cut off at its time limit, the host packages the newest whole export as the run's network; a run with no
whole export is lost. Keep this (or your own equivalent) working if you change the loop.

The host's own training of a submitted recipe (/opt/v1/train.py, shown under simulator/) imports
`GpuResidentTrainer` from this file and wraps `run` and `_update`; it reads `root`, `params`, `run_config`, `version`,
`_host_state()` and `_write()`. Keep those names. It refuses a run whose final policy is unchanged from its initial
weights.

Stopping: `ar stop` makes the host create the file named by $POLYTOPIA_STOP_REQUEST. The loop checks it at every step
and then ends as it does at its time limit: final checkpoint, training-complete.json. A trainer that has not ended
180 seconds after the request is ended by the host and packaged from its newest whole export.
"""
import json
import os
import shutil
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from polytopia_bot.ruleset import RULESET_ID
from . import mode, training_log
from .infra import MAP_TABLE_DIRS, MapTable, build_network, game_outcomes

EXPORTS, EXPORTS_KEPT = 'exports', 2            # the folder of whole policy + critic exports, and how many stay
STOP_REQUEST_ENV = 'POLYTOPIA_STOP_REQUEST'     # names the file the host creates to ask this run to end


@dataclass
class LearnerConfig:
    """The recipe (the config JSON's top-level fields). Add fields here for your own options."""
    size: int = mode.SIZE             # board, players, maps, round limit, map seeds: the frozen mode (rl/mode.py)
    players: int = mode.PLAYERS
    seed: int = 801
    map_seed_low: int = mode.MAP_SEED_LOW
    map_seed_high: int = mode.MAP_SEED_HIGH
    maps: str = mode.MAPS
    max_turns: int = mode.ROUND_LIMIT  # rounds before a game is cut (the evaluator then scores it by in-game score)
    batch: int = 65536                # decisions between two calls of `_update`
    learning_rate: float = 3e-4       # required by the host's recipe check (0 < lr <= 0.01); unused here
    entropy_coef: float = 0.          # required by the host's recipe check (0 <= coef <= 0.1); unused here
    checkpoint_every: int = 25        # updates between latest.npz writes
    export_seconds: float = 60.       # seconds between whole exports of the policy and critic (exports/update-N)
    snapshot_every: int = 50          # updates between snapshots/update-N.npz
    snapshot_at: tuple = field(default_factory=tuple)
    training_updates: int = 0         # stop after this many updates (0: by training_seconds)
    training_seconds: int = 600

    def validate(self):
        if self.players != 2:
            raise ValueError('two-player games only')
        if self.batch < 1:
            raise ValueError('batch is a positive number of decisions')
        if isinstance(self.max_turns, bool) or not isinstance(self.max_turns, int) or self.max_turns < 1:
            raise ValueError('max_turns is the round limit: a positive whole number')


def network_spec_of(config, network=None):
    """The architecture spec of a fresh network: the defaults below, overridden by the recipe's arch JSON."""
    from .observation import OBS_VERSION
    spec = dict(size=config.size, width=32, blocks=0, architecture='linear', obs_version=OBS_VERSION)
    spec.update(network or {})
    return spec


def policy_template(spec, key=None):
    from . import network
    return build_network(network, spec, key)


def critic_template(spec, key=None):
    """A second network of the same structure, written as the run's -critic files."""
    from . import network
    return build_network(network, spec, key)


class _Checkpointer:
    """Writes checkpoints on a background thread; submit() joins the previous write first so files land in order."""

    def __init__(self):
        self.thread, self.error, self.seconds, self.writes = None, None, 0., 0

    def submit(self, fn):
        self.wait()

        def run():
            t0 = time.perf_counter()
            try:
                fn()
            except BaseException as exc:   # noqa: BLE001 - re-raised on the loop thread
                self.error = exc
            self.seconds = time.perf_counter() - t0
            self.writes += 1

        self.thread = threading.Thread(target=run, name='checkpoint-writer', daemon=True)
        self.thread.start()

    def wait(self):
        if self.thread is not None:
            self.thread.join()
            self.thread = None
        if self.error is not None:
            exc, self.error = self.error, None
            raise RuntimeError('the background checkpoint write failed') from exc


class GpuResidentTrainer:
    """`config`: the recipe dict or a LearnerConfig; `run_dir`: the run folder; `network`: the architecture dict (the
    recipe's arch JSON) of the fresh network. `ring` is accepted for the launcher's command line and unused here."""

    def __init__(self, config, run_dir, *, network=None, training_seconds=None, games=4096, sets=4, ring=160,
                 batch=None, checkpoint_every=None, snapshot_every=None, map_table=None, device=0, seed=None, log=print,
                 commit=lambda: None):
        import torch
        import jax
        self.torch, self.jax = torch, jax
        cfg = dict(asdict(config) if isinstance(config, LearnerConfig) else config)
        overrides = dict(training_seconds=training_seconds, batch=batch, checkpoint_every=checkpoint_every,
                         snapshot_every=snapshot_every, seed=seed)
        cfg.update({k: int(v) for k, v in overrides.items() if v})
        cfg['snapshot_at'] = tuple(cfg.get('snapshot_at', ()))
        self.config = cfg = LearnerConfig(**cfg)
        cfg.validate()
        self.root, self.log, self.commit = Path(run_dir), log, commit
        self.G, self.sets = int(games), int(sets)
        if self.sets < 1 or self.G % self.sets:
            raise ValueError('games must be a positive multiple of sets')
        self.Gs = self.G // self.sets
        self.map_table_path, self.device_index = map_table, device
        self.async_ticks = self.sets > 1
        if self.root.exists() and (self.root / 'latest.npz').exists():
            raise RuntimeError(f'{self.root} already has a checkpoint: runs start from scratch')
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / 'snapshots').mkdir(exist_ok=True)
        self.rng = np.random.default_rng(cfg.seed)
        self.version = 0
        self.decisions = 0
        self.stop_request = os.environ.get(STOP_REQUEST_ENV) or None
        self._build_network(network)
        self._build_engine()
        self._build_jax()

    # ---- setup
    def _build_network(self, network):
        jax = self.jax
        from .runner import Config
        cfg = self.config
        self.network = spec = network_spec_of(cfg, network)
        key = jax.random.key(cfg.seed)
        self.params = policy_template(spec, jax.random.fold_in(key, 0))
        self.critic = critic_template(spec, jax.random.fold_in(key, 1))
        # the checkpoint's JSON config (runner.Config): the evaluator rebuilds the network from it + ar_arch.json
        self.run_config = Config(size=cfg.size, width=spec['width'], blocks=spec['blocks'], architecture=spec['architecture'],
                                 players=cfg.players, obs_version=spec.get('obs_version', 1), no_resign=True,
                                 reward_mode='native', learning_rate=cfg.learning_rate, entropy_coef=cfg.entropy_coef,
                                 seed=cfg.seed)

    def _build_engine(self):
        torch = self.torch
        from .cpu_engine import view_layout
        from .gpu_engine import GpuGames
        cfg = self.config
        size, players, Gs = cfg.size, cfg.players, self.Gs
        self.device = torch.device('cuda', self.device_index)
        torch.cuda.set_device(self.device)
        self.layout = view_layout(size, players)
        offsets, at = {}, 0
        for name, shape in self.layout:
            offsets[name] = at
            at += int(np.prod(shape))
        self.vw, self.player_at = at, offsets['player']
        self.round_limit = limit = int(cfg.max_turns)
        path = self.map_table_path or MapTable.find(cfg.maps, size, players, cfg.map_seed_low, cfg.map_seed_high)
        if not path or not os.path.exists(path):
            raise FileNotFoundError(f'map table for {cfg.maps} {size} {players}p [{cfg.map_seed_low}, {cfg.map_seed_high}) '
                                    f'not found in {MAP_TABLE_DIRS}; pass map_table')
        table = MapTable(path)
        if table.lo != cfg.map_seed_low or table.hi != cfg.map_seed_high:
            raise ValueError(f'map table seeds [{table.lo}, {table.hi}) differ from the config')
        self.map_lo, self.map_hi, self.map_types, self.map_kind = table.lo, table.hi, table.types, table.kind
        n = size * size
        self.map_gen = torch.Generator().manual_seed(int(cfg.seed))
        self.clock, self.calls = {}, {}
        # the map table on the device ([3, M, n] terrain / structure / resource, [M, players] capitals)
        self.table3 = torch.from_numpy(np.stack(table.arrays[:3])).to(self.device)
        self.table_caps = torch.from_numpy(table.arrays[3]).to(self.device)
        self.playing_kind = np.zeros(self.G, np.int8)

        class GameSet:
            pass

        self.gsets = []
        for k in range(self.sets):
            gs = GameSet()
            gs.k, gs.lo, gs.hi = k, k * Gs, (k + 1) * Gs
            gs.games = np.arange(gs.lo, gs.hi)
            gs.env = GpuGames(size=size, games=Gs, players=players, max_turns=limit, no_resign=True, max_actions=10_000,
                              device=self.device_index)
            if gs.env.view_width != self.vw:
                raise RuntimeError(f'packed view width differs: layout {self.vw} vs C engine {gs.env.view_width}')
            seeds = self._draw_seeds(Gs)
            self.playing_kind[gs.lo:gs.hi] = self.map_kind[seeds - self.map_lo]
            gs.env.reset(seeds, tuple(np.ascontiguousarray(a[seeds - self.map_lo]) for a in table.arrays))
            gs.seeds = torch.from_numpy(seeds).to(self.device)
            gs.seeds_host = seeds
            # the next step's seeds and table rows: pinned host buffers, two non-blocking H2D copies per step
            gs.seeds_pin = torch.empty(Gs, dtype=torch.int32).pin_memory()
            gs.rows_pin = torch.empty(Gs, dtype=torch.int64).pin_memory()
            gs.rows = torch.zeros(Gs, dtype=torch.int64, device=self.device)
            gs.layouts3 = torch.zeros((3, Gs, n), dtype=torch.int32, device=self.device)
            gs.layout_caps = torch.zeros((Gs, players), dtype=torch.int32, device=self.device)
            gs.layouts = (gs.layouts3[0], gs.layouts3[1], gs.layouts3[2], gs.layout_caps)
            gs.out = gs.env.alloc_device_outputs(self.device)
            # the first state's masks and views from the host-pointer calls
            gs.out['masks'].copy_(torch.from_numpy(gs.env.legal_masks(packed=True)))
            view = gs.env.observe()
            mover = view[:, self.player_at].astype(np.int32)
            others = np.stack([gs.env.observe(player=p) for p in range(1, players + 1)], 1)      # [Gs, players, vw]
            opp = others[np.arange(Gs), np.where(mover == 1, 1, 0)]
            gs.out['view'].copy_(torch.from_numpy(view))
            gs.out['opp_view'][:, 0].copy_(torch.from_numpy(np.ascontiguousarray(opp)))
            # the tick's outcomes: one device staging buffer (scores, rewards, status, flags as bytes) copied to pinned
            # host memory in one non-blocking transfer, read in _outcomes
            seg_bytes = (4 * players * Gs, 4 * players * Gs, 4 * Gs, 3 * Gs)
            offs = np.concatenate([[0], np.cumsum(seg_bytes)])
            gs.stage_dev = torch.empty(int(offs[-1]), dtype=torch.uint8, device=self.device)
            gs.stage_host = torch.empty(int(offs[-1]), dtype=torch.uint8).pin_memory()
            sl = [slice(int(offs[i]), int(offs[i + 1])) for i in range(4)]
            gs.stage_in = (gs.stage_dev[sl[0]].view(torch.int32).view(Gs, players),
                           gs.stage_dev[sl[1]].view(torch.float32).view(Gs, players),
                           gs.stage_dev[sl[2]].view(torch.int32), gs.stage_dev[sl[3]])
            host = gs.stage_host.numpy()
            gs.stage_out = dict(scores=host[sl[0]].view(np.int32).reshape(Gs, players),
                                rewards=host[sl[1]].view(np.float32).reshape(Gs, players),
                                status=host[sl[2]].view(np.int32), flags=host[sl[3]].reshape(3, Gs))
            gs.stage_event = torch.cuda.Event()
            gs.in_flight, gs.pending, gs.hold = False, None, None
            self.gsets.append(gs)
        for gs in self.gsets:
            # JAX aliases of the engine's output tensors (fixed buffers)
            gs.view_j = self._to_jax(gs.out['view'])
            gs.masks_j = self._to_jax(gs.out['masks'])
        self.stats = dict(steps=0, finished=0, terminated=0, truncated=0, drawn=0, nonok=0, maps={})
        self.t = 0
        torch.cuda.synchronize(self.device)

    def _draw_seeds(self, m):
        return self.torch.randint(self.map_lo, self.map_hi, (m,), generator=self.map_gen,
                                  dtype=self.torch.int64).numpy().astype(np.int32)

    def _build_jax(self):
        import functools
        import jax
        import jax.numpy as jnp
        from polytopia_bot.gpu.actions import Masks
        from . import network
        from .actions import leaf_shapes
        from .cpu_engine import view_features
        cfg = self.config
        size, layout = cfg.size, self.layout
        mask_width, player_at = self.gsets[0].env.mask_width, self.player_at
        self.mask_bytes, self.mask_width = self.gsets[0].env.mask_bytes, mask_width
        encode_one = lambda v: view_features(v, layout, size)

        def dense_masks(packed):
            bits, leaves, at = jnp.unpackbits(packed, axis=1)[:, :mask_width].astype(bool), [], 0
            for shape in leaf_shapes(size):
                k = int(np.prod(shape)) if shape else 1
                leaves.append(bits[:, at:at + k].reshape((bits.shape[0],) + tuple(shape)))
                at += k
            return Masks(*leaves)

        sample_one = functools.partial(network.sample, size=size)

        @jax.jit
        def fused(params, view, packed, key):
            """One set's JAX work for a step: encode, sample the policy's choice, the packed masks of the chosen path,
            log pi(choice). Returns choice [Gs, 3], choice.T [3, Gs] (the engine's inputs, read in place), path masks,
            logp [Gs], mover [Gs]."""
            features = jax.vmap(encode_one)(view)
            mover = view[:, player_at].astype(jnp.int32)
            keys = jax.random.split(key, view.shape[0])
            choice, path, logp, _ = jax.vmap(sample_one, in_axes=(None, 0, 0, 0))(params, features, dense_masks(packed), keys)
            path_masks = jnp.concatenate([jnp.packbits(path.family, axis=-1), jnp.packbits(path.source, axis=-1),
                                          jnp.packbits(path.target, axis=-1)], -1)
            choice3 = jnp.stack(choice, 1)
            return choice3, choice3.T, path_masks, logp, mover

        self.dense_masks, self.encode_one = dense_masks, encode_one
        self._fused = fused
        self.params, self.critic = jax.device_put(self.params), jax.device_put(self.critic)
        self.n_params = int(sum(x.size for x in jax.tree.leaves(self.params)))
        self.n_critic = int(sum(x.size for x in jax.tree.leaves(self.critic)))
        self.key = jax.random.key(int(self.rng.integers(1 << 31)))

    # ---- rollout
    def _to_jax(self, t):
        import jax.dlpack
        return jax.dlpack.from_dlpack(t)

    def _to_torch(self, a):
        import torch.utils.dlpack
        self.jax.block_until_ready(a)
        return torch.utils.dlpack.from_dlpack(a)

    def _add(self, name, ms):
        self.clock[name] = self.clock.get(name, 0.) + ms
        self.calls[name] = self.calls.get(name, 0) + 1

    def _lap(self, name, since):
        now = time.perf_counter()
        self._add(name, (now - since) * 1000.)
        return now

    def _record(self, gs, t, rows):
        """Each step's decisions of one set (`rows`: view, opp_view, masks, choice, path, logp, mover; see the module
        docstring), for games `gs.games` at step `t`. The buffers behind `view`, `opp_view` and `masks` are the
        engine's and change at the next step: copy what you keep. The boilerplate keeps nothing."""

    def _finished(self, games, rewards):
        """The games (lane indices) that ended in a step and their results, `rewards` [k, 2] float32 (seat 1, seat 2:
        +1 win, -1 loss, 0 draw). The boilerplate only counts them."""

    def _finish(self, gs):
        """Complete the set's queued tick and stage its outcomes for the host."""
        if gs.in_flight:
            t0 = time.perf_counter()
            gs.env.sync()
            self._lap('engine_wait', t0)
            gs.in_flight = False
        if gs.pending is not None:
            out = gs.out
            scores, rewards, status, flags = gs.stage_in
            scores.copy_(out['scores'])
            rewards.copy_(out['rewards'])
            status.copy_(out['status'])
            self.torch.cat((out['done'], out['terminated'], out['truncated']), out=flags)
            gs.stage_host.copy_(gs.stage_dev, non_blocking=True)
            gs.stage_event.record()

    def _outcomes(self, gs):
        """The finished games of the set's last tick: their results for each seat (infra.game_outcomes: elimination,
        or the higher in-game score at the round limit), statistics, new maps' kinds."""
        gs.stage_event.synchronize()
        host = gs.stage_out
        done, term_all, trunc_all = host['flags'].astype(bool)
        st = self.stats
        st['steps'] += self.Gs
        st['nonok'] += int((host['status'] != 0).sum())
        if not done.any():
            return
        d = np.flatnonzero(done)
        g = gs.games[d]
        term, trunc = term_all[d], trunc_all[d]
        rewards = game_outcomes(term, trunc, host['rewards'][d], host['scores'][d])
        self._finished(g, rewards)
        st['finished'] += len(d)
        st['drawn'] += int((rewards[:, 0] == 0).sum())      # equal scores at the round limit
        st['terminated'] += int(term.sum())
        st['truncated'] += int(trunc.sum())
        for i in range(len(d)):
            rec = st['maps'].setdefault(self.map_types[int(self.playing_kind[g[i]])], [0, 0])
            rec[0] += 1
            rec[1] += int(trunc[i])
        # the finished lanes restarted inside this tick on the seeds and layouts it was given
        self.playing_kind[g] = self.map_kind[gs.seeds_host[d] - self.map_lo]

    def _launch(self, gs, t, key):
        """The set's decisions from its current state, `_record`, the new maps, and the tick."""
        import jax
        torch = self.torch
        Gs = self.Gs
        t0 = time.perf_counter()
        res = self._fused(self.params, gs.view_j, gs.masks_j, key)
        t1 = time.perf_counter()
        if gs.pending is not None:                                  # booked while the fused call runs
            self._outcomes(gs)
            gs.pending = None
        self._lap('outcomes', t1)
        jax.block_until_ready(res)
        choice3, choiceT, path_masks, logp, mover = res
        self._add('fused', (time.perf_counter() - t0) * 1000.)
        t0 = time.perf_counter()
        self._record(gs, t, dict(view=gs.out['view'], opp_view=gs.out['opp_view'][:, 0], masks=gs.out['masks'],
                                 choice=self._to_torch(choice3), path=self._to_torch(path_masks),
                                 logp=self._to_torch(logp), mover=self._to_torch(mover)))
        self.decisions += Gs
        # the engine's family / source / target inputs: the [3, Gs] output read in place (held until the tick synced)
        choiceT_t = self._to_torch(choiceT)
        gs.hold = (choiceT, choiceT_t)
        # maps for the games that finish in this tick: fresh seeds (also the game RNG seeds) and their table rows
        seeds = self._draw_seeds(Gs)
        gs.seeds_host = seeds
        gs.seeds_pin.numpy()[:] = seeds
        gs.rows_pin.numpy()[:] = seeds.astype(np.int64) - self.map_lo
        gs.seeds.copy_(gs.seeds_pin, non_blocking=True)
        gs.rows.copy_(gs.rows_pin, non_blocking=True)
        torch.index_select(self.table3, 1, gs.rows, out=gs.layouts3)
        torch.index_select(self.table_caps, 0, gs.rows, out=gs.layout_caps)
        self._lap('book', t0)
        t0 = time.perf_counter()
        gs.env.tick_device(choiceT_t[0], choiceT_t[1], choiceT_t[2], gs.seeds, *gs.layouts, True, gs.out,
                           sync=not self.async_ticks)
        gs.pending = t
        if self.async_ticks:
            gs.in_flight = True
        else:
            self._lap('tick', t0)

    def _step(self):
        """One engine step of every set: a round robin finish(k) -> launch(k), so the other sets' ticks run while set
        k's JAX work does."""
        import jax
        t = self.t
        self.key, *keys = jax.random.split(self.key, self.sets + 1)
        t0 = time.perf_counter()
        for gs, k in zip(self.gsets, keys):
            if self.sets > 1:
                self._finish(gs)
            self._launch(gs, t, k)
            if self.sets == 1:
                self._finish(gs)
        self._lap('step', t0)
        self.t += 1

    # ---- the update
    def _update(self):
        """Called every `batch` decisions. This is where a learning update belongs; the boilerplate has none, so the
        weights never change. Returns the row's own fields for training.jsonl."""
        self.version += 1
        return dict(optimizer_updates=0)

    # ---- checkpoints
    def _host_state(self):
        jax = self.jax
        return dict(stage=f'update-{self.version}', params=jax.device_get(self.params), critic=jax.device_get(self.critic))

    def _write(self, name, state):
        from .runner import save_checkpoint
        save_checkpoint(self.root / f'{name}.npz', state['params'], self.run_config, stage=state['stage'])
        save_checkpoint(self.root / f'{name}-critic.npz', state['critic'], self.run_config, stage=state['stage'])

    def _export(self, state, record):
        """The policy and critic of `state` as exports/update-N: written under another name and renamed, so a folder
        with this name is always whole, whenever the process is ended. The newest EXPORTS_KEPT stay."""
        from .runner import save_checkpoint, write_json
        folder = self.root / EXPORTS
        target, scratch = folder / f"update-{record['update']:09d}", folder / f"writing-{record['update']}"
        if target.exists():
            return
        shutil.rmtree(scratch, ignore_errors=True)
        scratch.mkdir(parents=True)
        save_checkpoint(scratch / 'final.npz', state['params'], self.run_config, stage=state['stage'])
        save_checkpoint(scratch / 'final-critic.npz', state['critic'], self.run_config, stage=state['stage'])
        write_json(scratch / 'export.json', record)
        os.replace(scratch, target)
        for old in sorted(folder.glob('update-*'))[:-EXPORTS_KEPT]:
            shutil.rmtree(old, ignore_errors=True)

    def _stop_requested(self):
        """True once the host has asked this run to end (`ar stop`): it creates the file $POLYTOPIA_STOP_REQUEST names."""
        return self.stop_request is not None and os.path.exists(self.stop_request)

    def _checkpoint(self, rows, names, export=None):
        """Log rows + checkpoints (+ a whole export, given its record) on the background writer; returns the loop's
        share (seconds). The rows are appended first, so the log always covers an export that exists."""
        t0 = time.perf_counter()
        state = self._host_state()
        path, commit, write, write_export = self.root / training_log.JSONL, self.commit, self._write, self._export

        def job():
            training_log.append(path, rows)
            for name in names:
                write(name, state)
            if export is not None:
                write_export(state, export)
            commit()

        self.ckpt.submit(job)
        return time.perf_counter() - t0

    def _timing(self):
        return {k: round(self.clock[k] / max(self.calls.get(k, 1), 1), 2) for k in self.clock}

    # ---- the run
    def run(self):
        import jax
        from .runner import write_json
        torch = self.torch
        cfg = self.config
        self.ckpt = _Checkpointer()
        write_json(self.root / 'protocol.json', dict(
            config=asdict(cfg), network=self.network, trainer='gpu_resident (boilerplate, no learning)', games=self.G,
            sets=self.sets, engine='rl.gpu_engine.GpuGames (engine-cu, CUDA)', ruleset=RULESET_ID))
        gpu_name = torch.cuda.get_device_name(self.device_index)
        self.log(json.dumps(dict(stage='gpu_resident_start', gpu=gpu_name, games=self.G, sets=self.sets,
                                 params=self.n_params, critic_params=self.n_critic, network=self.network,
                                 batch=cfg.batch, round_limit=self.round_limit)))
        self.clock, self.calls = {}, {}
        history, logged = [], 0
        started = time.monotonic()
        last_report = time.monotonic()
        steps_at_update = finished_at_update = decisions_at_update = 0
        bookkeeping = 0.
        exported = None                 # when the last whole export was queued
        for gs in self.gsets:
            self._finish(gs)
        while (time.monotonic() - started < cfg.training_seconds
               and not (cfg.training_updates and self.version >= cfg.training_updates) and not self._stop_requested()):
            self._step()
            if self.decisions - decisions_at_update < cfg.batch:
                continue
            tick = time.monotonic()
            info = self._update()
            now = time.monotonic()
            st = self.stats
            decisions = self.decisions - decisions_at_update
            row = dict(update=self.version, samples=self.decisions, seconds=now - last_report, elapsed=now - started,
                       update_seconds=now - tick, bookkeeping_seconds=bookkeeping, decisions=decisions,
                       engine_steps=st['steps'] - steps_at_update,
                       samples_per_second=decisions / max(now - last_report, 1e-9),
                       finished=st['finished'] - finished_at_update, finished_total=st['finished'],
                       terminated=st['terminated'], truncated=st['truncated'], drawn=st['drawn'], nonok=st['nonok'],
                       maps=dict(st['maps']), timing_ms=self._timing(), **info)
            last_report = now
            steps_at_update, finished_at_update, decisions_at_update = st['steps'], st['finished'], self.decisions
            history.append(row)
            self.log(json.dumps(dict(stage='update', **row)))
            snapshot = self.version % cfg.snapshot_every == 0 or self.version in {int(u) for u in cfg.snapshot_at}
            checkpoint = self.version % cfg.checkpoint_every == 0 or snapshot
            # a usable policy early, kept current: the first update, then every export_seconds
            export = exported is None or now - exported >= cfg.export_seconds
            if checkpoint or export:
                names = (['latest'] if checkpoint else []) + ([f'snapshots/update-{self.version}'] if snapshot else [])
                record = dict(update=self.version, samples=self.decisions, elapsed=now - started)
                bookkeeping = self._checkpoint(history[logged:], names, export=record if export else None)
                logged = len(history)
                if export:
                    exported = now
            else:
                bookkeeping = 0.
        elapsed = time.monotonic() - started
        stopped = self._stop_requested()
        self.ckpt.wait()
        training_log.append(self.root / training_log.JSONL, history[logged:])
        write_json(self.root / training_log.JSON, history)
        state = self._host_state()
        self._write('latest', state)
        self._write('final', state)
        write_json(self.root / 'training-complete.json', dict(updates=self.version, seconds=elapsed, stopped=stopped))
        self.commit()
        st = self.stats
        jax_stats = jax.local_devices()[0].memory_stats() or {}
        self.result = dict(
            ok=True, stopped=stopped, gpu=gpu_name, run_dir=str(self.root), updates=self.version,
            samples=self.decisions, seconds=round(elapsed, 1), games=self.G, sets=self.sets, batch=cfg.batch,
            params=self.n_params, critic_params=self.n_critic,
            samples_per_s=round(self.decisions / max(elapsed, 1e-9)), engine_steps=st['steps'],
            finished=st['finished'], terminated=st['terminated'], truncated=st['truncated'],
            nonok_fraction=st['nonok'] / max(st['steps'], 1), timing_ms=self._timing(),
            gpu_mem_gib=dict(torch_max_allocated=round(torch.cuda.max_memory_allocated() / 2 ** 30, 2),
                             jax_peak_in_use=round(jax_stats.get('peak_bytes_in_use', 0) / 2 ** 30, 2)),
            maps=dict(st['maps']))
        return history

    def close(self):
        for gs in self.gsets:
            gs.env.close()
