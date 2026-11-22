"""The starter trainer: plain PPO self-play on one GPU, with the CUDA engine stepping thousands of games in place.

What it does (and, by design, nothing more):

  * self-play: one policy plays both seats of every game, and every decision is a training row;
  * rewards: only the game's result, by the rule the evaluator scores with (infra.game_outcomes): +1 win / -1 loss
    when a player is eliminated; a game cut at the round limit goes to the higher in-game score (+1 / -1) and equal
    scores draw (0). Nothing else is rewarded;
  * a separate critic network of the same structure as the policy (`critic_template`), reading the mover's own view
    (the same features as the policy), trained on the returns with a mean squared error;
  * GAE per decision (gamma / gae_lambda per decision, the sign flips when the next decision is the other seat's, as
    the value is from the mover's side), advantages normalized over the batch;
  * PPO: `ppo.loss` (clipped surrogate + entropy bonus), `epochs` passes over each batch in `minibatch` pieces,
    one Adam step per minibatch over (policy, critic), constant learning rate.

Rollout mechanics (frozen in spirit; change them if you need to, but they are tuned for throughput): the games are split
into `sets` engine batches stepped in a round robin, so the engine of one set runs while the GPU samples another; each
set's encode + sampling + critic value is one fused jit; every step writes each game's (view, the other seat's view,
packed action masks, choice, logp, value) into slot t % L of a device ring of L steps. The host `SegmentLog` groups a
seat's rows into segments of `segment` decisions; a segment closes at the seat's next decision (its boundary), at the
game's end, or when it gets `ring` steps old (forced; the boundary is the seat's value of the current state). When the
closed rows reach `batch`, the oldest `batch` rows are gathered, valued (rows valued by an older critic are re-scored),
GAE'd per segment, and trained on.

Checkpoints: runner.save_checkpoint, as latest / latest-critic / snapshots/update-N / final (+ -critic) in the run
folder, with protocol.json and training.jsonl. Runs always start from scratch.

Exports: after the first update and then every `export_seconds`, the current policy and critic are also written as
`exports/update-N/` (final.npz, final.json, final-critic.npz, final-critic.json, export.json). A folder appears under
that name only once it is whole (it is written under another name and renamed), and the two newest are kept. If the
run is cut off at its time limit, the host packages the newest whole export as the run's network; a run with no whole
export is lost. Keep this (or your own equivalent) working if you change the loop.

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
from .infra import END_FAMILY, VALUE_CHUNK, MAP_TABLE_DIRS, MapTable, build_network, game_outcomes

RING_SLACK_STEPS = 8          # ring steps past `ring` + batch / games: closed segments wait this long for the assembly
EXPORTS, EXPORTS_KEPT = 'exports', 2            # the folder of whole policy + critic exports, and how many stay
STOP_REQUEST_ENV = 'POLYTOPIA_STOP_REQUEST'     # names the file the host creates to ask this run to end


@dataclass
class LearnerConfig:
    """The recipe (the config JSON's top-level fields). Add fields here for your own options."""
    size: int = mode.SIZE             # board, players, maps, round limit, map seeds: the frozen mode (rl/mode.py)
    players: int = mode.PLAYERS
    gamma: float = .999               # per decision
    gae_lambda: float = .95
    segment: int = 32                 # decisions of one seat per GAE segment
    batch: int = 65536                # rows per update
    minibatch: int = 8192
    grad_accum: int = 1               # micro-batches per minibatch (same step, less memory)
    epochs: int = 1
    learning_rate: float = 3e-4
    entropy_coef: float = .01
    clip: float = .2
    seed: int = 801
    map_seed_low: int = mode.MAP_SEED_LOW
    map_seed_high: int = mode.MAP_SEED_HIGH
    maps: str = mode.MAPS
    max_turns: int = mode.ROUND_LIMIT  # rounds before a game is cut (the evaluator then scores it by in-game score)
    checkpoint_every: int = 25        # updates between latest.npz writes
    export_seconds: float = 60.       # seconds between whole exports of the policy and critic (exports/update-N)
    snapshot_every: int = 50          # updates between snapshots/update-N.npz
    snapshot_at: tuple = field(default_factory=tuple)
    training_updates: int = 0         # stop after this many updates (0: by training_seconds)
    training_seconds: int = 600

    def validate(self):
        if self.players != 2:
            raise ValueError('two-player games only')
        if self.batch % self.minibatch or self.minibatch % self.grad_accum:
            raise ValueError('batch must be a multiple of minibatch, minibatch a multiple of grad_accum')
        if not 0 < self.gamma <= 1 or not 0 <= self.gae_lambda <= 1:
            raise ValueError('gamma in (0, 1], gae_lambda in [0, 1]')
        if isinstance(self.max_turns, bool) or not isinstance(self.max_turns, int) or self.max_turns < 1:
            raise ValueError('max_turns is the round limit: a positive whole number')


def network_spec_of(config, network=None):
    """The architecture spec of a fresh network: the defaults below, overridden by the recipe's arch JSON."""
    from .observation import OBS_VERSION
    spec = dict(size=config.size, width=32, blocks=2, architecture='normalized-v2', obs_version=OBS_VERSION)
    spec.update(network or {})
    return spec


def policy_template(spec, key=None):
    from . import network
    return build_network(network, spec, key)


def critic_template(spec, key=None):
    """The critic: the same network structure, reading the mover's own view (the policy's features)."""
    from . import network
    return build_network(network, spec, key)


def segment_gae(values, seg, pos, length, terminal, reward, boot_values, gamma, lam):
    """GAE of flat rows grouped into same-seat segments, vectorized over segments. values [R] per row (the critic's,
    from the row's mover's side), seg / pos [R] the row's segment and position, length / terminal / reward /
    boot_values [S] per segment (reward: the seat's terminal reward, used when terminal; boot_values: the critic's value
    of the boundary state from the seat's side, used when not terminal). Every row of a segment belongs to the same
    seat, so no sign flips inside a segment. Returns (advantage, returns) per row."""
    values = np.asarray(values, np.float32)
    seg, pos, length = np.asarray(seg), np.asarray(pos), np.asarray(length)
    n_seg, width = len(length), int(length.max()) if len(length) else 1
    rows, last = np.arange(n_seg), length - 1
    V = np.zeros((n_seg, width), np.float32)
    valid = np.zeros((n_seg, width), bool)
    V[seg, pos] = values
    valid[seg, pos] = True
    terminal = np.asarray(terminal, bool)
    next_v = np.zeros_like(V)
    next_v[:, :-1] = V[:, 1:]
    next_v[rows, last] = np.where(terminal, 0., np.asarray(boot_values, np.float32))
    rewards = np.zeros_like(V)
    rewards[rows, last] = np.where(terminal, np.asarray(reward, np.float32), 0.)
    delta = (rewards + gamma * next_v - V).astype(np.float32)
    A = np.zeros_like(V)
    running = np.zeros(n_seg, np.float32)
    for t in range(width - 1, -1, -1):
        running = np.where(valid[:, t], (delta[:, t] + gamma * lam * running).astype(np.float32), np.float32(0.))
        A[:, t] = running
    adv = A[seg, pos]
    return adv, (adv + values).astype(np.float32)


def _ceil_to(n, k):
    return -(-n // k) * k


def value_sources(take, step_version, version):
    """Where each critic value comes from at assembly. The rollout stores every row's value in the ring;
    step_version[step % L] is the update count the critic had when that step was valued. Returns (row_stale [n],
    boundary_ring [S], boundary_call [S]): a stale row is re-scored with the current critic; a boundary is read from the
    ring when it is the seat's own current row, else it needs a value call (a forced cut seen from the other seat, or
    a stale row); a terminal segment (its game finished) has no boundary."""
    L = len(step_version)
    segs = take['segments']
    row_stale = step_version[take['step'] % L] != version
    nonterminal = ~np.asarray(segs['terminal'], bool)
    own_row = ~np.asarray(segs['boot_swap'], bool)
    current = step_version[segs['boot_step'] % L] == version
    boundary_ring = nonterminal & own_row & current
    return row_stale, boundary_ring, nonterminal & ~boundary_ring


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


# ----------------------------------------------------------------------------- rows and segments

class SegmentLog:
    """Host bookkeeping of the rows in the device ring and their same-seat segments. A row is (step, game); per
    (game, seat) the steps of the open segment's rows sit in `steps[game, seat, :count]`. Closed segments queue as
    events until take() hands out the oldest `batch` rows."""

    KEYS = ('game', 'seat', 'steps', 'length', 'terminal', 'reward', 'boot_step', 'boot_swap')

    def __init__(self, games, players, segment, max_age):
        self.G, self.players, self.segment, self.max_age = games, players, segment, max_age
        self.W = segment + 1
        self.count = np.zeros((games, players), np.int32)
        self.first = np.full((games, players), -1, np.int64)
        self.steps = np.full((games, players, self.W), -1, np.int64)
        self.events = []
        self.closed_rows = self.closed_segments = self.recorded = 0
        self.forced_rows = self.forced_segments = self.cut_segments = self.finished_segments = 0

    def _close(self, g, s, length, terminal, reward, boot_step, boot_swap):
        if len(g) == 0:
            return
        steps = self.steps[g, s].copy()
        steps[np.arange(self.W)[None, :] >= length[:, None]] = -1
        self.events.append(dict(game=g.astype(np.int64), seat=s.astype(np.int32), steps=steps,
                                length=length.astype(np.int32), terminal=np.asarray(terminal, bool),
                                reward=np.asarray(reward, np.float32), boot_step=boot_step.astype(np.int64),
                                boot_swap=np.asarray(boot_swap, bool)))
        self.closed_rows += int(length.sum())
        self.closed_segments += len(g)

    def force_close(self, t, games, mover0):
        """Open segments of `games` whose first row is max_age steps old: closed with the boundary at the seat's
        current state (the ring row of step t, seen from the other seat's side when the seat is not the mover)."""
        cnt = self.count[games]
        old = (cnt > 0) & (self.first[games] <= t - self.max_age)
        if not old.any():
            return 0
        gi, si = np.nonzero(old)
        g, s = games[gi], si.astype(np.int32)
        length = cnt[gi, si]
        self._close(g, s, length, np.zeros(len(g), bool), np.zeros(len(g), np.float32), np.full(len(g), t, np.int64),
                    s != mover0[gi])
        self.count[g, s] = 0
        self.first[g, s] = -1
        self.forced_rows += int(length.sum())
        self.forced_segments += len(g)
        return len(g)

    def record(self, t, games, seats0):
        """This step's rows: games and the mover seat (0-based) of each. A seat's (segment + 1)-th row cuts the first
        `segment` rows (boundary = this row) and starts the next segment."""
        if len(games) == 0:
            return
        c = self.count[games, seats0]
        self.steps[games, seats0, c] = t
        new = c == 0
        if new.any():
            self.first[games[new], seats0[new]] = t
        c1 = c + 1
        self.count[games, seats0] = c1
        full = c1 > self.segment
        if full.any():
            gf, sf = games[full], seats0[full]
            k = len(gf)
            self._close(gf, sf, np.full(k, self.segment, np.int32), np.zeros(k, bool), np.zeros(k, np.float32),
                        np.full(k, t, np.int64), np.zeros(k, bool))
            self.steps[gf, sf, 0] = t
            self.count[gf, sf] = 1
            self.first[gf, sf] = t
            self.cut_segments += k
        self.recorded += len(games)

    def finish(self, games, rewards):
        """Finished games: rewards [k, players], each seat's result of its game (infra.game_outcomes). Every finished
        game is an outcome: the seats' open segments close as terminal, with that reward on their last row."""
        if len(games) == 0:
            return
        cnt = self.count[games]
        has = cnt > 0
        if has.any():
            gi, si = np.nonzero(has)
            g, s = games[gi], si.astype(np.int32)
            length = cnt[gi, si]
            last = self.steps[g, s, length - 1]
            self._close(g, s, length, np.ones(len(g), bool), rewards[gi, si], last, s == 1)
            self.finished_segments += len(g)
        self.count[games] = 0
        self.first[games] = -1

    def take(self, batch, t, ring_len):
        """The oldest closed segments covering `batch` rows (the last may straddle): flat rows (step, game, seg, pos),
        the per-segment tables and n (>= batch; rows past batch are dropped by the caller after GAE)."""
        if self.closed_rows < batch:
            return None
        cat = {k: np.concatenate([e[k] for e in self.events]) for k in self.KEYS}
        length = cat['length']
        cum = np.cumsum(length, dtype=np.int64)
        used = int(np.searchsorted(cum, batch, side='left')) + 1
        n = int(cum[used - 1])
        seg = np.repeat(np.arange(used, dtype=np.int32), length[:used])
        start = cum[:used] - length[:used]
        pos = (np.arange(n, dtype=np.int64) - np.repeat(start, length[:used])).astype(np.int32)
        steps = cat['steps'][:used][seg, pos]
        game = cat['game'][:used][seg]
        if (steps < 0).any():
            raise RuntimeError('segment bookkeeping error: a row has no step')
        if t - int(steps.min()) >= ring_len:
            raise RuntimeError(f'row {t - int(steps.min())} steps old has left the device ring ({ring_len} steps); '
                               f'raise --ring or lower --batch / raise --games')
        rest = {k: v[used:] for k, v in cat.items()}
        self.events = [rest] if len(rest['length']) else []
        self.closed_rows -= n
        self.closed_segments -= used
        return dict(step=steps, game=game, seg=seg, pos=pos, n=n, segments={k: cat[k][:used] for k in self.KEYS})


# ----------------------------------------------------------------------------- trainer

class GpuResidentTrainer:
    """`config`: the recipe dict or a LearnerConfig; `run_dir`: the run folder; `network`: the architecture dict (the
    recipe's arch JSON) of the fresh network."""

    def __init__(self, config, run_dir, *, network=None, training_seconds=None, games=4096, sets=4, ring=160,
                 minibatch=None, batch=None, grad_accum=None, checkpoint_every=None, snapshot_every=None, map_table=None,
                 device=0, seed=None, log=print, commit=lambda: None):
        import torch
        import jax
        self.torch, self.jax = torch, jax
        cfg = dict(asdict(config) if isinstance(config, LearnerConfig) else config)
        overrides = dict(training_seconds=training_seconds, minibatch=minibatch, batch=batch, grad_accum=grad_accum,
                         checkpoint_every=checkpoint_every, snapshot_every=snapshot_every, seed=seed)
        cfg.update({k: int(v) for k, v in overrides.items() if v})
        cfg['snapshot_at'] = tuple(cfg.get('snapshot_at', ()))
        self.config = cfg = LearnerConfig(**cfg)
        cfg.validate()
        self.root, self.log, self.commit = Path(run_dir), log, commit
        self.G, self.sets, self.ring = int(games), int(sets), int(ring)
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
                                 gamma=cfg.gamma, gae_lambda=cfg.gae_lambda, minibatch=cfg.minibatch, epochs=cfg.epochs,
                                 seed=cfg.seed)

    def _build_engine(self):
        torch = self.torch
        from .cpu_engine import view_layout
        from .gpu_engine import GpuGames
        cfg = self.config
        size, players, G, Gs = cfg.size, cfg.players, self.G, self.Gs
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
        self.fam_bytes, self.src_bytes, self.tgt_bytes = 3, -(-n // 8), -(-max(n, 128) // 8)
        self.row_mask_bytes = self.fam_bytes + self.src_bytes + self.tgt_bytes
        self.L = L = self.ring + -(-cfg.batch // G) + RING_SLACK_STEPS
        self.N_cap = cfg.batch + VALUE_CHUNK      # assembled rows: the batch + the straddling segment's tail
        self.map_gen = torch.Generator().manual_seed(int(cfg.seed))
        self.clock, self.calls = {}, {}
        # the map table on the device ([3, M, n] terrain / structure / resource, [M, players] capitals)
        self.table3 = torch.from_numpy(np.stack(table.arrays[:3])).to(self.device)
        self.table_caps = torch.from_numpy(table.arrays[3]).to(self.device)
        self.playing_kind = np.zeros(G, np.int8)

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
            # host memory in one non-blocking transfer (_stage_outcomes), read in _outcomes
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
            gs.opp_j = self._to_jax(gs.out['opp_view'][:, 0])
            gs.masks_j = self._to_jax(gs.out['masks'])
        z = lambda shape, dt: torch.zeros(shape, dtype=dt, device=self.device)
        # the ring: per row the mover's view, the other seat's view (a boundary seen from the non-moving seat, and
        # available to your own code), packed masks, choice, (logp, value)
        self.ring_view, self.ring_opp = z((L * G, self.vw), torch.int16), z((L * G, self.vw), torch.int16)
        self.ring_masks = z((L * G, self.row_mask_bytes), torch.uint8)
        self.ring_choice = z((L * G, 3), torch.int32)
        self.ring_fv = z((2, L * G), torch.float32)                 # row 0 logp, row 1 the critic's value
        self.ring_logp, self.ring_value = self.ring_fv[0], self.ring_fv[1]
        self.ring_bytes = sum(x.numel() * x.element_size() for x in
                              (self.ring_view, self.ring_opp, self.ring_masks, self.ring_choice, self.ring_fv))
        N_cap = self.N_cap
        self.buf_view = z((N_cap, self.vw), torch.int16)
        self.buf_masks = z((N_cap, self.row_mask_bytes), torch.uint8)
        self.buf_choice, self.buf_logp = z((N_cap, 3), torch.int32), z((N_cap,), torch.float32)
        self.buf_value = z((N_cap,), torch.float32)
        self.tmp_view = z((VALUE_CHUNK, self.vw), torch.int16)       # scratch of the small value calls
        self.seglog = SegmentLog(G, players, cfg.segment, self.ring)
        self.stats = dict(steps=0, finished=0, terminated=0, truncated=0, drawn=0, nonok=0, maps={})
        self.t = 0
        self.h_step_version = np.full(L, -1, np.int64)
        torch.cuda.synchronize(self.device)

    def _draw_seeds(self, m):
        return self.torch.randint(self.map_lo, self.map_hi, (m,), generator=self.map_gen,
                                  dtype=self.torch.int64).numpy().astype(np.int32)

    def _build_jax(self):
        import functools
        import jax
        import jax.numpy as jnp
        from polytopia_bot.gpu.actions import Masks
        from . import network, ppo
        from .actions import Choice, PathMasks, leaf_shapes
        from .cpu_engine import view_features
        cfg = self.config
        size, layout, n = cfg.size, self.layout, cfg.size * cfg.size
        mask_width, player_at = self.gsets[0].env.mask_width, self.player_at
        self.mask_bytes, self.mask_width = self.gsets[0].env.mask_bytes, mask_width
        fb, sb = self.fam_bytes, self.src_bytes
        encode_one = lambda v: view_features(v, layout, size)

        def dense_masks(packed):
            bits, leaves, at = jnp.unpackbits(packed, axis=1)[:, :mask_width].astype(bool), [], 0
            for shape in leaf_shapes(size):
                k = int(np.prod(shape)) if shape else 1
                leaves.append(bits[:, at:at + k].reshape((bits.shape[0],) + tuple(shape)))
                at += k
            return Masks(*leaves)

        sample_one = functools.partial(network.sample, size=size)
        critic_value = lambda critic, features: jax.vmap(lambda x: network.trunk(critic, x)[2])(features)

        @jax.jit
        def fused(params, critic, view, packed, key):
            """One set's JAX work for a step: encode, sample the policy's choice, the packed PathMasks rows, behaviour
            logp, the critic's value. Returns choice [Gs, 3], choice.T [3, Gs] (the engine's inputs, read in place),
            row masks, (logp, value) [2, Gs], mover [Gs]."""
            features = jax.vmap(encode_one)(view)
            mover = view[:, player_at].astype(jnp.int32)
            keys = jax.random.split(key, view.shape[0])
            choice, path, logp, _ = jax.vmap(sample_one, in_axes=(None, 0, 0, 0))(params, features, dense_masks(packed), keys)
            row_masks = jnp.concatenate([jnp.packbits(path.family, axis=-1), jnp.packbits(path.source, axis=-1),
                                         jnp.packbits(path.target, axis=-1)], -1)
            choice3 = jnp.stack(choice, 1)
            return choice3, choice3.T, row_masks, jnp.stack([logp, critic_value(critic, features)], 0), mover

        @jax.jit
        def values_chunk(critic, view):
            """The critic's value of a chunk of views (each from its own seat's side)."""
            return critic_value(critic, jax.vmap(encode_one)(view))

        def unpack_bits(a, width):
            return jnp.unpackbits(a, axis=-1)[..., :width].astype(bool)

        def loss(trainable, mb, entropy_coef):
            params, critic = trainable
            features = jax.vmap(encode_one)(mb['view'])
            masks = PathMasks(unpack_bits(mb['masks'][:, :fb], 17), unpack_bits(mb['masks'][:, fb:fb + sb], n),
                              unpack_bits(mb['masks'][:, fb + sb:], max(n, 128)))
            batch = dict(features=features, masks=masks, choice=Choice(mb['choice'][:, 0], mb['choice'][:, 1], mb['choice'][:, 2]),
                         logp=mb['logp'], advantage=mb['advantage'], returns=mb['returns'])
            total, metrics = ppo.loss(params, batch, clip=cfg.clip, entropy_coef=entropy_coef, value_coef=0.)
            v = critic_value(critic, features)
            critic_loss = .5 * jnp.mean((v - batch['returns']) ** 2)
            return total + .5 * critic_loss, dict(metrics, value_loss=critic_loss, value_mean=v.mean(), value_std=v.std())

        accum, micro = int(cfg.grad_accum), cfg.minibatch // int(cfg.grad_accum)

        @jax.jit
        def train_step(trainable, opt, dev, idx, learning_rate, entropy_coef):
            """One minibatch (rows idx of the batch): the loss, its gradient (mean over grad_accum micro-batches), one
            Adam step over (policy, critic)."""
            if accum == 1:
                mb = jax.tree.map(lambda a: a[idx], dev)
                (_, metrics), grads = jax.value_and_grad(loss, has_aux=True)(trainable, mb, entropy_coef)
            else:
                def part(acc, j):
                    sub = jax.lax.dynamic_slice(idx, (j * micro,), (micro,))
                    mb = jax.tree.map(lambda a: a[sub], dev)
                    (_, m), g = jax.value_and_grad(loss, has_aux=True)(trainable, mb, entropy_coef)
                    return jax.tree.map(jnp.add, acc, g), m
                grads, ms = jax.lax.scan(part, jax.tree.map(jnp.zeros_like, trainable), jnp.arange(accum))
                grads = jax.tree.map(lambda g: g / accum, grads)
                metrics = jax.tree.map(jnp.mean, ms)
            trainable, opt, norm = ppo.adam_step(trainable, grads, opt, learning_rate=learning_rate)
            return trainable, opt, dict(metrics, grad_norm=norm)

        self._fused, self._values_chunk, self._train_step = fused, values_chunk, train_step
        self.params, self.critic = jax.device_put(self.params), jax.device_put(self.critic)
        self.opt = ppo.adam_init((self.params, self.critic))
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
        """The finished games of the set's last tick: their segments close with the game's result for each seat
        (infra.game_outcomes: elimination, or the higher in-game score at the round limit), statistics, new maps'
        kinds. Every finished game is an outcome, so nothing bootstraps past a game's end."""
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
        self.seglog.finish(g, rewards)
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
        """The set's decisions from its current state, its rows into the ring, the new maps, and the tick."""
        import jax
        torch = self.torch
        lo, Gs = gs.lo, self.Gs
        t0 = time.perf_counter()
        res = self._fused(self.params, self.critic, gs.view_j, gs.masks_j, key)
        t1 = time.perf_counter()
        if gs.pending is not None:                                  # booked while the fused call runs
            self._outcomes(gs)
            gs.pending = None
        t1 = self._lap('outcomes', t1)
        jax.block_until_ready(res)
        choice3, choiceT, row_masks, fv, mover_j = res
        mover = np.asarray(mover_j)
        self._add('fused', (time.perf_counter() - t0) * 1000.)
        t0 = time.perf_counter()
        # rows: force-close stale open segments against this state, then record this step's rows (every game's mover)
        self.seglog.force_close(t, gs.games, mover - 1)
        self.seglog.record(t, gs.games, (mover - 1).astype(np.int32))
        base = (t % self.L) * self.G + lo
        self.ring_view[base:base + Gs].copy_(gs.out['view'])
        self.ring_opp[base:base + Gs].copy_(gs.out['opp_view'][:, 0])
        self.ring_masks[base:base + Gs].copy_(self._to_torch(row_masks))
        self.ring_choice[base:base + Gs].copy_(self._to_torch(choice3))
        self.ring_fv[:, base:base + Gs].copy_(self._to_torch(fv))
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
        self.h_step_version[t % self.L] = self.version          # the critic this step's rows are valued with
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

    # ---- assembly and update
    def _values_gathered(self, src_view, flat):
        """The critic's values of rows `flat` (device int64 [k]) of src_view, VALUE_CHUNK at a time; numpy [k]."""
        torch = self.torch
        C, k, out = VALUE_CHUNK, int(flat.shape[0]), []
        for c in range(0, k, C):
            idx = flat[c:c + C]
            m = int(idx.shape[0])
            torch.index_select(src_view, 0, idx, out=self.tmp_view[:m])
            out.append(np.asarray(self._values_chunk(self.critic, self._to_jax(self.tmp_view)))[:m])
        return np.concatenate(out) if out else np.zeros(0, np.float32)

    def _assemble(self):
        """The oldest `batch` closed rows: gathered from the ring, valued (stale rows and boundaries re-scored), GAE per
        segment, advantages normalized over the batch. Returns (device batch, info)."""
        jax, torch = self.jax, self.torch
        cfg = self.config
        G, L, C = self.G, self.L, VALUE_CHUNK
        t0 = time.perf_counter()
        take = self.seglog.take(cfg.batch, self.t - 1, L)
        n, segs = take['n'], take['segments']
        S = len(segs['length'])
        dev64 = lambda a: torch.from_numpy(np.ascontiguousarray(a, np.int64)).to(self.device)
        flat = dev64((take['step'] % L) * G + take['game'])
        torch.index_select(self.ring_view, 0, flat, out=self.buf_view[:n])
        torch.index_select(self.ring_masks, 0, flat, out=self.buf_masks[:n])
        torch.index_select(self.ring_choice, 0, flat, out=self.buf_choice[:n])
        torch.index_select(self.ring_logp, 0, flat, out=self.buf_logp[:n])
        # boundaries: the ring row of the boundary step seen from the segment's seat (the other seat's view when the
        # seat was not the mover)
        bflat = dev64((segs['boot_step'] % L) * G + segs['game'])
        swap = torch.from_numpy(segs['boot_swap']).to(self.device)[:, None]
        boot_view = torch.zeros((_ceil_to(S, C), self.vw), dtype=torch.int16, device=self.device)
        boot_view[:S] = torch.where(swap, torch.index_select(self.ring_opp, 0, bflat), torch.index_select(self.ring_view, 0, bflat))
        torch.cuda.synchronize(self.device)
        t0 = self._lap('gather', t0)
        row_stale, b_ring, b_call = value_sources(take, self.h_step_version, self.version)
        torch.index_select(self.ring_value, 0, flat, out=self.buf_value[:n])
        values = self.buf_value[:n].cpu().numpy().copy()
        boots = np.zeros(S, np.float32)
        if b_ring.any():
            boots[b_ring] = self.ring_value[bflat[torch.from_numpy(b_ring).to(self.device)]].cpu().numpy()
        stale = np.flatnonzero(row_stale)
        if len(stale):
            values[stale] = self._values_gathered(self.buf_view, dev64(stale))
        call = np.flatnonzero(b_call)
        if len(call):
            boots[call] = self._values_gathered(boot_view, dev64(call))
        t0 = self._lap('value', t0)
        adv, ret = segment_gae(values, take['seg'], take['pos'], segs['length'], segs['terminal'], segs['reward'], boots,
                               cfg.gamma, cfg.gae_lambda)
        adv, ret = adv[:cfg.batch], ret[:cfg.batch]       # the straddling segment's tail past the batch is dropped
        raw_std = float(adv.std())
        norm = ((adv - adv.mean()) / (raw_std + 1e-8)).astype(np.float32)
        pad = np.zeros(self.N_cap - cfg.batch, np.float32)
        dev = dict(view=self._to_jax(self.buf_view), masks=self._to_jax(self.buf_masks), choice=self._to_jax(self.buf_choice),
                   logp=self._to_jax(self.buf_logp), advantage=jax.numpy.asarray(np.concatenate([norm, pad])),
                   returns=jax.numpy.asarray(np.concatenate([ret.astype(np.float32), pad])))
        self._lap('gae', t0)
        info = dict(rows_gathered=n, dropped_rows=n - cfg.batch, segments=S, terminal_fraction=float(np.mean(segs['terminal'])),
                    stale_rows=int(row_stale.sum()), boundary_calls=int(len(call)), raw_advantage_std=raw_std,
                    return_mean=float(ret.mean()), value_mean_batch=float(values.mean()))
        return dev, info

    def _update(self):
        """One PPO update on the assembled batch: `epochs` passes in minibatches."""
        import jax
        cfg = self.config
        tick = time.monotonic()
        dev, info = self._assemble()
        assembled = time.monotonic()
        trainable = (self.params, self.critic)
        metrics_all = []
        for _ in range(cfg.epochs):
            self.update_key, sub = jax.random.split(self.update_key)
            perm = jax.random.permutation(sub, cfg.batch).astype(jax.numpy.int32)
            acc = None
            for i in range(cfg.batch // cfg.minibatch):
                idx = jax.lax.dynamic_slice(perm, (i * cfg.minibatch,), (cfg.minibatch,))
                trainable, self.opt, m = self._train_step(trainable, self.opt, dev, idx, np.float32(cfg.learning_rate),
                                                          np.float32(cfg.entropy_coef))
                acc = m if acc is None else jax.tree.map(jax.numpy.add, acc, m)
            metrics = {k: float(v) / (cfg.batch // cfg.minibatch) for k, v in jax.device_get(acc).items()}
            if not all(np.isfinite(list(metrics.values()))):
                raise RuntimeError(f'nonfinite PPO metrics {metrics}')
            metrics_all.append(metrics)
        self.params, self.critic = trainable
        jax.block_until_ready(self.params)
        trained = time.monotonic()
        self._add('update', (trained - assembled) * 1000.)
        self.version += 1
        mean = {k: float(np.mean([m[k] for m in metrics_all])) for k in metrics_all[0]}
        return dict(info, learning_rate=cfg.learning_rate, assemble_seconds=assembled - tick,
                    train_seconds=trained - assembled, **mean)

    # ---- checkpoints
    def _host_state(self):
        jax = self.jax
        return dict(stage=f'update-{self.version}', params=jax.device_get(self.params), opt=jax.device_get(self.opt),
                    critic=jax.device_get(self.critic))

    def _write(self, name, state):
        from .runner import save_checkpoint
        save_checkpoint(self.root / f'{name}.npz', state['params'], self.run_config, stage=state['stage'], optimizer=state['opt'])
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
            config=asdict(cfg), network=self.network, trainer='gpu_resident (starter)', games=self.G, sets=self.sets,
            ring=self.ring, ring_steps=self.L, engine='rl.gpu_engine.GpuGames (engine-cu, CUDA)', ruleset=RULESET_ID))
        gpu_name = torch.cuda.get_device_name(self.device_index)
        self.log(json.dumps(dict(stage='gpu_resident_start', gpu=gpu_name, games=self.G, sets=self.sets, ring=self.ring,
                                 ring_steps=self.L, params=self.n_params, critic_params=self.n_critic, network=self.network,
                                 batch=cfg.batch, minibatch=cfg.minibatch, grad_accum=cfg.grad_accum,
                                 round_limit=self.round_limit, ring_gib=round(self.ring_bytes / 2 ** 30, 2))))
        self.clock, self.calls = {}, {}
        history, logged = [], 0
        started = time.monotonic()
        self.update_key = jax.random.key(cfg.seed)
        last_report = time.monotonic()
        steps_at_update = rows_at_update = finished_at_update = 0
        bookkeeping = 0.
        exported = None                 # when the last whole export was queued
        for gs in self.gsets:
            self._finish(gs)
        while (time.monotonic() - started < cfg.training_seconds
               and not (cfg.training_updates and self.version >= cfg.training_updates) and not self._stop_requested()):
            self._step()
            if self.seglog.closed_rows < cfg.batch:
                continue
            tick = time.monotonic()
            info = self._update()
            now = time.monotonic()
            st = self.stats
            row = dict(update=self.version, samples=self.version * cfg.batch, seconds=now - last_report, elapsed=now - started,
                       update_seconds=now - tick, bookkeeping_seconds=bookkeeping,
                       engine_steps=st['steps'] - steps_at_update, learner_rows=self.seglog.recorded - rows_at_update,
                       samples_per_second=cfg.batch / max(now - last_report, 1e-9),
                       finished=st['finished'] - finished_at_update, finished_total=st['finished'],
                       terminated=st['terminated'], truncated=st['truncated'], drawn=st['drawn'], nonok=st['nonok'],
                       forced_close_fraction=self.seglog.forced_rows / max(self.seglog.recorded, 1),
                       maps=dict(st['maps']), timing_ms=self._timing(), **info)
            last_report = now
            steps_at_update, rows_at_update, finished_at_update = st['steps'], self.seglog.recorded, st['finished']
            history.append(row)
            self.log(json.dumps(dict(stage='update', **row)))
            snapshot = self.version % cfg.snapshot_every == 0 or self.version in {int(u) for u in cfg.snapshot_at}
            checkpoint = self.version % cfg.checkpoint_every == 0 or snapshot
            # a usable policy early, kept current: the first update, then every export_seconds
            export = exported is None or now - exported >= cfg.export_seconds
            if checkpoint or export:
                names = (['latest'] if checkpoint else []) + ([f'snapshots/update-{self.version}'] if snapshot else [])
                record = dict(update=self.version, samples=self.version * cfg.batch, elapsed=now - started)
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
            samples=self.version * cfg.batch,
            seconds=round(elapsed, 1), games=self.G, sets=self.sets, ring=self.ring, batch=cfg.batch, minibatch=cfg.minibatch,
            params=self.n_params, critic_params=self.n_critic,
            samples_per_s=round(self.version * cfg.batch / max(elapsed, 1e-9)), engine_steps=st['steps'],
            learner_rows=self.seglog.recorded, finished=st['finished'], terminated=st['terminated'], truncated=st['truncated'],
            nonok_fraction=st['nonok'] / max(st['steps'], 1),
            forced_close_fraction=self.seglog.forced_rows / max(self.seglog.recorded, 1), timing_ms=self._timing(),
            gpu_mem_gib=dict(torch_max_allocated=round(torch.cuda.max_memory_allocated() / 2 ** 30, 2),
                             jax_peak_in_use=round(jax_stats.get('peak_bytes_in_use', 0) / 2 ** 30, 2)),
            maps=dict(st['maps']),
            metrics={k: round(float(v), 5) for k, v in history[-1].items() if isinstance(v, float)} if history else {})
        return history

    def close(self):
        for gs in self.gsets:
            gs.env.close()
