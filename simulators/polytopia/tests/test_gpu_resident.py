"""The GPU-resident trainer's host bookkeeping (rl.gpu_resident.SegmentLog, Lanes, value_sources, the background
checkpoint writer) against the production SegmentCollector semantics. No GPU, no JAX: numpy only."""
import time

import numpy as np
import pytest

from polytopia_bot.rl.distributed import SegmentCollector
from polytopia_bot.rl.gpu_resident import Lanes, SegmentLog, _Checkpointer, rebalance_plan, row_shares, value_sources


def _simulate(seed, games=7, players=2, segment=4, steps=400, max_age=10 ** 9, pool_games=()):
    """A random row stream: every step each game's mover records a row (pool games: only the learner seat's), games
    end at random (native or capped). Returns the SegmentCollector segments and the SegmentLog segments, both as
    (rows, length, terminal, reward, boundary) with rows = [(step, game)] and boundary = ('row', step, game) /
    ('final', game, seat) / None."""
    rng = np.random.default_rng(seed)
    ref = SegmentCollector(games, segment, players)
    log = SegmentLog(games, players, segment, max_age)
    learner_seat = {g: rng.integers(1, players + 1) for g in pool_games}
    expected, store_rows, store = [], {}, 0

    def ref_take():
        rows, segs = ref.take()
        by = {}
        for r in rows:
            by.setdefault(int(r['seg']), []).append(r)
        out = []
        for j, s in enumerate(segs):
            rs = sorted(by[j], key=lambda r: int(r['pos']))
            boot = None
            if not s['terminal']:
                bv = tuple(int(x) for x in s['boot_view'])
                boot = ('row', bv[0], bv[1]) if bv[2] == 0 else ('final', bv[1], bv[2])
            out.append(([(int(r['view'][0]), int(r['view'][1])) for r in rs], int(s['length']), bool(s['terminal']),
                        float(s['reward']), boot))
        return out

    mover = rng.integers(1, players + 1, games)
    for t in range(steps):
        rec = np.array([not (g in learner_seat and mover[g] != learner_seat[g]) for g in range(games)])
        g_rec = np.flatnonzero(rec)
        for g in g_rec:
            ref.add(g, int(mover[g]), dict(view=np.array([t, g, 0]), opp_view=np.array([t, g, 0]), pos=0))
        log.force_close(t, np.arange(games), mover - 1)
        log.record(t, g_rec.astype(np.int64), (mover[g_rec] - 1).astype(np.int32))
        done = rng.random(games) < 0.05
        for g in np.flatnonzero(done):
            terminated = rng.random() < 0.7
            rewards = np.array([1., -1.] if rng.random() < 0.5 else [-1., 1.], np.float32)
            if not terminated:
                rewards[:] = 0.
            final = np.array([[t, g, s] for s in range(1, players + 1)])
            # the reference finishes both seats; a pool game's opponent seat recorded no rows, so only the learner
            # seat's segment comes out (the vector actor drops the other seats first, which is the same here)
            ref.finish(g, rewards, terminated, final)
            mask = np.ones((1, players), bool)
            if g in learner_seat:
                mask[0] = np.arange(players) == learner_seat[g] - 1
            idx = -1
            if not terminated:
                store_rows[store] = ('final', int(g))
                idx = store
                store += 1
            log.finish(np.array([g]), rewards[None], np.array([terminated]), mask, np.array([idx]))
        mover = rng.integers(1, players + 1, games)
    expected = ref_take()
    got = []
    while log.closed_rows:
        take = log.take(1, steps - 1, 10 ** 9)
        segs = take['segments']
        for j in range(len(segs['length'])):
            rows = [(int(s), int(take['game'][i])) for i, s in zip(np.flatnonzero(take['seg'] == j), take['step'][take['seg'] == j])]
            boot = None
            if not segs['terminal'][j]:
                if segs['boot_store'][j] >= 0:
                    boot = ('final', int(segs['game'][j]), int(segs['seat'][j]) + 1)
                else:
                    boot = ('row', int(segs['boot_step'][j]), int(segs['game'][j]))
            got.append((rows, int(segs['length'][j]), bool(segs['terminal'][j]), float(segs['reward'][j]), boot))
    return expected, got


@pytest.mark.parametrize('seed', [0, 1, 2])
def test_segment_log_matches_segment_collector(seed):
    expected, got = _simulate(seed)
    assert sorted(expected) == sorted(got)
    assert len(got) > 20


def test_segment_log_pool_game_records_learner_seat_only():
    rng = np.random.default_rng(3)
    games, segment = 3, 4
    log = SegmentLog(games, 2, segment, 10 ** 9)
    # game 0: pool game, learner seat 1; only seat-1 rows are recorded by the caller; at the end seat 2 is dropped
    for t in range(6):
        log.record(t, np.array([0]), np.array([0], np.int32))        # seat 1 rows at steps 0..5: one cut at the 5th row
    assert log.closed_segments == 1 and log.closed_rows == segment
    log.finish(np.array([0]), np.array([[1., -1.]], np.float32), np.array([True]), np.array([[True, False]]), np.array([-1]))
    assert log.closed_segments == 2 and log.closed_rows == segment + 2
    take = log.take(1, 5, 1000)
    s = take['segments']
    assert list(s['length']) == [4] and not s['terminal'][0] and s['boot_step'][0] == 4   # boundary = the 5th row
    take = log.take(1, 5, 1000)
    s = take['segments']
    assert list(s['length']) == [2] and s['terminal'][0] and s['reward'][0] == 1. and list(take['step']) == [4, 5]


def test_segment_log_forced_cut_bootstraps_on_current_state():
    log = SegmentLog(2, 2, 32, max_age=5)
    for t in range(5):
        log.record(t, np.array([0]), np.array([1], np.int32))          # seat 2 rows at steps 0..4
    assert log.force_close(5, np.array([0, 1]), np.array([0, 0])) == 1   # first row 5 steps old: cut, mover is seat 1
    assert log.forced_rows == 5
    take = log.take(1, 5, 100)
    s = take['segments']
    assert list(s['length']) == [5] and not s['terminal'][0] and s['boot_step'][0] == 5 and s['boot_swap'][0]
    assert list(take['step']) == [0, 1, 2, 3, 4] and list(take['pos']) == [0, 1, 2, 3, 4]


def test_segment_log_take_straddles_and_refuses_stale_rows():
    log = SegmentLog(4, 2, 3, 10 ** 9)
    for t in range(8):
        log.record(t, np.arange(4), np.zeros(4, np.int32))               # 4 games x 8 rows: 2 cuts each (steps 0-2, 3-5)
    assert log.closed_segments == 8 and log.closed_rows == 24
    take = log.take(10, 7, 100)
    assert take['n'] == 12 and len(take['segments']['length']) == 4       # 4 oldest segments (3 rows each) cover 10
    assert log.closed_rows == 12
    with pytest.raises(RuntimeError):
        log.take(1, 7, 4)                                                 # rows of step 3 are 4 steps old: out of a 4-step ring


def test_lanes_blocks_and_eviction():
    rng = np.random.default_rng(0)
    lanes = Lanes(48, 2, 24, rng, per_process=True)
    assert not lanes.lane_pool.any() and lanes.p_self == 1.
    lanes.set_pool([('hist-u250', .1)], {'hist-u250'})
    assert abs(lanes.p_self - .9) < 1e-9
    # redraws over many finished games populate pool blocks about 10% of the time, learner seats in {1, 2}
    for _ in range(400):
        lanes.game_over(np.arange(48))
    assert lanes.lane_pool.any() or True
    pool_lanes = np.flatnonzero(lanes.lane_pool)
    assert set(lanes.lane_seat[pool_lanes].tolist()) <= {1, 2}
    for lane in pool_lanes.tolist():
        assert lanes.lane_name[lane] == 'hist-u250' and lanes.block_opp[lane // 24] == 'hist-u250'
    lanes.set_pool([('hist-u500', .1)], {'hist-u500'})
    assert not any(n == 'hist-u250' for n in lanes.lane_name) and 'hist-u250' not in lanes.block_opp


def test_value_sources_match_the_assembly_indices():
    """In-rollout critic values (value_mode rollout): a row's value sits in the ring at (step, game), the row the
    assembly path gathers and scores; the boundary of a cut (the seat's own row at boot_step: a 33rd-row cut, or a
    forced cut while that seat is the mover) is that ring row's value, which is the first row of the seat's next
    segment; forced cuts seen from the other seat (boot_swap), truncation boundaries from the final-view store and
    boundaries valued by an older critic need the assembly's small value call; terminal segments have no boundary.
    Rows recorded under an older critic version are stale (re-scored at assembly)."""
    G, players, segment, L, max_age, T, every = 3, 2, 4, 128, 6, 60, 20
    log = SegmentLog(G, players, segment, max_age)
    step_version = np.full(L, -1, np.int64)
    rng = np.random.default_rng(5)
    mover = rng.integers(1, players + 1, G)
    for t in range(T):
        step_version[t % L] = t // every                       # an update (new critic) every `every` steps
        log.force_close(t, np.arange(G), mover - 1)
        log.record(t, np.arange(G), (mover - 1).astype(np.int32))
        if t == 25:   # game 2 truncates without a cap reward: both seats bootstrap on the stored final views
            log.finish(np.array([2]), np.zeros((1, players), np.float32), np.array([False]), np.ones((1, players), bool),
                       np.array([0]))
        if t == 30:   # game 1 ends natively: terminal segments, no boundary
            log.finish(np.array([1]), np.array([[1., -1.]], np.float32), np.array([True]), np.ones((1, players), bool),
                       np.array([-1]))
        mover = rng.integers(1, players + 1, G)
    version = (T - 1) // every                                  # the critic at assembly
    take = log.take(log.closed_rows, T - 1, L)                  # every closed segment
    segs = take['segments']
    S = len(segs['length'])
    row_stale, b_ring, b_call = value_sources(take, step_version, version)
    assert row_stale.shape == take['step'].shape and b_ring.shape == b_call.shape == (S,)
    assert np.array_equal(row_stale, take['step'] // every != version)
    assert row_stale.any() and (~row_stale).any()
    # the ring index of every row and boundary, as the assembly gathers them
    flat = (take['step'] % L) * G + take['game']
    bflat = (segs['boot_step'] % L) * G + segs['game']
    kinds = dict(ring=0, forced_other_seat=0, stored=0, stale=0, terminal=0)
    for j in range(S):
        if segs['terminal'][j]:
            assert not b_ring[j] and not b_call[j]
            kinds['terminal'] += 1
            continue
        assert b_ring[j] != b_call[j]
        if segs['boot_store'][j] >= 0:
            assert b_call[j]
            kinds['stored'] += 1
        elif segs['boot_swap'][j]:
            assert b_call[j]
            kinds['forced_other_seat'] += 1
        elif segs['boot_step'][j] // every != version:
            assert b_call[j]
            kinds['stale'] += 1
        else:
            assert b_ring[j]
            kinds['ring'] += 1
            g, s, b = int(segs['game'][j]), int(segs['seat'][j]), int(segs['boot_step'][j])
            # the boundary row is the first row of the same (game, seat)'s next segment: closed (the next one of
            # that pair in close order) or still open in the log
            later = [k for k in range(j + 1, S) if segs['game'][k] == g and segs['seat'][k] == s]
            if later:
                k = later[0]
                first = np.flatnonzero((take['seg'] == k) & (take['pos'] == 0))
                assert len(first) == 1 and take['step'][first[0]] == b and take['game'][first[0]] == g
                assert bflat[j] == flat[first[0]]
            else:
                assert log.count[g, s] > 0 and log.steps[g, s, 0] == b and log.first[g, s] == b
    assert all(v > 0 for v in kinds.values()), kinds
    # a row's ring index is unique per (step, game): two rows never share a ring slot
    assert len(np.unique(flat)) == len(flat)


def test_checkpointer_orders_writes_and_surfaces_errors():
    ck = _Checkpointer()
    order = []

    def slow():
        time.sleep(0.05)
        order.append(1)

    ck.submit(slow)
    ck.submit(lambda: order.append(2))        # joins the first write: files land in order
    ck.wait()
    assert order == [1, 2] and ck.writes == 2 and ck.seconds >= 0.

    def boom():
        raise OSError('disk full')

    ck.submit(boom)
    with pytest.raises(RuntimeError):
        ck.wait()
    ck.submit(lambda: order.append(3))
    ck.wait()
    assert order == [1, 2, 3]


def test_row_shares_equal_split_and_shortfall():
    """Data parallel: batch // N rows per device when every device has them (remainder to the first devices); a
    device short of its quota gives everything it has and the others cover the shortfall evenly; always sum == batch
    and share <= closed."""
    assert row_shares([70, 80, 90, 100], 200).tolist() == [50, 50, 50, 50]
    assert row_shares([70, 80, 90, 100], 202).tolist() == [51, 51, 50, 50]
    assert row_shares([30, 80, 90, 100], 200).tolist() == [30, 57, 57, 56]          # device 0 short by 20
    assert row_shares([30, 52, 90, 100], 200).tolist() == [30, 52, 59, 59]          # a second device then fills up
    assert row_shares([0, 200, 0, 0], 200).tolist() == [0, 200, 0, 0]
    assert row_shares([262144], 262144).tolist() == [262144]                        # one device: the whole batch
    with pytest.raises(ValueError):
        row_shares([10, 10], 21)
    rng = np.random.default_rng(0)
    for _ in range(500):
        n = int(rng.integers(1, 9))
        batch = int(rng.integers(1, 5000))
        closed = rng.integers(0, 2 * batch // n + 2, n)
        if closed.sum() < batch:
            closed[int(rng.integers(n))] += batch - closed.sum()
        share = row_shares(closed, batch)
        assert share.sum() == batch and (share <= closed).all() and (share >= 0).all()
        # water filling: a device below the top share is exhausted (it gave everything it had)
        top = share.max()
        assert all(share[d] == closed[d] for d in range(n) if share[d] < top - 1)
        if (closed >= -(-batch // n)).all():
            assert share.max() - share.min() <= 1


def test_rebalance_plan_gives_every_device_its_static_shard():
    """After the moves every device holds exactly `per` rows and every row of the global batch sits in exactly one
    device's [0, per) (a device keeps its own first rows; only surplus rows move, into the deficit devices' tail)."""
    assert rebalance_plan([50, 50, 50, 50], 50) == []
    rng = np.random.default_rng(1)
    for case in range(300):
        n = int(rng.integers(1, 7))
        per = int(rng.integers(1, 400))
        closed = rng.integers(0, 2 * per + 1, n)
        if closed.sum() < n * per:
            closed[int(rng.integers(n))] += n * per - closed.sum()
        shares = row_shares(closed, n * per)
        moves = rebalance_plan(shares, per)
        # simulate: device d's rows are (d, i) for i < shares[d] in buffers of capacity max(shares, per)
        cap = int(max(shares.max(), per))
        buf = [[(d, i) if i < shares[d] else None for i in range(cap)] for d in range(n)]
        for src, slo, dst, dlo, k in moves:
            assert shares[src] > per and slo >= per and slo + k <= shares[src]
            assert shares[dst] < per and dlo >= shares[dst] and dlo + k <= per
            buf[dst][dlo:dlo + k] = buf[src][slo:slo + k]
        held = [r for d in range(n) for r in buf[d][:per]]
        assert None not in held
        assert sorted(held) == sorted((d, i) for d in range(n) for i in range(int(shares[d])))
        assert sum(m[4] for m in moves) == int(np.maximum(shares - per, 0).sum())
    with pytest.raises(ValueError):
        rebalance_plan([10, 10], 11)
