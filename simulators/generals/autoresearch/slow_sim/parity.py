"""Host check: the task's plain-Python simulator (`generals_sim`) plays exactly the frozen JAX engine's games.

Both simulators are stepped in lock-step from the same maps with the same joint actions, and after every tick the
whole true state, both players' observations and legal masks, the encoded inputs and the step outcome
(terminated, truncated, reward) are compared. States, observations, masks and outcomes must be identical; the
float32 inputs may differ by rounding in log1p (at most 1e-6).

Actions come from mixed policies so that rare rules happen: uniform legal moves, an aggressive policy that
attacks the cheapest visible enemy or neutral target and heads for the enemy general, out-of-mask ids (illegal
moves that must change nothing) and passes; and scripted head-on general trades.

    python -m simulators.generals.autoresearch.slow_sim.parity --games 64 --ticks 1000
"""
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import generals_sim as slow  # noqa: E402


def fast_modules():
    import jax
    import jax.numpy as jnp
    from integrations.generals import environment as env
    from generals.core import game as engine
    return jax, jnp, env, engine


def to_fast(jnp, engine, games):
    """A batched JAX GameState from slow games."""
    arrays = [g.as_arrays() for g in games]
    return engine.GameState(**{k: jnp.asarray(np.stack([a[k] for a in arrays])) for k in slow.FIELDS})


def compare_state(fast_states, games, where):
    for name in slow.FIELDS:
        a = np.asarray(getattr(fast_states, name))
        b = np.stack([g.as_arrays()[name] for g in games])
        if a.shape != b.shape or not np.array_equal(a.astype(b.dtype), b):
            bad = np.argwhere((a.astype(b.dtype) != b).reshape(len(games), -1).any(-1)).ravel()
            raise AssertionError(f'{where}: state field {name} differs in games {bad[:8].tolist()}')


def aggressive(rng, view, mask):
    """An attacking policy over the legal moves: capture what the stack can beat, prefer generals and enemies."""
    legal = np.flatnonzero(mask[1:]) + 1
    if not len(legal) or rng.random() < .05:
        return 0
    best, score = [], -1e9
    for a in legal[rng.permutation(len(legal))[:96]]:
        p, r, c, d, s = slow.decode_action(a)
        rr, cc = r + slow.DIRECTIONS[d][0], c + slow.DIRECTIONS[d][1]
        army = view['armies'][r][c]
        moved = army // 2 if s else army - 1
        target = view['armies'][rr][cc]
        mine = view['owned_cells'][rr][cc]
        value = (moved > target) * (3 + 4 * view['opponent_cells'][rr][cc] + 20 * view['generals'][rr][cc]
                                    + 2 * view['castles'][rr][cc]) * (not mine) + .01 * moved + rng.random()
        if value > score:
            best, score = a, value
    return int(best)


def choose(rng, mode, view, mask):
    if mode == 'uniform':
        legal = np.flatnonzero(mask)
        return int(legal[rng.integers(len(legal))])
    if mode == 'aggressive':
        return aggressive(rng, view, mask)
    # noisy: mostly aggressive, sometimes any id at all (illegal moves must leave the board unchanged)
    if rng.random() < .15:
        return int(rng.integers(slow.ACTION_COUNT))
    return aggressive(rng, view, mask)


def run(maps_path, games=32, ticks=1000, seed=0, check_every=1):
    jax, jnp, env, engine = fast_modules()
    rng = np.random.default_rng(seed)
    bank = slow.load_maps(maps_path)
    picks = rng.integers(len(bank), size=games)
    played = [bank[i].copy() for i in picks]
    modes = [('uniform', 'aggressive', 'noisy')[i % 3] for i in range(games)]
    fast = to_fast(jnp, engine, played)
    step = jax.jit(env.transition)
    observe = jax.jit(lambda s: (env.observations(s), env.encode(env.observations(s)), env.action_mask(env.observations(s))))
    stats = dict(ticks=0, captures=0, truncations=0, illegal_ids=0, trades=0, max_encode_diff=0.0)
    for tick in range(ticks):
        obs, x_fast, m_fast = observe(fast)
        obs = jax.tree.map(np.asarray, obs)
        x_fast, m_fast = np.asarray(x_fast), np.asarray(m_fast)
        ids = np.zeros((games, 2), np.int32)
        for g, game in enumerate(played):
            for p in range(2):
                view = slow.observe(game, p)
                if tick % check_every == 0:
                    for name in slow.SPATIAL:
                        if not np.array_equal(np.asarray(view[name]).astype(np.asarray(getattr(obs, name)).dtype), getattr(obs, name)[g, p]):
                            raise AssertionError(f'tick {tick} game {g} player {p}: observation {name} differs')
                    for name in slow.SCALARS:
                        if int(view[name]) != int(getattr(obs, name)[g, p]):
                            raise AssertionError(f'tick {tick} game {g} player {p}: {name} differs')
                    x, m = slow.encode(view), slow.legal_mask(view)
                    if not np.array_equal(m, m_fast[g, p]):
                        raise AssertionError(f'tick {tick} game {g} player {p}: legal mask differs')
                    diff = float(np.abs(x - x_fast[g, p]).max())
                    stats['max_encode_diff'] = max(stats['max_encode_diff'], diff)
                    if diff > 1e-6:
                        raise AssertionError(f'tick {tick} game {g} player {p}: encoded input differs by {diff}')
                else:
                    m = slow.legal_mask(view)
                ids[g, p] = choose(rng, modes[g], view, m)
                stats['illegal_ids'] += int(not m[ids[g, p]])
        result = step(fast, env.decode_action(jnp.asarray(ids)))
        for g, game in enumerate(played):
            terminated, truncated, reward = game.step([slow.decode_action(i) for i in ids[g]])
            if (bool(result.terminated[g]) != terminated or bool(result.truncated[g]) != truncated
                    or not np.array_equal(np.asarray(result.reward[g]), np.asarray(reward, np.float32))):
                raise AssertionError(f'tick {tick} game {g}: step outcome differs')
            stats['captures'] += int(terminated)
            stats['truncations'] += int(truncated)
        compare_state(result.final_state, played, f'tick {tick}')
        stats['ticks'] += games
        # finished games restart on another map, in both simulators
        done = np.asarray(result.terminated | result.truncated)
        if done.any():
            fresh = rng.integers(len(bank), size=games)
            for g in np.flatnonzero(done):
                played[g] = bank[fresh[g]].copy()
            fast = env.reset_done(result.final_state, jnp.asarray(done), to_fast(jnp, engine, bank), jnp.asarray(fresh))
            compare_state(fast, played, f'tick {tick} reset')
        else:
            fast = result.final_state
    return stats


def trade_cases(maps_path, seed=1, cases=64):
    """Scripted mutual general captures and near-misses: both generals attacked head-on on one tick."""
    jax, jnp, env, engine = fast_modules()
    rng = np.random.default_rng(seed)
    bank = slow.load_maps(maps_path)
    checked = trades = 0
    for k in range(cases):
        game = bank[rng.integers(len(bank))].copy()
        (g0r, g0c), (g1r, g1c) = game.general_positions
        # give each player a strip of land next to the other's general and a stack on it
        for p, (tr, tc) in ((0, (g1r, g1c)), (1, (g0r, g0c))):
            for d, (dr, dc) in enumerate(slow.DIRECTIONS):
                r, c = tr - dr, tc - dc
                if 0 <= r < slow.SIZE and 0 <= c < slow.SIZE and game.passable[r][c] and not game.generals[r][c]:
                    for q in range(2):
                        game.ownership[q][r][c] = q == p
                    game.ownership_neutral[r][c] = False
                    game.castles[r][c] = False
                    game.armies[r][c] = int(rng.integers(1, 30))
                    break
        for p, (r, c) in enumerate(((g0r, g0c), (g1r, g1c))):
            game.armies[r][c] = int(rng.integers(1, 25))
        game.time = int(rng.integers(0, 999))
        fast = to_fast(jnp, engine, [game])
        actions = []
        for p, (tr, tc) in ((0, (g1r, g1c)), (1, (g0r, g0c))):
            options = []
            for d, (dr, dc) in enumerate(slow.DIRECTIONS):
                r, c = tr - dr, tc - dc
                if 0 <= r < slow.SIZE and 0 <= c < slow.SIZE and game.ownership[p][r][c]:
                    options.append([0, r, c, d, int(rng.integers(2))])
            # sometimes the defender's own general moves out instead (the garrison it leaves is what counts)
            if not options or rng.random() < .2:
                gr, gc = game.general_positions[p]
                options = [[0, gr, gc, int(rng.integers(4)), int(rng.integers(2))]]
            actions.append(options[0])
        ids = np.array([[slow.encode_action(a) for a in actions]], np.int32)
        before = game.copy()
        trades += int(before.is_general_trade([slow.decode_action(i) for i in ids[0]], 0, 1)
                      or before.is_general_trade([slow.decode_action(i) for i in ids[0]], 1, 0))
        result = jax.jit(env.transition)(fast, env.decode_action(jnp.asarray(ids)))
        terminated, truncated, reward = game.step([slow.decode_action(i) for i in ids[0]])
        if bool(result.terminated[0]) != terminated or bool(result.truncated[0]) != truncated:
            raise AssertionError(f'trade case {k}: outcome differs')
        compare_state(result.final_state, [game], f'trade case {k}')
        checked += 1
    return dict(cases=checked, trades=trades)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--maps', required=True)
    p.add_argument('--games', type=int, default=32)
    p.add_argument('--ticks', type=int, default=1000)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--trade-cases', type=int, default=256)
    a = p.parse_args()
    started = time.time()
    report = dict(play=run(a.maps, a.games, a.ticks, a.seed), trades=trade_cases(a.maps, a.seed + 1, a.trade_cases))
    report['seconds'] = round(time.time() - started, 1)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
