"""Host-only panel calibration: opponent-vs-opponent matches and a Bradley-Terry (Elo) fit.

Every pair plays the same 2-ply openings from both seats, all games of all pairs in one
lockstep batch so the MoHex pool stays busy. No candidate policy, GPU or cloud is
involved. Ratings are relative; `--anchor NAME=ELO` pins one player.

    .venv-hex/bin/python -m simulators.hex.autoresearch.ladder --players players.json \
        --openings 32 --out NEW_FILE.json [--anchor mohex-100=-365]
"""
import argparse
from itertools import combinations
import json
import math
from pathlib import Path
import time

import numpy as np

from .arena import MoHexPool, check_panel, opening_bank, opponent_moves, start_games


def play(pairs, openings, seed=1, workers=7, binary=None):
    """[(spec a, spec b)] -> rows {a, b, opening_index, a_seat, a_won, plies}, every opening from both seats."""
    import torch
    check_panel(list({spec['name']: spec for pair in pairs for spec in pair}.values()))
    bank = opening_bank(seed, openings)
    jobs = [(a, b, i, a_seat) for a, b in pairs for i in range(openings) for a_seat in (1, 2)]
    games, history = start_games([bank[i] for _, _, i, _ in jobs])
    rngs = [np.random.default_rng([seed, index]) for index in range(len(jobs))]
    with MoHexPool(workers, seed, binary) as pool:
        while not bool(games.done.all()):
            running = (~games.done).nonzero().squeeze(1).tolist()
            movers = []
            for index in running:
                a, b, _, a_seat = jobs[index]
                movers.append((index, a if int(games.to_play[index]) == a_seat else b))
            moves = opponent_moves(movers, games, games.legal(), rngs, history, pool)
            actions = torch.zeros(games.n, dtype=torch.long)
            for index, cell in moves.items():
                actions[index] = cell
                history[index].append(cell)
            games.step(actions)
    return [{'a': a['name'], 'b': b['name'], 'opening_index': i, 'a_seat': 'black' if s == 1 else 'white',
             'a_won': int(games.winner[k]) == s, 'plies': len(history[k])}
            for k, (a, b, i, s) in enumerate(jobs)]


def fit(rows, names, prior=1e-5):
    """Maximum-likelihood Elo with a tiny ridge (keeps all-win pairs finite) and Fisher standard errors.

    Ratings are centred on zero; P(i beats j) = 1 / (1 + 10 ** ((r_j - r_i) / 400)).
    """
    index = {n: k for k, n in enumerate(names)}
    wins = np.zeros((len(names), len(names)))
    for r in rows:
        i, j = index[r['a']], index[r['b']]
        if r['a_won']:
            wins[i, j] += 1
        else:
            wins[j, i] += 1
    games = wins + wins.T
    c = math.log(10) / 400
    r = np.zeros(len(names))
    for _ in range(200):
        p = 1 / (1 + np.exp(-c * (r[:, None] - r[None, :])))
        gradient = c * (wins - games * p).sum(1) - prior * r
        hessian = -c * c * np.diag((games * p * (1 - p)).sum(1)) + c * c * games * p * (1 - p) - prior * np.eye(len(r))
        step = np.linalg.solve(hessian - 1e-9 * np.eye(len(r)), gradient)
        r -= step
        r -= r.mean()
        if np.abs(step).max() < 1e-6:
            break
    p = 1 / (1 + np.exp(-c * (r[:, None] - r[None, :])))
    # Without the ridge: the pseudo-inverse is then the covariance of the mean-centred ratings.
    information = c * c * (np.diag((games * p * (1 - p)).sum(1)) - games * p * (1 - p))
    covariance = np.linalg.pinv(information)
    return {n: {'elo': float(r[k]), 'se': float(math.sqrt(max(covariance[k, k], 0)))} for n, k in index.items()}


def pairwise(rows):
    table = {}
    for r in rows:
        key = (r['a'], r['b'])
        won, total = table.get(key, (0, 0))
        table[key] = (won + r['a_won'], total + 1)
    return [{'a': a, 'b': b, 'a_wins': w, 'games': n, 'a_win_rate': w / n} for (a, b), (w, n) in sorted(table.items())]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--players', required=True, type=Path, help='JSON list of opponent specs')
    p.add_argument('--openings', type=int, default=32)
    p.add_argument('--seed', type=int, default=20261112)
    p.add_argument('--workers', type=int, default=7)
    p.add_argument('--neighbours', type=int, default=0,
                   help='play each player only against this many neighbours in list order (0: everyone)')
    p.add_argument('--anchor', help='NAME=ELO: shift ratings so NAME has this Elo')
    p.add_argument('--out', required=True, type=Path)
    a = p.parse_args()
    if a.out.exists():
        raise SystemExit('Choose a new output file; calibration records are never overwritten')
    players = json.loads(a.players.read_text())
    check_panel(players)
    pairs = [(x, y) for i, x in enumerate(players) for j, y in enumerate(players)
             if i < j and (not a.neighbours or j - i <= a.neighbours)]
    started = time.monotonic()
    rows = play(pairs, a.openings, a.seed, a.workers)
    ratings = fit(rows, [x['name'] for x in players])
    if a.anchor:
        name, value = a.anchor.split('=')
        shift = float(value) - ratings[name]['elo']
        for r in ratings.values():
            r['elo'] += shift
    record = {'players': players, 'openings': a.openings, 'seed': a.seed, 'pairs': len(pairs), 'games': len(rows),
              'seconds': round(time.monotonic() - started, 1), 'anchor': a.anchor, 'ratings': ratings,
              'pairwise': pairwise(rows), 'rows': rows}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open('x') as f:
        json.dump(record, f, indent=1)
    for x in players:
        print(f"{x['name']:>18} {ratings[x['name']]['elo']:8.0f} ± {ratings[x['name']]['se']:.0f}")
    print(f"{len(rows)} games in {record['seconds']} s")


if __name__ == '__main__':
    main()
