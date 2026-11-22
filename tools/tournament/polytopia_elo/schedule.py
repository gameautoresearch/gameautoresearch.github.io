"""Which pairs play which blocks, and the Bradley-Terry fit.

A block is one call: one pair on `chunk` maps x both seats; block b of every pair uses the same maps (schedule seeds
[b*chunk, (b+1)*chunk)), so pairs are compared on common maps.

full    every pair (references never play each other) plays blocks 0..rounds-1: N^2/2 pairs.
sparse  two stages, about 20 % of the full design's games for ~150 entrants:
          A  every entrant plays every anchor and reference and R random other entrants: block 0;
          B  provisional fit; every entrant plays its K nearest neighbours by provisional Elo up to block B-1
             (pairs from stage A that are neighbours get their remaining blocks).
        Checked offline against the full 753,600-game round robin of polytopia-empty15-20261118 (schedule_sim.py):
        R=4, K=24, B=4 played 21 % of the games; for networks above -800 Elo the mean absolute error was 11 Elo (max
        65), against a leave-one-block-out spread of the full run itself of ~16-27 Elo (map sampling: its 32 maps).
"""
import math
import random

import numpy as np


def fit(names, pairs, iterations=5000, tol=1e-10):
    """Bradley-Terry by MM on pair points; Elo = 400 log10 strength (mean 0). `pairs`: (a, b, points of a, games)."""
    index = {n: i for i, n in enumerate(names)}
    k = len(names)
    wins = np.zeros((k, k)); games = np.zeros((k, k))
    for a, b, points, n in pairs:
        i, j = index[a], index[b]
        wins[i, j] += points; wins[j, i] += n - points
        games[i, j] += n; games[j, i] += n
    p = np.ones(k)
    total = wins.sum(1) + 0.5  # tiny prior keeps a network with zero wins finite
    games_prior = games + 1.0 / max(k - 1, 1) * (1 - np.eye(k))
    for _ in range(iterations):
        new = total / (games_prior / (p[:, None] + p[None, :])).sum(1)
        new /= math.exp(np.log(new).mean())
        if np.abs(new - p).max() < tol:
            p = new
            break
        p = new
    return 400 * np.log10(p)


def eligible(kind, a, b):
    return a != b and not (kind[a] == 'reference' and kind[b] == 'reference')


def all_pairs(names, kind):
    return [(names[i], names[j]) for i in range(len(names)) for j in range(i + 1, len(names))
            if eligible(kind, names[i], names[j])]


def key(a, b):
    return tuple(sorted((a, b)))


def stage_a(names, kind, partners=4, seed=0):
    """Pairs of stage A: every non-reference entrant vs every anchor and reference and `partners` random others."""
    rng = random.Random(seed)
    fixed = [n for n in names if kind[n] in ('anchor', 'reference')]
    others = [n for n in names if kind[n] not in ('anchor', 'reference')]
    out = set()
    for n in names:
        if kind[n] == 'reference':
            continue
        pool = [m for m in others if m != n]
        for m in [f for f in fixed if f != n] + rng.sample(pool, min(partners, len(pool))):
            if eligible(kind, n, m):
                out.add(key(n, m))
    return sorted(out)


def neighbours(names, kind, provisional, k=24):
    """Pairs of stage B: every non-reference entrant and its k nearest entrants by provisional Elo."""
    out = set()
    for n in names:
        if kind[n] == 'reference' or n not in provisional:
            continue
        near = sorted((m for m in names if m in provisional and eligible(kind, n, m)),
                      key=lambda m: abs(provisional[m] - provisional[n]))[:k]
        out.update(key(n, m) for m in near)
    return sorted(out)


def provisional_elo(names, games):
    """Elo (mean 0) from game rows {a, b, a_points}."""
    agg = {}
    for g in games:
        k = key(g['a'], g['b'])
        pts = g['a_points'] if k[0] == g['a'] else 1 - g['a_points']
        v = agg.setdefault(k, [0.0, 0])
        v[0] += pts; v[1] += 1
    played = sorted({x for k in agg for x in k})
    e = fit(played, [(a, b, s, n) for (a, b), (s, n) in agg.items()])
    return dict(zip(played, (float(x) for x in e)))
