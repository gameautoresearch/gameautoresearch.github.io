"""Offline check of sparse tournament schedules against a finished full round robin (its games.jsonl, elo.json).

Every pair of the full run played 4 blocks of 16 games (8 maps x both seats per block). A schedule picks blocks:
  stage A  each entrant vs the anchor, every reference and R random partners: 1 block each;
  stage B  a provisional Bradley-Terry fit, then each entrant vs its K nearest neighbours by provisional Elo: up to
           B blocks each (pairs already played in stage A get their extra blocks).
The schedule's Elo (anchor = 0) is compared with the full run's: rank correlation, mean / p90 / max absolute error over
research networks and finals, and how often each model's best network is the same.

    PYTHONPATH=$PWD .venv-pokemon/bin/python -m tools.tournament.polytopia_elo.schedule_sim --config CONFIG.json
"""
import collections, itertools, json, random, sys
from pathlib import Path

import numpy as np

from tools.tournament.polytopia_elo import config as configs
from tools.tournament.polytopia_elo.schedule import fit

HERE = None      # the tournament folder (set from --config)
ANCHOR = None    # the first anchor of the entrants


def load():
    blocks = collections.defaultdict(lambda: [0.0, 0])          # (a, b, round) -> [points of a, games]
    for line in (HERE / 'games.jsonl').open():
        g = json.loads(line)
        k = (g['a'], g['b'], g['round'])
        blocks[k][0] += g['a_points']
        blocks[k][1] += 1
    kinds = {n['network']: n['kind'] for n in json.loads((HERE / 'elo.json').read_text())['networks']}
    return blocks, kinds


def fit_sparse(names, pairs, iterations=20000, tol=1e-10):
    """Bradley-Terry (MM) where the half-win prior applies only to pairs that were played: with a sparse schedule
    the full-run fit's prior on every pair (1/(k-1) game each) pulls isolated networks toward the middle."""
    index = {n: i for i, n in enumerate(names)}
    k = len(names)
    wins = np.zeros((k, k)); games = np.zeros((k, k))
    for a, b, points, n in pairs:
        i, j = index[a], index[b]
        wins[i, j] += points + .5; wins[j, i] += n - points + .5
        games[i, j] += n + 1; games[j, i] += n + 1
    p = np.ones(k)
    total = wins.sum(1)
    for _ in range(iterations):
        new = total / (games / (p[:, None] + p[None, :])).sum(1)
        new /= np.exp(np.log(new).mean())
        if np.abs(new - p).max() < tol:
            p = new
            break
        p = new
    return 400 * np.log10(p)


FIT = fit


def elo(names, blocks, chosen):
    pairs = [(a, b, blocks[(a, b, r)][0], blocks[(a, b, r)][1]) for a, b, r in chosen]
    e = FIT(names, pairs)
    return dict(zip(names, e - e[names.index(ANCHOR)]))


def spearman(x, y):
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def simulate(blocks, kinds, R, K, B, seed=0):
    rng = random.Random(seed)
    names = sorted(kinds)
    key = {}
    for a, b, r in blocks:
        key.setdefault(frozenset((a, b)), set()).add((a, b, r))
    def block(a, b, i):
        rows = sorted(key.get(frozenset((a, b)), ()), key=lambda t: t[2])
        return rows[i] if i < len(rows) else None
    fixed = [n for n in names if kinds[n] in ('reference', 'anchor')]
    chosen, used = set(), collections.Counter()
    for n in names:
        if kinds[n] == 'reference':
            continue
        partners = [f for f in fixed if f != n] + rng.sample([m for m in names if m != n and m not in fixed], R)
        for m in partners:
            p = frozenset((n, m))
            if used[p] == 0 and block(n, m, 0):
                chosen.add(block(n, m, 0)); used[p] = 1
    provisional = elo(names, blocks, chosen)
    for n in names:
        if kinds[n] == 'reference':
            continue
        near = sorted((m for m in names if m != n and not (kinds[m] == 'reference' and kinds[n] == 'reference')),
                      key=lambda m: abs(provisional[m] - provisional[n]))[:K]
        for m in near:
            p = frozenset((n, m))
            while used[p] < B and block(n, m, used[p]):
                chosen.add(block(n, m, used[p])); used[p] += 1
    return elo(names, blocks, chosen), sum(blocks[c][1] for c in chosen)


def main():
    global FIT, HERE, ANCHOR
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--sparse-fit', action='store_true')
    a = ap.parse_args()
    HERE = Path(configs.load(a.config)['dir'])
    blocks, kinds = load()
    ANCHOR = next(n for n, k in sorted(kinds.items()) if k == 'anchor')
    names = sorted(kinds)
    full = elo(names, blocks, list(blocks))          # the reported fit, as published
    FIT = fit_sparse if a.sparse_fit else fit
    total = sum(v[1] for v in blocks.values())
    scored = [n for n in names if kinds[n] in ('research', 'final')]
    best_full = {}
    for n in scored:
        m = n.split(':')[0]
        if kinds[n] == 'research' and (m not in best_full or full[n] > full[best_full[m]]):
            best_full[m] = n
    print(f'full: {total:,} games, {len(names)} entrants')
    print(' R  K  B   games   share   rho    MAE   p90   max  best-same  top(>-800) MAE  max')
    rows = []
    for R, K, B in itertools.product((4, 8), (6, 10, 16, 24), (2, 4)):
        errs, rhos, same, games, errs_top = [], [], [], [], []
        for seed in range(3):
            est, g = simulate(blocks, kinds, R, K, B, seed)
            e = np.array([abs(est[n] - full[n]) for n in scored])
            errs.append(e)
            errs_top.append([est[n] - full[n] for n in scored if full[n] > -800])
            rhos.append(spearman([full[n] for n in scored], [est[n] for n in scored]))
            bs = {}
            for n in scored:
                m = n.split(':')[0]
                if kinds[n] == 'research' and (m not in bs or est[n] > est[bs[m]]):
                    bs[m] = n
            same.append(sum(bs[m] == best_full[m] for m in best_full) / len(best_full))
            games.append(g)
        e = np.concatenate(errs)
        top = np.abs(np.concatenate([np.array(x) for x in errs_top]))
        row = dict(R=R, K=K, B=B, games=int(np.mean(games)), share=np.mean(games) / total, rho=float(np.mean(rhos)),
                   mae=float(e.mean()), p90=float(np.percentile(e, 90)), max=float(e.max()), best_same=float(np.mean(same)),
                   top_mae=float(top.mean()), top_max=float(top.max()))
        rows.append(row)
        print(f"{R:2d} {K:2d} {B:2d} {row['games']:8,d} {row['share']*100:5.1f}%  {row['rho']:.4f} {row['mae']:5.1f} "
              f"{row['p90']:5.1f} {row['max']:5.1f}  {row['best_same']*100:4.0f}%      {row['top_mae']:5.1f} {row['top_max']:5.1f}")
    (HERE / 'schedule_sim.json').write_text(json.dumps(dict(full_games=total, rows=rows), indent=1) + '\n')


if __name__ == '__main__':
    main()
