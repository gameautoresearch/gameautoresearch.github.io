"""Host-only panel calibration: an Elo ladder fitted to recorded games, and the sealed panel a release reads.

Generals games need GPU workers, so nothing is played here. `fit` reads games that were already recorded:

- a `tools.tournament.generals` result folder (sealed policy against sealed policy on a release's private maps);
- the release grader's private result of a policy (its games against the scripted opponents of that release).

Both use the same map indices of one release, so the interval resamples whole maps. A turn-cap draw is half a
win. Ratings are relative; `--anchor NAME=ELO` pins one player (the scripted Hunter at 0 is the convention: it
is the same program in every release).

    python -m simulators.generals.autoresearch.ladder fit --tournament DIR --graded NAME=RESULT.json ... \
        --anchor hunter=0 --out NEW_FILE.json
    python -m simulators.generals.autoresearch.ladder seal --ladder FILE.json --member NAME=PLAYER=BUNDLE ... \
        [--practice NAME] --out NEW_DIR

`seal` writes the panel folder that `prepare(panel=NEW_DIR)` turns into a release's private opponents. It checks
each package against its own manifest only; `prepare` checks the binding to the release's frozen engine.
"""
import argparse
import json
import math
import re
import shutil
from pathlib import Path

import numpy as np

from .contract import digest, sha

SCRIPTED = ('random', 'expansion', 'hunter')


def tournament_rows(folder):
    """Rows {a, b, map_index, a_seat, a_points} of a tournament result folder; forfeited games are left out."""
    rows, faults = [], 0
    for path in sorted(Path(folder).glob('pairs/*/games.jsonl')):
        for line in path.read_text().splitlines():
            r = json.loads(line)
            if r['outcome'] == 'fault':
                faults += 1
                continue
            rows.append({k: r[k] for k in ('a', 'b', 'map_index', 'a_seat', 'a_points')})
    if not rows:
        raise ValueError('No tournament games under ' + str(folder))
    return rows, faults


def graded_rows(name, result):
    """The same rows from one policy's private grading result (`grading-private/<identity>.json`)."""
    value = json.loads(Path(result).read_text())
    return [dict(a=name, b=r['opponent'], map_index=r['map_index'], a_seat=r['seat'], a_points=r['score'])
            for r in value['private']['rows']]


def fit(rows, names, prior=1e-5):
    """Maximum-likelihood Elo, centred on zero, with a tiny ridge that keeps all-win pairs finite.

    P(i beats j) = 1 / (1 + 10 ** ((r_j - r_i) / 400)); a draw adds half a win to each side.
    """
    index = {n: k for k, n in enumerate(names)}
    wins = np.zeros((len(names), len(names)))
    for r in rows:
        i, j = index[r['a']], index[r['b']]
        wins[i, j] += r['a_points']
        wins[j, i] += 1 - r['a_points']
    games, c, r = wins + wins.T, math.log(10) / 400, np.zeros(len(names))
    for _ in range(500):
        p = 1 / (1 + np.exp(-c * (r[:, None] - r[None, :])))
        gradient = c * (wins - games * p).sum(1) - prior * r
        hessian = -c * c * np.diag((games * p * (1 - p)).sum(1)) + c * c * games * p * (1 - p) - prior * np.eye(len(r))
        step = np.linalg.solve(hessian - 1e-9 * np.eye(len(r)), gradient)
        r -= np.clip(step, -200, 200)
        r -= r.mean()
        if np.abs(step).max() < 1e-6:
            break
    return r


def ladder(rows, anchor=None, replicates=1000, seed=83):
    """{name: {elo, ci95, games, points}} with a whole-map bootstrap; `anchor` is (name, elo) or None."""
    names = sorted({r['a'] for r in rows} | {r['b'] for r in rows})
    pairs = {(min(r['a'], r['b']), max(r['a'], r['b'])) for r in rows}
    linked = {names[0]}
    while True:  # one ladder needs every player connected to every other through recorded games
        more = {x for pair in pairs if linked & set(pair) for x in pair} - linked
        if not more:
            break
        linked |= more
    if linked != set(names):
        raise ValueError('The games do not connect these players into one ladder: ' + ', '.join(sorted(set(names) - linked)))
    index = {n: k for k, n in enumerate(names)}

    def shifted(values):
        return values - values[index[anchor[0]]] + anchor[1] if anchor else values

    by_map = {}
    for r in rows:
        by_map.setdefault(r['map_index'], []).append(r)
    maps, rng = sorted(by_map), np.random.default_rng(seed)
    elo = shifted(fit(rows, names))
    boots = np.array([shifted(fit([r for m in rng.choice(maps, len(maps)) for r in by_map[m]], names))
                      for _ in range(replicates)])
    low, high = np.quantile(boots, [.025, .975], axis=0)
    totals = {n: [0.0, 0] for n in names}
    for r in rows:
        for who, points in ((r['a'], r['a_points']), (r['b'], 1 - r['a_points'])):
            totals[who][0] += points
            totals[who][1] += 1
    return {n: dict(elo=float(elo[index[n]]), ci95=[float(low[index[n]]), float(high[index[n]])],
                    se=float(boots[:, index[n]].std()), games=totals[n][1], points=totals[n][0]) for n in names}


def pairwise(rows):
    table = {}
    for r in rows:
        points, total = table.get((r['a'], r['b']), (0.0, 0))
        table[(r['a'], r['b'])] = (points + r['a_points'], total + 1)
    return [dict(a=a, b=b, a_points=p, games=n, a_score=p / n) for (a, b), (p, n) in sorted(table.items())]


def fit_record(tournaments=(), graded=(), anchor=None, replicates=1000, players=None):
    """The calibration record: ratings, the pair table and the identity of every file it was fitted to."""
    rows, sources, faults = [], [], 0
    for folder in tournaments:
        found, skipped = tournament_rows(folder)
        if players:  # only games between the players being rated; the folder's other entrants are left out
            found = [r for r in found if r['a'] in players and r['b'] in players]
        rows += found
        faults += skipped
        sources.append(dict(kind='tournament', path=str(folder), games=len(found), forfeits_left_out=skipped,
                            result_sha256=sha(Path(folder) / 'result.json')))
    for name, result in graded:
        found = graded_rows(name, result)
        rows += found
        sources.append(dict(kind='graded', player=name, path=str(result), games=len(found), sha256=sha(result)))
    keys = [(r['a'], r['b'], r['map_index'], r['a_seat']) for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError('A game is recorded twice')
    ratings = ladder(rows, anchor, replicates)
    return dict(method='Bradley-Terry maximum likelihood; a turn-cap draw is half a win; 95% interval and se from a '
                       f'bootstrap over whole maps ({replicates} replicates)',
                anchor=dict(name=anchor[0], elo=anchor[1]) if anchor else None, games=len(rows),
                maps=len({r['map_index'] for r in rows}), forfeits_left_out=faults,
                players=sorted(players) if players else None, ratings=ratings,
                pairwise=pairwise(rows), sources=sources)


def _self_consistent(directory):
    """A package's manifest and files agree with each other. The engine binding is checked by `prepare`."""
    from .bundle import FILES
    root = Path(directory)
    manifest = json.loads((root / 'manifest.json').read_text())
    if digest({k: v for k, v in manifest.items() if k != 'bundle_sha256'}) != manifest['bundle_sha256']:
        raise ValueError('Manifest drift: ' + str(root))
    if set(manifest['files']) != set(FILES):
        raise ValueError('Unexpected package members: ' + str(root))
    for name, value in manifest['files'].items():
        if (root / name).is_symlink() or sha(root / name) != value:
            raise ValueError('Package file drift: ' + str(root / name))
    return manifest


def seal(out, record, members, practice=(), note=None):
    """Write a panel folder: one sealed package per member, `bots.json` and the calibration record.

    members: [(panel name, player name in the record, package folder)]. Panel names are what a researcher sees
    in practice feedback, so they should not name where a network came from.
    """
    from .bundle import FILES
    from .contract import write_json
    out, ratings = Path(out), record['ratings']
    if out.exists():
        raise FileExistsError('Choose a new panel folder; sealed panels are never overwritten')
    names = [name for name, _, _ in members]
    if len(set(names)) != len(names) or set(names) & {*SCRIPTED, 'candidate'} or set(practice) - set(names):
        raise ValueError('Panel names must be unique, not a scripted opponent, and include every practice opponent')
    if any(not re.fullmatch(r'[a-z0-9][a-z0-9-]{0,31}', name) for name in names):
        raise ValueError('Panel names are short lower-case labels (letters, digits, hyphens)')
    bots, engines = [], set()
    out.mkdir(parents=True)
    for name, player, folder in members:
        manifest = _self_consistent(folder)
        (out / name).mkdir()
        for filename in (*FILES, 'manifest.json'):
            shutil.copyfile(Path(folder) / filename, out / name / filename)
        _self_consistent(out / name)
        engines.add(manifest['engine'])
        bots.append(dict(name=name, kind='policy', bundle=name, bundle_sha256=manifest['bundle_sha256'],
                         tensors=manifest['tensors'], elo=ratings[player]['elo'], elo_ci95=ratings[player]['ci95'],
                         practice=name in practice, origin=player,
                         qualification='sealed submitted policy; rated on the panel ladder'))
    if len(engines) != 1:
        raise ValueError('Panel members are sealed against different frozen engines')
    write_json(out / 'ladder.json', record, exclusive=True)
    write_json(out / 'bots.json', dict(
        bots=bots, engine=engines.pop(), anchor=record['anchor'],
        scripted={n: dict(elo=ratings[n]['elo'], elo_ci95=ratings[n]['ci95']) for n in SCRIPTED if n in ratings},
        ladder='ladder.json', ladder_sha256=sha(out / 'ladder.json'), note=note), exclusive=True)
    return bots


def read_panel(folder):
    """A sealed panel folder's `bots.json`, with its calibration record checked against the recorded hash."""
    root = Path(folder)
    panel = json.loads((root / 'bots.json').read_text())
    if panel.get('ladder') and sha(root / panel['ladder']) != panel['ladder_sha256']:
        raise ValueError('Panel calibration record drift')
    return panel


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='op', required=True)
    c = sub.add_parser('fit')
    c.add_argument('--tournament', action='append', default=[], type=Path)
    c.add_argument('--graded', action='append', default=[], metavar='NAME=RESULT.json')
    c.add_argument('--anchor', metavar='NAME=ELO')
    c.add_argument('--players', help='comma-separated tournament entrants to rate (default: all of them)')
    c.add_argument('--replicates', type=int, default=1000)
    c.add_argument('--out', required=True, type=Path)
    c = sub.add_parser('seal')
    c.add_argument('--ladder', required=True, type=Path)
    c.add_argument('--member', action='append', required=True, metavar='NAME=PLAYER=BUNDLE')
    c.add_argument('--practice', action='append', default=[])
    c.add_argument('--note')
    c.add_argument('--out', required=True, type=Path)
    a = p.parse_args()
    if a.op == 'fit':
        if a.out.exists():
            raise SystemExit('Choose a new output file; calibration records are never overwritten')
        anchor = (a.anchor.split('=')[0], float(a.anchor.split('=')[1])) if a.anchor else None
        record = fit_record(a.tournament, [tuple(v.split('=', 1)) for v in a.graded], anchor, a.replicates,
                            set(a.players.split(',')) if a.players else None)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        with a.out.open('x') as f:
            json.dump(record, f, indent=1)
        for name, r in sorted(record['ratings'].items(), key=lambda item: -item[1]['elo']):
            print(f"{name:>12} {r['elo']:8.0f}  [{r['ci95'][0]:.0f}, {r['ci95'][1]:.0f}]  {r['games']} games")
    else:
        bots = seal(a.out, json.loads(a.ladder.read_text()), [tuple(v.split('=', 2)) for v in a.member], a.practice, a.note)
        for bot in bots:
            print(f"{bot['name']:>12} {bot['elo']:8.0f}  {bot['bundle_sha256'][:12]}  practice={bot['practice']}")


if __name__ == '__main__':
    main()
