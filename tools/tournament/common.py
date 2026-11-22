"""Host-side tournament plumbing shared by every game: entrants, pair schedule, scoring and the result file.

Nothing here runs entrant code or talks to Modal; the game modules supply the evaluator.
"""
import argparse
import itertools
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.dont_write_bytecode = True  # never drop __pycache__ into harness/ or simulators/ while a campaign runs

NAME = re.compile(r'[A-Za-z0-9_-]{1,40}')
ENVIRONMENT = 'game-autoresearch-project'


def parser(game, unit):
    p = argparse.ArgumentParser(prog='python -m tools.tournament.' + game,
                                description=f'Head-to-head {game} tournament between finished research networks.')
    p.add_argument('--entrant', action='append', default=[], metavar='NAME=SESSION_DIR',
                   help='a finished session: the fresh final network of a recipe session, or the network a '
                        'network session submitted last')
    p.add_argument('--bundle', action='append', default=[], metavar='NAME=DIR',
                   help='a sealed research-run bundle, for testing; needs --release')
    p.add_argument('--release', help='release every entrant must be bound to (required with --bundle)')
    p.add_argument('--out', required=True, help='new output directory (must not exist)')
    p.add_argument('--smoke', action='store_true', help=f'tiny slice: 2 {unit}')
    p.add_argument('--slice', type=int, metavar='N', help=f'play only N {unit} (testing)')
    p.add_argument('--parallel', type=int, default=4, help='concurrent CPU workers')
    p.add_argument('--anchor', action='append', default=[], metavar='NAME',
                   help='an anchor entrant (the trained starter); with anchors only pairs including one are played')
    return p


def entrant_specs(args):
    """[(name, kind, path)] in command-line order; names unique and file-system safe."""
    specs = []
    for kind, values in (('session', args.entrant), ('bundle', args.bundle)):
        for value in values:
            name, sep, path = value.partition('=')
            if not sep or not NAME.fullmatch(name) or not path:
                raise ValueError(f'Use NAME=PATH with NAME matching {NAME.pattern}: {value!r}')
            specs.append((name, kind, Path(path).resolve()))
    names = [s[0] for s in specs]
    if len(set(names)) != len(names):
        raise ValueError('Entrant names must be unique')
    if len(specs) < 2:
        raise ValueError('A tournament needs at least two entrants')
    if any(kind == 'bundle' for _, kind, _ in specs) and not args.release:
        raise ValueError('--bundle entrants need --release')
    return specs


def session_settings(session):
    session = Path(session).resolve()
    run = session.with_name(session.name + '.harness-run.json')
    if not run.is_file():
        raise ValueError(f'Not a harness session directory: {session}')
    return json.loads(run.read_text())['settings']


def network_submission(session):
    """The harness record of the network a finalized network session submitted last (the one its final grade scored)."""
    path = Path(session) / 'session.json'
    record = json.loads(path.read_text()) if path.is_file() else {}
    submitted = [s for s in record.get('submissions') or [] if s.get('kind') == 'network']
    if not record.get('finalized') or not submitted:
        raise ValueError(f'Session is not finalized with a submitted network: {session}')
    return submitted[-1]


def session_release(session):
    """The release a finished session is bound to. A recipe session needs a complete fresh final training, a
    network session a finalized submitted network."""
    session = Path(session).resolve()
    settings = session_settings(session)
    if settings.get('variant') == 'network':
        network_submission(session)
    elif settings.get('variant') == 'recipe':
        trained = session / 'final-train.json'
        if not trained.is_file() or json.loads(trained.read_text()).get('status') != 'complete':
            raise ValueError(f'Session has no completed fresh final training: {session}')
    else:
        raise ValueError(f'Not a recipe or network session: {session}')
    return Path(settings['release']).resolve()


def common_release(specs, override):
    """One release for every entrant: session bindings must agree with each other and with --release."""
    releases = {session_release(path) for _, kind, path in specs if kind == 'session'}
    if override:
        releases.add(Path(override).resolve())
    if len(releases) != 1:
        raise ValueError('Entrants come from different releases: ' + ', '.join(sorted(map(str, releases))))
    return releases.pop()


def harness_release_sha256(release):
    """The harness's own release record identity, read only (the release may predate the current source)."""
    sidecar = Path(str(release) + '.harness.json')
    return json.loads(sidecar.read_text()).get('sha256') if sidecar.is_file() else None


def new_out(out, protected):
    """Create the output directory; it must be new and outside every release, session and bundle."""
    out = Path(out).resolve()
    if out.exists() or out.is_symlink():
        raise FileExistsError(f'--out must not exist: {out}')
    for path in protected:
        path = Path(path).resolve()
        if out == path or path in out.parents:
            raise ValueError(f'--out may not be inside {path}')
    out.mkdir(parents=True)
    return out


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=False, default=str) + '\n')
    os.replace(temporary, path)


def pairs(names, anchors=()):
    """Every unordered pair once, in entrant order; with anchors, only pairs that include an anchor.

    The anchors schedule is what lift over the starter needs: every entrant against every anchor, and the
    anchors against each other. It skips entrant-vs-entrant games, so field scores then mean "vs anchors".
    """
    anchors = list(anchors)
    unknown = sorted(set(anchors) - set(names))
    if unknown:
        raise ValueError('Anchors that are not entrants: ' + ', '.join(unknown))
    every = list(itertools.combinations(names, 2))
    if not anchors:
        return every
    if len(anchors) == len(names):
        return every
    return [(a, b) for a, b in every if a in anchors or b in anchors]


def chunks(count, size):
    return [list(range(start, min(count, start + size))) for start in range(0, count, size)]


def bootstrap_ci(rows, replicates=4000, seed=20261111):
    """95% interval of A's 0-100 score, resampling evaluation units (map/seed clusters, both seats together)."""
    import numpy as np
    units = {}
    for row in rows:
        if row['a_points'] is not None:
            units.setdefault(row['unit'], []).append(row['a_points'])
    if not units:
        return None
    means = np.array([np.mean(v) for _, v in sorted(units.items())])
    rng = np.random.default_rng(seed)
    boot = means[rng.integers(len(means), size=(replicates, len(means)))].mean(1) * 100
    return [float(x) for x in np.quantile(boot, [.025, .975])]


def pair_summary(a, b, rows, scheduled, details, errors=(), faults_by=None):
    """Normalized rows: {unit, seat, a_points (None = not scored), natural, truncated, fault}."""
    scored = [r for r in rows if r['a_points'] is not None]
    points = float(sum(r['a_points'] for r in scored))
    return dict(a=a, b=b, games=len(scored), a_points=points,
                a_score_0_100=100 * points / len(scored) if scored else None,
                ci95=bootstrap_ci(rows), scheduled_games=scheduled, attempted_games=len(rows),
                unplayed_games=scheduled - len(rows), natural_games=sum(bool(r['natural']) for r in rows),
                faults=sum(r['fault'] is not None for r in rows), truncations=sum(bool(r['truncated']) for r in rows),
                faults_by=faults_by or {}, by_a_seat=by_seat(rows), errors=list(errors),
                complete=len(scored) == scheduled and not errors, details=details)


def by_seat(rows):
    seats = {}
    for row in rows:
        if row['a_points'] is not None:
            seats.setdefault(str(row['seat']), []).append(row['a_points'])
    return {seat: dict(games=len(v), a_score_0_100=100 * sum(v) / len(v)) for seat, v in sorted(seats.items())}


def leaderboard(names, results):
    """Mean score over all of an entrant's scored games; B's points in a game are 1 - A's."""
    totals = {name: [0.0, 0] for name in names}
    for pair in results:
        totals[pair['a']][0] += pair['a_points']
        totals[pair['a']][1] += pair['games']
        totals[pair['b']][0] += pair['games'] - pair['a_points']
        totals[pair['b']][1] += pair['games']
    board = [dict(name=name, score_0_100=100 * p / g if g else None, games=g, points=p)
             for name, (p, g) in totals.items()]
    return sorted(board, key=lambda r: (r['score_0_100'] is None, -(r['score_0_100'] or 0), r['name']))


def run_tasks(tasks, parallel):
    """Run callables concurrently; returns [(value, error)] in task order. One failure never stops the rest."""
    def guarded(task):
        try:
            return task(), None
        except Exception as exc:
            return None, type(exc).__name__ + ': ' + str(exc)[:800]
    with ThreadPoolExecutor(max_workers=max(1, parallel)) as pool:
        return list(pool.map(guarded, tasks))


class Clock:
    """Provider-reported worker seconds (finished_at - started_at), summed across threads."""

    def __init__(self):
        self.lock = threading.Lock()
        self.seconds = 0.0
        self.workers = []

    def add(self, receipt, label):
        if receipt and receipt.get('started_at') and receipt.get('finished_at'):
            seconds = float(receipt['finished_at'] - receipt['started_at'])
        else:
            seconds = None
        with self.lock:
            self.workers.append(dict(label=label, seconds=seconds,
                                     provider_id=(receipt or {}).get('provider_id')))
            if seconds is not None:
                self.seconds += seconds


def result(game, release, release_sha256, smoke, entrants, pair_results, clock, started, points_rule, **extra):
    return dict(game=game, release=str(release), release_sha256=release_sha256, smoke=smoke, entrants=entrants,
                pairs=pair_results, leaderboard=leaderboard(list(entrants), pair_results),
                worker_seconds=clock.seconds, workers=clock.workers, started_at=started, finished_at=time.time(),
                points_rule=points_rule, **extra)
