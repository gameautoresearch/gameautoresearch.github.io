"""Host side of the Generals and Dou Dizhu tournaments: one Modal sandbox per pair.

A game module supplies an adapter (bundle verification, private schedule, Modal image, staging and the
worker script it uploads). This module resolves entrants, schedules every unordered pair, runs each pair in
its own sandbox built from the game's grading image with the grading isolation arguments, checks the
returned rows against the schedule, scores them and writes `--out/result.json`. Nothing here writes outside
`--out`. See tools/tournament/README.md.
"""
import sys

sys.dont_write_bytecode = True  # never drop __pycache__ into harness/ or simulators/ while a campaign runs

from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
from pathlib import Path
import argparse
import hashlib
import json
import re
import time
import traceback

ENVIRONMENT = 'game-autoresearch-project'
NAME = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,63}')
FINAL_BUNDLE = ('final', 'jobs', 'final', 'bundle')
WORKER_PATH = '/private/tournament_worker.py'
REQUEST_PATH = '/private/tournament.json'
RESULT_PATH = '/private/tournament-result.json'


def canonical_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def write_json(path, value, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n'
    with path.open('x' if exclusive else 'w') as handle:
        handle.write(data)


# ---------------------------------------------------------------- entrants

def parse_entrant(spec):
    """NAME=SESSION_DIR (the session's graded network) or NAME=SESSION_DIR@JOB (one network the session packaged)."""
    name, sep, path = spec.partition('=')
    path, at, job = path.partition('@')
    if not sep or not path or not NAME.fullmatch(name) or (at and not NAME.fullmatch(job)):
        raise ValueError('Entrant must be NAME=SESSION_DIR[@JOB] with NAME matching ' + NAME.pattern + ': ' + spec)
    return name, Path(path).expanduser().resolve(), (job or None)


def resolve_entrant(name, session, job, verify, release_engine_digest, local_release=None):
    """Read-only resolution of one entrant's sealed final bundle: a recipe session's fresh final network, or the
    network a network session submitted last.

    `verify(bundle, expected)` is the game's own bundle verifier and returns its manifest;
    `release_engine_digest(release)` is the digest of that release's frozen engine.
    """
    session = Path(session)
    if not session.is_dir():
        raise ValueError(name + ': no session directory ' + str(session))
    run_path = Path(str(session) + '.harness-run.json')
    if not run_path.is_file():
        raise ValueError(name + ': missing ' + run_path.name + ' beside the session')
    run = json.loads(run_path.read_text())
    settings = run.get('settings', {})
    if job is not None:
        # One packaged network of the session (a checkpoint), read-only from its ledger: finished or not.
        import sqlite3
        ledger = sqlite3.connect(f'file:{session / "research" / "ledger.sqlite3"}?mode=ro', uri=True)
        try:
            row = ledger.execute('select artifact_sha from jobs where name=?', (job,)).fetchone()
        finally:
            ledger.close()
        if not row or not row[0]:
            raise ValueError(name + ': ' + job + ' packaged no network in ' + str(session))
        expected = row[0]
        bundle = session / 'research' / 'jobs' / job / 'bundle'
    elif settings.get('variant') == 'network':
        # The network the session submitted last: the sealed bundle its final grade scored.
        record = json.loads((session / 'session.json').read_text()) if (session / 'session.json').is_file() else {}
        submitted = [s for s in record.get('submissions') or [] if s.get('kind') == 'network']
        if not record.get('finalized') or not submitted:
            raise ValueError(name + ': the session is not finalized with a submitted network')
        expected = submitted[-1]['native']['bundle_sha256']
        bundle = session / 'research' / 'jobs' / submitted[-1]['name'] / 'bundle'
    elif settings.get('variant') == 'recipe':
        final_train = json.loads((session / 'final-train.json').read_text())
        if final_train.get('status') != 'complete':
            raise ValueError(name + ': fresh final training is not complete (' + str(final_train.get('status')) + ')')
        expected = (final_train.get('submission') or {}).get('bundle_sha256')
        if not expected:
            raise ValueError(name + ': final-train.json names no submitted bundle')
        bundle = session.joinpath(*FINAL_BUNDLE)
    else:
        raise ValueError(name + ': not a recipe or network session (variant ' + str(settings.get('variant')) + ')')
    manifest = verify(bundle, expected)
    release = Path(settings['release'])
    if not release.exists() and local_release is not None:
        # A session run by a cloud coordinator records the container's copy (/work/releases/...). The local copy
        # stands in for it only when its binding is the one the session ran under.
        sidecar = Path(str(Path(local_release).resolve()) + '.harness.json')
        bound = json.loads(sidecar.read_text()).get('sha256') if sidecar.is_file() else None
        if not bound or bound != run.get('release_sha256'):
            raise ValueError(name + ': --release ' + str(local_release) + ' is not the release this session ran under')
        release = Path(local_release)
    release = release.resolve()
    if manifest['engine'] != release_engine_digest(release):
        raise ValueError(name + ': bundle is not sealed against the frozen engine of ' + str(release))
    return dict(name=name, session=str(session), job=job, bundle=str(bundle), bundle_sha256=manifest['bundle_sha256'],
                release=str(release), release_sha256=run.get('release_sha256'), game=run.get('game'),
                model=settings.get('model'))


def common_release(entrants, game):
    if len(entrants) < 2:
        raise ValueError('A tournament needs at least two entrants')
    names = [e['name'] for e in entrants]
    if len(set(names)) != len(names):
        raise ValueError('Duplicate entrant names')
    if any(e['game'] != game for e in entrants):
        raise ValueError('Every entrant must be a ' + game + ' session')
    releases = {e['release'] for e in entrants}
    digests = {e['release_sha256'] for e in entrants}
    if len(releases) != 1 or len(digests) != 1:
        raise ValueError('Entrants come from different releases: ' + ', '.join(sorted(releases)))
    release = Path(releases.pop())
    digest = digests.pop()
    sidecar = Path(str(release) + '.harness.json')
    if sidecar.is_file() and json.loads(sidecar.read_text()).get('sha256') != digest:
        raise ValueError('Entrant release_sha256 differs from the release binding ' + sidecar.name)
    return release, digest


def check_out(out, release, entrants):
    out = Path(out).expanduser().resolve()
    if out.exists() or out.is_symlink():
        raise FileExistsError('--out must not exist: ' + str(out))
    protected = [Path(release).resolve()] + [Path(e['session']).resolve() for e in entrants]
    for root in protected:
        if out == root or root in out.parents:
            raise ValueError('--out may not be inside a release or session directory: ' + str(root))
    return out


def pairs_of(entrants, anchors=()):
    """Every unordered pair; with anchors, only the pairs that include one (see common.pairs)."""
    names = [e['name'] for e in entrants]
    anchors = list(anchors)
    unknown = sorted(set(anchors) - set(names))
    if unknown:
        raise ValueError('Anchors that are not entrants: ' + ', '.join(unknown))
    every = list(combinations(names, 2))
    if not anchors or len(anchors) == len(names):
        return every
    return [(a, b) for a, b in every if a in anchors or b in anchors]


# ---------------------------------------------------------------- scoring

def validate_rows(rows, expected_keys, key):
    seen = [key(r) for r in rows]
    if len(seen) != len(set(seen)):
        raise ValueError('Duplicate game rows')
    if set(seen) != set(expected_keys):
        raise ValueError('Worker rows differ from the schedule (' + str(len(seen)) + ' rows, ' +
                         str(len(expected_keys)) + ' scheduled)')
    for r in rows:
        if r['a_points'] not in (0.0, 0.5, 1.0) or r['outcome'] not in ('natural', 'truncated', 'fault'):
            raise ValueError('Invalid game row: ' + json.dumps(r, sort_keys=True))
        if (r['outcome'] == 'fault') != bool(r.get('fault_by')):
            raise ValueError('Fault rows must name the faulting entrant')


def cluster_ci(rows, unit, replicates=2000, seed=83):
    """95% interval of A's 0-100 score; resamples whole maps/deals (both seats or orderings together)."""
    import numpy as np
    clusters = {}
    for r in rows:
        clusters.setdefault(r[unit], []).append(r['a_points'])
    if not clusters:
        return [0.0, 0.0]
    values = np.asarray([np.mean(clusters[k]) for k in sorted(clusters)])
    rng = np.random.default_rng(seed)
    boots = values[rng.integers(len(values), size=(replicates, len(values)))].mean(1)
    return [float(100 * q) for q in np.quantile(boots, [.025, .975])]


def summarize_pair(a, b, rows, unit, details):
    games = len(rows)
    points = float(sum(r['a_points'] for r in rows))
    faults_by = {}
    for r in rows:
        if r['outcome'] == 'fault':
            faults_by[r['fault_by']] = faults_by.get(r['fault_by'], 0) + 1
    return dict(a=a, b=b, games=games, a_points=points,
                a_score_0_100=100 * points / games if games else 0.0,
                ci95=cluster_ci(rows, unit),
                faults=sum(r['outcome'] == 'fault' for r in rows), faults_by=faults_by,
                truncations=sum(r['outcome'] == 'truncated' for r in rows),
                details=details)


def leaderboard(names, pair_rows):
    """Mean points over every game an entrant played, from that entrant's side."""
    scored = {n: [] for n in names}
    for (a, b), rows in pair_rows.items():
        for r in rows:
            scored[a].append(r['a_points'])
            scored[b].append(1.0 - r['a_points'])
    board = [dict(name=n, score_0_100=100 * sum(v) / len(v) if v else 0.0, games=len(v)) for n, v in scored.items()]
    return sorted(board, key=lambda x: (-x['score_0_100'], x['name']))


# ---------------------------------------------------------------- Modal pair runner

class ModalPairRunner:
    """One sandbox per pair: the game's grading image, no network, no secrets, no volumes."""

    def __init__(self, adapter, app, image, timeout, rpc=None):
        self.adapter = adapter
        self.app = app
        self.image = image
        self.timeout = int(timeout)
        self.rpc = rpc or (lambda call: call())

    def __call__(self, pair, staging, folder):
        import modal
        adapter = self.adapter
        rpc = self.rpc
        sb = None
        host_start = time.time()
        record = dict(sandbox_id=None, app_id=self.app.app_id, environment=ENVIRONMENT, gpu=adapter.gpu,
                      cpu=4, memory_mib=16384, timeout=self.timeout)
        try:
            sb = modal.Sandbox.create('/bin/sleep', '86400', app=self.app, image=self.image, cpu=(4, 4),
                                      memory=(16384, 16384), timeout=self.timeout, block_network=True, secrets=[],
                                      volumes={}, include_oidc_identity_token=False, environment_name=ENVIRONMENT,
                                      tags={'tournament_pair': pair[0] + '-vs-' + pair[1]},
                                      **({'gpu': adapter.gpu} if adapter.gpu else {}))
            record['sandbox_id'] = sb.object_id
            write_json(folder / 'sandbox.json', record)
            for remote in staging['directories']:
                sb.filesystem.make_directory(remote)
            for local, remote in staging['files']:
                sb.filesystem.copy_from_local(local, remote)
            sb.filesystem.write_text(json.dumps(staging['request']), REQUEST_PATH)
            sb.filesystem.write_text(Path(adapter.worker_source).read_text(), WORKER_PATH)
            proc = sb.exec(adapter.python, WORKER_PATH, REQUEST_PATH, timeout=max(30, self.timeout - 20),
                           env=adapter.worker_env)
            while (code := rpc(proc.poll)) is None:
                time.sleep(1)
            for stream in ('stdout', 'stderr'):
                try:
                    text = getattr(proc, stream).read()[-200000:]
                except Exception as exc:
                    text = 'unavailable: ' + repr(exc)
                (folder / ('worker-' + stream + '.log')).write_text(text)
            if code != 0:
                raise RuntimeError('Tournament worker failed (exit ' + str(code) + '); see worker-stderr.log')
            return json.loads(rpc(lambda: sb.filesystem.read_text(RESULT_PATH)))
        finally:
            if sb is not None:
                try:
                    sb.terminate(wait=True)
                except Exception as exc:
                    record['termination_error'] = repr(exc)
                info = None
                for _ in range(40):
                    try:
                        info = adapter.allocation_info(self.app.app_id, sb.object_id)
                    except Exception as exc:
                        record['allocation_error'] = repr(exc)
                    if info and info.get('finished_at'):
                        break
                    time.sleep(.5)
                record['allocation'] = info
                if info and info.get('started_at') and info.get('finished_at'):
                    record['worker_seconds'] = float(info['finished_at'] - info['started_at'])
                    record['worker_seconds_source'] = 'provider allocation receipt'
                else:
                    record['worker_seconds'] = float(time.time() - host_start)
                    record['worker_seconds_source'] = 'host wall time (provider receipt unavailable)'
                if adapter.gpu:
                    record['gpu_type'] = (info or {}).get('gpu_type')
                    record['gpu_matches'] = adapter.gpu_matches((info or {}).get('gpu_type'))
                record['host_seconds'] = time.time() - host_start
                write_json(folder / 'sandbox.json', record)


# ---------------------------------------------------------------- driver

def run(adapter, entrants_spec, out, smoke=False, units=None, parallel=4, timeout=None, runner=None, anchors=(),
        local_release=None):
    """Resolve, play every unordered pair, score, write result.json. Returns the result dict."""
    started = time.time()
    entrants = [resolve_entrant(*parse_entrant(s), adapter.verify, adapter.release_engine_digest, local_release)
                for s in entrants_spec]
    release, release_sha = common_release(entrants, adapter.game)
    out = check_out(out, release, entrants)
    schedule = adapter.schedule(release, smoke, units)
    out.mkdir(parents=True)
    names = [e['name'] for e in entrants]
    by_name = {e['name']: e for e in entrants}
    pairs = pairs_of(entrants, anchors)
    write_json(out / 'plan.json', dict(game=adapter.game, release=str(release), release_sha256=release_sha,
                                       smoke=bool(smoke), entrants=entrants, pairs=pairs,
                                       schedule=adapter.public_schedule(schedule), scoring=adapter.scoring,
                                       started_at=started), exclusive=True)
    context = None
    try:
        if runner is None:
            context, runner = adapter.modal_runner(release, timeout or (900 if smoke else 3600))
            write_json(out / 'app.json', dict(app_id=runner.app.app_id, environment=ENVIRONMENT))

        def play(pair):
            a, b = pair
            folder = out / 'pairs' / (a + '-vs-' + b)
            folder.mkdir(parents=True)
            try:
                with adapter.staging(by_name[a], by_name[b], release, schedule) as staging:
                    raw = runner(pair, staging, folder)
                rows = raw['rows']
                validate_rows(rows, adapter.expected_keys(schedule), adapter.row_key)
                with (folder / 'games.jsonl').open('x') as handle:
                    for r in rows:
                        handle.write(json.dumps(dict(r, a=a, b=b), sort_keys=True) + '\n')
                write_json(folder / 'worker-summary.json', {k: v for k, v in raw.items() if k != 'rows'})
                return pair, rows, None
            except BaseException as exc:
                (folder / 'host-error.txt').write_text(traceback.format_exc())
                if not isinstance(exc, Exception):
                    raise
                return pair, None, type(exc).__name__ + ': ' + str(exc)[:2000]

        with ThreadPoolExecutor(max_workers=max(1, min(parallel, len(pairs)))) as pool:
            outcomes = list(pool.map(play, pairs))
    finally:
        if context is not None:
            context.__exit__(None, None, None)
    pair_rows = {}
    summaries = []
    worker_seconds = 0.0
    for (a, b), rows, error in outcomes:
        folder = out / 'pairs' / (a + '-vs-' + b)
        receipt = folder / 'sandbox.json'
        if receipt.is_file():
            worker_seconds += json.loads(receipt.read_text()).get('worker_seconds') or 0.0
        details = 'pairs/' + a + '-vs-' + b + '/'
        if rows is None:
            summaries.append(dict(summarize_pair(a, b, [], adapter.unit, details), error=error))
            continue
        pair_rows[(a, b)] = rows
        summaries.append(summarize_pair(a, b, rows, adapter.unit, details))
    result = dict(game=adapter.game, release=str(release), release_sha256=release_sha, smoke=bool(smoke),
                  complete=len(pair_rows) == len(pairs),
                  entrants={e['name']: dict(session=e['session'], bundle=e['bundle'], bundle_sha256=e['bundle_sha256'],
                                            model=e['model']) for e in entrants},
                  schedule=adapter.public_schedule(schedule), scoring=adapter.scoring,
                  anchors=list(anchors), pairing='anchors' if anchors else 'round-robin',
                  pairs=summaries, leaderboard=leaderboard(names, pair_rows),
                  worker_seconds=worker_seconds, started_at=started, finished_at=time.time())
    write_json(out / 'result.json', result, exclusive=True)
    return result


def main(adapter, argv=None):
    parser = argparse.ArgumentParser(prog='python -m tools.tournament.' + adapter.game,
                                     description='Head-to-head ' + adapter.game + ' tournament between finished research networks.')
    parser.add_argument('--entrant', action='append', required=True, metavar='NAME=SESSION_DIR')
    parser.add_argument('--out', required=True, help='new directory for result.json and per-pair details')
    parser.add_argument('--smoke', action='store_true', help='tiny slice of the private schedule, for testing')
    parser.add_argument('--' + adapter.unit_flag, type=int, default=None, dest='units',
                        help='play only the first N private ' + adapter.unit_flag + ' (testing)')
    parser.add_argument('--parallel', type=int, default=4, help='pairs played concurrently (one worker each)')
    parser.add_argument('--timeout', type=int, default=None, help='sandbox lifetime per pair, seconds')
    parser.add_argument('--anchor', action='append', default=[], metavar='NAME',
                        help='an anchor entrant (the trained starter); with anchors only pairs including one are played')
    parser.add_argument('--release', default=None, help='local copy of the release, for sessions run by a cloud '
                        'coordinator (they record the container\'s path); its binding must match theirs')
    args = parser.parse_args(argv)
    result = run(adapter, args.entrant, args.out, smoke=args.smoke, units=args.units, parallel=args.parallel,
                 timeout=args.timeout, anchors=args.anchor, local_release=args.release)
    brief = [dict({k: p[k] for k in ('a', 'b', 'games', 'a_score_0_100', 'ci95', 'faults', 'truncations')},
                  **({'error': p['error']} if 'error' in p else {})) for p in result['pairs']]
    print(json.dumps(dict(complete=result['complete'], leaderboard=result['leaderboard'], pairs=brief,
                          worker_seconds=result['worker_seconds']), indent=2))
    return 0 if result['complete'] else 1
