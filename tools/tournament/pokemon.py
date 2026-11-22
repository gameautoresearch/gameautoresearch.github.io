"""Pokémon head-to-head tournament on the release's frozen evaluator-v2 arena, on native CPU workers.

Each unordered pair (A, B) is one or more `assess` jobs of the release's own worker (`runtime/worker.py` ->
`runtime/arena.py run`) through `harness.games.pokemon.provider.Provider`: A is the `bundle-native` candidate
and the panel is exactly `[{kind: 'bundle-native', name: B, path: /pilotjob/input/peer-0}]`. The arena runs
both sealed graphs as isolated unprivileged policy processes (candidate uid 65534, peer uid 65532) and plays
every seed with A in seat 0 and in seat 1, so B-vs-A would only repeat the same seat assignments.

Seeds are a fresh host-secret bank written to `OUT/private/bank.json` (128 seeds, twice session grading, smoke 2),
shared by every pair. Workers are CPU-only (`kind='final'`, 4 cores / 16 GiB, network blocked) and are charged
on a tournament-local ledger in `OUT/ledger.json`; no session or release ledger is touched. Run from the
repository root:

    PYTHONPATH=$PWD MODAL_PROFILE=duper MODAL_DISABLE_API_PROXY=1 .venv-pokemon/bin/python \
        -m tools.tournament.pokemon --entrant A=SESSION --entrant B=SESSION --out NEW_DIR [--smoke]
"""
import sys

sys.dont_write_bytecode = True

import json
import secrets
import sqlite3
import threading
import time
import uuid
from pathlib import Path

from tools.tournament import common

UNITS = {True: 2, False: 128}   # seeds (clusters): twice session grading, for tighter lift intervals
CHUNK = 8                       # seeds per CPU worker: 16 games
POINTS_RULE = ('per game: natural engine outcome from A\'s seat, win 1, tie 0.5, loss 0 (the arena\'s own points); '
               'administrative truncations, faults and unplayed games are counted separately and never scored')


def native():
    from simulators.pokemon.autoresearch.runtime import bundle as module
    return module


def release_manifest(release):
    return json.loads((Path(release) / 'release.json').read_text())


def session_bundle(session):
    """The locked final identity of the session's fresh final training, read without writing to its ledger."""
    db = Path(session) / 'final/ledger.sqlite3'
    if not db.is_file():
        raise ValueError(f'{session} has no final-training ledger')
    connection = sqlite3.connect(f'file:{db}?mode=ro', uri=True)
    try:
        row = connection.execute("SELECT value FROM metadata WHERE name='final_locked'").fetchone()
    finally:
        connection.close()
    if not row:
        raise ValueError(f'Final training of {session} never locked a submission')
    final = json.loads(row[0])
    return Path(final['bundle']), final['bundle_sha256']


def entrant(kind, path, manifest):
    bundle, expected = session_bundle(path) if kind == 'session' else (Path(path), None)
    sealed = native().verify(bundle, expected)
    binding = json.loads((bundle / 'binding.json').read_text())
    if binding.get('runtime_sha256') != manifest['protected_runtime_sha256']:
        raise ValueError(f'{bundle} is bound to a different native runtime than the release')
    if binding.get('benchmark_id') != manifest['benchmark_id']:
        raise ValueError(f'{bundle} belongs to a different benchmark')
    return dict(session=str(path) if kind == 'session' else None, bundle=str(bundle),
                bundle_sha256=sealed['bundle_sha256'])


def private_bank(out, count):
    seeds = secrets.SystemRandom().sample(range(1_900_000_000, 2_000_000_000), count)
    private = out / 'private'
    private.mkdir(mode=0o700)
    common.write_json(private / 'bank.json', dict(seeds=seeds))
    return seeds


def lease(games):
    return min(7200, max(300, 180 + 30 * games))


def bundle_files(bundle, prefix):
    """Every file the native verifier binds (the runtime FILES include graph.json) plus the manifest."""
    return [(Path(bundle) / name, prefix + '/' + name) for name in (*native().FILES, 'manifest.json')]


def build_request(release, a, b, a_name, b_name, seeds):
    """The release's frozen evaluator request with A as candidate and the panel replaced by B alone."""
    from harness.games.pokemon.provider import assessment_files
    request, files = assessment_files(release)
    files = [(src, name) for src, name in files if not name.startswith('peer-')]
    files += bundle_files(b['bundle'], 'peer-0') + bundle_files(a['bundle'], 'candidate')
    request.update(panel=[dict(kind='bundle-native', name=b_name, path='/pilotjob/input/peer-0',
                               bundle_sha256=b['bundle_sha256'])],
                   candidate=dict(kind='bundle-native', name=a_name, path='/pilotjob/input/candidate',
                                  bundle_sha256=a['bundle_sha256']),
                   seeds=list(seeds), public_development=False)
    return request, files


class Ledger:
    """Tournament-local allocation journal with the calls `Provider.launch`/`collect` make.

    Like the session ledger, a reservation settles only on the provider's stop receipt, and one that
    never reached the provider settles at zero; unlike it, several CPU workers may run at once.
    """

    def __init__(self, path):
        self.path = Path(path)
        self.lock = threading.Lock()
        self.jobs = {}
        self.save()

    def save(self):
        common.write_json(self.path, dict(jobs=list(self.jobs.values())))

    def reserve(self, name, seconds):
        with self.lock:
            ident = uuid.uuid4().hex
            self.jobs[ident] = dict(id=ident, name=name, kind='final', seconds=seconds, status='reserved',
                                    provider=None, actual_seconds=None, receipt=None, created=time.time())
            self.save()
            return ident

    def cancel_unattached(self, ident, reason):
        with self.lock:
            job = self.jobs[ident]
            if job['status'] != 'reserved' or job['provider'] is not None:
                raise ValueError('Only an unattempted unattached reservation can settle at zero')
            job.update(status='settled', actual_seconds=0, receipt=dict(provider_created=False, reason=str(reason)))
            self.save()

    def attach(self, ident, provider):
        with self.lock:
            if self.jobs[ident]['status'] != 'reserved':
                raise ValueError('Provider already attached')
            self.jobs[ident].update(status='active', provider=provider)
            self.save()

    def settle(self, ident, seconds, receipt):
        if not receipt.get('provider_verified_stopped'):
            raise ValueError('Complete provider stop receipt required')
        with self.lock:
            job = self.jobs[ident]
            if job['status'] != 'settled':
                job.update(status='settled', actual_seconds=seconds, receipt=receipt)
                self.save()

    def settle_unstarted(self, ident, receipt, reason):
        """A worker the provider never started (harness/workers.py): settled at zero, receipt kept."""
        receipt = dict(receipt or {})
        if receipt.get('started_at') or not receipt.get('provider_id'):
            raise ValueError("Only the provider's record of an unstarted worker settles at zero")
        value = dict(receipt, provider_never_started=True, charged_seconds=0,
                     reason=str(reason or 'The provider never started this worker; it was terminated'))
        with self.lock:
            job = self.jobs[ident]
            if job['status'] == 'settled':
                return job['receipt']
            job.update(status='settled', actual_seconds=0, receipt=value)
            self.save()
        return value

    def unknown(self, ident, detail):
        with self.lock:
            if self.jobs[ident]['status'] != 'settled':
                self.jobs[ident].update(status='unknown', receipt=detail)
                self.save()

    def job(self, ident):
        with self.lock:
            return dict(self.jobs[ident])


def normalize(raw, part):
    rows = []
    for r in raw:
        clean = r.get('natural') and r.get('fault') is None
        rows.append(dict(unit=part[r['seed_index']], seat=r['seat'], a_points=r['points'] if clean else None,
                         natural=bool(r.get('natural')), truncated=bool(r.get('truncated')),
                         fault=r.get('fault_category') or ('fault' if r.get('fault') else None),
                         terminal_reason=r.get('terminal_reason'), engine_turns=r.get('engine_turns'),
                         wall_seconds=r.get('wall_seconds')))
    return rows


def fault_owner(category):
    if not category:
        return None
    return 'a' if category.startswith('candidate') else 'b' if category.startswith('reference') else 'other'


def play(provider, ledger, clock, release, entrants, pair, part, seeds, poll=5.0):
    a, b = pair
    label = f'{a}-vs-{b}-{part[0]:04d}'
    request, files = build_request(release, entrants[a], entrants[b], a, b, [seeds[i] for i in part])
    seconds = lease(2 * len(part))
    operation = ledger.reserve(label, seconds)
    try:
        provider.launch(ledger, operation, 'final', dict(action='assess', assessment=request), files, seconds)
        deadline = time.monotonic() + seconds + 900
        while True:
            try:
                outcome = provider.collect(ledger, operation)
            except Exception as exc:
                if ledger.job(operation)['status'] == 'unknown':
                    raise
                outcome = dict(status='active-or-unknown', error=type(exc).__name__)
            if outcome['status'] in ('settled', 'unknown'):
                break
            if time.monotonic() > deadline:
                raise TimeoutError(f'{label}: worker unresolved; reconcile operation {operation}')
            time.sleep(poll)
    finally:
        if ledger.job(operation)['status'] != 'settled':
            try:
                provider.stop(operation)  # never leave a created worker running past an interruption
            except Exception:
                pass
    clock.add(outcome.get('provider'), label)
    if outcome['status'] == 'unknown':
        raise RuntimeError(f'{label}: provider lifetime unknown; operation {operation}')
    try:
        provider.verify_profile(operation, None)
    except Exception as exc:
        outcome['profile_error'] = type(exc).__name__ + ': ' + str(exc)[:200]
    output = Path(outcome['output']) if outcome.get('output') else None
    raw = json.loads((output / 'rows.json').read_text()) if output and (output / 'rows.json').is_file() else []
    summary = json.loads((output / 'summary.json').read_text()) if output and (output / 'summary.json').is_file() else None
    errors = []
    if not outcome.get('successful') or summary is None:
        failure = output / 'failure.json' if output else None
        reason = json.loads(failure.read_text()).get('fault') if failure and failure.is_file() else None
        errors.append(dict(seeds=[part[0], part[-1]], operation=operation,
                           error=reason or 'worker did not complete the assessment'))
    if outcome.get('profile_error'):
        errors.append(dict(seeds=[part[0], part[-1]], operation=operation, error=outcome['profile_error']))
    return dict(rows=normalize(raw, part), errors=errors, operation=operation, output=str(output) if output else None,
                status=summary and summary.get('status'))


def main(argv=None, provider=None, poll=5.0):
    p = common.parser('pokemon', 'seeds')
    p.add_argument('--chunk', type=int, default=CHUNK, help='seeds per CPU worker')
    args = p.parse_args(argv)
    started = time.time()
    specs = common.entrant_specs(args)
    release = common.common_release(specs, args.release)
    manifest = release_manifest(release)
    entrants = {name: entrant(kind, path, manifest) for name, kind, path in specs}
    out = common.new_out(args.out, [release, *(path for _, _, path in specs),
                                    *(Path(e['bundle']) for e in entrants.values())])
    count = args.slice or UNITS[args.smoke]
    common.write_json(out / 'request.json', dict(argv=sys.argv if argv is None else argv, seeds=count,
                                                 chunk=args.chunk))
    seeds = private_bank(out, count)
    if provider is None:
        from harness.policy import require_modal
        from harness.games.pokemon.provider import Provider
        require_modal()
        provider = Provider(release, out / 'provider')  # audits the release read-only; app/Volume private to OUT
    ledger = Ledger(out / 'ledger.json')
    clock = common.Clock()
    pairs = common.pairs(list(entrants), args.anchor)
    plan = [(pair, part) for pair in pairs for part in common.chunks(count, args.chunk)]
    outcomes = common.run_tasks([lambda pair=pair, part=part: play(provider, ledger, clock, release, entrants, pair,
                                                                    part, seeds, poll) for pair, part in plan],
                                args.parallel)
    results = []
    for pair in pairs:
        a, b = pair
        rows, errors, chunks = [], [], []
        for (owner, part), (value, error) in zip(plan, outcomes):
            if owner != pair:
                continue
            if error:
                errors.append(dict(seeds=[part[0], part[-1]], error=error))
                continue
            rows.extend(value['rows'])
            errors.extend(value['errors'])
            chunks.append(dict(seeds=[part[0], part[-1]], operation=value['operation'], output=value['output'],
                               status=value['status']))
        rows.sort(key=lambda r: (r['unit'], r['seat']))
        owners = {}
        for row in rows:
            owner = fault_owner(row['fault'])
            if owner:
                key = {'a': a, 'b': b}.get(owner, 'other')
                owners[key] = owners.get(key, 0) + 1
        details = out / 'pairs' / f'{a}-vs-{b}'
        common.write_json(details / 'rows.json', rows)
        common.write_json(details / 'chunks.json', chunks)
        results.append(common.pair_summary(a, b, rows, 2 * count, f'pairs/{a}-vs-{b}/', errors, owners))
    value = common.result('pokemon', release, manifest['release_sha256'], bool(args.smoke), entrants, results, clock,
                          started, POINTS_RULE, harness_release_sha256=common.harness_release_sha256(release),
                          seeds=count, seats='A in seat 0 and seat 1 on every seed', anchors=list(args.anchor),
                          schedule='anchors' if args.anchor else 'round-robin',
                          evaluator='runtime/arena.py run (frozen release), CPU workers',
                          ledger='ledger.json')
    common.write_json(out / 'result.json', value)
    print(json.dumps(dict(out=str(out), leaderboard=value['leaderboard'], worker_seconds=value['worker_seconds'],
                          errors=sum(len(r['errors']) for r in results)), indent=2))
    return value


if __name__ == '__main__':
    main()
