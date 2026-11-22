"""Tavern head-to-head tournament in full lobbies on the release's CPU evaluation image.

Tavern is a multiplayer game (N = 4 in the small form, 8 in the full-size form), so a head-to-head pair (A, B)
is played inside full lobbies: A and B sit opposite each other (seats k and k+N/2) and the N-2 other seats are
the release registry's trusted visible scripts (the first N-2 of its `scripts` profile, in a fixed clockwise
order from seat k). Every lobby is played twice with A and B swapped and everything else identical, and k runs
over 0..N/2-1, so each entrant plays every seat and every lobby has an exact mirror: identical networks score
exactly 50.

Both entrants run as the mixed-v2 evaluator runs its candidate (the frozen `policy_worker.py` behind
`IsolatedVisible`, each under its own unprivileged uid with an owner-only copy of its own bundle); the worker
`tavern_worker.py` is uploaded into a CPU sandbox built from the release's frozen image (4 cores, 16 GiB, no
GPU, no network, no secrets, `harness.games.tavern.provider.ModalProvider`) and drives the frozen
`evaluation_readiness.lobby.run_lobby`.

Scoring: a game is one lobby; A's points are 1 if A finished above B, else 0 (natural placements are a
permutation, so there are no ties). The leaderboard is the mean of these head-to-head points. Each pair also
reports both entrants' mean placement points 100*(N - placement)/(N-1) in the same lobbies. A lobby voided by an
entrant's fault (illegal action, forward deadline, process failure) is forfeited by that entrant; a lobby voided
by anything else (simulator fault, administrative cap) is recorded but not scored.

Seeds are fresh 64-bit host secrets (288 lobbies per pair, the private final bank's lobby count: 72 groups of 4
lobbies at N=4, 36 groups of 8 at N=8; smoke 2 groups), screened like the final
bank against the release's public/registry schedules and every entrant's declared training streams, and
written to `OUT/private/schedule.json`. Run from the repository root:

    PYTHONPATH=$PWD MODAL_PROFILE=duper MODAL_DISABLE_API_PROXY=1 .venv-tavern-ar/bin/python \
        -m tools.tournament.tavern --entrant A=SESSION --entrant B=SESSION --out NEW_DIR [--smoke]
"""
import sys

sys.dont_write_bytecode = True

import contextlib
import hashlib
import json
import secrets
import time
from pathlib import Path

from tools.tournament import common, tavern_worker

HERE = Path(__file__).resolve().parent
LOBBIES = 288                   # lobbies per pair, the private final bank's lobby count
SMOKE_UNITS = 2                 # seed groups in a smoke
CHUNK_LOBBIES = 96              # lobbies per CPU worker (about 1 s each at eight players)
LOBBY_SECONDS = 120             # run_lobby's wall cap, as in the evaluator's screen
SCRIPT_PROFILE = 'scripts'      # registry profile the N-2 script seats are taken from (its first N-2 slots)
FILES = tavern_worker.BUNDLE_FILES
POINTS_RULE = ('per lobby: A gets 1 if A placed above B, else 0 (natural placements are distinct). A lobby voided '
               'by one entrant\'s fault (illegal action, inference deadline, policy process failure) is forfeited '
               'by that entrant and counted in faults; a lobby voided by a simulator fault or administrative cap '
               'is counted (truncations/faults) and not scored. Placement points 100*(N-placement)/(N-1) of both '
               'entrants in the same lobbies are reported per pair as a secondary measure')
SCHEME = ('Each seed group is N lobbies (N players): for k in 0..N/2-1, A and B sit in seats k and k+N/2 (then '
          'swapped), the N-2 other seats are trusted registry scripts filled clockwise from seat k in a fixed order. '
          'Every lobby has a mirror with A and B exchanged; each entrant plays all N seats per seed group.')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def verify_release(release):
    """The mixed-v2 release, checked against the current source by its own verifier and the harness binding."""
    from harness.policy import verify_release as harness_verify
    from simulators.tavern.autoresearch.evaluation_v2.release import verify
    verify(release)
    return harness_verify('tavern', release)['sha256']


def session_bundle(session):
    """The read-only submission copy of the session's fresh final training, checked against its record."""
    record = json.loads((Path(session) / 'final-train.json').read_text())
    submission = record.get('submission') or {}
    if record.get('status') != 'complete' or not submission.get('submission'):
        raise ValueError(f'Final training of {session} submitted no network')
    folder = Path(session) / 'final/submissions' / submission['submission']
    manifest = json.loads((folder / 'submission.json').read_text())
    if manifest['files'] != submission['files'] or any(sha(folder / f) != h for f, h in manifest['files'].items()):
        raise ValueError(f'Final submission of {session} differs from its record')
    return folder


def admit(folder, release):
    """Host-side static admission: the evaluator's own checks that need no native module or entrant code."""
    import numpy as np
    from simulators.tavern.autoresearch.contract import EDITABLE, validate
    folder = Path(folder)
    for name in FILES:
        if (folder / name).is_symlink() or not (folder / name).is_file():
            raise ValueError(f'{folder} lacks {name}')
    ident = json.loads((folder / 'identity.json').read_text())
    summary = json.loads((folder / 'summary.json').read_text())
    if ident['binding'] != json.loads((Path(release) / 'binding.json').read_text()):
        raise ValueError(f'{folder} is bound to a different task/native source than the release')
    validate(ident['config'])
    if any(sha(folder / f) != ident['source'][f] for f in EDITABLE):
        raise ValueError(f'{folder}: editable source differs from its identity')
    if sha(folder / 'identity.json') != summary['identity_sha256'] or sha(folder / 'weights.npz') != summary['weights_sha256']:
        raise ValueError(f'{folder}: weights/identity digest mismatch')
    if (folder / 'weights.npz').stat().st_size > 32000000:
        raise ValueError(f'{folder}: weights exceed 32 MB')
    with np.load(folder / 'weights.npz', allow_pickle=False) as z:
        if sum(v.size for v in z.values()) > 2000000 or any(not np.isfinite(v).all() for v in z.values()):
            raise ValueError(f'{folder}: nonfinite or oversized network')
    return {name: sha(folder / name) for name in FILES}


def entrant(kind, path, release):
    folder = session_bundle(path) if kind == 'session' else Path(path)
    files = admit(folder, release)
    return dict(session=str(path) if kind == 'session' else None, bundle=str(folder), bundle_sha256=digest(files),
                files=files)


def script_styles(release):
    registry = json.loads((Path(release) / 'private/registry/registry.json').read_text())
    items = {r['policy_id']: r for r in registry['opponents']}
    rules = registry['binding']['rules']
    lineup = registry['profiles'][SCRIPT_PROFILE][:int(rules['num_players']) - 2]
    if any(items[i]['kind'] != 'script' for i in lineup):
        raise ValueError('The script profile holds non-script references')
    return [items[i]['style'] for i in lineup], lineup, rules


def private_schedule(out, count, roots):
    """Fresh 64-bit secrets screened like the private final bank (harness/games/tavern/final_bank.py)."""
    from simulators.tavern.autoresearch.evaluation_v2.seed_exclusions import collect
    from simulators.tavern.autoresearch.evaluation_v2.seed_keys import lobby_key, training_overlap
    excluded, configs, evidence = collect(roots=roots)
    seeds = []
    for _ in range(100000):
        seed = secrets.randbits(64)
        if seed not in seeds and lobby_key(seed) not in excluded and not any(training_overlap(seed, c) for c in configs):
            seeds.append(seed)
        if len(seeds) == count:
            break
    if len(seeds) != count:
        raise ValueError('Unable to draw a disjoint tournament bank')
    private = out / 'private'
    private.mkdir(mode=0o700)
    common.write_json(private / 'schedule.json', dict(seeds=seeds, exclusions=evidence))
    return seeds


def score(row, a, b):
    """Normalized row (common.pair_summary) from one worker lobby row."""
    owner = row['fault_by']
    if row['valid']:
        points, fault = float(row['a_placement'] < row['b_placement']), None
    elif owner in ('a', 'b'):
        points, fault = (0.0 if owner == 'a' else 1.0), (row['failure'] or {}).get('kind', 'fault')
    else:
        points, fault = None, (row['failure'] or {}).get('kind', 'fault')
    return dict(unit=row['unit'], seat=row['a_seat'], a_points=points, natural=bool(row['valid']),
                truncated=fault == 'administrative_cap', fault=fault,
                fault_by={'a': a, 'b': b}.get(owner, owner) if fault else None,
                k=row['k'], swap=row['swap'], b_seat=row['b_seat'], a_placement=row['a_placement'],
                b_placement=row['b_placement'], process_seconds=row['process_seconds'])


def placement_points(rows, players=8):
    valid = [r for r in rows if r['natural']]
    if not valid:
        return None, None
    a = sum(100 * (players - r['a_placement']) / (players - 1) for r in valid) / len(valid)
    b = sum(100 * (players - r['b_placement']) / (players - 1) for r in valid) / len(valid)
    return a, b


class Runner:
    """One CPU sandbox of the release's frozen image per worker chunk, always terminated."""

    def __init__(self, provider, lifetime):
        self.provider, self.lifetime = provider, int(lifetime)

    def __call__(self, request, a_bundle, b_bundle, folder, private, clock, label):
        import shutil
        import tempfile
        provider = self.provider
        sb = None
        try:
            sb = provider.create(False, self.lifetime)
            common.write_json(folder / 'sandbox.json', dict(app_id=provider.app_id, sandbox_id=provider.identity(sb),
                                                            lifetime=self.lifetime, cpu=4, memory_mib=16384))
            request = dict(request, absolute_stop=time.time() + self.lifetime - 60,
                           bundles=dict(a='/private/entrants/a', b='/private/entrants/b'))
            with tempfile.TemporaryDirectory() as temp:  # exactly the admitted files, never resume.pt
                for tag, source in (('a', a_bundle), ('b', b_bundle)):
                    staged = Path(temp) / tag
                    staged.mkdir()
                    for name in FILES:
                        shutil.copyfile(Path(source) / name, staged / name)
                    provider.upload(sb, staged, '/private/entrants/' + tag)
            common.write_json(private, request)  # holds the secret seeds: kept under OUT/private
            provider.copy(sb, private, '/private/tournament.json')
            provider.copy(sb, HERE / 'tavern_worker.py', '/private/tournament_worker.py')
            text = provider.command(sb, ['python', '/private/tournament_worker.py', '/private/tournament.json'],
                                    timeout=self.lifetime - 30)
            (folder / 'worker-stdout.log').write_text(text[-200000:])
            return json.loads(provider.read(sb, '/private/tournament-result.json'))
        finally:
            if sb is not None:
                try:
                    (folder / 'entrant-stderr.log').write_bytes(provider.read(sb, '/private/entrant-stderr.log')[-200000:])
                except Exception:
                    pass
                receipt = None
                try:
                    provider.terminate(sb)
                    for _ in range(100):
                        receipt = provider.allocation(provider.app_id, provider.identity(sb))
                        if receipt and receipt.get('finished_at', 0) > 0:
                            break
                        time.sleep(.2)
                except Exception as exc:
                    common.write_json(folder / 'termination-error.json', dict(error=repr(exc)))
                common.write_json(folder / 'provider.json', receipt)
                clock.add(receipt if receipt and receipt.get('finished_at') else None, label)


def open_provider(release, stack, out):
    import modal
    from harness.policy import ENVIRONMENT, require_modal
    from harness.games.tavern.provider import ModalProvider
    require_modal()
    app = modal.App('game-autoresearch-tournament-tavern')
    stack.enter_context(app.run(environment_name=ENVIRONMENT))
    common.write_json(out / 'app.json', dict(app_id=app.app_id, environment=ENVIRONMENT))
    provider = ModalProvider(release, app)
    provider.ready()  # the frozen-source image, built (or reused) before any worker exists
    return provider


def main(argv=None, provider=None):
    p = common.parser('tavern', 'seed groups (N lobbies each in an N-player lobby)')
    p.add_argument('--chunk', type=int, default=None, help='seed groups per CPU worker (default: 96 lobbies)')
    p.add_argument('--cpu-timeout', type=int, default=1500, help='lifetime of each CPU worker, seconds')
    args = p.parse_args(argv)
    started = time.time()
    specs = common.entrant_specs(args)
    release = common.common_release(specs, args.release)
    release_sha256 = verify_release(release)
    entrants = {name: entrant(kind, path, release) for name, kind, path in specs}
    out = common.new_out(args.out, [release, *(path for _, _, path in specs),
                                    *(Path(e['bundle']) for e in entrants.values())])
    styles, lineup, rules = script_styles(release)
    players = int(rules['num_players'])
    args.chunk = args.chunk or CHUNK_LOBBIES // players
    count = args.slice or (SMOKE_UNITS if args.smoke else LOBBIES // players)
    common.write_json(out / 'request.json', dict(argv=sys.argv if argv is None else argv, seed_groups=count,
                                                 chunk=args.chunk, cpu_timeout=args.cpu_timeout))
    roots = [release / 'public', release / 'private/registry',
             *{(path if kind == 'session' else Path(entrants[name]['bundle'])) for name, kind, path in specs}]
    seeds = private_schedule(out, count, roots)
    clock = common.Clock()
    pairs = common.pairs(list(entrants), args.anchor)  # anchors-only when --anchor is given
    plan = [(pair, part) for pair in pairs for part in common.chunks(count, args.chunk)]

    def task(runner, pair, part):
        a, b = pair
        label = f'{a}-vs-{b}'
        folder = out / 'pairs' / label / 'chunks' / f'{part[0]:04d}'
        folder.mkdir(parents=True)
        request = dict(lobbies=tavern_worker.lobbies([(i, seeds[i]) for i in part], styles), rules=rules,
                       lobby_seconds=LOBBY_SECONDS)
        raw = runner(request, entrants[a]['bundle'], entrants[b]['bundle'], folder,
                     out / 'private/requests' / f'{label}-{part[0]:04d}.json', clock, label)
        common.write_json(folder / 'worker.json', raw)  # raw lobby rows (no seeds) and the privacy probes
        expected = {(i, k, s) for i in part for k in range(players // 2) for s in (0, 1)}
        keys = [(r['unit'], r['k'], r['swap']) for r in raw['rows']]
        if len(keys) != len(set(keys)) or not set(keys) <= expected:
            raise ValueError('worker rows differ from the schedule')
        if any(c.get('private_paths_denied') != 3 or not c.get('credential_env_absent') for c in raw['privacy'].values()):
            raise ValueError('entrant privacy probe failed')
        return [score(r, a, b) for r in raw['rows']], raw.get('pending', [])

    with contextlib.ExitStack() as stack:
        if provider is None:
            provider = open_provider(release, stack, out)
        runner = Runner(provider, args.cpu_timeout)
        outcomes = common.run_tasks([lambda pair=pair, part=part: task(runner, pair, part) for pair, part in plan],
                                    args.parallel)
    results = []
    for pair in pairs:
        a, b = pair
        rows, errors = [], []
        for (owner, part), (value, error) in zip(plan, outcomes):
            if owner != pair:
                continue
            if error:
                errors.append(dict(seeds=[part[0], part[-1]], error=error))
                continue
            rows.extend(value[0])
            if value[1]:
                errors.append(dict(seeds=[part[0], part[-1]], error=f'{len(value[1])} lobbies not reached before '
                                                                     'the worker deadline'))
        rows.sort(key=lambda r: (r['unit'], r['k'], r['swap']))
        faults_by = {}
        for row in rows:
            if row['fault_by']:
                faults_by[row['fault_by']] = faults_by.get(row['fault_by'], 0) + 1
        common.write_json(out / 'pairs' / f'{a}-vs-{b}' / 'rows.json', rows)
        summary = common.pair_summary(a, b, rows, players * count, f'pairs/{a}-vs-{b}/', errors, faults_by)
        summary['a_placement_points_0_100'], summary['b_placement_points_0_100'] = placement_points(rows, players)
        results.append(summary)
    value = common.result('tavern', release, release_sha256, bool(args.smoke), entrants, results, clock, started,
                          POINTS_RULE, harness_release_sha256=release_sha256, seed_groups=count, players=players,
                          scheme=SCHEME,
                          script_seats=dict(profile=SCRIPT_PROFILE, policy_ids=lineup, styles=styles),
                          placement_leaderboard=placement_board(list(entrants), results),
                          evaluator='frozen evaluation_readiness.lobby.run_lobby in tools/tournament/tavern_worker.py '
                                    'on the release image (CPU 4 cores, 16 GiB); both entrants IsolatedVisible-style '
                                    'processes under uids 65534/65533')
    common.write_json(out / 'result.json', value)
    print(json.dumps(dict(out=str(out), leaderboard=value['leaderboard'], placement=value['placement_leaderboard'],
                          worker_seconds=value['worker_seconds'],
                          errors=sum(len(r['errors']) for r in results)), indent=2))
    return value


def placement_board(names, results):
    """Secondary: each entrant's mean placement points over its natural lobbies, weighted by lobbies."""
    totals = {n: [0.0, 0] for n in names}
    for pair in results:
        valid = pair['natural_games']
        if pair['a_placement_points_0_100'] is None:
            continue
        totals[pair['a']][0] += pair['a_placement_points_0_100'] * valid
        totals[pair['b']][0] += pair['b_placement_points_0_100'] * valid
        totals[pair['a']][1] += valid
        totals[pair['b']][1] += valid
    return sorted([dict(name=n, placement_points_0_100=s / g if g else None, lobbies=g) for n, (s, g) in totals.items()],
                  key=lambda r: (r['placement_points_0_100'] is None, -(r['placement_points_0_100'] or 0), r['name']))


if __name__ == '__main__':
    main()
