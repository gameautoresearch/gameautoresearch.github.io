"""Polytopia head-to-head tournament on the release's frozen CPU grader.

Each unordered pair (A, B) is played through `Provider.evaluate(A, maps, 0, out, opponent_bundle=B)`, the same
4-core CPU worker and `/opt/v1/grade.py` that grade a session: A's network runs as the candidate (uid 65534)
and B's as the opponent (uid 65533), each in its own unprivileged `/opt/v1/policy.py` process, and every map
is played twice with A in seat 1 and then seat 2. Both policies are deterministic, so B-vs-A would replay
exactly the same games; one direction covers both seats.

`--reference NAME` adds a panel opponent from the release's `host-private-panel/` (checked against its inventory
checksums). A reference is always seated as B, the way session grading seats it, and references do not play each
other. `--equivalent-release DIR` accepts bundles sealed against another release whose manifest differs from the
tournament release only in files that do not reach play (`benchmark.json`: the research worker shape; `public/`:
researcher-facing text); each such bundle is still verified against its own seal and its own release identity.

Maps are a fresh host-secret schedule written to `OUT/private/schedule.json` (128 maps, smoke 2), shared by
every pair. Run from the repository root:

    PYTHONPATH=$PWD MODAL_PROFILE=duper MODAL_DISABLE_API_PROXY=1 .venv-polytopia-ar/bin/python \
        -m tools.tournament.polytopia --entrant A=SESSION --entrant B=SESSION --out NEW_DIR [--smoke]
"""
import sys

sys.dont_write_bytecode = True

import contextlib
import json
import secrets
import time
from pathlib import Path

from tools.tournament import common

UNITS = {True: 2, False: 128}   # maps, as in session grading
CHUNK = 2                       # maps per CPU worker: 4 games
POINTS_RULE = ('per game: (reward + 1) / 2 from A\'s side, i.e. win 1, draw 0.5, loss 0; a game reaching the '
               'round cap is adjudicated by the engine score difference (the grader\'s own rule) and also '
               'counted in truncations')


def core():
    from simulators.polytopia.autoresearch.v1 import core as module
    return module


def session_bundle(session):
    """The sealed bundle a session enters: a network session's last submission, or a recipe session's fresh final."""
    run = Path(str(Path(session).resolve()) + '.harness-run.json')
    if run.is_file() and json.loads(run.read_text())['settings'].get('variant') == 'network':
        sha256 = common.network_submission(session)['native']['sha256']
        bundle = Path(session) / 'research' / 'submissions' / sha256
        if core().verify(bundle) != sha256:
            raise ValueError(f'Submitted bundle of {session} differs from its submission record')
        return bundle
    final = Path(session) / 'final'
    if (final / 'ledger.sqlite3').is_file():
        from simulators.polytopia.autoresearch.ledger import Ledger
        ledger = Ledger(final)
        entries = ledger.submissions()
        submission = entries and dict(sha256=entries[-1]['artifact_sha'],
                                      bundle=final / 'submissions' / entries[-1]['artifact_sha'])
        finalized = ledger.lifecycle() == 'finalized'
    else:  # a session of the earlier contract, budgeted in seconds
        state = json.loads((final / 'state.json').read_text())
        submission, finalized = state.get('submission'), state.get('finalized')
    if not finalized or not submission:
        raise ValueError(f'Final training of {session} has no finalized submission')
    bundle = Path(submission['bundle'])
    if core().verify(bundle) != submission['sha256']:
        raise ValueError(f'Final bundle of {session} differs from its submission record')
    return bundle


# Release files that never reach the CPU grader: the research worker shape, researcher-facing text, and the starter's
# trainer files and choice (an empty-starter and a PPO-starter release of the same code differ only there). Each
# bundle plays with its own sealed network.py; the release's rl/network.py is only the default for an opponent that
# has none, and a tournament entrant always has one.
PLAY_NEUTRAL = ('benchmark.json', 'public/', 'starter.json', 'runtime/experiments/gpu_resident_train.py',
                'runtime/src/polytopia_bot/rl/gpu_resident.py', 'runtime/src/polytopia_bot/rl/network.py',
                'runtime/src/polytopia_bot/rl/ppo.py')


def equivalent_releases(release, others, neutral=()):
    """{sha256: path} of the tournament release and every release whose play-path files are identical to it.
    `neutral` names further files the caller has checked do not reach play (recorded in request.json)."""
    def files(path):
        core().verify(path)
        return json.loads((Path(path) / 'manifest.json').read_text())['files']
    base = files(release)
    accepted = {core().verify(release): str(release)}
    for other in others:
        theirs = files(other)
        differ = sorted(k for k in set(base) | set(theirs) if base.get(k) != theirs.get(k))
        played = [k for k in differ if not k.startswith(PLAY_NEUTRAL) and k not in neutral]
        if played:
            raise ValueError(f'{other} differs from {release} in files that reach play: {played[:5]}')
        accepted[core().verify(other)] = str(Path(other).resolve())
    return accepted


def entrant(kind, path, accepted):
    bundle = session_bundle(path) if kind == 'session' else Path(path)
    identity = core().verify(bundle)
    binding = json.loads((bundle / 'binding.json').read_text())
    if binding.get('benchmark_id') != core().BENCHMARK or binding.get('release_sha256') not in accepted:
        raise ValueError(f'{bundle} is sealed against a different release')
    if not (bundle / 'final.npz').is_file():
        raise ValueError(f'{bundle} has no trained network')  # the policy worker would play a random init
    return dict(session=str(path) if kind == 'session' else None, bundle=str(bundle), bundle_sha256=identity,
                release_sha256=binding['release_sha256'], release=accepted[binding['release_sha256']])


def reference(release, name):
    """A panel opponent of the release, verified against the panel inventory; it only ever plays as B."""
    import hashlib
    inventory = json.loads((Path(release) / 'host-private-panel' / 'inventory.json').read_text())['panel']
    found = [p for p in inventory if p['name'] == name]
    if not found:
        raise ValueError(f'No panel opponent {name!r} in {release}; have {[p["name"] for p in inventory]}')
    item, folder = found[0], Path(release) / found[0]['directory']
    for f, key in (('final.npz', 'checkpoint_sha256'), ('network.py', 'network_sha256')):
        if hashlib.sha256((folder / f).read_bytes()).hexdigest() != item[key]:
            raise ValueError(f'{folder / f} differs from the panel inventory')
    return dict(session=None, bundle=str(folder), bundle_sha256=item['checkpoint_sha256'], reference=True,
                provenance=item.get('provenance'))


def private_schedule(out, count):
    """Fresh host secrets, drawn exactly like a session's private grading schedule."""
    seeds = set()
    while len(seeds) < count:
        seeds.add(int.from_bytes(secrets.token_bytes(4), 'big') % 1_000_000_000 + 1_000_000)
    private = out / 'private'
    private.mkdir(mode=0o700)
    value = dict(seeds=sorted(seeds), opponent_seed=0)
    common.write_json(private / 'schedule.json', value)
    return value


def normalize(raw, index):
    return [dict(unit=index[r['seed']], seat=r['seat'], a_points=(float(r['reward']) + 1) / 2,
                 natural=not r['capped'], truncated=bool(r['capped']), fault=None, turn=r.get('turn'),
                 steps=r.get('steps'), scores=r.get('scores')) for r in raw]


def open_provider(release, args, stack, out):
    import modal
    from harness.policy import ENVIRONMENT, require_modal
    from harness.games.polytopia.provider import Provider
    require_modal()
    app = stack.enter_context(modal.App('game-autoresearch-tournament-polytopia').run(environment_name=ENVIRONMENT))
    common.write_json(out / 'app.json', dict(app_id=app.app_id, environment=ENVIRONMENT))
    options = {k: getattr(args, k) for k in ('base_image_id', 'reuse_image_id') if getattr(args, k)}
    return Provider(release, app, cpu_timeout=args.cpu_timeout, **options)


def main(argv=None, provider=None):
    # Direct bot-writing sessions use the local 14x14 Wasm referee, not the
    # network release's 11x11 cloud grader. No researcher is launched here.
    direct_args = sys.argv[1:] if argv is None else argv
    if '--botwriter-workspace' in direct_args:
        from simulators.polytopia.botwriter.evaluate import main as evaluate_bots
        return evaluate_bots(direct_args)
    p = common.parser('polytopia', 'maps')
    p.add_argument('--chunk', type=int, default=CHUNK, help='maps per CPU worker')
    p.add_argument('--cpu-timeout', type=int, default=900, help='seconds per CPU worker')
    p.add_argument('--base-image-id')
    p.add_argument('--reuse-image-id')
    p.add_argument('--reference', action='append', default=[], metavar='PANEL_NAME',
                   help='add a panel opponent of the release (always seated as B; references do not play each other)')
    p.add_argument('--equivalent-release', action='append', default=[], metavar='DIR',
                   help='also accept bundles sealed against this release, if it differs only in files outside play')
    p.add_argument('--play-neutral', action='append', default=[], metavar='RELEASE_FILE',
                   help='a release file (manifest path) the caller has checked does not reach play, for '
                        '--equivalent-release; recorded in request.json')
    args = p.parse_args(argv)
    started = time.time()
    specs = common.entrant_specs(args)
    release = common.common_release(specs, args.release)
    release_sha256 = core().verify(release)
    accepted = equivalent_releases(release, args.equivalent_release, tuple(args.play_neutral))
    entrants = {name: entrant(kind, path, accepted) for name, kind, path in specs}
    for name in args.reference:
        if not common.NAME.fullmatch(name) or name in entrants:
            raise ValueError(f'Reference names must be unique and match {common.NAME.pattern}: {name!r}')
        entrants[name] = reference(release, name)
    out = common.new_out(args.out, [release, *map(Path, args.equivalent_release), *(path for _, _, path in specs),
                                    *(Path(e['bundle']) for e in entrants.values())])
    count = args.slice or UNITS[args.smoke]
    common.write_json(out / 'request.json', dict(argv=sys.argv if argv is None else argv, maps=count,
                                                 chunk=args.chunk, cpu_timeout=args.cpu_timeout,
                                                 references=args.reference, accepted_releases=accepted,
                                                 play_neutral=args.play_neutral))
    schedule = private_schedule(out, count)
    index = {seed: i for i, seed in enumerate(schedule['seeds'])}
    clock = common.Clock()
    pairs = [(a, b) for a, b in common.pairs(list(entrants), args.anchor)  # anchors-only when --anchor is given
             if not entrants[a].get('reference')]  # references are listed last, so they are always B
    plan = [(pair, part) for pair in pairs for part in common.chunks(count, args.chunk)]

    def task(pair, part):
        a, b = pair
        path = out / 'pairs' / f'{a}-vs-{b}' / 'chunks' / f'{part[0]:04d}.json'
        try:
            provider.evaluate(entrants[a]['bundle'], [schedule['seeds'][i] for i in part],
                              schedule['opponent_seed'], path, entrants[b]['bundle'])
        finally:
            worker = path.with_suffix('.worker.json')
            if worker.is_file():
                record = json.loads(worker.read_text())
                try:
                    clock.add(provider.lookup(record['app_id'], record['sandbox_id']), path.parent.parent.name)
                except Exception:
                    clock.add(None, path.parent.parent.name)
        return normalize(json.loads(path.read_text())['details'], index)

    with contextlib.ExitStack() as stack:
        if provider is None:
            provider = open_provider(release, args, stack, out)
        outcomes = common.run_tasks([lambda pair=pair, part=part: task(pair, part) for pair, part in plan],
                                    args.parallel)
    results = []
    for pair in pairs:
        a, b = pair
        rows, errors = [], []
        for (owner, part), (value, error) in zip(plan, outcomes):
            if owner != pair:
                continue
            if error:
                errors.append(dict(maps=[part[0], part[-1]], error=error))
            else:
                rows.extend(value)
        rows.sort(key=lambda r: (r['unit'], r['seat']))
        details = out / 'pairs' / f'{a}-vs-{b}'
        common.write_json(details / 'rows.json', rows)
        results.append(common.pair_summary(a, b, rows, 2 * count, f'pairs/{a}-vs-{b}/', errors))
    value = common.result('polytopia', release, release_sha256, bool(args.smoke), entrants, results, clock, started,
                          POINTS_RULE, harness_release_sha256=common.harness_release_sha256(release), maps=count,
                          accepted_releases=accepted, references=args.reference,
                          seats='A in seat 1 and seat 2 on every map', evaluator='/opt/v1/grade.py (frozen release)')
    common.write_json(out / 'result.json', value)
    print(json.dumps(dict(out=str(out), leaderboard=value['leaderboard'], worker_seconds=value['worker_seconds'],
                          errors=sum(len(r['errors']) for r in results)), indent=2))
    return value


if __name__ == '__main__':
    main()
