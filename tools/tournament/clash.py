"""Clash head-to-head tournament on the release's own CPU grader (`FastProvider.cpu_grade`).

Each unordered pair (A, B) is played through the same 8-core / 32 GiB CPU sandbox and evaluator that grade a
session (`simulators.clash.autoresearch.evaluation_v3.evaluate`): A's sealed bundle is the candidate and the
schedule's only opponent is `{name: B, kind: 'checkpoint', identity}`. Inside the sandbox both bundles pass the
evaluator's checkpoint admission (`readiness_contract.admit`: bundle verification and greedy-policy source
review), the runtime determinism probes, and then play as separate unprivileged `PolicyProcess` children with
different uids; neither is ever imported by the evaluator. Every engine seed is played twice, with A in seat 0
and in seat 1, so B-vs-A would only repeat the same games.

The release grades its six-opponent panel with three pair workers (`parallel3`, one simulation thread each);
a tournament call holds a single pair, so the provider runs the release's sequential evaluator module
(`evaluation_v3`, four simulation threads) instead. The release interface declaration is passed unchanged.

Engine seeds are a fresh host-secret schedule written to `OUT/private/schedule.json` (32 seeds as the release's
private bank, smoke 2), shared by every pair; the release's own `private/grade.json` is never read for seeds.
Run from the repository root:

    PYTHONPATH=$PWD MODAL_PROFILE=duper .venv-clash-ar/bin/python \
        -m tools.tournament.clash --entrant A=SESSION --entrant B=SESSION --out NEW_DIR [--smoke] [--seeds N]

Do not set MODAL_DISABLE_API_PROXY=1 on the operator's Mac: the tournament's ephemeral app then misses heartbeats,
Modal stops it and its CPU workers are killed mid-schedule (seen 2026-11-14).
"""
import sys

sys.dont_write_bytecode = True

import contextlib
import json
import os
import secrets
import time
from pathlib import Path

from tools.tournament import common

UNITS = {True: 2, False: 32}   # engine seeds; the release's private bank has 32 units
CHUNK = 8                      # seeds per CPU worker: 16 games
RESERVED = 'candidate'         # the evaluator's own name for the graded bundle
POINTS_RULE = ('per game from A\'s seat: natural win 1, tie 0.5, loss 0 (the evaluator\'s own points). The '
               'evaluator scores only complete natural schedules: an entrant admission failure, policy fault, '
               'administrative cap or simulator fault voids the whole worker chunk, which is recorded under '
               '`errors` and left unplayed, never scored')


def modules():
    # The active Clash release constructor (short-v1); the fast-v4 one is at git tag full-size-20261113.
    from simulators.clash.autoresearch import bundle, common as clash_common, short_release, readiness_contract
    return bundle, clash_common, short_release, readiness_contract


def verify_release(release):
    """(harness release sha256, release binding): the harness record and the release checked against the current
    source (engine and match form, fast runtime, panel). Both verifiers only read."""
    from harness.policy import verify_release as harness_verify
    _, _, release_module, _ = modules()
    return harness_verify('clash', release)['sha256'], release_module.verify(release)


def session_bundle(session):
    """The sealed bundle of the session's fresh final training and the identity it submitted."""
    record = json.loads((Path(session) / 'final-train.json').read_text())
    identity = (record.get('submission') or {}).get('identity')
    if record.get('status') != 'complete' or not identity:
        raise ValueError(f'Final training of {session} submitted no network')
    return Path(session) / 'final/jobs/final/bundle', identity


def entrant(kind, path, binding):
    bundle, clash_common, _, readiness = modules()
    folder, expected = session_bundle(path) if kind == 'session' else (Path(path), None)
    manifest = bundle.verify(folder, expected)
    if manifest['engine'] != binding['engine']:
        raise ValueError(f'{folder} is sealed against a different engine than the release')
    if 'policy_runtime' in manifest and manifest['policy_runtime'] != binding['trusted_fast_runtime']:
        raise ValueError(f'{folder} is bound to a different fast policy runtime than the release')
    review = readiness.review_policy(folder / 'policy.py')  # the evaluator refuses it later anyway; fail before paying
    return dict(session=str(path) if kind == 'session' else None, bundle=str(folder),
                bundle_sha256=manifest['identity'], parameters=manifest['parameters'],
                policy_source_sha256=review['source_sha256'])


def excluded_seeds(release):
    from simulators.clash.autoresearch import release_v3
    seeds = set(release_v3.PUBLIC_SEEDS + release_v3.VALIDATION_SEEDS + release_v3.PUBLIC_DEMO_SEEDS +
                release_v3.PUBLIC_CAPACITY_SEEDS)
    seeds.update(json.loads((Path(release) / 'public/practice.json').read_text())['engine_seeds'])
    return seeds


def private_schedule(out, count, excluded):
    """Fresh host secrets, drawn like the release's reconstructed private registry."""
    seeds = []
    while len(seeds) < count:
        seed = secrets.randbelow(2 ** 31 - 1) + 1
        if seed not in excluded and seed not in seeds:
            seeds.append(seed)
    value = dict(engine_seeds=seeds, policy_seed=secrets.randbelow(2 ** 31 - 1) + 1)
    private = out / 'private'
    private.mkdir(mode=0o700)
    common.write_json(private / 'schedule.json', value)
    return value


def chunk_schedule(schedule, part, b_name, b):
    """The evaluator schedule of one worker: A's opponent is B alone, on this chunk's seeds."""
    return dict(engine_seeds=[schedule['engine_seeds'][i] for i in part], policy_seed=schedule['policy_seed'],
                opponents=[dict(name=b_name, kind='checkpoint', identity=b['bundle_sha256'], bundle=b['bundle'])])


def canonical(value):
    """The schedule as the sandbox sees it (cpu_grade rewrites bundle paths under /private)."""
    value = json.loads(json.dumps(value))
    for actor in value['opponents']:
        actor['bundle'] = '/private/' + actor['name']
    return value


def normalize(grade, part, a, b, a_name, b_name, schedule):
    """Checked rows of one worker: identities, schedule commitment and the full seat x seed grid."""
    _, clash_common, _, _ = modules()
    if grade.get('submission_identity') != a['bundle_sha256']:
        raise ValueError('worker graded a different candidate identity')
    if grade.get('schedule_commitment') != clash_common.digest(canonical(schedule)):
        raise ValueError('worker played a different schedule')
    rows = grade['rows']
    keys = [(r['seed_index'], r['seat']) for r in rows]
    if sorted(keys) != [(i, s) for i in range(len(part)) for s in (0, 1)]:
        raise ValueError(f'worker rows differ from the schedule ({len(rows)} rows, {2 * len(part)} scheduled)')
    out = []
    for r in rows:
        if r['opponent'] != b_name or r['points'] not in (0, .5, 1) or not r['natural'] or r['fault']:
            raise ValueError('invalid worker row: ' + json.dumps({k: r.get(k) for k in ('opponent', 'points', 'natural')}))
        out.append(dict(unit=part[r['seed_index']], seat=r['seat'], a_points=float(r['points']), natural=True,
                        truncated=bool(r['truncated']), fault=None, winner=r['winner'], decisions=r['decisions']))
    return out


def open_provider(release, args, stack, out):
    import modal
    from harness.policy import ENVIRONMENT, require_modal
    from harness.games.clash.provider import FastProvider, snapshot
    require_modal()
    if args.base_image_id:
        os.environ['CLASH_AR_BASE_IMAGE_ID'] = args.base_image_id
    archive = snapshot(out / 'uploads')  # the worker source, content-addressed, inside OUT
    app = modal.App('game-autoresearch-tournament-clash')
    stack.enter_context(app.run(environment_name=ENVIRONMENT))
    common.write_json(out / 'app.json', dict(app_id=app.app_id, environment=ENVIRONMENT, source_archive=archive.name))
    return FastProvider(app, archive, out / 'provider')


def main(argv=None, provider=None):
    p = common.parser('clash', 'engine seeds')
    p.add_argument('--chunk', type=int, default=CHUNK, help='engine seeds per CPU worker')
    p.add_argument('--cpu-timeout', type=int, default=1800, help='seconds per CPU worker')
    p.add_argument('--base-image-id', help='runtime base image override (CLASH_AR_BASE_IMAGE_ID)')
    p.add_argument('--seeds', type=int, help=f'engine seeds per pair (default {UNITS[False]}, smoke {UNITS[True]}); '
                                             'a fresh host-secret schedule either way')
    args = p.parse_args(argv)
    started = time.time()
    specs = common.entrant_specs(args)
    if any(name == RESERVED for name, _, _ in specs):
        raise ValueError(f'The entrant name {RESERVED!r} is reserved by the evaluator')
    release = common.common_release(specs, args.release)
    release_sha256, binding = verify_release(release)
    entrants = {name: entrant(kind, path, binding) for name, kind, path in specs}
    out = common.new_out(args.out, [release, *(path for _, _, path in specs),
                                    *(Path(e['bundle']) for e in entrants.values())])
    count = args.slice or args.seeds or UNITS[args.smoke]
    common.write_json(out / 'request.json', dict(argv=sys.argv if argv is None else argv, engine_seeds=count,
                                                 chunk=args.chunk, cpu_timeout=args.cpu_timeout))
    schedule = private_schedule(out, count, excluded_seeds(release))
    interface = json.loads((release / 'public/interface.json').read_text())
    clock = common.Clock()
    pairs = common.pairs(list(entrants), args.anchor)  # anchors-only when --anchor is given
    plan = [(pair, part) for pair in pairs for part in common.chunks(count, args.chunk)]

    def task(pair, part):
        a, b = pair
        label = f'{a}-vs-{b}'
        value = chunk_schedule(schedule, part, b, entrants[b])
        path = out / 'private/chunks' / label / f'{part[0]:04d}.json'
        common.write_json(path, value)
        result = out / 'pairs' / label / 'chunks' / f'{part[0]:04d}.json'
        result.parent.mkdir(parents=True, exist_ok=True)
        try:
            grade = provider.cpu_grade(entrants[a]['bundle'], path, {b: entrants[b]['bundle']}, result,
                                       seconds=args.cpu_timeout, split='tournament-private')
        finally:
            receipt = result.with_suffix('.provider.json')
            clock.add(json.loads(receipt.read_text()).get('provider') if receipt.is_file() else None, label)
        return normalize(grade, part, entrants[a], entrants[b], a, b, value)

    with contextlib.ExitStack() as stack:
        if provider is None:
            provider = open_provider(release, args, stack, out)
        provider.interface = interface   # the release's own declaration, as the session controller sets it
        provider.pair_workers = 1        # one pair per call: the sequential evaluator module
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
                errors.append(dict(seeds=[part[0], part[-1]], error=error))
            else:
                rows.extend(value)
        rows.sort(key=lambda r: (r['unit'], r['seat']))
        common.write_json(out / 'pairs' / f'{a}-vs-{b}' / 'rows.json', rows)
        results.append(common.pair_summary(a, b, rows, 2 * count, f'pairs/{a}-vs-{b}/', errors))
    value = common.result('clash', release, release_sha256, bool(args.smoke), entrants, results, clock, started,
                          POINTS_RULE, release_commitment=modules()[1].digest(binding), engine_seeds=count,
                          seats='A in seat 0 and seat 1 on every engine seed',
                          evaluator='simulators.clash.autoresearch.evaluation_v3.evaluate via FastProvider.cpu_grade '
                                    '(8 CPU cores, 32 GiB, no GPU, no network); A = candidate, B = sole checkpoint '
                                    'opponent, both isolated PolicyProcess children',
                          interface_sha256=modules()[1].digest(interface))
    common.write_json(out / 'result.json', value)
    print(json.dumps(dict(out=str(out), leaderboard=value['leaderboard'], worker_seconds=value['worker_seconds'],
                          errors=sum(len(r['errors']) for r in results)), indent=2))
    return value


if __name__ == '__main__':
    main()
