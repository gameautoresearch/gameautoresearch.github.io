"""Local Pokémon 3v3 ladder: sealed networks against Foul Play at several search times, on this Mac.

Reuses the release's frozen evaluator pieces exactly as `runtime/arena.py` `run()` plays a `search` panel member:
`EvaluationVector(1, team_size=3)`, the candidate through `runtime.policy_process.PolicyProcess` (bundle verified
against its sha256, runtime binding checked), Foul Play through `strong_bots.foul_process_v4.FoulPlayProcess`
fed `vector.visible(...)`. Differences from the graded arena (all host-local, recorded in the output):
  * macOS has no root/UID isolation, so both inference children run with `development=True` and no deny file;
  * Foul Play `time_ms` varies per rung (10 / 100 / 1000 ms, `samples=2`, `mode='production'`), its per-decision
    timeout is 3 * samples * time_ms / 1000 + 2 s, and the whole-game wall cap is raised per rung;
  * the pinned Foul Play source archive is unpacked once (sha256 checked) instead of `prepare_search` installing
    the pinned Linux wheels; the local `.venv-pokemon` poke-engine 0.0.48 is used.
One process-pool task per game (fresh simulator, network and Foul Play processes). Rows are appended to
`OUT/rows.jsonl` as games finish; `--report` rebuilds result.json / summary.md from rows.jsonl.
Holds `evidence/local-performance.lock` exclusively for the whole run.
"""
import sys

sys.dont_write_bytecode = True

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import argparse
import contextlib
import fcntl
import hashlib
import json
import multiprocessing
import os
import secrets
import shutil
import tarfile
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
RELEASE = ROOT / 'evidence/releases-20261114-main3/pokemon'
REP = ROOT / 'runs/local/poke3v3-rep-20261115'
DEFAULT_OUT = ROOT / 'runs/local/poke3v3-foulplay-ladder-20261115'
LOCK = ROOT / 'evidence/local-performance.lock'
RUNGS = {'foulplay-10ms': 10, 'foulplay-100ms': 100, 'foulplay-1s': 1000}
WALL_CAP = {'foulplay-10ms': 300, 'foulplay-100ms': 600, 'foulplay-1s': 900}
SAMPLES = 2


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    tmp = Path(str(path) + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    tmp.replace(path)


def foul_timeout(time_ms):
    return 3 * SAMPLES * time_ms / 1000 + 2.0


def entrants():
    combined = json.loads((REP / 'tournament-combined38/result.json').read_text())['entrants']
    listed = json.loads((REP / 'tournament-entrants.json').read_text())['entrants']
    out = {}
    for name, value in sorted(combined.items()):
        if name in listed and Path(listed[name]['bundle']) != Path(value['bundle']):
            raise ValueError(f'Bundle path disagreement for {name}')
        out[name] = dict(bundle=value['bundle'], bundle_sha256=value['bundle_sha256'])
    return out


def spec():
    request = json.loads((RELEASE / 'private/assessment/registry.json').read_text())['evaluator_request']
    pk = ROOT / 'simulators/pokemon/autoresearch'
    ext = request['search_extension_file_sha256']
    if set(ext) != {'foul_v4.py', 'foul_process_v4.py', 'arena_v4.py'}:
        raise ValueError('Complete forced-search extension binding required')
    for name, digest in ext.items():
        if sha(pk / 'strong_bots' / name) != digest:
            raise ValueError('Forced-search extension changed: ' + name)
    for name, digest in request['qualified_core_file_sha256'].items():
        if sha(pk / 'evaluation_v2' / name) != digest:
            raise ValueError('Qualified evaluator core changed: ' + name)
    archive = RELEASE / 'private/assessment' / request['source_archive']
    sets = RELEASE / 'private/assessment' / request['public_sets']
    if sha(archive) != request['source_archive_sha256']:
        raise ValueError('Foul Play archive changed')
    if sha(sets) != request['public_sets_sha256']:
        raise ValueError('Public random-set data changed')
    return request, archive, sets


def verify_bundle(path, digest):
    from simulators.pokemon.autoresearch.runtime import bundle as sealer, sdk
    meta = sealer.verify(path, digest)
    binding = json.loads((Path(path) / 'binding.json').read_text())
    if binding['runtime_sha256'] != sdk.runtime_signature():
        raise ValueError('Native candidate runtime differs')
    return meta


def play(task):
    """One game. Returns a row from the NETWORK's perspective; never raises."""
    import numpy as np  # noqa: F401  (EvaluationVector imports it anyway)
    from simulators.pokemon.autoresearch.runtime.policy_process import PolicyProcess
    from simulators.pokemon.autoresearch.evaluation_v2.env import EvaluationVector
    from simulators.pokemon.autoresearch.strong_bots.foul_process_v4 import FoulPlayProcess
    from integrations.pokemon.env import WorkerError
    rung, name, bundle, digest, index, seed, seat, source, sets, time_ms, cap = (
        task[k] for k in ('rung', 'network', 'bundle', 'bundle_sha256', 'seed_index', 'seed', 'seat', 'source',
                          'sets', 'time_ms', 'cap'))
    row = dict(rung=rung, network=name, seed_index=index, seat=seat, natural=False, truncated=False, points=None,
               fault=None, fault_category=None, network_decisions=0, foul_decisions=0,
               decision_counts=dict(forced_public_actions=0, revival_target_actions=0, actual_search_actions=0))
    actor = 'setup'
    before = time.monotonic()
    cpu0 = os.times()
    try:
        verify_bundle(bundle, digest)
        with contextlib.ExitStack() as game:
            process = game.enter_context(PolicyProcess(bundle, development=True))
            vector = game.enter_context(EvaluationVector(1, team_size=3))
            actor = 'reference'
            foul = game.enter_context(FoulPlayProcess(source, sets, mode='production', iterations=1000,
                                                      time_ms=time_ms, samples=SAMPLES,
                                                      timeout=foul_timeout(time_ms), development=True))
            offset = 0
            states = vector.reset([seed], 4096)
            while not (states[0]['terminated'] or states[0]['truncated']):
                if time.monotonic() - before > cap:
                    raise TimeoutError('__wall_cap__')
                state = states[0]
                actions = [None, None]
                if state['ready'][seat]:
                    actor = 'candidate'
                    actions[seat] = process.choose([state['observations'][seat]])[0]
                    row['network_decisions'] += 1
                if state['ready'][1 - seat]:
                    actor = 'reference'
                    visible = vector.visible([1 - seat], [offset])[0]
                    offset = visible['protocol_offset']
                    actions[1 - seat], metadata = foul.choose(visible)
                    row['foul_decisions'] += 1
                    if metadata.get('visits'):
                        row['search_samples'] = row.get('search_samples', 0) + len(metadata['visits'])
                        row['search_visits'] = row.get('search_visits', 0) + int(sum(metadata['visits']))
                        row['search_seconds'] = row.get('search_seconds', 0.) + float(metadata.get('seconds') or 0.)
                    row['decision_counts']['revival_target_actions' if metadata.get('revival_target_action') else
                                           'forced_public_actions' if metadata.get('forced_public_action') else
                                           'actual_search_actions'] += 1
                actor = 'engine'
                states = vector.step([actions])
            final = states[0]
            row.update(natural=bool(final['terminated']), truncated=bool(final['truncated']),
                       points=(final['returns'][seat] + 1) / 2 if final['terminated'] else None,
                       terminal_reason=final['terminal_reason'], engine_turns=final['engine_turns'],
                       adapter_phases=final['adapter_phases'], accepted_choices=final['accepted_choices'],
                       unavailable_choice_retries=final['unavailable_choice_retries'],
                       visible_choice_rejections=sum(final['rejections']))
    except Exception as exc:
        if isinstance(exc, TimeoutError) and str(exc) == '__wall_cap__':
            row.update(truncated=True, terminal_reason='process_wall_safety_cap')
        else:
            kind = (exc.category if isinstance(exc, WorkerError) else
                    actor + '_inference_timeout' if isinstance(exc, TimeoutError) else
                    actor + '_illegal_action' if 'illegal action' in str(exc).lower() or 'masked action' in str(exc).lower()
                    else actor + '_adapter_fault')
            row.update(fault=type(exc).__name__ + ': ' + str(exc)[:2000], fault_category=kind,
                       traceback=traceback.format_exc()[-4000:])
    cpu1 = os.times()
    row['wall_seconds'] = time.monotonic() - before
    # children_* covers the reaped simulator/network/Foul Play processes of this game (pool worker's own CPU too).
    row['cpu_seconds'] = (cpu1.user - cpu0.user + cpu1.system - cpu0.system
                          + cpu1.children_user - cpu0.children_user + cpu1.children_system - cpu0.children_system)
    row['finished_at'] = time.time()
    return row


def schedule(names, rung_plan, seeds):
    """rung_plan: [(rung, networks, n_seeds)] in play order; seed-major within a rung so a cut stays balanced."""
    tasks = []
    for rung, networks, count in rung_plan:
        for index in range(count):
            for name in networks:
                for seat in (0, 1):
                    tasks.append((rung, name, index, seat))
    return tasks


def run(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    lock = open(LOCK, 'a+')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        sys.exit(f'{LOCK} is held by another process')
    request, archive, sets = spec()
    entr = entrants()
    if len(entr) != 38:
        raise ValueError('Expected 38 entrants')
    names = sorted(entr)
    if args.networks:
        names = [n for n in args.networks.split(',')]
    for n in names:
        verify_bundle(entr[n]['bundle'], entr[n]['bundle_sha256'])
    private = out / 'private'
    private.mkdir(mode=0o700, exist_ok=True)
    bank = private / 'bank.json'
    if not bank.exists():
        write_json(bank, dict(seeds=secrets.SystemRandom().sample(range(1_900_000_000, 2_000_000_000), 32)))
    seeds = json.loads(bank.read_text())['seeds']
    source_root = private / 'public-source'
    if not source_root.exists():
        source_root.mkdir()
        with tarfile.open(archive) as tar:
            tar.extractall(source_root, filter='data')
    source = next(source_root.iterdir())
    tonight = [n for n in names if n.endswith(('-a4', '-a5'))] + [n for n in names if n == 'starter-s2']
    plan = []
    for item in args.plan.split(','):
        rung, count = item.split(':')
        nets = tonight if rung == 'foulplay-1s' and not args.networks else names
        plan.append((rung, nets, int(count)))
    tasks = schedule(names, plan, seeds)
    rows_path = out / args.rows
    done = set()
    if rows_path.exists():
        for line in rows_path.read_text().splitlines():
            r = json.loads(line)
            done.add((r['rung'], r['network'], r['seed_index'], r['seat']))
    tasks = [t for t in tasks if t not in done]
    stop_at = args.stop_at
    meta = dict(started_at=time.time(), plan=[dict(rung=r, networks=n, seeds=c) for r, n, c in plan],
                concurrency=args.jobs, stop_starting_after=stop_at, pending_games=len(tasks),
                machine='Apple M5 Max, 18 cores, 64 GB', pid=os.getpid(), argv=sys.argv)
    write_json(out / f'run-{args.label}.json', meta)
    print(f'{len(tasks)} games to play, {args.jobs} in parallel', flush=True)
    ctx = multiprocessing.get_context('spawn')
    with ProcessPoolExecutor(max_workers=args.jobs, mp_context=ctx) as pool, rows_path.open('a') as rows:
        pending = iter(tasks)
        running = {}

        def submit():
            if stop_at and time.time() > stop_at:
                return False
            t = next(pending, None)
            if t is None:
                return False
            rung, name, index, seat = t
            task = dict(rung=rung, network=name, bundle=entr[name]['bundle'], bundle_sha256=entr[name]['bundle_sha256'],
                        seed_index=index, seed=seeds[index], seat=seat, source=str(source), sets=str(sets),
                        time_ms=RUNGS[rung], cap=WALL_CAP[rung])
            running[pool.submit(play, task)] = t
            return True

        for _ in range(args.jobs):
            if not submit():
                break
        finished = 0
        while running:
            future = next(as_completed(list(running)))
            running.pop(future)
            try:
                row = future.result()
            except Exception as exc:  # pool-level failure: record as a fault, never a result
                rung, name, index, seat = None, None, None, None
                row = dict(fault='pool: ' + repr(exc), fault_category='host_pool_fault')
            row.pop('traceback', None) if not row.get('fault') else None
            rows.write(json.dumps(row, sort_keys=True) + '\n')
            rows.flush()
            finished += 1
            if finished % 20 == 0 or row.get('fault'):
                print(time.strftime('%H:%M:%S'), finished, row['rung'], row['network'], row.get('points'),
                      row.get('fault_category'), round(row['wall_seconds'], 1), flush=True)
            submit()
    meta.update(finished_at=time.time(), games_finished=finished, unstarted=sum(1 for _ in pending))
    write_json(out / f'run-{args.label}.json', meta)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', default=str(DEFAULT_OUT))
    p.add_argument('--plan', default='foulplay-100ms:32,foulplay-1s:32,foulplay-10ms:32')
    p.add_argument('--networks', default='', help='comma list overriding all rungs (smoke)')
    p.add_argument('--jobs', type=int, default=16)
    p.add_argument('--rows', default='rows.jsonl')
    p.add_argument('--label', default='main')
    p.add_argument('--stop-at', type=float, default=0, help='epoch seconds after which no new game starts')
    run(p.parse_args())


if __name__ == '__main__':
    main()
