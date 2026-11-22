"""Play a Polytopia checkpoint-Elo tournament: the sparse (default) or full schedule on Modal Functions or Daytona.

    PYTHONPATH=$PWD MODAL_PROFILE=duper MODAL_ENVIRONMENT=game-autoresearch-project caffeinate -dimsu \
        .venv-polytopia-ar/bin/python -m tools.tournament.polytopia_elo.run --config CONFIG.json \
        [--backend modal|daytona] [--schedule sparse|full] [--hours 1.5] [--dry-run] [--smoke]

Daytona needs the Daytona SDK in the interpreter (`.venv-daytona`: `uv pip install daytona==0.232.0 numpy`, plus the
repository on PYTHONPATH) and the CLI logged in to the organization whose credits pay.

Writes (beside the config): games.jsonl, jobs.jsonl, errors.jsonl, status.json, then report.py's elo.json, curves.png,
REPORT.md. Resumable: a (block, pair) with a complete job line is never played again. The private map schedule is
private/schedule.json (host-secret seeds; never publish). No block starts after the deadline; a call that starts
after deadline + grace plays nothing.
"""
import argparse
import json
import os
import secrets
import subprocess
import sys
import time
from pathlib import Path

from tools.tournament.polytopia_elo import config as configs, schedule

HERE = Path(__file__).resolve().parent


def jwrite(path, value):
    tmp = Path(str(path) + '.tmp')
    tmp.write_text(json.dumps(value, indent=1) + '\n')
    os.replace(tmp, path)


def bank(out, count):
    """The private map schedule, extended with fresh host-secret seeds (existing ones kept first) when too short."""
    path = Path(out) / 'private/schedule.json'
    seeds = json.loads(path.read_text())['seeds'] if path.is_file() else []
    if len(seeds) < count:
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        pool = set(seeds)
        while len(seeds) < count:
            s = int.from_bytes(secrets.token_bytes(4), 'big') % 1_000_000_000 + 1_000_000
            if s not in pool:
                seeds.append(s); pool.add(s)
        jwrite(path, dict(seeds=seeds, note='host-secret tournament map schedule; never publish'))
        path.chmod(0o600)
    return seeds[:count]


def read_jsonl(path):
    rows = []
    if Path(path).is_file():
        for line in Path(path).read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except ValueError:
                pass
    return rows


def plan(args, names, kind, games):
    """[(stage label, [(block, a, b)])] still to play; stage B needs stage A's games (call again after A)."""
    if args.smoke:
        pairs = schedule.all_pairs(names, kind)[:args.smoke_pairs]
        return [('smoke', [(0, a, b) for a, b in pairs])]
    if args.schedule == 'full':
        pairs = schedule.all_pairs(names, kind)
        return [(f'full block {r}', [(r, a, b) for a, b in pairs]) for r in range(args.blocks)]
    a_pairs = schedule.stage_a(names, kind, partners=args.partners, seed=args.seed)
    stages = [('A', [(0, a, b) for a, b in a_pairs])]
    if games is not None:
        provisional = schedule.provisional_elo(names, games)
        b_pairs = schedule.neighbours(names, kind, provisional, k=args.neighbours)
        stages.append(('B', [(r, a, b) for a, b in b_pairs for r in range(args.blocks)]))
    return stages


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', required=True)
    p.add_argument('--backend', choices=('modal', 'daytona'), default='modal')
    p.add_argument('--schedule', choices=('sparse', 'full'), default='sparse')
    p.add_argument('--blocks', type=int, default=4, help='blocks per neighbour pair (sparse) or per pair (full)')
    p.add_argument('--chunk', type=int, default=8, help='maps per block (a call plays 2 x chunk games)')
    p.add_argument('--partners', type=int, default=4, help='sparse stage A: random partners per entrant')
    p.add_argument('--neighbours', type=int, default=24, help='sparse stage B: nearest neighbours per entrant')
    p.add_argument('--seed', type=int, default=0, help='stage A partner draw')
    p.add_argument('--hours', type=float, default=2.0)
    p.add_argument('--grace-minutes', type=float, default=5.0)
    p.add_argument('--smoke', action='store_true', help='--smoke-pairs pairs, one block, in smoke/')
    p.add_argument('--smoke-pairs', type=int, default=3)
    p.add_argument('--dry-run', action='store_true', help='print the plan and an estimate; play nothing')
    p.add_argument('--workers', type=int, default=120, help='Daytona: worker sandboxes (one call at a time each)')
    p.add_argument('--daytona-cpu', type=int, default=2)
    p.add_argument('--daytona-memory', type=int, default=3, help='Daytona: GiB per worker (a call peaks ~1.4 GB)')
    p.add_argument('--max-containers', type=int, default=600, help='Modal: concurrent Function containers')
    p.add_argument('--image', choices=('cpu', 'release'), default='cpu',
                   help="Modal: the slim CPU image (default) or the release's CUDA worker image")
    p.add_argument('--memory-mib', type=int, default=None, help='Modal: memory request per call (cpu 2048, release 4096)')
    p.add_argument('--report-python', default=os.environ.get('ELO_REPORT_PY', sys.executable))
    args = p.parse_args()
    cfg = configs.load(args.config)
    out = Path(cfg['dir']) / ('smoke' if args.smoke else '')
    out.mkdir(exist_ok=True)
    entrants = json.loads((Path(cfg['dir']) / 'entrants.json').read_text())
    kind = {e['name']: e['kind'] for e in entrants}
    names = [e['name'] for e in entrants]
    seeds = bank(cfg['dir'], args.blocks * args.chunk)
    done = {(j['round'], *schedule.key(j['a'], j['b'])) for j in read_jsonl(out / 'jobs.jsonl')
            if j.get('games') == j.get('expected')}
    games = read_jsonl(out / 'games.jsonl')
    if args.dry_run:
        stages = plan(args, names, kind, games or None)
        for label, items in stages:
            todo = [x for x in items if (x[0], *schedule.key(x[1], x[2])) not in done]
            print(f'stage {label}: {len(items):,} calls ({len(todo):,} to play), {2 * args.chunk * len(todo):,} games; '
                  f'~{len(todo) * 37 / 3600:.0f} call-hours at 37 s a call')
        if args.schedule == 'sparse' and not args.smoke and not games:
            n = sum(1 for x in names if kind[x] != 'reference')
            print(f'stage B (after A): ~{n * args.neighbours // 2 * args.blocks:,} calls at most')
        return
    if args.backend == 'modal':
        from tools.tournament.polytopia_elo.modal_backend import ModalBackend
        backend = ModalBackend(cfg, entrants, memory_mib=args.memory_mib or (2048 if args.image == 'cpu' else 4096),
                               max_containers=args.max_containers, image=args.image)
    else:
        from tools.tournament.polytopia_elo.daytona_backend import DaytonaBackend
        backend = DaytonaBackend(cfg, entrants, workers=args.workers, cpu=args.daytona_cpu,
                                 memory_gib=args.daytona_memory, ttl_hours=args.hours + 1)
    started = time.time()
    deadline = started + args.hours * 3600
    hard = deadline + args.grace_minutes * 60
    state = dict(schedule=args.schedule, blocks=args.blocks, chunk=args.chunk, partners=args.partners,
                 neighbours=args.neighbours, release=cfg['release'], started_unix=started, deadline_unix=deadline,
                 hard_deadline_unix=hard, entrants=len(names), jobs_done=0, jobs_failed=0, games=0, capped=0,
                 unplayed=0, cpu_seconds=0.0, call_seconds=0.0, max_total_rss=0, stages=[], running=True,
                 pid=os.getpid())
    jwrite(out / 'status.json', state)
    with backend:
        state.update(backend.info())
        with open(out / 'jobs.jsonl', 'a') as jl, open(out / 'games.jsonl', 'a') as gl:
            stage_index = 0
            while True:
                stages = plan(args, names, kind, read_jsonl(out / 'games.jsonl') if stage_index else None)
                if stage_index >= len(stages):
                    break
                label, items = stages[stage_index]
                stage_index += 1
                if time.time() >= deadline:
                    state['stop_reason'] = 'deadline'
                    break
                jobs = []
                for r, a, b in items:
                    if (r, *schedule.key(a, b)) in done:
                        continue
                    if r % 2:  # alternate which network is the grader's candidate; both play both seats every block
                        a, b = b, a
                    jobs.append(dict(round=r, a=a, b=b, first=r * args.chunk,
                                     seeds=seeds[r * args.chunk:(r + 1) * args.chunk], hard_deadline=hard))
                state['stages'].append(dict(stage=label, calls=len(items), to_play=len(jobs), pairs=len({(a, b) for _, a, b in items})))
                jwrite(out / 'status.json', state)
                for res in backend.map(jobs):
                    if isinstance(res, BaseException):
                        state['jobs_failed'] += 1
                        with open(out / 'errors.jsonl', 'a') as e:
                            e.write(json.dumps(dict(stage=label, error=repr(res)[:2000])) + '\n')
                        continue
                    j = res['job']
                    rows = [dict(round=j['round'], a=j['a'], b=j['b'], seed=j['first'] + j['seeds'].index(x['seed']),
                                 a_seat=x['seat'], a_points=(x['reward'] + 1) / 2, capped=bool(x['capped']),
                                 turn=x.get('turn'), steps=x.get('steps'), scores=x.get('scores'))
                            for x in res.get('rows', [])]
                    for row in rows:
                        gl.write(json.dumps(row) + '\n')
                    gl.flush()
                    expected = 2 * len(j['seeds'])
                    meta = dict(round=j['round'], a=j['a'], b=j['b'], stage=label, games=len(rows), expected=expected,
                                exit=res.get('exit'), seconds=res.get('seconds'), cpu_seconds=res.get('cpu_seconds'),
                                total_rss_peak=res.get('total_rss_peak'), cgroup_memory_peak=res.get('cgroup_memory_peak'),
                                container=res.get('container'), skipped=res.get('skipped'), finished=time.time())
                    jl.write(json.dumps(meta) + '\n')
                    jl.flush()
                    if len(rows) == expected:
                        done.add((j['round'], *schedule.key(j['a'], j['b'])))
                    state['jobs_done'] += 1
                    state['games'] += len(rows)
                    state['capped'] += sum(1 for x in rows if x['capped'])
                    state['unplayed'] += expected - len(rows)
                    state['cpu_seconds'] += res.get('cpu_seconds') or 0
                    state['call_seconds'] += res.get('seconds') or 0
                    state['max_total_rss'] = max(state['max_total_rss'], res.get('total_rss_peak') or 0)
                    if len(rows) != expected or res.get('exit') != 0:
                        state['jobs_failed'] += 1
                        with open(out / 'errors.jsonl', 'a') as e:
                            e.write(json.dumps(dict(meta, stderr=res.get('stderr'), policy_stderr=res.get('policy_stderr'))) + '\n')
                    elapsed = time.time() - started
                    state.update(now_unix=time.time(), elapsed_h=round(elapsed / 3600, 3),
                                 games_per_s=round(state['games'] / max(elapsed, 1), 2),
                                 est_compute_usd=round(backend.cost(state['call_seconds'], state['cpu_seconds'], elapsed), 2))
                    if state['jobs_done'] % 25 == 0:
                        jwrite(out / 'status.json', state)
                jwrite(out / 'status.json', state)
                if args.smoke:
                    break
            state.setdefault('stop_reason', 'schedule complete')
        state['est_compute_usd'] = round(backend.cost(state['call_seconds'], state['cpu_seconds'], time.time() - started), 2)
    state.update(running=False, finished_unix=time.time())
    jwrite(out / 'status.json', state)
    if not args.smoke:
        subprocess.run([args.report_python, '-m', 'tools.tournament.polytopia_elo.report', '--config', args.config],
                       env=dict(os.environ, PYTHONPATH=str(configs.REPO)))
    print(json.dumps({k: state.get(k) for k in ('games', 'jobs_done', 'jobs_failed', 'unplayed', 'cpu_seconds',
                                                'call_seconds', 'est_compute_usd', 'stop_reason')}))


if __name__ == '__main__':
    main()
