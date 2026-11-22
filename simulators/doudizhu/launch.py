#!/usr/bin/env python3
"""Meter a complete allocated worker subprocess, including Python startup and teardown.

Use this entry point on the allocated machine. A lease begun earlier can provide
--allocated-start-unix; preparation/transfer inside that lease is then charged too.
No GPU is provisioned here and no cloud APIs or paid resources are invoked.
"""
import json
import fcntl
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def main():
    started = time.time()
    argv = sys.argv[1:]
    here = Path(__file__).resolve().parent
    root = here.parents[1]

    def value(flag, default=None):
        return argv[argv.index(flag)+1] if flag in argv else default

    def resolve(path):
        p = Path(path)
        return p if p.is_absolute() else root / p

    out = resolve(value('--out', 'evidence/doudizhu-implementation/run'))
    out.mkdir(parents=True,exist_ok=True)
    if '--resume' not in argv and any((out/name).exists() for name in ('manifest.json','allocation.json','result.json')):
        raise RuntimeError('Output run already exists; choose a fresh directory or explicit training resume')
    lock_paths=[out/'.worker.lock']
    if '--resume' in argv:
        parent_lock=resolve(value('--resume')).parent/'.worker.lock'
        if parent_lock.resolve()!=lock_paths[0].resolve():
            lock_paths.append(parent_lock)
    locks=[]
    for path in lock_paths:
        handle=path.open('a')
        try:
            fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f'Run is already active: {path.parent}')
        locks.append(handle)
    prior = 0.
    # Only training resume extends an earlier experiment allocation. Evaluations are
    # separate full-worker allocations and must be added to the experiment ledger.
    if '--resume' in argv:
        for path in (out/'allocation.json', resolve(value('--resume')).parent/'allocation.json'):
            if path.exists():
                prior = max(prior, json.loads(path.read_text())['cumulative_allocated_worker_seconds'])
    allocated_start = float(value('--allocated-start-unix', started))
    if '--allocated-start-unix' not in argv:
        argv += ['--allocated-start-unix', str(allocated_start)]
    argv += ['--prior-worker-seconds', str(prior)]
    recipe = resolve(value('--recipe', 'integrations/doudizhu/recipes/smoke.json'))
    config = json.loads(recipe.read_text())
    limit = float(config['max_seconds'])
    if config['allocation']['gpus'] and 'budget_gpu_hours' in config:
        limit = min(limit,3600*config['budget_gpu_hours'])
    remaining = max(.001,limit-prior-(time.time()-allocated_start))
    process = subprocess.Popen([sys.executable, str(here/'run.py'), *argv], start_new_session=True)
    deadline_hit = False
    interrupted_signal = None

    class WorkerInterrupted(Exception):
        def __init__(self,signum):self.signum=signum

    def interrupt(signum,frame):raise WorkerInterrupted(signum)

    def stop_worker():
        # A cancelled launcher must not leave its separate worker session alive.
        try:os.killpg(process.pid,signal.SIGTERM)
        except ProcessLookupError:return process.wait()
        try:return process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            try:os.killpg(process.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            return process.wait()

    previous={s:signal.signal(s,interrupt) for s in (signal.SIGINT,signal.SIGTERM)}
    try:
        status = process.wait(timeout=remaining)
    except subprocess.TimeoutExpired:
        deadline_hit = True
        # Terminate the whole worker group, including persistent CPU simulators.
        # The last atomic checkpoint remains resumable. Teardown is still charged.
        status = stop_worker()
    except WorkerInterrupted as exc:
        interrupted_signal=exc.signum
        for signum in previous:signal.signal(signum,signal.SIG_IGN)
        stop_worker()
        status=128+exc.signum
    finally:
        for signum,handler in previous.items():signal.signal(signum,handler)
    ended = time.time()
    seconds = max(0., ended-allocated_start)
    gpus = config['allocation']['gpus']
    summary = {'returncode': status, 'worker_pid':process.pid, 'started_unix': allocated_start, 'ended_unix': ended,
               'hard_deadline_hit': deadline_hit, 'limit_seconds':limit,
               'interrupted_signal':interrupted_signal,
               'invocation_allocated_worker_seconds': seconds,
               'cumulative_allocated_worker_seconds': prior+seconds,
               'cumulative_allocated_gpu_hours': (prior+seconds)/3600 if gpus else 0.,
               'allocation': config['allocation'],
               'scope': 'entire child lifetime plus supplied prelaunch allocated lease time; '
                        'CPU simulation and idle GPU included; no sample-based budget'}
    out.mkdir(parents=True, exist_ok=True)
    temporary=out/'allocation.json.tmp'
    temporary.write_text(json.dumps(summary, indent=2)+'\n')
    temporary.replace(out/'allocation.json')
    print(json.dumps(summary), flush=True)
    raise SystemExit(status)


if __name__ == '__main__':
    main()
