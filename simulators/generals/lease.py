"""Supervise an already allocated worker; does NOT acquire compute or spend credits.

Keep setup/training/profiling/evaluation commands in one child script. Every second
of its lifetime counts, including idle GPU time and CPU-only development work.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import uuid


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--recipe', required=True)
    p.add_argument('--ledger', required=True, help='persistent allocation ledger across restarts/commands')
    p.add_argument('command', nargs=argparse.REMAINDER)
    a = p.parse_args()
    command = a.command[1:] if a.command[:1] == ['--'] else a.command
    if not command:
        p.error('a worker command is required after --')
    cfg = json.loads(Path(a.recipe).read_text())
    path = Path(a.ledger)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_suffix(path.suffix+'.lock')
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(fd)
    proc = None
    recorded = False
    start_mono = time.monotonic()
    def cancelled(signum, frame):
        raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM, cancelled)
    try:
        old = json.loads(path.read_text()) if path.exists() else dict(allocated_worker_seconds=0., leases=[])
        prior = old['allocated_worker_seconds']
        limit = float(cfg['allocated_worker_seconds'])
        if prior >= limit:
            raise RuntimeError('Allocation exhausted; no worker launched')
        start = time.time()
        lease_id = str(uuid.uuid4())
        env = dict(os.environ, GENERALS_ALLOCATION_START_UNIX=str(start),
                   GENERALS_ALLOCATION_DEADLINE_UNIX=str(start+limit-prior),
                   GENERALS_ALLOCATION_PRIOR_SECONDS=str(prior), GENERALS_ALLOCATION_ID=lease_id)
        proc = subprocess.Popen(command, env=env, start_new_session=True)
        reason = 'completed'
        try:
            code = proc.wait(timeout=limit-prior)
        except subprocess.TimeoutExpired:
            reason = 'allocated_worker_deadline'
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            code = 124
        elapsed = time.monotonic()-start_mono
        old['leases'].append(dict(id=lease_id, command=command, start_unix=start, elapsed_seconds=elapsed,
                                  exit_code=code, stop_reason=reason, hardware_contract=cfg['hardware']))
        old.update(allocated_worker_seconds=prior+elapsed, limit_seconds=limit,
                   allocated_worker_hours=(prior+elapsed)/3600,
                   accounting='Full elapsed worker allocation, including CPU work, idle GPU, compilation, training, evaluation and shutdown.',
                   overrun_seconds=max(0., prior+elapsed-limit))
        tmp = path.with_suffix(path.suffix+'.tmp')
        tmp.write_text(json.dumps(old, indent=2)+'\n')
        tmp.replace(path)
        recorded = True
        raise SystemExit(code)
    finally:
        if proc is not None and proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
        if proc is not None and not recorded:
            elapsed = time.monotonic()-start_mono
            old['leases'].append(dict(id=lease_id, command=command, start_unix=start,
                                     elapsed_seconds=elapsed, exit_code=proc.returncode,
                                     stop_reason='interrupted', hardware_contract=cfg['hardware']))
            old.update(allocated_worker_seconds=prior+elapsed, limit_seconds=limit,
                       allocated_worker_hours=(prior+elapsed)/3600,
                       overrun_seconds=max(0., prior+elapsed-limit))
            tmp = path.with_suffix(path.suffix+'.tmp')
            tmp.write_text(json.dumps(old, indent=2)+'\n')
            tmp.replace(path)
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
