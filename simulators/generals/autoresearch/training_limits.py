"""Bounded training-only Linux limits; inference keeps its separate limits."""
from pathlib import Path
import ctypes
import os
import resource
import subprocess
import time

RUNTIME_ID = 'generals-training-runtime-v2-nproc256-nofile512'
NPROC = 256
NOFILE = 512


def drop_training_privileges():
    os.setgroups([])
    os.setgid(65534)
    os.setuid(65534)
    if ctypes.CDLL(None, use_errno=True).prctl(38, 1, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), 'PR_SET_NO_NEW_PRIVS failed')
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_NPROC, (NPROC, NPROC))
    resource.setrlimit(resource.RLIMIT_NOFILE, (NOFILE, NOFILE))


def process_sample(pid, proc_root=Path('/proc')):
    """Root supervisor reads child metadata only; no policy/private inputs."""
    base = Path(proc_root) / str(pid)
    result = {}
    try:
        entries = dict(line.split(':', 1) for line in (base/'status').read_text().splitlines() if ':' in line)
        for key in ('Uid', 'Gid', 'Threads', 'NoNewPrivs', 'VmRSS'):
            if key in entries: result[key] = entries[key].strip()
        result['limits'] = (base/'limits').read_text()
        result['open_fds'] = len(list((base/'fd').iterdir()))
    except OSError as exc:
        result['read_error'] = type(exc).__name__
    return result


def wait_monitored(proc, seconds, record):
    record.update(runtime_id=RUNTIME_ID, configured_nproc=NPROC, configured_nofile=NOFILE,
                  samples=0, peak_threads=0, peak_open_fds=0)
    # Unified cgroup readings are diagnostic; worker CPU/RAM and GPU are unchanged.
    record['cgroup'] = {}
    for name in ('pids.max', 'pids.current', 'memory.max'):
        try: record['cgroup'][name] = (Path('/sys/fs/cgroup')/name).read_text().strip()
        except OSError: pass
    deadline = time.monotonic() + max(.01, seconds)
    while True:
        sample = process_sample(proc.pid)
        if 'Threads' in sample:
            record['samples'] += 1
            record.setdefault('first_observation', sample)
            record['last_observation'] = sample
            record['peak_threads'] = max(record['peak_threads'], int(sample['Threads']))
            record['peak_open_fds'] = max(record['peak_open_fds'], sample.get('open_fds', 0))
        rc = proc.poll()
        if rc is not None: return rc
        remaining = deadline - time.monotonic()
        if remaining <= 0: raise subprocess.TimeoutExpired(proc.args, seconds)
        time.sleep(min(1., remaining))
