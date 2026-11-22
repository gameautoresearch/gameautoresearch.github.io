"""Root-owned child supervisor. The host controller owns allocation time."""
from pathlib import Path
import argparse
import json
import os
import signal
import subprocess
import sys

from .arena import _drop_privileges
from .contract import ROOT, write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', default='/job/input')
    p.add_argument('--output', default='/job/output')
    p.add_argument('--seconds', required=True, type=float)
    a = p.parse_args()
    source = Path(a.input)
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    os.chown(out, 65534, 65534)
    out.chmod(0o700)
    env = {'PATH': os.environ['PATH'], 'HOME': str(out), 'TMPDIR': str(out), 'PYTHONPATH': str(ROOT),
           'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '4',
           'CUBLAS_WORKSPACE_CONFIG': ':4096:8'}
    request = json.loads((source / 'request.json').read_text())
    command = [sys.executable, str(source / 'train.py'), '--config', str(source / 'config.json'), '--out', str(out),
               '--seed', str(request['seed']), '--seconds', str(max(0., a.seconds - 2))]
    log = Path('/job/trainer.log').open('wb')
    proc = subprocess.Popen(command, cwd=source, env=env, preexec_fn=_drop_privileges, start_new_session=True,
                            stdout=log, stderr=subprocess.STDOUT)
    reason = 'completed'
    try:
        rc = proc.wait(timeout=max(.01, a.seconds))
    except subprocess.TimeoutExpired:
        reason = 'deadline'
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            rc = proc.wait(timeout=.5)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            rc = proc.wait()
    write_json('/job/supervisor.json', {'status': reason, 'returncode': rc, 'child_pid': proc.pid,
                                        'scope': 'diagnostic only; authoritative provider allocation timestamps '
                                                 'are recorded outside this sandbox'})
    log.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
