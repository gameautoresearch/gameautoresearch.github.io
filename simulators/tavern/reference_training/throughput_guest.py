"""Finite performance verification child; durable outputs and no successor jobs."""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def main():
    entered = time.time()
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--seconds', type=int, required=True)
    p.add_argument('--graphs', action='store_true')
    a = p.parse_args()
    a.root.mkdir(parents=True, exist_ok=False)
    status = dict(entry_unix=entered, status='started', automatic_retries=0)
    def save():
        (a.root/'lifecycle.json').write_text(json.dumps(status, indent=2)+'\n')
        subprocess.run(['sync', str(a.root)], check=True, timeout=10)
    save()
    command = [sys.executable, '-u', '-m', 'simulators.tavern.reference_training.strong.throughput',
               '--inputs', '/inputs', '--out', str(a.root/'suite'), '--device', 'cuda:0',
               '--trials', '2', '--updates', '2', '--sizes', '48', '96', '192', '256',
               '--seconds', str(a.seconds-65), '--proof']
    if a.graphs:
        command += ['--graphs']
    with (a.root/'trainer.log').open('w') as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = child.wait(timeout=a.seconds-45)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait(timeout=10)
            code = 124
    status.update(status='completed' if code == 0 else 'failed', code=code, finished_unix=time.time())
    save()
    raise SystemExit(code)


if __name__ == '__main__':
    main()
