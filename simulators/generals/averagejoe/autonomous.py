"""Sandbox entrypoint: durable files, bounded child process, no laptop heartbeat."""
from pathlib import Path
import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import traceback


def write(path, value):
    path = Path(path); temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n'); temporary.replace(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--deadline', type=float, required=True)
    a = p.parse_args(); a.root.mkdir(parents=True, exist_ok=True)
    claim = a.root / 'entry-claim'
    claim.mkdir(exist_ok=False)  # a duplicate allocation must never silently resume/spend.
    started = time.time()
    def sync():
        subprocess.run(['sync', str(a.root)], check=True, timeout=35)
    write(a.root/'lifecycle.json', {'status': 'initializing', 'started_unix': started,
          'deadline_unix': a.deadline, 'task_id': os.environ.get('MODAL_TASK_ID'),
          'laptop_dependency': False, 'automatic_gpu_retries': 0})
    sync()
    child = None
    try:
        for source, name in [('/inputs', 'inputs'), ('/parent', 'parent'),
                             ('/parent-source', 'parent-source'), ('/workspace', 'source'),
                             ('/job-metadata', 'launch-metadata')]:
            shutil.copytree(source, a.root/name, ignore=shutil.ignore_patterns('__pycache__', '.pytest_cache'))
        sync()
        seconds = max(60., a.deadline-time.time()-35)
        command = [sys.executable, '-u', '-m', 'simulators.generals.averagejoe.continuation',
                   '--inputs', str(a.root/'inputs'), '--parent', str(a.root/'parent'),
                   '--parent-source', str(a.root/'parent-source'), '--output', str(a.root/'output'),
                   '--seconds', str(seconds)]
        write(a.root/'command.json', {'argv': command, 'started_unix': time.time(),
              'soft_deadline_unix': a.deadline-35, 'hard_deadline_unix': a.deadline+35})
        with (a.root/'trainer.log').open('w', buffering=1) as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            while child.poll() is None and time.time() < a.deadline-35:
                write(a.root/'heartbeat.json', {'unix': time.time(), 'child_pid': child.pid,
                      'status': 'running', 'deadline_unix': a.deadline})
                time.sleep(5)
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=max(1, a.deadline+35-time.time()))
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL); child.wait(timeout=10)
            code = child.returncode
        output = a.root/'output'
        complete = code == 0 and (output/'result.json').exists()
        write(a.root/'lifecycle.json', {'status': 'completed' if complete else 'failed_or_deadline',
              'trainer_returncode': code, 'started_unix': started, 'ended_unix': time.time(),
              'deadline_unix': a.deadline, 'task_id': os.environ.get('MODAL_TASK_ID'),
              'laptop_dependency': False, 'automatic_gpu_retries': 0,
              'latest_checkpoint_present': (output/'latest.json').exists()})
        sync()
        if not complete:
            raise RuntimeError('Trainer did not complete cleanly; durable partial evidence retained')
    except BaseException:
        (a.root/'entry-error.txt').write_text(traceback.format_exc())
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGKILL); child.wait(timeout=10)
        sync()
        raise


if __name__ == '__main__':
    main()
