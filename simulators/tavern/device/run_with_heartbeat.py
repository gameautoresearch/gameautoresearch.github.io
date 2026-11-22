"""Bound an owned compiler child and keep long image-build streams active."""
import argparse
import json
import os
import signal
import subprocess
import time


def main():
    p=argparse.ArgumentParser();p.add_argument('--timeout',type=int,required=True)
    p.add_argument('--label',required=True);p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
    if not a.command or a.timeout<=0:raise ValueError('bounded compiler command required')
    started=time.perf_counter();child=subprocess.Popen(a.command,start_new_session=True)
    try:
        while True:
            elapsed=time.perf_counter()-started
            print(json.dumps({'build_phase':a.label,'elapsed_seconds':round(elapsed,3)}),flush=True)
            left=a.timeout-elapsed
            if left<=0:
                os.killpg(child.pid,signal.SIGKILL);child.wait();raise SystemExit(124)
            try:raise SystemExit(child.wait(timeout=min(10,left)))
            except subprocess.TimeoutExpired:pass
    finally:
        if child.poll() is None:os.killpg(child.pid,signal.SIGKILL);child.wait()


if __name__=='__main__':main()
