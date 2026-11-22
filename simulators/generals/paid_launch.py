"""Cost preflight BEFORE Modal image builds; invokes only the Generals-owned app.

This script does not create infrastructure outside its bounded Modal invocation.
Reservations remain charged after failures until provider usage can be reconciled.
"""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import time


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--tag',required=True)
    p.add_argument('--stage',choices=['probe','compare','learning','cli'],required=True)
    p.add_argument('--batches',default='128')
    p.add_argument('--steps',type=int,default=256)
    p.add_argument('--seeds',default='42,43,44')
    a=p.parse_args()
    root=Path(__file__).resolve().parents[2]
    folder=root/'evidence/generals-implementation/cuda';folder.mkdir(parents=True,exist_ok=True)
    with (folder/'preflight.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        ledger=json.loads((folder/'budget.json').read_text()) if (folder/'budget.json').exists() else {'runs':[]}
        if any(r['tag']==a.tag for r in ledger['runs']):raise RuntimeError('Tag already reserved; no automatic retries')
        reserve=1140*.00030992+.50
        if len(ledger['runs'])>=8 or sum(r['reserved_usd'] for r in ledger['runs'])+reserve>12 or sum(r['reserved_gpu_seconds'] for r in ledger['runs'])+1140>7200:
            raise RuntimeError('Exclusive Generals cap exhausted before any image build')
        attempts=folder/'launch-attempts.json'
        records=json.loads(attempts.read_text()) if attempts.exists() else []
        row=dict(tag=a.tag,start_unix=time.time(),prebuild_reserved_usd=reserve,reserved_gpu_seconds=1140,
                 scope='Full cold launch wall time, including image build, client wait and teardown. Function resource usage is reconciled separately.')
        records.append(row);attempts.write_text(json.dumps(records,indent=2)+'\n')
        command=['modal','run','--env','staging','integrations/generals/modal_efficiency.py','--tag',a.tag,
                 '--stage',a.stage,'--batches',a.batches,'--steps',str(a.steps),'--seeds',a.seeds]
        started=time.monotonic()
        try:
            with (folder/f'{a.tag}-modal.log').open('w') as log:
                result=subprocess.run(command,cwd=root,stdout=log,stderr=subprocess.STDOUT,timeout=1200)
            row.update(exit_code=result.returncode)
        except subprocess.TimeoutExpired:
            # Only stop this exact game-specific app, identified by its reserved tag.
            latest=json.loads((folder/'budget.json').read_text())
            own=next((r for r in latest['runs'] if r['tag']==a.tag),None)
            if own and own.get('app_id'):
                subprocess.run(['modal','app','stop',own['app_id'],'--env','staging'],timeout=30,check=False)
            row.update(exit_code=124,error='cold-launch watchdog; reservation retained')
        finally:
            row['cold_launch_wall_seconds']=time.monotonic()-started
            attempts.write_text(json.dumps(records,indent=2)+'\n')
        raise SystemExit(row['exit_code'])


if __name__=='__main__':main()
