#!/usr/bin/env python3
"""New-authorized performance campaign: immutable payload/attempt records, bounded jobs.

The user's Nov 8 campaign update supersedes the old dollar/hour/attempt gates. Old
ledgers remain immutable. Every experiment still has one fixed allocation, no retry,
no warm pool, a hard timeout, retained costs/failures and game-owned cleanup.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import hashlib

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'evidence/tavern-implementation/structural-20261108'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=['probe','learning','refine','massive','final','packed','wall'],default='probe')
    p.add_argument('--run-id',required=True)
    p.add_argument('--seed',type=int,default=12345)
    p.add_argument('--gpu',choices=['L4','H100'],default='L4')
    p.add_argument('--cpu-cores',type=int,default=4)
    p.add_argument('--ram-gib',type=int,default=16)
    p.add_argument('--pilot-checkpoint',type=Path,default=ROOT/'evidence/tavern-implementation/cuda/cuda-learning-v3/learning/seed12345-optimized/latest.pt')
    a=p.parse_args()
    if not a.run_id.replace('-','').isalnum():raise ValueError('simple unique run ID required')
    if not 1<=a.cpu_cores<=64 or not 4<=a.ram_gib<=256:raise ValueError('explicit bounded CPU/RAM required')
    out=BASE/a.run_id;out.mkdir(parents=True,exist_ok=False)
    pilot=out/'pilot-source.pt';shutil.copyfile(a.pilot_checkpoint,pilot)
    payload=out/'payload'
    payload.mkdir()
    shutil.copytree(ROOT/'candidates/AutoTavern',payload/'candidates/AutoTavern',ignore=shutil.ignore_patterns('target','.git','__pycache__','*.so','*.dylib','*.pyc','.pytest_cache'))
    shutil.copytree(ROOT/'integrations/tavern',payload/'integrations/tavern',ignore=shutil.ignore_patterns('__pycache__','*.pyc','target','.git','tavern.ptx'))
    record={'authorization':'Nov 8 explicit campaign update: paid structural experiments authorized, previous game/shared dollar/hour/attempt ceilings superseded; historical ledgers unchanged',
        'run_id':a.run_id,'phase':a.phase,'seed':a.seed,'status':'started','start_unix':time.time(),
        'pilot_checkpoint_sha256':hashlib.sha256(pilot.read_bytes()).hexdigest(),
        'allocation':{'gpu':a.gpu,'cpu_cores':a.cpu_cores,'memory_gib':a.ram_gib,'max_containers':1,'timeout_seconds':3600,'retries':0,'min_containers':0}}
    (out/'attempt.json').write_text(json.dumps(record,indent=2)+'\n')
    started=time.perf_counter()
    cmd=['modal','run','--env','staging',str(ROOT/'integrations/tavern/structural_modal.py'),'--phase',a.phase,'--run-id',a.run_id,'--seed',str(a.seed)]
    try:
        with (out/'modal.log').open('w') as log:
            result=subprocess.run(cmd,env=dict(os.environ,TAVERN_GPU_PAYLOAD=str(payload),TAVERN_GPU=a.gpu,
                TAVERN_CPU_CORES=str(a.cpu_cores),TAVERN_RAM_GIB=str(a.ram_gib),TAVERN_PILOT_CHECKPOINT=str(pilot)),stdout=log,stderr=subprocess.STDOUT,timeout=5400)
        record.update(returncode=result.returncode,status='succeeded' if result.returncode==0 else 'failed')
    except BaseException as exc:
        record.update(status='client_failed',error=repr(exc));raise
    finally:
        record['client_build_and_call_seconds']=time.perf_counter()-started
        (out/'attempt.json').write_text(json.dumps(record,indent=2)+'\n')
    # Modal treats a nonexistent destination as a single file, even for recursive
    # remote directories. Establish the parent first; preserve run-ID nesting.
    (out/'retained').mkdir()
    download=subprocess.run(['modal','volume','get','autoresearch-tavern-structural-evidence-20261108',a.run_id,str(out/'retained'),'--env','staging'],capture_output=True,text=True)
    record.update(artifact_download_returncode=download.returncode,artifact_download_stderr=download.stderr[-2000:])
    (out/'attempt.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))
    if result.returncode:raise SystemExit(result.returncode)


if __name__=='__main__':main()
