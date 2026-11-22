#!/usr/bin/env python3
"""Pre-reserve game-only budget before image building or GPU calls; freeze upload source."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

ROOT=Path(__file__).resolve().parents[2]
EVIDENCE=ROOT/'evidence/tavern-implementation'
LEDGER=EVIDENCE/'gpu-budget.json'
RATES={'gpu_L4_second':.000222,'cpu_core_second':.0000131,'memory_GiB_second':.00000222}
RATE=.000222+4*.0000131+16*.00000222


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=['baseline','optimized','learning','sampling'],required=True)
    p.add_argument('--run-id',required=True)
    p.add_argument('--seed',type=int,choices=[12345,45678,78901],default=12345)
    a=p.parse_args()
    if not a.run_id.replace('-','').isalnum():raise ValueError('simple unique run-id required')
    reservation={'run_id':a.run_id,'phase':a.phase,'seed':a.seed,'reserved_gpu_seconds':1082,
      'reserved_usd':1082*RATE+1.0+.10,'status':'reserved','start_unix':time.time(),
      'build_and_storage_reserve_usd':1.10}
    with (EVIDENCE/'gpu-budget.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        ledger=json.loads(LEDGER.read_text()) if LEDGER.exists() else {'hard_cap_usd':12,'hard_cap_gpu_seconds':7200,
          'rates':RATES,'rates_source':'https://modal.com/pricing','rates_verified_date':'2026-11-08','runs':[]}
        if any(r['run_id']==a.run_id for r in ledger['runs']):raise ValueError('run-id already reserved; no blind retry')
        if sum(r['reserved_gpu_seconds'] for r in ledger['runs'])+1082>7200:raise RuntimeError('two GPU-hour reservation cap')
        if sum(r['reserved_usd'] for r in ledger['runs'])+reservation['reserved_usd']>12:raise RuntimeError('$12 reservation cap')
        ledger['runs'].append(reservation);LEDGER.write_text(json.dumps(ledger,indent=2)+'\n')
    payload=EVIDENCE/'gpu-payloads'/a.run_id
    payload.mkdir(parents=True,exist_ok=False)
    engine=ROOT/'candidates/AutoTavern'
    shutil.copytree(engine,payload/'candidates/AutoTavern',ignore=shutil.ignore_patterns('target','.git','__pycache__','*.so','*.dylib','*.pyc','.pytest_cache'))
    shutil.copytree(ROOT/'integrations/tavern',payload/'integrations/tavern',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    command=['modal','run','--env','staging',str(ROOT/'integrations/tavern/modal_app.py'),'--phase',a.phase,'--run-id',a.run_id,'--seed',str(a.seed)]
    log=EVIDENCE/(a.run_id+'-modal.log')
    started=time.perf_counter()
    with log.open('w') as stream:
        result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,
            env=dict(os.environ,TAVERN_GPU_PAYLOAD=str(payload)),timeout=2400)
    destination=EVIDENCE/'cuda'/a.run_id
    destination.parent.mkdir(parents=True,exist_ok=True)
    download=subprocess.run(['modal','volume','get','autoresearch-tavern-cuda-evidence-20261108',a.run_id,str(destination.parent),'--force','--env','staging'],capture_output=True,text=True)
    with (EVIDENCE/'gpu-budget.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        ledger=json.loads(LEDGER.read_text());row=next(r for r in ledger['runs'] if r['run_id']==a.run_id)
        row.update(status='succeeded' if result.returncode==0 else 'failed',client_build_and_call_seconds=time.perf_counter()-started,
                   returncode=result.returncode,artifact_download_returncode=download.returncode)
        if (destination/'allocation.json').exists():
            allocation=json.loads((destination/'allocation.json').read_text());row['allocation']=allocation
            row['estimated_runtime_usd']=(allocation['worker_seconds']+182)*RATE
        LEDGER.write_text(json.dumps(ledger,indent=2)+'\n')
    print(json.dumps(row,indent=2))
    if download.returncode:print(download.stderr[-2000:])
    if result.returncode:raise SystemExit(result.returncode)


if __name__=='__main__':main()
