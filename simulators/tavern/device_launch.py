#!/usr/bin/env python3
"""Freeze and account one independent device-engine attempt; never reset old ledgers."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'evidence/tavern-implementation/device-port'


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True)
    p.add_argument('--split',action='store_true');p.add_argument('--arena',action='store_true');p.add_argument('--rust-opt',choices=['0','1','2','3'],default='3')
    p.add_argument('--ptxas-opt',choices=['0','1','2','3'],default='0');p.add_argument('--nvvm-opt',choices=['0','1'],default='1');p.add_argument('--diagnostic-ir',action='store_true');a=p.parse_args()
    out=BASE/a.run_id;out.mkdir(parents=True,exist_ok=False);payload=out/'payload'
    for source in ['candidates/AutoTavern','integrations/tavern/device']:
        shutil.copytree(ROOT/source,payload/source,ignore=shutil.ignore_patterns('target','target-*','.git','__pycache__','*.so','*.dylib','*.ptx','*.cubin'))
    for name in ['device_modal.py','device_probe.py','device_env.py','device_fixtures.py','device_training.py','device_learning_probe.py','model.py','learner_graph.py','rollout.py']:
        shutil.copyfile(ROOT/'integrations/tavern'/name,payload/'integrations/tavern'/name)
    record={'run_id':a.run_id,'status':'started','start_unix':time.time(),
        'authorization':'Nov 8 user explicitly authorizes complete GPU simulator implementation and paid validation without old caps',
        'compiler':{'split':a.split,'arena':a.arena,'rust_opt':a.rust_opt,'nvvm_opt':a.nvvm_opt,'ptxas_opt':a.ptxas_opt,'diagnostic_ir':a.diagnostic_ir},
        'resources':{'gpu':'B200','cpu_cores':4,'ram_gib':16,'timeout_seconds':1800,'retries':0,'warm_containers':0}}
    (out/'attempt.json').write_text(json.dumps(record,indent=2)+'\n');start=time.perf_counter()
    with (out/'modal.log').open('w') as stream:
        result=subprocess.run(['modal','run','--env','staging',str(payload/'integrations/tavern/device_modal.py'),
            '--run-id',a.run_id],env=dict(os.environ,TAVERN_DEVICE_PAYLOAD=str(payload),TAVERN_DEVICE_ARENA=str(int(a.arena)),
                TAVERN_DEVICE_SPLIT=str(int(a.split)),TAVERN_DEVICE_RUST_OPT=a.rust_opt,TAVERN_DEVICE_NVVM_OPT=a.nvvm_opt,TAVERN_DEVICE_ASM_OPT=a.ptxas_opt,
                **({'TAVERN_DEVICE_IR_TAG':a.run_id} if a.diagnostic_ir else {})),stdout=stream,stderr=subprocess.STDOUT,timeout=7200)
    record.update(returncode=result.returncode,client_build_and_call_seconds=time.perf_counter()-start,
        status='succeeded' if result.returncode==0 else 'failed')
    (out/'attempt.json').write_text(json.dumps(record,indent=2)+'\n')
    (out/'retained').mkdir()
    download=subprocess.run(['modal','volume','get','autoresearch-tavern-device-evidence-20261108',a.run_id,
        str(out/'retained'),'--env','staging'],capture_output=True,text=True)
    (out/'retrieval.json').write_text(json.dumps({'returncode':download.returncode,'stderr':download.stderr[-2000:]},indent=2)+'\n')
    print(json.dumps(record,indent=2))
    if result.returncode:raise SystemExit(result.returncode)


if __name__=='__main__':main()
