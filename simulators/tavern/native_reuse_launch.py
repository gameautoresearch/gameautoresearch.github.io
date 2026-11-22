"""Qualify a completed build-only attempt without compiling or overlapping GPUs."""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'evidence/tavern-implementation/device-port'


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-id',required=True)
    p.add_argument('--compiled-attempt',required=True)
    p.add_argument('--block-threads',type=int,choices=[1,32],default=32)
    a=p.parse_args()
    parent=BASE/a.compiled_attempt;prior=json.loads((parent/'attempt.json').read_text())
    if prior['status']!='succeeded_build_only' or prior.get('gpu_executed') is not False:
        raise ValueError('completed explicit CPU-only build attempt required')
    images=re.findall(r'Built image (im-[A-Za-z0-9]+)',(parent/'modal.log').read_text())
    if not images:raise ValueError('built image ID missing')
    image=images[-1];out=BASE/a.run_id;out.mkdir(parents=True,exist_ok=False)
    payload=out/'payload';shutil.copytree(parent/'payload',payload)
    shutil.copytree(parent/'compiler-receipt',out/'compiler-receipt')
    program=payload/'integrations/tavern/prebuilt_native_modal.py'
    shutil.copyfile(ROOT/'integrations/tavern/prebuilt_native_modal.py',program)
    for name in ['device_env.py','device_probe.py','device_wide_probe.py','device_divergence_probe.py','device_state_schema.py','device_training.py']:
        shutil.copyfile(ROOT/'integrations/tavern'/name,payload/'integrations/tavern'/name)
    shutil.copytree(ROOT/'integrations/tavern/device/fixtures',payload/'integrations/tavern/device/fixtures',dirs_exist_ok=True)
    record={'run_id':a.run_id,'status':'started','start_unix':time.time(),'compiled_attempt':a.compiled_attempt,
        'binary_image':image,'block_threads':a.block_threads,'compiler':prior['compiler'],'assembler_opt':prior['assembler_opt'],
        'state_assembler_opt':prior.get('state_assembler_opt'),'max_registers':prior.get('max_registers'),
        'resources':{'gpu':'B200','cpu_cores':4,'ram_gib':16,'timeout_seconds':1800,'retries':0,'warm_containers':0},
        'setup_accounting':'Parent local compilation and CPU image assembly remain retained/charged; this call records its complete GPU allocation.'}
    (out/'attempt.json').write_text(json.dumps(record,indent=2)+'\n');start=time.perf_counter()
    with (out/'modal.log').open('x') as log:
        proc=subprocess.run(['modal','run','--detach','--env','staging',str(program),'--run-id',a.run_id],
            env=dict(os.environ,TAVERN_NATIVE_PAYLOAD=str(payload),TAVERN_NATIVE_PREBUILT_IMAGE=image,TAVERN_NATIVE_BLOCK_THREADS=str(a.block_threads)),
            stdout=log,stderr=subprocess.STDOUT,timeout=3600)
    effective=proc.returncode
    if proc.returncode:
        from modal_call_wait import reattach
        result=reattach(out/'modal.log',out/'call-recovery.json',a.run_id)
        if result is not None:effective=int(result['returncode']);record['reattached_existing_call']=True
    record.update(returncode=effective,initial_client_returncode=proc.returncode,detached=True,status='succeeded' if not effective else 'failed',
                  client_build_and_call_seconds=time.perf_counter()-start)
    (out/'attempt.json').write_text(json.dumps(record,indent=2)+'\n');(out/'retained').mkdir()
    fetch=subprocess.run(['modal','volume','get','autoresearch-tavern-device-evidence-20261108',a.run_id,
        str(out/'retained'),'--env','staging'],capture_output=True,text=True)
    (out/'retrieval.json').write_text(json.dumps({'returncode':fetch.returncode,'stderr':fetch.stderr},indent=2)+'\n')
    print(json.dumps(record),flush=True)
    if effective:raise SystemExit(effective)


if __name__=='__main__':main()
