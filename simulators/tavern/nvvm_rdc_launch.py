"""Freeze and account a CPU-only alternate compiler experiment."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

ROOT=Path(__file__).resolve().parents[2]


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);a=p.parse_args()
    out=ROOT/'evidence/tavern-implementation/device-port'/a.run_id
    out.mkdir(exist_ok=False);payload=out/'payload'
    shutil.copytree(ROOT/'integrations/tavern',payload/'integrations/tavern',
        ignore=shutil.ignore_patterns('target','target-*','__pycache__','*.so','*.dylib','*.ptx','*.cubin'))
    manifest={str(p.relative_to(payload)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(payload.rglob('*')) if p.is_file()}
    (out/'payload-sha256.json').write_text(json.dumps(manifest,indent=2)+'\n')
    started=time.perf_counter();record={'run_id':a.run_id,'status':'started','start_unix':time.time(),
        'gpu_executed':False,'compiler':'pinned patched Rust-CUDA/NVVM; source-bound RDC CPU build',
        'resources':{'cpu_cores':4,'ram_gib':16,'retries':0,'warm_containers':0},
        'setup_accounting':'All image build, compiler, assembly, transfer and retain worker costs remain charged; no GPU speed claim.'}
    (out/'attempt.json').write_text(json.dumps(record,indent=2)+'\n')
    with (out/'modal.log').open('x') as stream:
        proc=subprocess.run(['modal','run','--detach','--env','staging',
            str(payload/'integrations/tavern/nvvm_rdc_modal.py'),'--run-id',a.run_id],
            env=dict(os.environ,TAVERN_NVVM_PAYLOAD=str(payload)),stdout=stream,stderr=subprocess.STDOUT,timeout=3600)
    record.update(returncode=proc.returncode,status='succeeded_build_only' if not proc.returncode else 'failed',
        client_build_and_call_seconds=time.perf_counter()-started)
    (out/'attempt.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record),flush=True)
    if proc.returncode:raise SystemExit(proc.returncode)


if __name__=='__main__':main()
