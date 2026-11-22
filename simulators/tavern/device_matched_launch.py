"""Freeze a qualified device build plus independent original/native baselines."""
import argparse
import hashlib
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import time

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'evidence/tavern-implementation'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-id',required=True)
    p.add_argument('--qualified-run',required=True);p.add_argument('--phase',choices=['simulation','learning','dispatch','dispatch_debug','dispatch_watch','dispatch_scaling','dispatch_sanitizer','dispatch_initcheck'],required=True)
    p.add_argument('--sizes',default='128,1024');p.add_argument('--trials',type=int,default=2);a=p.parse_args()
    if not 0<=a.trials<=3:raise ValueError('bounded repeat count required')
    qualified=BASE/'device-port'/a.qualified_run
    retained=qualified/'retained'/a.qualified_run
    log=(qualified/'modal.log').read_text()
    proof_path=retained/'differential.json'
    proof_origin='downloaded differential.json'
    try:proof=json.loads(proof_path.read_text())
    except (ValueError,FileNotFoundError):
        # A failed bulk download can leave sparse/preallocated files. The worker
        # prints the exact complete proof before returning; retain that source
        # explicitly rather than accepting an incomplete file or rerunning it.
        candidates=[]
        for line in log.splitlines():
            try:row=json.loads(line)
            except ValueError:continue
            if isinstance(row,dict) and row.get('passed') is True and 'runtime_options' in row and 'neural' in row:
                candidates.append(row)
        if len(candidates)!=1:raise ValueError('one complete worker qualification proof required')
        proof=candidates[0];proof_origin='complete worker stdout; bulk download incomplete'
    if not proof.get('passed') or not proof['neural']['complete_neural_learning']:
        raise ValueError('complete game, neural and resume qualification must pass first')
    receipt=json.loads((qualified/'compiler-receipt/build.json').read_text())
    images=re.findall(r'Built image (im-[A-Za-z0-9]+)',log)
    if not images:raise ValueError('qualified immutable image ID missing')
    base_image=images[-1]
    # Reuse the immutable qualified image. Module hashes come from the actual
    # successful device loads, and are checked again inside the next worker.
    # The independent full retention download/audit may finish concurrently.
    loaded=[]
    for line in log.splitlines():
        try:row=json.loads(line)
        except (ValueError,TypeError):continue
        if isinstance(row,dict) and 'limits' in row and 'modules' in row:loaded.append(row['modules'])
    if not loaded or any(row!=loaded[0] for row in loaded):raise ValueError('consistent qualified device module receipts required')
    modules={}
    for family,stem in [('main','tavern'),('state','tavern-state')]:
        row=loaded[0][family]
        if row['format']!='.cubin':raise ValueError('qualified native cubin required')
        modules[stem+'.ptx']=row['ptx_sha256'];modules[stem+'.cubin']=row['module_sha256']
    for name,row in receipt['modules'].items():
        if modules[name]!=row['sha256']:raise ValueError('loaded PTX differs from source-bound build')
    out=BASE/'device-port'/a.run_id;out.mkdir(parents=True,exist_ok=False);payload=out/'payload'
    shutil.copytree(qualified/'compiler-receipt',out/'compiler-receipt')
    ignore=shutil.ignore_patterns('target','target-*','__pycache__','*.so','*.dylib','*.pyc','*.ptx','*.cubin')
    for label in ['upstream','earliest']:
        shutil.copytree(BASE/'original-baselines-20261108'/label,payload/label,ignore=ignore)
    for name in ['upstream-source.json','earliest-source.json']:
        shutil.copyfile(BASE/'original-baselines-20261108'/name,payload/name)
    current=payload/'integrations/tavern'
    shutil.copytree(ROOT/'integrations/tavern',current,ignore=ignore)
    # The simulator and learning gate use the exact qualified frozen source.
    shutil.rmtree(current/'device')
    shutil.copytree(qualified/'payload/integrations/tavern/device',current/'device',ignore=ignore)
    frozen_python=['model.py','learner_graph.py','rollout.py']
    if not a.phase.startswith('dispatch'):frozen_python+=['device_env.py','device_training.py']
    for name in frozen_python:
        shutil.copyfile(qualified/'payload/integrations/tavern'/name,current/name)
    (out/'qualification-proof.json').write_text(json.dumps(proof,indent=2)+'\n')
    qualification={'passed':True,'qualified_run':a.qualified_run,'base_image_id':base_image,'proof_sha256':sha(out/'qualification-proof.json'),'proof_origin':proof_origin,
                   'module_sha256':modules,'proof':proof,'compiler_receipt':receipt,
                   'block_threads':proof.get('runtime_options',{}).get('block_threads',32),
                   'experimental_dispatch_runtime_requires_full_requalification':a.phase.startswith('dispatch')}
    (payload/'qualification.json').write_text(json.dumps(qualification,indent=2)+'\n')
    manifest={str(path.relative_to(payload)):sha(path) for path in sorted(payload.rglob('*')) if path.is_file()}
    (out/'payload-sha256.json').write_text(json.dumps(manifest,indent=2)+'\n')
    record={'run_id':a.run_id,'qualified_run':a.qualified_run,'phase':a.phase,'sizes':a.sizes,'trials':a.trials,
        'start_unix':time.time(),'status':'started','resources':{'gpu':'B200','cpu_cores':4,'ram_gib':16,
        'timeout_seconds':3600,'retries':0,'warm_containers':0},
        'authorization':'Complete GPU rewrite with original and strongest native comparisons; prior caps superseded. All prior failures/ledgers preserved.',
        'setup_accounting':'Qualified compile/assembly/image setup receipts remain charged separately; full allocated worker time is also retained.'}
    (out/'attempt.json').write_text(json.dumps(record,indent=2)+'\n');start=time.perf_counter()
    with (out/'modal.log').open('x') as stream:
        proc=subprocess.run(['modal','run','--detach','--env','staging',str(current/'device_matched_modal.py'),
            '--run-id',a.run_id,'--phase',a.phase,'--sizes',a.sizes,'--trials',str(a.trials)],
            env=dict(os.environ,TAVERN_MATCHED_PAYLOAD=str(payload),TAVERN_MATCHED_BASE_IMAGE=base_image,TAVERN_MATCHED_PHASE=a.phase),stdout=stream,stderr=subprocess.STDOUT,timeout=7200)
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
