"""Finite resident-simulator experiments, durable private artifacts, compact RPCs.

No historical ledger is reset. One L4 + four physical CPUs +16GiB per invocation.
Full weights/source/replays live on the game-owned staging Volume; download only
needed artifacts, rather than serializing hundreds of MB in a Function response.
"""
from pathlib import Path
import json
import time
import uuid
import modal

HERE=Path(__file__).resolve().parent
if not (HERE/'requirements-universal-lock.txt').exists():HERE=Path('/opt/arg/integrations/doudizhu')
ROOT=HERE.parents[1]
NAME='doudizhu-structural-evidence-20261108'
volume=modal.Volume.from_name(NAME,environment_name='staging',create_if_missing=True)
app=modal.App('arg-doudizhu-resident-20261108')
EXACT_PYTHON='/opt/exact-python/cpython-3.12.9-linux-x86_64-gnu/bin/python3.12'
image=(modal.Image.from_registry('nvidia/cuda:12.8.1-devel-ubuntu24.04',add_python='3.12')
       .apt_install('git','g++')
       .pip_install('uv==0.9.25')
       .add_local_file(HERE/'requirements-universal-lock.txt','/tmp/doudizhu-requirements.txt',copy=True)
       .run_commands('uv python install 3.12.9 --install-dir /opt/exact-python',
                     f'uv pip install --python {EXACT_PYTHON} --require-hashes -r /tmp/doudizhu-requirements.txt')
       .env({'PYTHONPATH':'/opt/arg','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1',
             'CUBLAS_WORKSPACE_CONFIG':':4096:8'})
       .run_commands('mkdir -p /opt/arg/candidates',
                     'git clone https://github.com/kwai/DouZero.git /opt/arg/candidates/DouZero',
                     'git -C /opt/arg/candidates/DouZero checkout --detach 718a5c920bf3361e34178a38f3b80458e176b351'))
original_archive=ROOT/'evidence/doudizhu-implementation/structural-20261108/original-integration/source.tar'
if original_archive.exists():
    image=image.add_local_file(original_archive,'/tmp/original-doudizhu.tar',copy=True)
    image=image.add_local_file(original_archive.with_name('manifest.json'),'/tmp/original-doudizhu-manifest.json',copy=True)
image=image.add_local_dir(HERE,'/opt/arg/integrations/doudizhu',ignore=['**/__pycache__/**'])


@app.function(image=image,gpu='L4',cpu=(4,4),memory=(16384,16384),volumes={'/retained':volume},
              max_containers=1,min_containers=0,buffer_containers=0,scaledown_window=2,
              retries=0,timeout=3600,startup_timeout=900,single_use_containers=True)
def experiment(run_id:str,stage:str):
    import os
    import shutil
    import subprocess
    import sys
    import tarfile
    started=time.time()
    out=Path('/retained')/run_id
    out.mkdir(parents=True,exist_ok=False)
    shutil.copytree('/opt/arg/integrations/doudizhu',out/'source',ignore=shutil.ignore_patterns('__pycache__'))
    with tarfile.open(out/'source.tar.gz','w:gz') as archive:archive.add(out/'source',arcname='source')
    command=[EXACT_PYTHON,'-u','integrations/doudizhu/cuda/experiment.py','--stage',stage,'--out',str(out)]
    status=subprocess.call(command,cwd='/opt/arg')
    for folder in ('cuda-build','cuda-build-cpu','native-build'):
        path=Path('/opt/arg/evidence/doudizhu-implementation')/folder
        if path.exists():shutil.copytree(path,out/folder)
    receipt={'run_id':run_id,'stage':stage,'status':status,'command':command,'allocated_start_unix':started,
             'work_end_unix':time.time(),'task_id':os.environ.get('MODAL_TASK_ID'),
             'allocation':{'gpu':'L4','gpus':1,'cpu_cores':4,'ram_gib':16},
             'retained_volume':NAME,'retained_path':'/'+run_id,'automatic_retries':0,
             'accounting':'full allocated function including cold imports, JIT compile, simulation, transfers, inference, learner, checkpoints and Volume commit; app startup/teardown separately retained'}
    (out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    volume.commit()
    receipt.update(allocated_end_unix=time.time())
    receipt['allocated_worker_seconds']=receipt['allocated_end_unix']-started
    receipt['allocated_gpu_hours']=receipt['allocated_worker_seconds']/3600
    (out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    volume.commit()
    print(json.dumps({'resident_complete':run_id,'status':status,'seconds':receipt['allocated_worker_seconds']}),flush=True)
    metadata=[name for name in ('receipt.json','result.json','matrix.json','simulation-matrix.json','validation.json','learning-summary-compact.json','precision-summary.json','baseline-matrix.json','historical-summary.json','original-source.json','operators.txt','final-learning.json','source.tar.gz') if (out/name).exists()]
    return {'receipt':receipt,'metadata':metadata}


@app.local_entrypoint()
def main(stage:str='validate'):
    if stage not in ('validate','profile','learning','precision','historical','long-matched','sim-original','scalar-matched','final'):
        raise ValueError('Choose a finite resident validation/profile/learning matrix')
    run_id='cuda-'+stage+'-'+time.strftime('%Y%m%dT%H%M%S',time.gmtime())+'-'+uuid.uuid4().hex[:6]
    out=ROOT/'evidence/doudizhu-implementation'/run_id
    out.mkdir(parents=True,exist_ok=False)
    start=time.time()
    (out/'authorization.json').write_text(json.dumps({'submitted_unix':start,'new_campaign':True,
        'authorization':'User explicitly requested full native/CUDA rewrite and GPU-resident training; old dollar/hour/attempt caps superseded',
        'historical_ledgers_modified':False},indent=2)+'\n')
    call=experiment.spawn(run_id,stage)
    (out/'provider-call.json').write_text(json.dumps({'function_call_id':call.object_id,'run_id':run_id},indent=2)+'\n')
    try:
        result=call.get()
        for name in result['metadata']:
            with (out/name).open('wb') as f:
                for chunk in volume.read_file('/'+run_id+'/'+name):f.write(chunk)
        print(json.dumps(result['receipt']),flush=True)
    finally:
        (out/'client-lifetime.json').write_text(json.dumps({'started_unix':start,'ended_unix':time.time(),
            'roundtrip_seconds':time.time()-start,'scope':'client elapsed; use app/container receipts for full allocation accounting'},indent=2)+'\n')
