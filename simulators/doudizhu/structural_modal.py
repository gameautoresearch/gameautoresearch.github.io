"""New user-authorized structural campaign; immutable per-invocation receipts.

Historical modal_gpu.py ledger and exhausted attempt scope are left untouched.
One finite single-use L4 worker; no warm pool or retries. Tracking is accounting,
not reuse/reset of the old campaign allowance.
"""
from pathlib import Path
import io
import json
import tarfile
import time
import uuid
import modal

HERE=Path(__file__).resolve().parent
if not (HERE/'requirements-universal-lock.txt').exists():
    HERE=Path('/opt/arg/integrations/doudizhu')
ROOT=HERE.parents[1]
app=modal.App('arg-doudizhu-structural-20261108')
image=(modal.Image.debian_slim(python_version='3.12.9').apt_install('git','g++')
       .pip_install_from_requirements(str(HERE/'requirements-universal-lock.txt'),extra_options='--require-hashes')
       .env({'PYTHONPATH':'/opt/arg','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1',
             'CUBLAS_WORKSPACE_CONFIG':':4096:8'})
       .run_commands('mkdir -p /opt/arg/candidates',
                     'git clone https://github.com/kwai/DouZero.git /opt/arg/candidates/DouZero',
                     'git -C /opt/arg/candidates/DouZero checkout --detach 718a5c920bf3361e34178a38f3b80458e176b351')
       .add_local_dir(HERE,'/opt/arg/integrations/doudizhu',ignore=['**/__pycache__/**']))


@app.function(image=image,gpu='L4',cpu=(4,4),memory=(16384,16384),
              max_containers=1,min_containers=0,buffer_containers=0,scaledown_window=2,
              retries=0,timeout=1800,startup_timeout=600,single_use_containers=True)
def experiment(run_id: str, small_checkpoint: bytes, mode: str='structural'):
    import os
    import shutil
    import subprocess
    import sys
    started=time.time()
    base=Path('/opt/arg/evidence/doudizhu-implementation')/run_id
    base.mkdir(parents=True,exist_ok=False)
    shutil.copytree('/opt/arg/integrations/doudizhu',base/'source',
                    ignore=shutil.ignore_patterns('__pycache__'))
    (base/'trained-small.pt').write_bytes(small_checkpoint)
    Path('/opt/arg/evidence/local-performance.lock').touch()
    receipt={'run_id':run_id,'allocated_start_unix':started,'task_id':os.environ.get('MODAL_TASK_ID'),
             'allocation':{'gpu':'L4','gpus':1,'cpu_cores':4,'ram_gib':16},'commands':[],'mode':mode}
    status=0
    try:
        if mode in ('learning','frontier-learning'):
            from integrations.doudizhu.learning_campaign import run
            run(base,mode)
            receipt['exact_parity']=True
            # The outer finally-equivalent accounting/packaging also covers failures.
            raise StopIteration('learning_complete')
        if mode in ('validation','native-validation'):
            command=[sys.executable,'integrations/doudizhu/cuda_validation.py',str(base)]
            if mode=='native-validation':
                command+=['--native']
            receipt['commands'].append(command)
            subprocess.run(command,cwd='/opt/arg',check=True,timeout=900)
            receipt['exact_parity']=True
            raise StopIteration('learning_complete')
        config=json.loads(Path('/opt/arg/integrations/doudizhu/recipes/validated-l4-smoke.json').read_text())
        config.update(architecture={'hidden':512,'depth':5,'history':128},max_episodes=1024,
                      max_seconds=450,workers=0,vector_games=64,engine_backend='view',encoding_backend='direct')
        config.pop('budget_gpu_hours',None)
        (base/'full-train.json').write_text(json.dumps(config,indent=2)+'\n')
        command=[sys.executable,'integrations/doudizhu/launch.py','train','--recipe',str(base/'full-train.json'),
                 '--out',str(base/'full-train')]
        receipt['commands'].append(command)
        subprocess.run(command,cwd='/opt/arg',check=True,timeout=480)
        rows=[]
        models=[('full',base/'full-train/latest.pt'),('trained-small',base/'trained-small.pt')]
        if mode in ('native','native-bulk','fixed-scaling'):
            models=models[:1]
        for label,checkpoint in models:
            for repeat in range(3):
                variants=[('upstream','upstream',0,False,'eager',False),('view','upstream',0,False,'eager',False),
                          ('view','direct',0,False,'eager',False),('view','direct',2,False,'eager',False),
                          ('view','direct',4,False,'eager',False)]
                if mode=='graph':
                    variants=[('view','direct',0,False,'eager',False),('view','direct',0,True,'eager',False),
                              ('view','direct',0,True,'graph',False)]
                if mode=='packed':
                    variants=[('view','direct',0,False,'eager',False),('view','direct',4,False,'eager',False),
                              ('view','direct',4,False,'eager',True),('view','direct',4,False,'graph',True)]
                if mode in ('native','native-bulk','fixed-scaling'):
                    # Batch scaling is an explicit recipe change. Compare source-
                    # preserving variants only within the same batch size.
                    variants=[('upstream','upstream',0,64,'eager',False),
                              ('native','direct',1,64,'eager',False),
                              ('native','direct',4,64,'eager',False),
                              ('native','direct',4,256,'eager',False),
                              ('native','direct',4,1024,'eager',False),
                              ('native','direct',4,2048,'eager',False)]
                if repeat%2:
                    variants.reverse()
                for engine,codec,workers,forced,inference,compact in variants:
                    target=base/f'{label}-{repeat}-{engine}-{codec}-w{workers}-f{int(forced)}-{inference}-p{int(compact)}'
                    command=[sys.executable,'integrations/doudizhu/structural_profile.py','--device','cuda:0',
                             '--checkpoint',str(checkpoint),'--games',{'graph':'2048','packed':'1024','native':'2048','native-bulk':'2048','fixed-scaling':'2048'}.get(mode,'256'),
                             '--batch',str(forced) if mode in ('native','native-bulk','fixed-scaling') else ('512' if mode=='packed' else '64'),
                             '--workers',str(workers),'--inference',inference,
                             '--engine',engine,'--codec',codec,'--out',str(target)]
                    if forced and mode not in ('native','native-bulk','fixed-scaling'):
                        command+=['--skip-forced']
                    if mode in ('native','native-bulk','fixed-scaling'):
                        command+=['--record-first-only','--record-games','4']
                    if mode in ('native-bulk','fixed-scaling') and engine=='native':
                        command+=['--native-fast']
                    if mode=='fixed-scaling':
                        command+=['--fixed-policy']
                    if compact:
                        command+=['--compact-workers']
                    receipt['commands'].append(command)
                    subprocess.run(command,cwd='/opt/arg',check=True,timeout=240)
                    row=json.loads((target/'result.json').read_text())
                    row.update(label=label,repeat=repeat)
                    rows.append(row)
                    (base/'matrix.json').write_text(json.dumps(rows,indent=2)+'\n')
        for label,_ in models:
            group=[r for r in rows if r['label']==label]
            if mode in ('native','native-bulk'):
                group=[r for r in group if r['batch']==64]
            for key in ('trajectory_sha256','samples_sha256','weights_sha256','evaluation_sha256'):
                if len({r[key] for r in group})!=1:
                    raise AssertionError(f'Exact {label} parity failed: {key}; all receipts retained')
        receipt['exact_parity']=True
    except StopIteration as exc:
        if str(exc)!='learning_complete':
            status=1
            receipt.update(error_type=type(exc).__name__,error=str(exc))
    except BaseException as exc:
        status=1
        receipt.update(error_type=type(exc).__name__,error=str(exc))
    receipt.update(status=status,allocated_end_unix=time.time())
    receipt['allocated_worker_seconds']=receipt['allocated_end_unix']-started
    receipt['allocated_gpu_hours']=receipt['allocated_worker_seconds']/3600
    print(json.dumps({'remote_complete':run_id,'mode':mode,'status':status,
        'exact_parity':receipt.get('exact_parity',False),
        'body_allocated_worker_seconds':receipt['allocated_worker_seconds']}),flush=True)
    # Preserve one resulting checkpoint per variant/model plus all initial training
    # checkpoints. Repeated identical weights are certified by the per-trial hashes.
    omitted=[]
    for target in base.glob('*-*/latest.pt'):
        if target.parent.name.startswith(('full-1-','full-2-','trained-small-1-','trained-small-2-')):
            omitted.append(str(target.relative_to(base)))
            target.unlink()
    receipt['duplicate_repeat_checkpoints_omitted']=omitted
    (base/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    buffer=io.BytesIO()
    with tarfile.open(fileobj=buffer,mode='w:gz') as archive:
        archive.add(base,arcname=run_id)
    return {'receipt':receipt,'archive':buffer.getvalue()}


@app.local_entrypoint()
def main(mode: str='structural'):
    if mode not in ('structural','graph','learning','packed','validation','native','native-bulk','frontier-learning','native-validation','fixed-scaling'):
        raise ValueError('Unknown finite experiment matrix')
    run_id='structural-'+mode+'-gpu-'+time.strftime('%Y%m%dT%H%M%S',time.gmtime())+'-'+uuid.uuid4().hex[:6]
    base=ROOT/'evidence/doudizhu-implementation'
    attempt=base/(run_id+'-client')
    attempt.mkdir(parents=True,exist_ok=False)
    started=time.time()
    (attempt/'authorization.json').write_text(json.dumps({
        'new_campaign':True,'authorization':'User explicitly superseded old campaign dollar/hour/attempt gates',
        'historical_ledgers_modified':False,'submitted_unix':started,
        'resource_bound':'one L4, 4 physical CPUs, 16GiB; single-use, 1800s timeout, retries=0'},indent=2)+'\n')
    try:
        call=experiment.spawn(run_id,(base/'gpu/followup/learning-seed-17/latest.pt').read_bytes(),mode)
        (attempt/'provider-call.json').write_text(json.dumps({'function_call_id':call.object_id,
            'submitted_unix':time.time(),'run_id':run_id},indent=2)+'\n')
        result=call.get()
        (attempt/'results.tar.gz').write_bytes(result['archive'])
        with tarfile.open(fileobj=io.BytesIO(result['archive'])) as archive:
            archive.extractall(base,filter='data')
        (attempt/'receipt.json').write_text(json.dumps(result['receipt'],indent=2)+'\n')
        print(json.dumps(result['receipt']),flush=True)
    finally:
        (attempt/'client-lifetime.json').write_text(json.dumps({'started_unix':started,'ended_unix':time.time(),
            'roundtrip_seconds':time.time()-started,'scope':'client wall time; obtain app lifecycle for allocation charge'},indent=2)+'\n')
