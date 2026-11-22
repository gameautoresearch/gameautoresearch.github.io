"""Dedicated bounded staging CUDA worker. Never touches another game's apps/jobs.

modal run --env staging integrations/generals/modal_efficiency.py --tag probe01 --stage probe
"""
from pathlib import Path
import json
import os
import time
import modal
ROOT=Path(__file__).resolve().parents[2] if modal.is_local() else Path('/workspace')
if os.environ.get('MODAL_ENVIRONMENT','staging')!='staging':raise RuntimeError('staging only')
APP_NAME='autoresearch-generals-efficiency-20261108'
RATE_SECOND=.000222+4*.0000131+16*.00000222
HARD_TIMEOUT=900
RESERVED_GPU_SECONDS=1140  # includes bounded startup and shutdown margin
image=(modal.Image.debian_slim(python_version='3.12').apt_install('git')
    .pip_install('jax[cuda12]==0.12.0.1','flax==0.12.12','optax==0.2.8','numpy==2.5.4','pygame==2.6.1','python-socketio[client]==5.18.1','pytest==9.2.0','psutil==7.2.2')
    .env({'PYTHONPATH':'/workspace:/workspace/candidates/generals-bots','JAX_PLATFORMS':'cuda','JAX_DEFAULT_MATMUL_PRECISION':'highest','XLA_PYTHON_CLIENT_PREALLOCATE':'false',
          'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4'}))
if modal.is_local():
    image=(image.add_local_dir(ROOT/'candidates/generals-bots','/workspace/candidates/generals-bots',ignore=['**/__pycache__/**','**/.pytest_cache/**'])
    .add_local_dir(ROOT/'integrations/generals','/workspace/integrations/generals',ignore=['**/__pycache__/**','**/.pytest_cache/**','autoresearch/**'])
    .add_local_file(ROOT/'evidence/generals-implementation/validated-smoke/latest.msgpack','/inputs/smoke.msgpack')
    .add_local_file(ROOT/'evidence/generals-implementation/validated-smoke/latest.msgpack.json','/inputs/smoke.msgpack.json'))
app=modal.App(APP_NAME)


@app.function(image=image,gpu='L4',cpu=(4,4),memory=(16384,16384),max_containers=1,min_containers=0,
              buffer_containers=0,retries=0,timeout=HARD_TIMEOUT,startup_timeout=180,scaledown_window=2,single_use_containers=True)
def worker(stage:str,batches:str,steps:int,seeds:str):
    import subprocess,sys,tarfile,io,time
    start=time.monotonic()
    output=Path('/artifacts');output.mkdir(exist_ok=True)
    command=[sys.executable,'-m','integrations.generals.cuda_experiment','--stage',stage,'--batches',batches,
             '--steps',str(steps),'--seeds',seeds,'--snapshot','/inputs/smoke.msgpack','--output',str(output)]
    worker_env=dict(os.environ,GENERALS_ALLOCATION_START_UNIX=str(time.time()),GENERALS_ALLOCATION_CONFIRMED='1')
    proc=subprocess.Popen(command,cwd='/workspace',stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,env=worker_env)
    with (output/'worker.log').open('w') as log:
        for line in proc.stdout:
            print(line,end='',flush=True);log.write(line);log.flush()
        code=proc.wait()
    receipt=dict(stage=stage,function_seconds=time.monotonic()-start,rate_usd_per_second=RATE_SECOND,
                 measured_function_estimate_usd=(time.monotonic()-start)*RATE_SECOND,exit_code=code,
                 allocation={'gpu':'L4','gpu_count':1,'cpu_physical_cores':4,'ram_gib':16},
                 scope='Measured function elapsed, not a provider invoice. Client reservation also charges startup/shutdown/build/egress conservatively.')
    (output/'provider-function.json').write_text(json.dumps(receipt,indent=2)+'\n')
    data=io.BytesIO()
    with tarfile.open(fileobj=data,mode='w:gz') as tar:
        for file in output.rglob('*'):
            if file.is_file():tar.add(file,arcname=str(file.relative_to(output)))
    return data.getvalue(),receipt


@app.local_entrypoint()
def main(tag:str,stage:str='probe',batches:str='16,64,128',steps:int=96,seeds:str='42,43,44'):
    import tarfile,io,hashlib
    from modal.config import config
    if config.get('environment')!='staging':raise RuntimeError('Pass --env staging')
    if not tag.replace('-','').isalnum() or stage not in ('probe','compare','learning','cli') or not 1<=steps<=256:
        raise ValueError('bounded tag/stage/steps required')
    if any(int(b) not in (16,32,64,128,256) for b in batches.split(',')) or len(seeds.split(','))>3:
        raise ValueError('batch/seed matrix too large')
    folder=ROOT/'evidence/generals-implementation/cuda'
    folder.mkdir(parents=True,exist_ok=True)
    import fcntl
    launch_lock=(folder/'launch.lock').open('a+')
    fcntl.flock(launch_lock,fcntl.LOCK_EX)
    ledger_path=folder/'budget.json'
    ledger=json.loads(ledger_path.read_text()) if ledger_path.exists() else dict(cap_usd=12.,gpu_seconds_cap=7200,runs=[])
    if any(r['tag']==tag for r in ledger['runs']):raise ValueError('Tag already reserved; inspect its state instead of resubmitting')
    # Reserve full timeouts for failures too, plus $0.50/run for CPU image builds
    # and transfer. This is intentionally above measured provider estimates.
    reserve=RESERVED_GPU_SECONDS*RATE_SECOND+.50
    if len(ledger['runs'])>=8 or sum(r['reserved_usd'] for r in ledger['runs'])+reserve>12 or sum(r['reserved_gpu_seconds'] for r in ledger['runs'])+RESERVED_GPU_SECONDS>7200:
        raise RuntimeError('Generals exclusive allocation exhausted')
    row=dict(tag=tag,stage=stage,app_id=app.app_id,reserved_usd=reserve,reserved_gpu_seconds=RESERVED_GPU_SECONDS,
             rate_usd_per_second=RATE_SECOND,status='reserved',start_unix=time.time())
    ledger['runs'].append(row);ledger_path.write_text(json.dumps(ledger,indent=2)+'\n')
    start=time.monotonic()
    try:
        call=worker.spawn(stage,batches,steps,seeds)
        row.update(status='submitted',call_id=call.object_id)
        ledger_path.write_text(json.dumps(ledger,indent=2)+'\n')
        binary,receipt=call.get()
        dest=folder/tag;dest.mkdir(exist_ok=False)
        (dest/'artifacts.tar.gz').write_bytes(binary)
        with tarfile.open(fileobj=io.BytesIO(binary),mode='r:gz') as tar:
            tar.extractall(dest,filter='data')
        row.update(status='succeeded' if receipt['exit_code']==0 else 'failed',receipt=receipt,
                   client_call_seconds=time.monotonic()-start,archive_bytes=len(binary),archive_sha256=hashlib.sha256(binary).hexdigest())
        print(json.dumps(row,indent=2),flush=True)
    except BaseException as exc:
        row.update(status='client_error_inspect_call_before_retry',error_type=type(exc).__name__,client_call_seconds=time.monotonic()-start)
        raise
    finally:
        ledger_path.write_text(json.dumps(ledger,indent=2)+'\n')
