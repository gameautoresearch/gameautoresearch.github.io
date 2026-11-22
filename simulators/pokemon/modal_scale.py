"""Newly authorized Pokemon structural campaign; historical ledger is immutable."""
import io,json,os,tarfile,time,hashlib,fcntl
from pathlib import Path
import modal
ROOT=Path(__file__).resolve().parents[2] if modal.is_local() else Path('/workspace')
SOURCE=ROOT/'evidence/pokemon-implementation/cuda-20261108'
OUT=ROOT/'evidence/pokemon-implementation/massive-cuda-20261108'
APP='autoresearch-pokemon-massive-20261108'
RATE=.000222+4*.0000131+8*.00000222
image=(modal.Image.debian_slim(python_version='3.12')
    .apt_install('git','curl','xz-utils')
    .run_commands('timeout 180s sh -c "curl -fsSL https://nodejs.org/dist/v24.18.1/node-v24.18.1-linux-x64.tar.xz | tar -xJ --strip-components=1 -C /usr/local"',
                  'timeout 600s python -m pip install torch==2.15.0 numpy==2.5.4')
    .env({'PYTHONPATH':'/workspace','CUBLAS_WORKSPACE_CONFIG':':4096:8','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'})
    .add_local_dir(SOURCE/'pinned-source/pokemon-showdown','/workspace/candidates/pokemon-showdown',copy=True,
                   ignore=['dist/**','node_modules/**','logs/*.txt','logs/repl/*.sock','logs/repl/app','logs/*.jsonl','config/config.js','**/__pycache__/**'])
    .run_commands('cd /workspace/candidates/pokemon-showdown && timeout 300s npm ci --omit=optional && timeout 120s node build')
    .add_local_dir(SOURCE/'pinned-source/poke-env','/workspace/candidates/poke-env',copy=True,
                   ignore=['**/__pycache__/**','.venv/**','dist/**'])
    .add_local_dir(ROOT/'integrations/pokemon','/workspace/integrations/pokemon',ignore=['**/__pycache__/**'])
    .add_local_file(ROOT/'evidence/pokemon-implementation/learning-verified/checkpoint.pt','/workspace/input-policy.pt')) if modal.is_local() else modal.Image.debian_slim()
artifact_volume=modal.Volume.from_name(APP+'-receipts',environment_name='staging',create_if_missing=True)
app=modal.App(APP,image=image)

@app.function(gpu='L4',cpu=(4,4),memory=(8192,8192),timeout=3600,startup_timeout=180,
              max_containers=1,min_containers=0,buffer_containers=0,scaledown_window=2,
              retries=0,single_use_containers=True,volumes={'/pokemon-receipts':artifact_volume})
def experiment(mode,label):
    started=time.perf_counter();import_started=time.perf_counter()
    import torch
    if not torch.cuda.is_available():raise RuntimeError('Actual NVIDIA CUDA is required')
    torch.cuda.init();torch.cuda.synchronize()
    cold_import_cuda_seconds=time.perf_counter()-import_started
    from integrations.pokemon import cuda_suite
    os.chdir('/workspace');out=Path('/pokemon-receipts')/label
    cuda_suite.PERSIST=artifact_volume.commit
    try:
        if mode=='wide-cli':
            from integrations.pokemon.scale_suite import cli_validation as run
        elif mode.endswith('sim'):
            from integrations.pokemon.parallel_native import run
        else:from integrations.pokemon.scale_suite import run
        result=run(out,cpus=4 if mode.startswith('small') else 32,**({'fast':True} if mode=='wide-fast' else {'frontier':True} if mode=='wide-frontier' else {}))
    except BaseException as exc:
        import traceback
        result={'error_type':type(exc).__name__,'error':str(exc),'traceback':traceback.format_exc(),'gpu_executed':True}
        out.mkdir(parents=True,exist_ok=True)
        print(result['traceback'],flush=True)
    result=json.loads(json.dumps(result,default=str))
    result['resource_limits']={'gpu':'L4','physical_cpu_request_limit':[4,4] if mode.startswith('small') else [32,32],'memory_mib_request_limit':[8192,8192] if mode.startswith('small') else [32768,32768]}
    for key,filename in [('cgroup_memory_peak_bytes','memory.peak'),('cgroup_cpu_max','cpu.max')]:
        path=Path('/sys/fs/cgroup')/filename
        if path.exists():result[key]=path.read_text().strip()
    result['cold_import_cuda_seconds']=cold_import_cuda_seconds
    result['allocated_body_seconds_before_archive']=time.perf_counter()-started
    result['estimated_body_cost_usd']=result['allocated_body_seconds_before_archive']*(RATE if mode.startswith('small') else .000222+32*.0000131+32*.00000222)
    (out/'allocation.json').write_text(json.dumps(result,indent=2))
    bio=io.BytesIO()
    with tarfile.open(fileobj=bio,mode='w:gz') as tar:
        tar.add(out,arcname='receipts',filter=lambda info: None if Path(info.name).name.startswith('checkpoint-') and Path(info.name).suffix=='.pt' else info)
    archive_bytes=bio.getvalue()
    (Path('/pokemon-receipts')/(label+'.tar.gz')).write_bytes(archive_bytes)
    artifact_volume.commit()
    return {'result':result,'artifact':archive_bytes,'body_seconds_with_archive':time.perf_counter()-started,'remote_archive':label+'.tar.gz'}

@app.function(gpu='L4',cpu=(32,32),memory=(32768,32768),timeout=3600,startup_timeout=180,
              max_containers=1,min_containers=0,buffer_containers=0,scaledown_window=2,
              retries=0,single_use_containers=True,volumes={'/pokemon-receipts':artifact_volume})
def experiment_wide(mode,label):
    return experiment.local(mode,label)

@app.local_entrypoint()
def main(mode:str='small',label:str='small-1'):
    from modal.config import config
    if config.get('environment')!='staging':raise RuntimeError('Use --env staging')
    if mode not in ('small','wide','small-sim','wide-sim','wide-fast','wide-frontier','wide-cli'):raise ValueError(mode)
    OUT.mkdir(parents=True,exist_ok=True)
    ledger_path=OUT/'attempts.json'
    with (OUT/'budget.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        ledger=json.loads(ledger_path.read_text()) if ledger_path.exists() else []
        if any(r['label']==label for r in ledger):raise ValueError('Use a unique attempt label')
        attempt={'label':label,'mode':mode,'start_unix':time.time(),'function_timeout_seconds':3600,'authorization':'User superseded previous budget/attempt caps for structural campaign','status':'reserved'}
        ledger.append(attempt);ledger_path.write_text(json.dumps(ledger,indent=2)+'\n')
    started=time.perf_counter()
    try:
        call=(experiment if mode.startswith('small') else experiment_wide).spawn(mode,label)
        attempt['function_call_id']=call.object_id
        with (OUT/'budget.lock').open('w') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            saved=json.loads(ledger_path.read_text());saved[next(i for i,r in enumerate(saved) if r['label']==label)]=attempt;ledger_path.write_text(json.dumps(saved,indent=2)+'\n')
        response=call.get(timeout=3650)
        target=OUT/label;target.mkdir(exist_ok=False)
        archive=target/'artifacts.tar.gz';archive.write_bytes(response.pop('artifact'))
        with tarfile.open(archive) as tar:tar.extractall(target,filter='data')
        response.update(client_remote_call_seconds=time.perf_counter()-started,artifact_sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
        (target/'receipt.json').write_text(json.dumps(response,indent=2)+'\n')
        attempt.update(status='completed' if 'error' not in response['result'] else 'failed_with_artifacts',body_seconds=response['body_seconds_with_archive'],client_remote_call_seconds=response['client_remote_call_seconds'],estimated_body_cost_usd=response['body_seconds_with_archive']*(RATE if mode.startswith('small') else .000222+32*.0000131+32*.00000222))
        print(json.dumps({k:v for k,v in response.items() if k!='result'},indent=2))
    except BaseException as exc:
        attempt.update(status='failed',error_type=type(exc).__name__,error=str(exc),client_remote_call_seconds=time.perf_counter()-started)
        raise
    finally:
        with (OUT/'budget.lock').open('w') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            ledger=json.loads(ledger_path.read_text());ledger[next(i for i,r in enumerate(ledger) if r['label']==label)]=attempt;ledger_path.write_text(json.dumps(ledger,indent=2)+'\n')
