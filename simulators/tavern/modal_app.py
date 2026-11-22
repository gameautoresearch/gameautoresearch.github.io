"""One dedicated staged L4, fixed CPU/RAM, no idle pool/retries; use launch_gpu.py."""
import json
import os
from pathlib import Path
import modal

ROOT=Path(__file__).resolve().parents[2] if modal.is_local() else Path('/workspace')
PAYLOAD=Path(os.environ.get('TAVERN_GPU_PAYLOAD',str(ROOT)))
APP_NAME='autoresearch-tavern-cuda-20261108'
VOLUME_NAME='autoresearch-tavern-cuda-evidence-20261108'
if modal.is_local():
    image=(modal.Image.debian_slim(python_version='3.12')
     .apt_install('curl','build-essential','git','ca-certificates')
     .pip_install('torch==2.9.1','numpy==2.3.5','gymnasium==1.2.2','maturin==1.10.2','pytest==9.0.2')
     .run_commands('curl --proto =https --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain 1.98.0')
     .env({'PATH':'/root/.cargo/bin:/usr/local/bin:/usr/bin:/bin','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','CUBLAS_WORKSPACE_CONFIG':':4096:8'})
     .add_local_dir(PAYLOAD/'candidates/AutoTavern','/workspace/candidates/AutoTavern',copy=True,
          ignore=['target/**','.git/**','**/__pycache__/**','**/*.so','**/*.dylib','**/*.pyc'])
     .run_commands('cd /workspace/candidates/AutoTavern && python -m pip install --no-build-isolation --editable .')
     .add_local_dir(PAYLOAD/'integrations/tavern','/workspace/integrations/tavern',copy=True,
          ignore=['**/__pycache__/**','**/*.pyc'])
     .add_local_file(ROOT/'evidence/tavern-implementation/learning-64/latest.pt','/workspace/pilot.pt',copy=True))
else:
    image=modal.Image.debian_slim(python_version='3.12')
app=modal.App(APP_NAME,image=image)
volume=modal.Volume.from_name(VOLUME_NAME,create_if_missing=True)


@app.function(gpu='L4',cpu=(4,4),memory=(16384,16384),timeout=900,startup_timeout=180,
              max_containers=1,min_containers=0,buffer_containers=0,scaledown_window=2,
              retries=0,volumes={'/results':volume},single_use_containers=True)
def experiment(phase:str,run_id:str,seed:int=12345):
    import subprocess,sys,time,traceback
    from pathlib import Path
    started=time.perf_counter()
    out=Path('/results')/run_id
    try:
        proc=subprocess.run([sys.executable,'-u','/workspace/integrations/tavern/gpu_suite.py',
            '--phase',phase,'--out',str(out),'--seed',str(seed)],timeout=850)
        return {'returncode':proc.returncode,'run_id':run_id,'worker_seconds':time.perf_counter()-started}
    except BaseException as exc:
        out.mkdir(parents=True,exist_ok=True)
        (out/'worker-error.txt').write_text(traceback.format_exc())
        return {'returncode':1,'run_id':run_id,'worker_seconds':time.perf_counter()-started,'error':repr(exc)}
    finally:
        out.mkdir(parents=True,exist_ok=True)
        (out/'allocation.json').write_text(json.dumps({'worker_seconds':time.perf_counter()-started,
            'gpu':'L4','cpu_physical_limit':4,'ram_GiB_limit':16,'price_per_second':.00030992,
            'startup_and_shutdown_reserve_seconds':182,'app':APP_NAME})+'\n')
        volume.commit()


@app.local_entrypoint()
def main(phase:str='baseline',run_id:str='baseline-v1',seed:int=12345):
    from modal.config import config
    if config.get('environment')!='staging':raise RuntimeError('staging only')
    result=experiment.remote(phase,run_id,seed)
    print(json.dumps(result),flush=True)
    if result['returncode']:raise RuntimeError('Remote experiment failed; preserve charged ledger and retrieve error evidence')
