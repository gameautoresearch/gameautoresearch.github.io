"""Matched fixed-resource historical-baseline qualification; independent source installs."""
from pathlib import Path
import json,os,modal
ROOT=Path(__file__).resolve().parents[2] if modal.is_local() else Path('/workspace')
PAYLOAD=Path(os.environ.get('TAVERN_ORIGINAL_PAYLOAD',str(ROOT)))
VOLUME='autoresearch-tavern-original-evidence-20261108'
if modal.is_local():
    image=(modal.Image.debian_slim(python_version='3.12')
     .apt_install('curl','build-essential','git','ca-certificates')
     .pip_install('torch==2.9.1','numpy==2.3.5','gymnasium==1.2.2','maturin==1.10.2','pytest==9.0.2')
     .run_commands('curl --proto =https --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain 1.98.0')
     .env({'PATH':'/root/.cargo/bin:/usr/local/bin:/usr/bin:/bin','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','CUBLAS_WORKSPACE_CONFIG':':4096:8'})
     .add_local_dir(PAYLOAD,'/workspace',copy=True,ignore=['**/target/**','**/__pycache__/**','**/*.so','**/*.dylib','**/*.pyc'])
     .run_commands('python -m venv --system-site-packages /venvs/upstream && /venvs/upstream/bin/python -m pip install --no-deps --no-build-isolation --editable /workspace/upstream/candidates/AutoTavern',
                   'python -m venv --system-site-packages /venvs/earliest && /venvs/earliest/bin/python -m pip install --no-deps --no-build-isolation --editable /workspace/earliest/candidates/AutoTavern',
                   'python -m venv --system-site-packages /venvs/candidate && /venvs/candidate/bin/python -m pip install --no-deps --no-build-isolation --editable /workspace/candidate/candidates/AutoTavern'))
else:image=modal.Image.debian_slim(python_version='3.12')
app=modal.App('autoresearch-tavern-original-baselines-20261108',image=image)
volume=modal.Volume.from_name(VOLUME,create_if_missing=True)


@app.function(gpu='L4',cpu=(4,4),memory=(16384,16384),timeout=3600,startup_timeout=180,
    max_containers=1,min_containers=0,buffer_containers=0,retries=0,scaledown_window=2,
    single_use_containers=True,volumes={'/results':volume})
def compare(run_id:str):
    import subprocess,sys,time,traceback
    start=time.perf_counter();out=Path('/results')/run_id
    try:
        p=subprocess.run([sys.executable,'-u','/workspace/candidate/integrations/tavern/original_baselines.py',str(out)],timeout=3500)
        return {'returncode':p.returncode,'run_id':run_id}
    except BaseException:
        out.mkdir(parents=True,exist_ok=True);(out/'failure.txt').write_text(traceback.format_exc());raise
    finally:
        out.mkdir(parents=True,exist_ok=True)
        (out/'allocation.json').write_text(json.dumps({'worker_seconds':time.perf_counter()-start,
            'gpu':'L4','physical_cpu_cores':4,'ram_gib':16,'retries':0,'warm_containers':0})+'\n')
        volume.commit()


@app.local_entrypoint()
def main(run_id:str):
    r=compare.remote(run_id);print(json.dumps(r),flush=True)
    if r['returncode']:raise RuntimeError('baseline qualification failed; retain all evidence')
