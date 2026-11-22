"""Authorized bounded L4 experiments: game cap $12 / 2 allocated GPU-hours.

modal run --env staging integrations/doudizhu/modal_gpu.py --stage probe
No deployments, warm pool, automatic retries, or other applications are modified.
"""
from pathlib import Path
import json
import time
import modal

HERE = Path(__file__).resolve().parent
if not (HERE/'requirements-universal-lock.txt').exists():
    HERE = Path('/opt/arg/integrations/doudizhu')
ROOT = HERE.parents[1]
app = modal.App('arg-doudizhu-l4-research-20261108')
image = (modal.Image.debian_slim(python_version='3.12.9')
         .apt_install('git')
         .pip_install_from_requirements(str(HERE/'requirements-universal-lock.txt'), extra_options='--require-hashes')
         .env({'PYTHONPATH':'/opt/arg','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1',
               'OPENBLAS_NUM_THREADS':'1','CUBLAS_WORKSPACE_CONFIG':':4096:8'})
         .run_commands('mkdir -p /opt/arg/candidates',
                       'git clone https://github.com/kwai/DouZero.git /opt/arg/candidates/DouZero',
                       'git -C /opt/arg/candidates/DouZero checkout --detach 718a5c920bf3361e34178a38f3b80458e176b351')
         .add_local_dir(HERE, '/opt/arg/integrations/doudizhu', ignore=['**/__pycache__/**']))


@app.function(image=image, gpu='L4', cpu=(4,4), memory=(16384,16384),
              max_containers=1, min_containers=0, buffer_containers=0,
              scaledown_window=2, retries=0, timeout=900, startup_timeout=300,
              single_use_containers=True)
def experiment(stage: str, checkpoint: bytes | None = None):
    allocated_start = time.time()
    import io
    import os
    import subprocess
    import sys
    import tarfile
    out = Path('/opt/arg/evidence/doudizhu-implementation/gpu')/stage
    out.mkdir(parents=True,exist_ok=True)
    command=[sys.executable,'-u','/opt/arg/integrations/doudizhu/cuda_experiment.py',
             '--stage',stage,'--out',str(out)]
    if checkpoint is not None:
        path=out/'trained-input.pt'
        path.write_bytes(checkpoint)
        command+=['--input-checkpoint',str(path)]
    status = subprocess.call(command,cwd='/opt/arg')
    receipt={'stage':stage,'status':status,'allocated_start_unix':allocated_start,
             'allocated_end_unix':time.time(),'allocation':{'gpu':'L4','gpus':1,'physical_cpu_cores':4,'ram_gib':16},
             'modal_task_id':os.environ.get('MODAL_TASK_ID'),
             'modal_environment':os.environ.get('MODAL_ENVIRONMENT')}
    receipt['allocated_worker_seconds']=receipt['allocated_end_unix']-allocated_start
    receipt['allocated_gpu_hours']=receipt['allocated_worker_seconds']/3600
    receipt['estimated_reserved_resource_cost_usd']=receipt['allocated_worker_seconds']*.00030992
    (out/'remote-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    buffer=io.BytesIO()
    with tarfile.open(fileobj=buffer,mode='w:gz') as archive:
        archive.add(out,arcname=stage)
    return {'receipt':receipt,'archive':buffer.getvalue()}


@app.local_entrypoint()
def main(stage: str='probe'):
    if stage not in ('probe','optimized','followup','parity'):
        raise ValueError('Only the bounded four-invocation experiment matrix is authorized')
    base=ROOT/'evidence/doudizhu-implementation/gpu'
    base.mkdir(parents=True,exist_ok=True)
    ledger_path=base/'cost-ledger.json'
    ledger=json.loads(ledger_path.read_text()) if ledger_path.exists() else {'invocations':[]}
    if any(item['stage']==stage for item in ledger['invocations']):
        raise RuntimeError('Stage already attempted; no automatic or unbounded retries')
    spent=sum(item.get('conservative_cost_bound_usd',0) for item in ledger['invocations'])
    hours=sum(item.get('gpu_hours_reserved_bound',0) for item in ledger['invocations'])
    # Reserve the entire timeout + startup at list rates, plus a $1 build/egress allowance.
    reservation=(900+300+2)*.00030992+1.
    if spent+reservation>12 or hours+1202/3600>2 or len(ledger['invocations'])>=4:
        raise RuntimeError('Dou Dizhu exclusive dollar/hour cap would be exceeded')
    row={'stage':stage,'submitted_unix':time.time(),'status':'submitted',
         'conservative_cost_bound_usd':reservation,'gpu_hours_reserved_bound':1202/3600}
    ledger['invocations'].append(row)
    ledger_path.write_text(json.dumps(ledger,indent=2)+'\n')
    call_start=time.time()
    try:
        checkpoint=(base/'followup/learning-seed-17/latest.pt').read_bytes() if stage=='parity' else None
        result=experiment.remote(stage,checkpoint)
        archive_path=base/f'{stage}.tar.gz'
        archive_path.write_bytes(result['archive'])
        import tarfile
        with tarfile.open(archive_path) as archive:
            archive.extractall(base,filter='data')
        row.update(status='completed' if result['receipt']['status']==0 else 'experiment_failed',
                   receipt=result['receipt'],roundtrip_seconds=time.time()-call_start)
        # Retain conservative reservation: it covers build, app load, teardown and egress
        # until provider billing settles. Function-body time is a separate measured receipt.
        print(json.dumps(result['receipt']),flush=True)
    except BaseException as exc:
        row.update(status='failed',error_type=type(exc).__name__,roundtrip_seconds=time.time()-call_start)
        raise
    finally:
        ledger_path.write_text(json.dumps(ledger,indent=2)+'\n')
