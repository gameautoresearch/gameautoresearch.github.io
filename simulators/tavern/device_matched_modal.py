"""Bounded same-B200 comparisons gated on retained full GPU qualification."""
import json
import os
from pathlib import Path
import modal

ROOT=Path(__file__).resolve().parents[2] if modal.is_local() else Path('/workspace')
PAYLOAD=Path(os.environ.get('TAVERN_MATCHED_PAYLOAD',str(ROOT)))
if modal.is_local():
    selected_width=json.loads((PAYLOAD/'qualification.json').read_text())['block_threads']
    image=modal.Image.from_id(os.environ['TAVERN_MATCHED_BASE_IMAGE'])
    if os.environ.get('TAVERN_MATCHED_PHASE') in ['dispatch_sanitizer','dispatch_initcheck']:
        image=image.apt_install('cuda-sanitizer-13-3')
    if not os.environ.get('TAVERN_MATCHED_PHASE','').startswith('dispatch'):
        image=(image.add_local_dir(PAYLOAD/'upstream','/workspace/upstream',copy=True)
        .add_local_dir(PAYLOAD/'earliest','/workspace/earliest',copy=True)
        .run_commands('python -m venv --system-site-packages /venvs/upstream && /venvs/upstream/bin/python -m pip install --no-deps --no-build-isolation --editable /workspace/upstream/candidates/AutoTavern',
                      'python -m venv --system-site-packages /venvs/earliest && /venvs/earliest/bin/python -m pip install --no-deps --no-build-isolation --editable /workspace/earliest/candidates/AutoTavern'))
    image=(image.add_local_dir(PAYLOAD/'integrations/tavern','/workspace/integrations/tavern',copy=True,ignore=['device/**'])
        .add_local_file(PAYLOAD/'qualification.json','/workspace/qualification.json',copy=True)
        .env({'TAVERN_DEVICE_ARENA':'1','TAVERN_DEVICE_STACK_BYTES':'262144',
              'TAVERN_DEVICE_TRACE_INIT':'0','TAVERN_DEVICE_TRACE_KERNELS':'0',
              'CUDA_LAUNCH_BLOCKING':'0','CUBLAS_WORKSPACE_CONFIG':':4096:8',
              'TAVERN_DEVICE_BLOCK_THREADS':str(selected_width),
              'TAVERN_SANITIZER_TOOL':'initcheck' if os.environ.get('TAVERN_MATCHED_PHASE')=='dispatch_initcheck' else 'memcheck'}))
else:image=modal.Image.debian_slim(python_version='3.12')
app=modal.App('autoresearch-tavern-device-matched-20261108',image=image)
volume=modal.Volume.from_name('autoresearch-tavern-device-evidence-20261108')


@app.function(gpu='B200',cpu=(4,4),memory=(16384,16384),timeout=3600,startup_timeout=180,
    max_containers=1,min_containers=0,buffer_containers=0,retries=0,scaledown_window=2,
    single_use_containers=True,volumes={'/results':volume})
def compare(run_id:str,phase:str,sizes:list[int],trials:int):
    import hashlib
    import subprocess
    import sys
    import time
    start=time.perf_counter();out=Path('/results')/run_id
    if out.exists():raise RuntimeError('existing worker claim; experiment will not be repeated')
    out.mkdir();(out/'worker-start.json').write_text(json.dumps({'unix':time.time(),
        'task_id':os.environ.get('MODAL_TASK_ID'),'phase':phase,'sizes':sizes})+'\n');volume.commit()
    try:
        receipt=json.loads(Path('/workspace/qualification.json').read_text())
        if not receipt['passed']:raise RuntimeError('full GPU qualification required')
        for name,sha in receipt['module_sha256'].items():
            p=Path('/workspace/integrations/tavern/device')/name
            if hashlib.sha256(p.read_bytes()).hexdigest()!=sha:raise RuntimeError('qualified module mismatch')
        proc=subprocess.run([sys.executable,'-u','/workspace/integrations/tavern/device_matched.py',
            '--out',str(out),'--phase',phase,'--trials',str(trials),'--sizes',*map(str,sizes)],timeout=3480)
        return {'returncode':proc.returncode,'run_id':run_id}
    finally:
        completed=(out/'matched.json').exists()
        passed=completed and (phase not in ['dispatch_debug','dispatch_watch','dispatch_scaling','dispatch_sanitizer','dispatch_initcheck'] or json.loads((out/'matched.json').read_text()).get('all_cases_passed',False))
        (out/'allocation.json').write_text(json.dumps({'worker_seconds':time.perf_counter()-start,
            'gpu':'B200','cpu_cores':4,'ram_gib':16,'retries':0,'warm_containers':0,
            'driver_completed':completed,'passed':passed})+'\n');volume.commit()


@app.local_entrypoint()
def main(run_id:str,phase:str,sizes:str='128,1024',trials:int=2):
    if phase not in ['simulation','learning','dispatch','dispatch_debug','dispatch_watch','dispatch_scaling','dispatch_sanitizer','dispatch_initcheck']:raise ValueError('invalid comparison phase')
    call=compare.spawn(run_id,phase,[int(x) for x in sizes.split(',')],trials)
    print(json.dumps({'run_id':run_id,'app_id':app.app_id,'function_call_id':call.object_id}),flush=True)
    r=call.get();print(json.dumps(r),flush=True)
    if r['returncode']:raise RuntimeError('matched comparison failed; retain all attempts')
