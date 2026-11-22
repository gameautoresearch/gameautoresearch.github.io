"""Continue full-device qualification from a retained immutable binary image."""
from pathlib import Path
import json,os,modal
PAYLOAD=Path(os.environ['TAVERN_NATIVE_PAYLOAD']) if modal.is_local() else Path('/workspace')
IMAGE=os.environ.get('TAVERN_NATIVE_PREBUILT_IMAGE','im-6HgysUX8nnvz9ApkWjhoNt')
if modal.is_local():
    image=(modal.Image.from_id(IMAGE)
        .add_local_dir(PAYLOAD/'integrations/tavern','/workspace/integrations/tavern',copy=True,ignore=['device/**'])
        .add_local_dir(PAYLOAD/'integrations/tavern/device/fixtures','/workspace/integrations/tavern/device/fixtures',copy=True)
        .env({'TAVERN_DEVICE_STACK_BYTES':'262144','TAVERN_DEVICE_TRACE_INIT':'1',
              'TAVERN_DEVICE_TRACE_KERNELS':'1','CUDA_LAUNCH_BLOCKING':'0','TAVERN_NATIVE_PREBUILT_IMAGE':IMAGE,
              'TAVERN_DEVICE_BLOCK_THREADS':os.environ.get('TAVERN_NATIVE_BLOCK_THREADS','32')}))
else:image=modal.Image.debian_slim(python_version='3.12')
app=modal.App('autoresearch-tavern-native-debug-20261108',image=image)
volume=modal.Volume.from_name('autoresearch-tavern-device-evidence-20261108')

@app.function(gpu='B200',cpu=(4,4),memory=(16384,16384),timeout=1800,startup_timeout=180,
    max_containers=1,min_containers=0,buffer_containers=0,retries=0,scaledown_window=2,
    single_use_containers=True,volumes={'/results':volume})
def qualify(run_id:str):
    import time,subprocess,sys
    start=time.perf_counter();out=Path('/results')/run_id
    if out.exists():raise RuntimeError('prior worker claim exists; no hidden repeated experiment')
    out.mkdir();(out/'worker-start.json').write_text(json.dumps({'unix':time.time(),'task_id':os.environ.get('MODAL_TASK_ID'),'binary_image':IMAGE})+'\n');volume.commit()
    try:
        proc=subprocess.run([sys.executable,'-u','/workspace/integrations/tavern/device_probe.py','--out',str(out),'--managed-out'],timeout=1700)
        return {'returncode':proc.returncode,'run_id':run_id}
    finally:
        (out/'allocation.json').write_text(json.dumps({'worker_seconds':time.perf_counter()-start,'gpu':'B200','cpu_cores':4,
            'ram_gib':16,'stack_bytes':262144,'cuda_launch_blocking':False,'binary_image':IMAGE,'gpu_simulation_qualified':(out/'differential.json').exists()})+'\n');volume.commit()

@app.local_entrypoint()
def main(run_id:str):
    call=qualify.spawn(run_id)
    print(json.dumps({'run_id':run_id,'app_id':app.app_id,'function_call_id':call.object_id}),flush=True)
    r=call.get();print(json.dumps(r),flush=True)
    if r['returncode']:raise RuntimeError('full-device diagnostic failed; retain evidence')
