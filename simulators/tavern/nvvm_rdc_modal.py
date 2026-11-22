"""CPU-only alternate compiler build; GPU execution requires a separate full gate."""
import hashlib
import json
import os
from pathlib import Path
import modal

ROOT=Path(__file__).resolve().parents[2] if modal.is_local() else Path('/workspace')
PAYLOAD=Path(os.environ.get('TAVERN_NVVM_PAYLOAD',str(ROOT)))
cache=modal.Volume.from_name('autoresearch-tavern-device-build-cache-20261108')
if modal.is_local():
    image=(modal.Image.from_id('im-JDvxQVGb7FtkAltPaQUBWK')
        .add_local_dir(PAYLOAD/'integrations/tavern/device','/workspace/integrations/tavern/device',copy=True,
            ignore=['**/target*/**','target*/**','*.ptx','*.cubin','**/__pycache__/**'])
        .env({'TAVERN_DEVICE_SPLIT':'1','TAVERN_DEVICE_ARENA':'1',
              'TAVERN_DEVICE_RUST_OPT':'2','TAVERN_DEVICE_NVVM_OPT':'1'})
        .run_commands('python /workspace/integrations/tavern/device/run_with_heartbeat.py --timeout 900 --label nvvm tavern-device-compiler /workspace/integrations/tavern/device',
            env={'CARGO_TARGET_DIR':'/compiler-cache/nvvm-rdc-v43','CARGO_HOME':'/compiler-cache/cargo'},
            volumes={'/compiler-cache':cache})
        .run_commands(*[
            f'python /workspace/integrations/tavern/device/run_with_heartbeat.py --timeout 300 --label ptxas /usr/local/cuda-13.3/bin/ptxas --compile-only --gpu-name=sm_100 --opt-level=1 --fmad=false --split-compile=4 /workspace/integrations/tavern/device/{stem}.ptx --output-file=/tmp/{stem}.o && '
            f'python /workspace/integrations/tavern/device/run_with_heartbeat.py --timeout 120 --label nvlink /usr/local/cuda-13.3/bin/nvlink --arch=sm_100 /tmp/{stem}.o /usr/local/cuda-13.3/lib64/libcudadevrt.a --output-file=/workspace/integrations/tavern/device/{stem}.cubin'
            for stem in ['tavern','tavern-state']])
        .add_local_dir(PAYLOAD/'integrations/tavern','/workspace/integrations/tavern',copy=True,ignore=['device/**'])
        .env({'TAVERN_DEVICE_ARENA':'1','TAVERN_DEVICE_STACK_BYTES':'262144',
              'CUDA_LAUNCH_BLOCKING':'0','CUBLAS_WORKSPACE_CONFIG':':4096:8'}))
else:image=modal.Image.debian_slim(python_version='3.12')
app=modal.App('autoresearch-tavern-nvvm-rdc-build-20261108',image=image)
volume=modal.Volume.from_name('autoresearch-tavern-device-evidence-20261108')


@app.function(cpu=(4,4),memory=(16384,16384),timeout=300,retries=0,
    max_containers=1,min_containers=0,buffer_containers=0,single_use_containers=True,
    scaledown_window=2,volumes={'/results':volume})
def retain(run_id:str):
    import shutil,time
    begin=time.perf_counter();out=Path('/results')/run_id
    out.mkdir(exist_ok=False)
    (out/'worker-start.json').write_text(json.dumps({'unix':time.time(),'gpu_executed':False,
        'task_id':os.environ.get('MODAL_TASK_ID')})+'\n');volume.commit()
    rows={}
    for name in ['tavern.ptx','tavern-state.ptx','tavern.cubin','tavern-state.cubin']:
        source=Path('/workspace/integrations/tavern/device')/name
        rows[name]={'bytes':source.stat().st_size,'sha256':hashlib.sha256(source.read_bytes()).hexdigest()}
        shutil.copyfile(source,out/name)
    receipt={'backend':'pinned patched Rust-CUDA/NVVM + separate device compile/link',
        'gpu_runtime_qualified':False,'gpu_executed':False,'modules':rows,
        'retain_worker_seconds':time.perf_counter()-begin}
    (out/'build.json').write_text(json.dumps(receipt,indent=2)+'\n');volume.commit()
    return receipt


@app.local_entrypoint()
def main(run_id:str):
    call=retain.spawn(run_id)
    print(json.dumps({'run_id':run_id,'app_id':app.app_id,'function_call_id':call.object_id}),flush=True)
    print(json.dumps(call.get()),flush=True)
