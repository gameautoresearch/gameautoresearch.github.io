"""Bounded complete-game stock-NVPTX validation with separate device linking."""
from pathlib import Path
import json,os,modal
ROOT=Path(__file__).resolve().parents[2] if modal.is_local() else Path('/workspace')
PAYLOAD=Path(os.environ.get('TAVERN_NATIVE_PAYLOAD',str(ROOT)))
BASE_IMAGE='im-JDvxQVGb7FtkAltPaQUBWK'
if modal.is_local():
    asm_opt=os.environ.get('TAVERN_NATIVE_ASM_OPT','1')
    state_asm_opt=os.environ.get('TAVERN_NATIVE_STATE_ASM_OPT',asm_opt)
    if asm_opt not in ['1','2','3'] or state_asm_opt not in ['1','2','3']:raise ValueError('invalid assembler optimization level')
    max_regs=os.environ.get('TAVERN_NATIVE_MAX_REGISTERS','')
    if max_regs and max_regs not in ['32','48','64','80','96','128','192','255']:raise ValueError('invalid register cap')
    register_flags=f' --maxrregcount={max_regs} --device-function-maxrregcount={max_regs}' if max_regs else ''
    image=(modal.Image.from_id(BASE_IMAGE)
        .add_local_dir(PAYLOAD/'integrations/tavern/device','/workspace/integrations/tavern/device',copy=True,
            ignore=['**/target*/**','target*/**','*.ptx','*.cubin','**/__pycache__/**'])
        .add_local_dir(PAYLOAD/'compiled','/workspace/compiled',copy=True,ignore=['abi.ptx'])
        .add_local_file(PAYLOAD/'compiled/abi.ptx','/workspace/native-abi/abi.ptx',copy=True)
        .run_commands('python /workspace/integrations/tavern/device/unpack_ptx.py /workspace/compiled /workspace/integrations/tavern/device',
            'timeout 60s /usr/local/cuda-13.3/bin/ptxas --compile-only --gpu-name=sm_100 --opt-level=1 --fmad=false --split-compile=4 /workspace/native-abi/abi.ptx --output-file=/tmp/abi.o && '
            'timeout 60s /usr/local/cuda-13.3/bin/nvlink --arch=sm_100 /tmp/abi.o /usr/local/cuda-13.3/lib64/libcudadevrt.a --output-file=/tmp/abi.cubin',
            f'python /workspace/integrations/tavern/device/run_with_heartbeat.py --timeout 300 --label ptxas /usr/local/cuda-13.3/bin/ptxas --compile-only --gpu-name=sm_100 --opt-level={asm_opt} --fmad=false --split-compile=4{register_flags} /workspace/integrations/tavern/device/tavern.ptx --output-file=/tmp/tavern.o && '
            'python /workspace/integrations/tavern/device/run_with_heartbeat.py --timeout 120 --label nvlink /usr/local/cuda-13.3/bin/nvlink --arch=sm_100 /tmp/tavern.o /usr/local/cuda-13.3/lib64/libcudadevrt.a --output-file=/workspace/integrations/tavern/device/tavern.cubin',
            f'python /workspace/integrations/tavern/device/run_with_heartbeat.py --timeout 300 --label ptxas /usr/local/cuda-13.3/bin/ptxas --compile-only --gpu-name=sm_100 --opt-level={state_asm_opt} --fmad=false --split-compile=4{register_flags} /workspace/integrations/tavern/device/tavern-state.ptx --output-file=/tmp/tavern-state.o && '
            'python /workspace/integrations/tavern/device/run_with_heartbeat.py --timeout 120 --label nvlink /usr/local/cuda-13.3/bin/nvlink --arch=sm_100 /tmp/tavern-state.o /usr/local/cuda-13.3/lib64/libcudadevrt.a --output-file=/workspace/integrations/tavern/device/tavern-state.cubin')
        .add_local_dir(PAYLOAD/'integrations/tavern','/workspace/integrations/tavern',copy=True,ignore=['device/**'])
        .env({'TAVERN_DEVICE_ARENA':'1','CUBLAS_WORKSPACE_CONFIG':':4096:8','TAVERN_DEVICE_STACK_BYTES':'262144','TAVERN_DEVICE_TRACE_INIT':'1','TAVERN_DEVICE_TRACE_KERNELS':'1','CUDA_LAUNCH_BLOCKING':'0'}))
else:image=modal.Image.debian_slim(python_version='3.12')
app=modal.App('autoresearch-tavern-native-full-20261108',image=image)
volume=modal.Volume.from_name('autoresearch-tavern-device-evidence-20261108')

@app.function(gpu='B200',cpu=(4,4),memory=(16384,16384),timeout=1800,startup_timeout=180,
    max_containers=1,min_containers=0,buffer_containers=0,retries=0,scaledown_window=2,
    single_use_containers=True,volumes={'/results':volume})
def qualify(run_id:str):
    import time,subprocess,sys
    start=time.perf_counter();out=Path('/results')/run_id
    if out.exists():raise RuntimeError('prior worker claim exists; infrastructure may not repeat a paid experiment')
    out.mkdir();(out/'worker-start.json').write_text(json.dumps({'unix':time.time(),'task_id':os.environ.get('MODAL_TASK_ID')})+'\n');volume.commit()
    try:
        proc=subprocess.run([sys.executable,'-u','/workspace/integrations/tavern/device_probe.py','--out',str(out),'--managed-out'],timeout=1700)
        return {'returncode':proc.returncode,'run_id':run_id}
    finally:
        (out/'allocation.json').write_text(json.dumps({'worker_seconds':time.perf_counter()-start,'gpu':'B200','cpu_cores':4,
            'ram_gib':16,'backend':'stock LLVM NVPTX; separately compiled/linked','gpu_simulation_qualified':(out/'differential.json').exists()})+'\n');volume.commit()

@app.local_entrypoint()
def main(run_id:str,build_only:bool=False):
    if build_only:
        print(json.dumps({'run_id':run_id,'build_only':True,'gpu_executed':False}),flush=True)
        return
    call=qualify.spawn(run_id)
    print(json.dumps({'run_id':run_id,'app_id':app.app_id,'function_call_id':call.object_id}),flush=True)
    r=call.get();print(json.dumps(r),flush=True)
    if r['returncode']:raise RuntimeError('full-device qualification failed; retain every artifact')
