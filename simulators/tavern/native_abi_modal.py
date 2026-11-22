"""Bounded real-CUDA validation of the stock NVPTX runtime (not game qualification)."""
from pathlib import Path
import json,os,modal
ROOT=Path(__file__).resolve().parents[2] if modal.is_local() else Path('/workspace')
PTX=ROOT/'integrations/tavern/device/native/target-frozen/nvptx64-nvidia-cuda/release/deps/tavern_native_abi.ptx'
BASE_IMAGE='im-JDvxQVGb7FtkAltPaQUBWK'
if modal.is_local():
    image=(modal.Image.from_id(BASE_IMAGE)
        .add_local_file(PTX,'/workspace/native-abi/abi.ptx',copy=True)
        .add_local_file(ROOT/'integrations/tavern/device_env.py','/workspace/native-abi/device_env.py',copy=True)
        .run_commands(('timeout 60s /usr/local/cuda-13.3/bin/ptxas --compile-only --gpu-name=sm_100 --opt-level=1 --fmad=false --split-compile=4 /workspace/native-abi/abi.ptx --output-file=/tmp/native-abi.o && timeout 60s /usr/local/cuda-13.3/bin/nvlink --arch=sm_100 /tmp/native-abi.o /usr/local/cuda-13.3/lib64/libcudadevrt.a --output-file=/workspace/native-abi/abi.cubin') if os.environ.get('TAVERN_NATIVE_ABI_RDC')=='1' else 'timeout 60s /usr/local/cuda-13.3/bin/ptxas --gpu-name=sm_100 --opt-level=1 --fmad=false --split-compile=4 /workspace/native-abi/abi.ptx --output-file=/workspace/native-abi/abi.cubin')
        .env({'TAVERN_DEVICE_TRACE_INIT':'1'}))
else:image=modal.Image.debian_slim(python_version='3.12')
app=modal.App('autoresearch-tavern-native-abi-20261108',image=image)
volume=modal.Volume.from_name('autoresearch-tavern-device-evidence-20261108')

@app.function(gpu='B200',cpu=(4,4),memory=(16384,16384),timeout=120,startup_timeout=120,
    max_containers=1,min_containers=0,buffer_containers=0,retries=0,scaledown_window=2,
    single_use_containers=True,volumes={'/results':volume})
def qualify(run_id:str):
    import time,os,sys,hashlib,shutil,traceback
    start=time.perf_counter();out=Path('/results')/run_id
    if out.exists():raise RuntimeError('prior worker claim exists; do not repeat')
    out.mkdir();(out/'worker-start.json').write_text(json.dumps({'unix':time.time(),'task_id':os.environ.get('MODAL_TASK_ID')})+'\n')
    for name in ['abi.ptx','abi.cubin']:shutil.copyfile('/workspace/native-abi/'+name,out/name)
    volume.commit()
    try:
        import torch,numpy as np,ctypes as ct
        sys.path.insert(0,'/workspace/native-abi');from device_env import DeviceEnv
        env=DeviceEnv(1,kernels_only=True,ptx='/workspace/native-abi/abi.ptx')
        result=torch.empty((64,6),dtype=torch.int64,device='cuda')
        env.launch('native_primitives',64,[result,ct.c_uint32(64)])
        actual=result.cpu().numpy().view(np.uint64)
        expected=np.array([[i,63,(i+1)%251,sum((i+j)%251 for j in range(64)),
            ((0xfedcba9876543210^i)*0xa1b2c3d4e5f60718)>>64,0x123456789abcdef0]for i in range(64)],np.uint64)
        checks={'thread_index':np.array_equal(actual[:,0],expected[:,0]),
            'wide_slice_abi':np.array_equal(actual[:,1:3],expected[:,1:3]),
            'cuda_malloc_vec_drop':np.array_equal(actual[:,3],expected[:,3]),
            'unsigned_high_multiply':np.array_equal(actual[:,4],expected[:,4]),
            'contended_once':np.array_equal(actual[:,5],expected[:,5])}
        receipt={'passed':all(checks.values()),'checks':checks,'gpu':torch.cuda.get_device_name(),
            'torch':torch.__version__,'module_receipts':env.module_receipts,'limits':env.limits,
            'actual':actual.tolist(),'expected':expected.tolist(),'base_image':BASE_IMAGE,
            'scope':'Primitive/runtime diagnostic only. No game simulation, neural learning or speed is qualified.'}
        (out/'result.json').write_text(json.dumps(receipt,indent=2)+'\n')
        if not receipt['passed']:raise AssertionError(checks)
        return json.loads(json.dumps(receipt))
    except BaseException:
        (out/'failure.txt').write_text(traceback.format_exc());raise
    finally:
        (out/'allocation.json').write_text(json.dumps({'worker_seconds':time.perf_counter()-start,
            'gpu':'B200','cpu_cores':4,'ram_gib':16,'full_game_qualified':False})+'\n');volume.commit()

@app.local_entrypoint()
def main(run_id:str):print(json.dumps(qualify.remote(run_id)),flush=True)
