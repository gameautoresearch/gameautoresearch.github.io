"""Separate, bounded complete-rule NVVM compiler/device-execution experiment.

The original CPU integration is the oracle, never a fallback for device transitions.
A failed or unqualified port is not reported as a GPU simulation result.
"""
import json
import hashlib
import os
from pathlib import Path
import modal

ROOT=Path(__file__).resolve().parents[2] if modal.is_local() else Path('/workspace')
PAYLOAD=Path(os.environ.get('TAVERN_DEVICE_PAYLOAD',str(ROOT)))
if modal.is_local():
    compiler_hash=hashlib.sha256()
    for path in sorted((PAYLOAD/'integrations/tavern/device/compiler').rglob('*')):
        if path.is_file() and 'target' not in path.parts:
            compiler_hash.update(str(path.relative_to(PAYLOAD)).encode());compiler_hash.update(path.read_bytes())
    COMPILER_ID=compiler_hash.hexdigest()[:16]
VOLUME='autoresearch-tavern-device-evidence-20261108'
build_cache=modal.Volume.from_name('autoresearch-tavern-device-build-cache-20261108',create_if_missing=True)
if modal.is_local():
    asm_opt=os.environ.get('TAVERN_DEVICE_ASM_OPT','0')
    if asm_opt not in ['0','1','2','3']:raise ValueError('invalid assembler optimization level')
    image=(modal.Image.from_registry('nvidia/cuda:13.0.2-devel-ubuntu24.04@sha256:0eee3094c71518ad31d011a594ae6ed6de72959ee07e318cb31cffe71690e90c',add_python='3.12')
        .apt_install('build-essential','curl','git','cmake','ninja-build','pkg-config','libxml2-dev','libssl-dev')
        .pip_install('torch==2.9.1','numpy==2.3.5')
        .run_commands('curl --proto =https --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain nightly-2026-04-02',
            '/root/.cargo/bin/rustup component add rust-src rustc-dev llvm-tools-preview --toolchain nightly-2026-04-02')
        .env({'PATH':'/root/.cargo/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin',
            'CUDA_PATH':'/usr/local/cuda','CUDA_ROOT':'/usr/local/cuda','CARGO_NET_GIT_FETCH_WITH_CLI':'true',
            'USE_PREBUILT_LLVM':'1','CUBLAS_WORKSPACE_CONFIG':':4096:8','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
        .apt_install('libclang-dev')
        .pip_install('maturin==1.10.2','pytest==9.0.2')
        .add_local_dir(PAYLOAD/'candidates/AutoTavern','/workspace/candidates/AutoTavern',copy=True,
            ignore=['target/**','.git/**','**/__pycache__/**','**/*.so','**/*.dylib'])
        .run_commands('cd /workspace/candidates/AutoTavern && python -m pip install --no-build-isolation --editable .')
        .apt_install('zlib1g-dev','libzstd-dev','cuda-nvcc-13-3=13.3.73-1','libnvvm-13-3=13.3.73-1','cuda-driver-dev-13-3')
        .env({'CUDA_PATH':'/usr/local/cuda-13.3','CUDA_ROOT':'/usr/local/cuda-13.3',
            'LD_LIBRARY_PATH':'/usr/local/cuda-13.3/nvvm/lib64:/usr/local/cuda-13.3/lib64:/usr/local/lib'})
        .add_local_dir(PAYLOAD/'integrations/tavern/device/compiler','/compiler-project',copy=True)
        .run_commands('cd /compiler-project && cargo build --release --locked && '
            'cp /compiler-cache/target/release/tavern-device-compiler /usr/local/bin/tavern-device-compiler && '
            'cp /compiler-cache/target/release/librustc_codegen_nvvm.so /usr/local/lib/librustc_codegen_nvvm.so',
            env={'CARGO_TARGET_DIR':'/compiler-cache/target','CARGO_HOME':'/compiler-cache/cargo'},
            volumes={'/compiler-cache':build_cache})
        .add_local_dir(PAYLOAD/'integrations/tavern/device','/workspace/integrations/tavern/device',copy=True,
            ignore=['target/**','**/__pycache__/**','*.ptx','*.cubin'])
        .env({'TAVERN_DEVICE_SPLIT':os.environ.get('TAVERN_DEVICE_SPLIT','0'),
            'TAVERN_DEVICE_ARENA':os.environ.get('TAVERN_DEVICE_ARENA','0'),
            'TAVERN_DEVICE_RUST_OPT':os.environ.get('TAVERN_DEVICE_RUST_OPT','3'),
            'TAVERN_DEVICE_NVVM_OPT':os.environ.get('TAVERN_DEVICE_NVVM_OPT','1'),
            **({'TAVERN_DEVICE_IR_TAG':os.environ['TAVERN_DEVICE_IR_TAG']} if 'TAVERN_DEVICE_IR_TAG' in os.environ else {})})
        .run_commands('timeout 900s tavern-device-compiler /workspace/integrations/tavern/device',
            env={'CARGO_TARGET_DIR':'/compiler-cache/device-target-'+COMPILER_ID+'-'+os.environ.get('TAVERN_DEVICE_SPLIT','0')+'-'+os.environ.get('TAVERN_DEVICE_ARENA','0')+'-'+os.environ.get('TAVERN_DEVICE_RUST_OPT','3')+'-'+os.environ.get('TAVERN_DEVICE_NVVM_OPT','1'),
                 'CARGO_HOME':'/compiler-cache/cargo'},
            volumes={'/compiler-cache':build_cache})
        .run_commands(f'timeout 900s /usr/local/cuda-13.3/bin/ptxas --gpu-name=sm_100 --opt-level={asm_opt} --fmad=false --split-compile=4 '
            '/workspace/integrations/tavern/device/tavern.ptx --output-file=/workspace/integrations/tavern/device/tavern.cubin'
            +(f' && timeout 900s /usr/local/cuda-13.3/bin/ptxas --gpu-name=sm_100 --opt-level={asm_opt} --fmad=false --split-compile=4 '
              '/workspace/integrations/tavern/device/tavern-state.ptx --output-file=/workspace/integrations/tavern/device/tavern-state.cubin'
              if os.environ.get('TAVERN_DEVICE_SPLIT')=='1' else ''))
        .add_local_dir(PAYLOAD/'integrations/tavern','/workspace/integrations/tavern',copy=True,ignore=['device/**']))
else:image=modal.Image.debian_slim(python_version='3.12')
app=modal.App('autoresearch-tavern-device-20261108-b200',image=image)
volume=modal.Volume.from_name(VOLUME,create_if_missing=True)


@app.function(gpu='B200',cpu=(4,4),memory=(16384,16384),timeout=1800,startup_timeout=180,
    max_containers=1,min_containers=0,buffer_containers=0,retries=0,scaledown_window=2,
    single_use_containers=True,volumes={'/results':volume})
def probe(run_id:str):
    import subprocess,sys,time
    started=time.perf_counter();out=Path('/results')/run_id
    if out.exists():
        marker=out/('duplicate-input-'+os.environ.get('MODAL_TASK_ID','unknown')+'.json')
        if not marker.exists():marker.write_text(json.dumps({'refused_duplicate':True,'unix':time.time()})+'\n')
        volume.commit()
        raise RuntimeError('prior worker claim exists; infrastructure restart may not repeat a paid experiment')
    out.mkdir(parents=True,exist_ok=False)
    (out/'worker-start.json').write_text(json.dumps({'start_unix':time.time(),'run_id':run_id,
        'task_id':os.environ.get('MODAL_TASK_ID'),'gpu':'B200','cpu_cores':4,'ram_gib':16})+'\n')
    volume.commit()
    try:
        proc=subprocess.run([sys.executable,'-u','/workspace/integrations/tavern/device_probe.py','--out',str(out),'--managed-out'],timeout=1700)
        return {'run_id':run_id,'returncode':proc.returncode}
    finally:
        out.mkdir(parents=True,exist_ok=True)
        (out/'allocation.json').write_text(json.dumps({'worker_seconds':time.perf_counter()-started,
            'gpu':'B200','cpu_cores':4,'ram_gib':16,'gpu_simulation_qualified':False})+'\n')
        volume.commit()


@app.local_entrypoint()
def main(run_id:str='device-port-v1'):
    r=probe.remote(run_id);print(json.dumps(r),flush=True)
    if r['returncode']:raise RuntimeError('Device port failed; preserve all build and allocation evidence')
