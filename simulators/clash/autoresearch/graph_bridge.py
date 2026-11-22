"""Graph bridges for the fast trainer: the frozen native CUDA SWITCH bridges, compiled so that captured child
graphs are copied into the conditional bodies instead of being nested in them (conditional_body.cuh says why:
the nested form segfaults in the GPU driver on worker hosts with NVIDIA driver 620.70.06).

The native tree stays byte for byte what releases pin. `install()` points the two native build entry points
at the builds made here; everything else in device_branch.py and packed_head_branch.py (loading, validation,
replay, qualification) runs unchanged. The worker image builds both ahead of time
(harness/games/clash/provider.py), because a worker's trainer cannot write build outputs.
"""
import importlib,os,re,shutil
from pathlib import Path
from .common import HERE,NATIVE,setup_native,sha

# native module -> (this directory's translation unit, the frozen unit it includes, library prefix, build name)
BRIDGES={'device_branch':('device_branch_flat.cu','device_branch.cu','libclash-device-branch-flat-','cuda-device-branch-flat-v1'),
         'packed_head_branch':('packed_head_branch_flat.cu','packed_head_branch.cu','libclash-packed-head-flat-','cuda-packed-head-branch-flat-v1')}
FILES=('graph_bridge.py','conditional_body.cuh','device_branch_flat.cu','packed_head_branch_flat.cu')


def build(name):
    """Build (or reuse) one bridge with the native build receipts, and return the library path."""
    setup_native()
    import torch
    from build_receipt import compiler_identity,build_identity,identity_sha256,materialize_build
    unit,frozen,prefix,backend=BRIDGES[name]
    nvcc=shutil.which('nvcc');host=shutil.which('g++') or shutil.which('clang++')
    if not nvcc or not host:raise RuntimeError('CUDA 12.8+ nvcc and a host C++ compiler are required; this does not launch a cloud job')
    compiler=compiler_identity(nvcc);host_compiler=compiler_identity(host);version=re.search(r'release\s+(\d+)\.(\d+)',compiler['version'])
    if version is None or tuple(map(int,version.groups()))<(12,8):raise RuntimeError('CUDA SWITCH requires nvcc 12.8 or newer')
    if not torch.cuda.is_available():raise RuntimeError('actual CUDA hardware required')
    architecture='sm_'+''.join(map(str,torch.cuda.get_device_capability()))
    # The same flags and command shape as the native bridges, which the build receipts require.
    flags=['-std=c++20','-O3','-arch='+architecture,'-Xcompiler=-fPIC','-shared','-ccbin='+host_compiler['path']]
    sources={n:sha(HERE/n) for n in FILES if n.endswith('.cuh') or n in (unit,'graph_bridge.py')}
    sources.update({'native/'+n:sha(NATIVE/n) for n in (frozen,name+'.py','build_receipt.py')})
    identity=build_identity(backend=backend,sources=sources,data_binding={'scope':'generic neural graph dispatch; no simulator data dependency','maximum_entities':200},
                            compiler=compiler,flags=flags,translation_units=[unit],
                            extra={'architecture':architecture,'host_compiler':host_compiler,'torch_version':torch.__version__,'torch_cuda_version':torch.version.cuda,'minimum_cudart':12080})
    output=NATIVE/'.build'/(prefix+identity_sha256(identity)[:16]+'.so')
    cache=Path(os.environ['CLASH_NATIVE_BUILD_CACHE']) if os.getenv('CLASH_NATIVE_BUILD_CACHE') else None
    return materialize_build(output,identity,lambda destination:[compiler['path'],*flags,str(HERE/unit),'-o',str(destination)],cache=cache)


def install():
    """Make the native modules build and load the bridges above. The native trainer imports them by their
    bare names; the package names are covered too so no import style can load the nested form."""
    setup_native()
    for name in BRIDGES:
        for module in (name,'simulators.clash.native.'+name):
            importlib.import_module(module).build=lambda name=name:build(name)
