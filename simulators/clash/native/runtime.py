"""Independent fixed8 native CPU and CUDA ABI adapters.

Java is used only by export_data qualification/setup, never by reset or stepping.
CUDA world state, observations, masks, actions and rewards remain device tensors.
"""
from pathlib import Path
import ctypes as C,hashlib,json,os,shutil,subprocess,sys
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from build import build

def pointers(lib):
    lib.cn_world_create.argtypes=[C.c_int];lib.cn_world_create.restype=C.c_void_p
    lib.cn_world_free.argtypes=[C.c_void_p]
    lib.cn_world_reset.argtypes=[C.c_void_p,C.c_int];lib.cn_world_reset.restype=C.c_int
    lib.cn_world_step.argtypes=[C.c_void_p,C.c_int,C.c_int,C.c_int];lib.cn_world_step.restype=C.c_int
    lib.cn_world_read.argtypes=[C.c_void_p]*5;lib.cn_world_read.restype=C.c_int
    lib.cn_world_bytes.restype=C.c_size_t
    lib.cn_world_save.argtypes=[C.c_void_p,C.c_void_p,C.c_size_t];lib.cn_world_load.argtypes=lib.cn_world_save.argtypes
    return lib

class NativeCPU:
    def __init__(self,seeds,form='full'):
        self.form=form;self.library=build(form);self.lib=pointers(C.CDLL(str(self.library)));self.handles=[self.lib.cn_world_create(int(seed)) for seed in seeds]
        if not all(self.handles):raise MemoryError('native world allocation failed')
        n=len(self.handles);self.observations=np.zeros((n,2,7024),np.int32);self.masks=np.zeros((n,2,2307),np.uint8);self.infos=np.zeros((n,12),np.int32);self.rewards=np.zeros((n,2),np.float32)
        self.read()
    def read(self):
        for i,p in enumerate(self.handles):
            code=self.lib.cn_world_read(p,*[C.c_void_p(a[i].ctypes.data) for a in (self.observations,self.masks,self.infos,self.rewards)])
            if code:raise RuntimeError(f'native simulator fault {code} lane {i}')
        return self.observations,self.masks,self.infos,self.rewards
    def step(self,actions,tick_limit=0):
        for p,a in zip(self.handles,np.asarray(actions)):self.lib.cn_world_step(p,int(a[0]),int(a[1]),int(tick_limit))
        return self.read()
    def reset(self,seeds,selected=None):
        for i,(p,seed) in enumerate(zip(self.handles,seeds)):
            if selected is None or selected[i]:self.lib.cn_world_reset(p,int(seed))
        return self.read()
    def state(self):
        size=self.lib.cn_world_bytes();out=[]
        for p in self.handles:
            buf=C.create_string_buffer(size)
            if self.lib.cn_world_save(p,buf,size):raise RuntimeError('native snapshot save failed')
            out.append(buf.raw)
        return out
    def load(self,values):
        for p,value in zip(self.handles,values):
            if self.lib.cn_world_load(p,C.c_char_p(value),len(value)):raise RuntimeError('native snapshot load failed')
        # Snapshot includes first-sight bookkeeping. Reading is an observation event
        # and must be caller-controlled on resume, exactly as at a rollout boundary.
    def close(self):
        for p in self.handles:self.lib.cn_world_free(p)
        self.handles=[]

def build_cuda(form='full'):
    """Build (or reuse) the CUDA library for a match form (native/match_forms.py); `full` is the pinned form."""
    from data_binding import validate_data
    from build_receipt import build_identity,compiler_identity,identity_sha256,materialize_build
    import match_forms
    data_binding=validate_data()
    nvcc=shutil.which('nvcc')
    if not nvcc:raise RuntimeError('nvcc CUDA development toolchain is required')
    # Explicitly select and bind nvcc's host compiler, rather than letting a PATH
    # change silently reuse a binary built by a different host toolchain.
    host=shutil.which('g++') or shutil.which('clang++')
    if not host:raise RuntimeError('CUDA host C++ compiler required')
    compiler=compiler_identity(nvcc);host_compiler=compiler_identity(host)
    sources=sorted(p for p in HERE.rglob('*') if p.is_file() and (p.suffix in ('.hpp','.cuh') or p.name=='cuda_engine.cu') and '.build' not in p.parts)
    hashes={str(p.relative_to(HERE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    import torch
    capability=torch.cuda.get_device_capability();architecture='sm_'+''.join(map(str,capability))
    flags=['-std=c++20','-O3','-arch='+architecture,'--ptxas-options=-O1,-v','--extended-lambda','--expt-relaxed-constexpr','-Xcompiler=-fPIC','-shared','-ccbin='+host_compiler['path'],*match_forms.flags(form)]
    extra={'architecture':architecture,'host_compiler':host_compiler}
    if form!='full':extra['match_form']=match_forms.describe(form)  # the pinned form keeps its historical identity shape
    identity=build_identity(backend='native-cuda',sources=hashes,data_binding=data_binding,compiler=compiler,flags=flags,translation_units=['cuda_engine.cu'],extra=extra)
    out=HERE/'.build'/f'libclash-cuda-v2{match_forms.library_suffix(form)}-{identity_sha256(identity)[:16]}.so'
    cache=Path(os.environ['CLASH_NATIVE_BUILD_CACHE']) if os.getenv('CLASH_NATIVE_BUILD_CACHE') else None
    # The compiled form is checked where the library is loaded (NativeCUDA), not here: image builds have no GPU.
    return materialize_build(out,identity,lambda path:[compiler['path'],*flags,str(HERE/'cuda_engine.cu'),'-o',str(path)],cache=cache)

class NativeCUDA:
    def __init__(self,seeds,device='cuda',*,launch_threads=None,render_mode=None,form='full'):
        import torch
        import match_forms
        self.form=form;self.torch=torch;self.device=torch.device(device);self.library=build_cuda(form);self.lib=C.CDLL(str(self.library));l=self.lib
        self.match_form=match_forms.check_library(l,form,prefix='cn_cuda')
        l.cn_cuda_create.restype=C.c_void_p;l.cn_cuda_free.argtypes=[C.c_void_p];l.cn_cuda_world_bytes.restype=C.c_size_t
        l.cn_cuda_reset_read.argtypes=[C.c_void_p]*8+[C.c_int,C.c_void_p];l.cn_cuda_reset.argtypes=[C.c_void_p]*4+[C.c_int,C.c_void_p];l.cn_cuda_rebind.argtypes=[C.c_void_p,C.c_void_p,C.c_int,C.c_void_p]
        l.cn_cuda_step.argtypes=[C.c_void_p,C.c_void_p,C.c_int,C.c_int,C.c_void_p];l.cn_cuda_read.argtypes=[C.c_void_p]*5+[C.c_int,C.c_void_p]
        l.cn_cuda_error_string.argtypes=[C.c_int];l.cn_cuda_error_string.restype=C.c_char_p
        l.cn_cuda_set_launch_threads.argtypes=[C.c_int];l.cn_cuda_set_render_mode.argtypes=[C.c_int]
        l.cn_cuda_get_launch_threads.argtypes=[];l.cn_cuda_get_launch_threads.restype=C.c_int
        l.cn_cuda_get_render_mode.argtypes=[];l.cn_cuda_get_render_mode.restype=C.c_int
        self.launch_threads=int(os.getenv('CLASH_CUDA_LAUNCH_THREADS','32')) if launch_threads is None else int(launch_threads)
        self.render_mode=int(os.getenv('CLASH_CUDA_RENDER_MODE','0')) if render_mode is None else int(render_mode)
        if l.cn_cuda_set_launch_threads(self.launch_threads):raise ValueError('invalid CUDA launch threads')
        if l.cn_cuda_set_render_mode(self.render_mode):raise ValueError('invalid CUDA render mode')
        self.handle=l.cn_cuda_create()
        if not self.handle:raise RuntimeError('native CUDA task allocation failed')
        self.n=len(seeds);self.states=torch.zeros((self.n,l.cn_cuda_world_bytes()),dtype=torch.uint8,device=self.device)
        self.observations=torch.empty((self.n,2,7024),dtype=torch.int32,device=self.device);self.masks=torch.empty((self.n,2,2307),dtype=torch.uint8,device=self.device);self.infos=torch.empty((self.n,12),dtype=torch.int32,device=self.device);self.rewards=torch.empty((self.n,2),dtype=torch.float32,device=self.device)
        self.reset(seeds)
    def stream(self):return C.c_void_p(self.torch.cuda.current_stream(self.device).cuda_stream)
    def runtime_policy(self):return {'cuda_launch_threads':int(self.lib.cn_cuda_get_launch_threads()),'cuda_render_mode':int(self.lib.cn_cuda_get_render_mode())}
    def ptr(self,value):return C.c_void_p(value.data_ptr()) if value is not None else C.c_void_p()
    def check(self,code):
        if code:raise RuntimeError(self.lib.cn_cuda_error_string(code).decode())
    def read(self):
        self.check(self.lib.cn_cuda_read(*[self.ptr(v) for v in (self.states,self.observations,self.masks,self.infos,self.rewards)],self.n,self.stream()))
        return self.observations,self.masks,self.infos,self.rewards
    def reset(self,seeds,selected=None):
        seed_tensor=self.torch.as_tensor(seeds,dtype=self.torch.int32,device=self.device).contiguous();flags=None if selected is None else self.torch.as_tensor(selected,dtype=self.torch.int32,device=self.device).contiguous()
        self.check(self.lib.cn_cuda_reset_read(self.handle,self.ptr(self.states),self.ptr(seed_tensor),self.ptr(flags),*[self.ptr(v) for v in (self.observations,self.masks,self.infos,self.rewards)],self.n,self.stream()))
        # C ABI kernels launch on Torch's current stream: allocator lifetime is
        # naturally ordered; only selected lanes render, preserving every other view.
        return self.observations,self.masks,self.infos,self.rewards
    def step(self,actions,tick_limit=0):
        actions=self.torch.as_tensor(actions,device=self.device,dtype=self.torch.int32).contiguous()
        if actions.shape!=(self.n,2):raise ValueError('native CUDA action batch shape differs')
        self.check(self.lib.cn_cuda_step(self.ptr(self.states),self.ptr(actions),self.n,int(tick_limit),self.stream()))
        return self.read()
    def state(self):return self.states.clone()
    def load(self,states):
        if not isinstance(states,self.torch.Tensor) or states.dtype!=self.torch.uint8 or states.shape!=self.states.shape:raise ValueError('native CUDA checkpoint matrix shape/dtype differs')
        self.states.copy_(states);self.check(self.lib.cn_cuda_rebind(self.handle,self.ptr(self.states),self.n,self.stream()))
    def close(self):
        self.torch.cuda.synchronize(self.device);self.lib.cn_cuda_free(self.handle);self.handle=None
