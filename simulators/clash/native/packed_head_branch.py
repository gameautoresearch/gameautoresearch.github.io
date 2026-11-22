"""One RNG-free CUDA SWITCH over <=4 entity bounds x <=4 head capacities.

Warp-per-row coalesced mask prepare ->4N retained flags ->count/entity SWITCH.
No CPU tensor-content read, nested conditional graph, or sampler. Children must
be complete unconditional loss/backward/optimizer graphs with shared fixed output
addresses. This prototype needs actual-device qualification before integration.
"""
from pathlib import Path
import ctypes as C,os,re,shutil
import torch
HERE=Path(__file__).resolve().parent

def build():
    from build_receipt import sha,compiler_identity,build_identity,identity_sha256,materialize_build
    nvcc=shutil.which('nvcc');host=shutil.which('g++') or shutil.which('clang++')
    if not nvcc or not host:raise RuntimeError('existing CUDA12.8+ development and host compiler required; no cloud job is launched')
    compiler=compiler_identity(nvcc);host=compiler_identity(host);v=re.search(r'release\s+(\d+)\.(\d+)',compiler['version'])
    if not v or tuple(map(int,v.groups()))<(12,8):raise RuntimeError('CUDA12.8+ SWITCH required')
    if not torch.cuda.is_available():raise RuntimeError('actual CUDA hardware required')
    arch='sm_'+''.join(map(str,torch.cuda.get_device_capability()));flags=['-std=c++20','-O3','-arch='+arch,'-Xcompiler=-fPIC','-shared','-ccbin='+host['path']]
    names=('packed_head_branch.cu','packed_head_branch.py','packed_head.py','packed_head_graph.py','build_receipt.py')
    identity=build_identity(backend='cuda-packed-head-branch-v1',sources={n:sha(HERE/n) for n in names},data_binding={'scope':'neural-only device active-count and entity-bound graph selection','maximum_entities':200},compiler=compiler,flags=flags,translation_units=['packed_head_branch.cu'],extra={'architecture':arch,'host_compiler':host,'torch_version':torch.__version__,'torch_cuda_version':torch.version.cuda})
    out=HERE/'.build'/('libclash-packed-head-'+identity_sha256(identity)[:16]+'.so');cache=Path(os.environ['CLASH_NATIVE_BUILD_CACHE']) if os.getenv('CLASH_NATIVE_BUILD_CACHE') else None
    return materialize_build(out,identity,lambda p:[compiler['path'],*flags,str(HERE/'packed_head_branch.cu'),'-o',str(p)],cache=cache)

class PackedHeadBranchGraph:
    """Retained child/parent lifetime; selection[flat,active,maxentity,ei,hi,error]."""
    def __init__(self,obs,masks,column,entity_bounds,head_caps,children):
        from build_receipt import backend_build_binding
        if not obs.is_cuda or obs.dtype!=torch.int32 or obs.ndim!=2 or not obs.is_contiguous():raise ValueError('owned contiguous CUDA int32 observation matrix required')
        if masks.device!=obs.device or masks.ndim!=2 or masks.shape[0]!=obs.shape[0] or masks.dtype not in (torch.bool,torch.uint8) or not masks.is_contiguous():raise ValueError('owned full CUDA bool/uint8 masks required')
        if type(column) is not int or not 0<=column<obs.shape[1]:raise ValueError('invalid public entity count column')
        self.entities=tuple(entity_bounds);self.heads=tuple(head_caps)
        for bounds,last in [(self.entities,200),(self.heads,obs.shape[0])]:
            if not 1<=len(bounds)<=4 or any(type(x) is not int or x<1 for x in bounds) or tuple(sorted(set(bounds)))!=bounds or bounds[-1]!=last:raise ValueError('increasing bounds ending at full200/fullN required')
        self.children=tuple(children)
        if len(self.children)!=len(self.entities)*len(self.heads):raise ValueError('entity-major/head-minor Cartesian child layout required')
        self.obs=obs;self.masks=masks;self.column=column;self.device=obs.device;self.handle=C.c_void_p()
        with torch.cuda.device(self.device):
            if torch.cuda.is_current_stream_capturing():raise RuntimeError('construct conditional graph outside capture')
            self.library=build();self.lib=C.CDLL(str(self.library));l=self.lib;self.build_binding=backend_build_binding(self.library)
            l.cn_head_branch_create.argtypes=[C.c_void_p,C.c_void_p,C.c_int,C.c_int64,C.c_int64,C.c_int,C.c_int,C.POINTER(C.c_int),C.c_int,C.POINTER(C.c_int),C.c_int,C.POINTER(C.c_void_p),C.c_void_p,C.POINTER(C.c_void_p)]
            l.cn_head_branch_probe.argtypes=[C.c_void_p,C.c_void_p,C.c_void_p];l.cn_head_branch_probe.restype=C.c_int
            l.cn_head_branch_create.restype=C.c_int;l.cn_head_branch_launch.argtypes=[C.c_void_p,C.c_void_p];l.cn_head_branch_destroy.argtypes=[C.c_void_p]
            l.cn_head_branch_validate_child.argtypes=[C.c_void_p,C.POINTER(C.c_int),C.c_int];l.cn_head_branch_error_string.restype=C.c_char_p;l.cn_head_branch_error_string.argtypes=[C.c_int];l.cn_head_branch_test_unsupported_children.argtypes=[];l.cn_head_branch_test_unsupported_children.restype=C.c_int
            self.selector_scratch_bytes=4*len(obs)
            self.selection=torch.empty(6,device=self.device,dtype=torch.int32)
            self.flat_bucket=self.selection[0];self.selected_bucket=self.selection[3];self.selected_head_bucket=self.selection[4]
            self.child_node_types=[];handles=[]
            for child in self.children:
                if not isinstance(child,torch.cuda.CUDAGraph):raise TypeError('retained Torch keep_graph=True child required')
                handle=int(child.raw_cuda_graph());handles.append(handle);counts=(C.c_int*32)();self._check(l.cn_head_branch_validate_child(C.c_void_p(handle),counts,32));self.child_node_types.append({str(i):n for i,n in enumerate(counts) if n})
            entities=(C.c_int*len(self.entities))(*self.entities);heads=(C.c_int*len(self.heads))(*self.heads);graphs=(C.c_void_p*len(handles))(*handles)
            self._check(l.cn_head_branch_create(C.c_void_p(obs.data_ptr()),C.c_void_p(masks.data_ptr()),len(obs),obs.stride(0),masks.stride(0),column,masks.shape[1],entities,len(entities),heads,len(heads),graphs,C.c_void_p(self.selection.data_ptr()),C.byref(self.handle)))
    def _check(self,code):
        if code:raise RuntimeError('packed-head CUDA branch: '+self.lib.cn_head_branch_error_string(code).decode())
    def replay(self):
        if not self.handle:raise RuntimeError('closed packed-head branch')
        with torch.cuda.device(self.device):
            if torch.cuda.is_current_stream_capturing():raise RuntimeError('conditional parent cannot be captured as an outer child')
            self._check(self.lib.cn_head_branch_launch(self.handle,C.c_void_p(torch.cuda.current_stream(self.device).cuda_stream)))
    def probe(self):
        """Diagnostic only: same two kernels, no child update or device assertion."""
        if not self.handle:raise RuntimeError('closed packed-head branch')
        with torch.cuda.device(self.device):
            if torch.cuda.is_current_stream_capturing():raise RuntimeError('probe is outside outer capture')
            self._check(self.lib.cn_head_branch_probe(self.handle,C.c_void_p(self.selection.data_ptr()),C.c_void_p(torch.cuda.current_stream(self.device).cuda_stream)))
        return self.selection
    def close(self):
        if self.handle:
            with torch.cuda.device(self.device):torch.cuda.synchronize(self.device);self._check(self.lib.cn_head_branch_destroy(self.handle));self.handle=C.c_void_p()
            self.children=()
