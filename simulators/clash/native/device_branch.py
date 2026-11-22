"""CUDA 12.8 SWITCH over retained RNG-free Torch child graphs.

No simulator state, CPU entity-count scalar, policy math, optimizer or sampling
is introduced here. Children must write shared fixed-shape output buffers and
must remain on the same device/context. A whole conditional graph cannot itself
be a child, so replay explicitly rejects outer stream capture.

NVIDIA references:
https://developer.nvidia.com/blog/dynamic-control-flow-in-cuda-graphs-with-conditional-nodes/
https://docs.nvidia.com/cuda/archive/12.8.1/cuda-c-programming-guide/index.html
"""
from __future__ import annotations
from pathlib import Path
import argparse,ctypes as C,json,os,re,shutil,subprocess,sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
DEFAULT_LIMITS=(32,64,96,200)

def _limits(values):
    values=tuple(values)
    if not values or len(values)>16 or any(isinstance(v,bool) or not isinstance(v,int) for v in values):raise ValueError('one to 16 integer bucket bounds required')
    if any(a>=b for a,b in zip((0,*values[:-1]),values)) or values[-1]!=200:raise ValueError('strictly increasing positive bounds ending at full 200 are required')
    return values

def build():
    import torch
    from build_receipt import sha,compiler_identity,build_identity,identity_sha256,materialize_build
    nvcc=shutil.which('nvcc')
    if not nvcc:raise RuntimeError('CUDA 12.8+ nvcc required; this helper does not launch a cloud job')
    host=shutil.which('g++') or shutil.which('clang++')
    if not host:raise RuntimeError('CUDA host C++ compiler required')
    compiler=compiler_identity(nvcc);host_compiler=compiler_identity(host);version=re.search(r'release\s+(\d+)\.(\d+)',compiler['version'])
    if version is None or tuple(map(int,version.groups()))<(12,8):raise RuntimeError('CUDA SWITCH requires nvcc 12.8 or newer')
    if not torch.cuda.is_available():raise RuntimeError('actual CUDA hardware required')
    architecture='sm_'+''.join(map(str,torch.cuda.get_device_capability()))
    flags=['-std=c++20','-O3','-arch='+architecture,'-Xcompiler=-fPIC','-shared','-ccbin='+host_compiler['path']]
    identity=build_identity(backend='cuda-device-branch-v1',sources={name:sha(HERE/name) for name in ('device_branch.cu','device_branch.py','build_receipt.py')},data_binding={'scope':'generic neural graph dispatch; no simulator data dependency','maximum_entities':200},compiler=compiler,flags=flags,translation_units=['device_branch.cu'],extra={'architecture':architecture,'host_compiler':host_compiler,'torch_version':torch.__version__,'torch_cuda_version':torch.version.cuda,'minimum_cudart':12080})
    output=HERE/'.build'/('libclash-device-branch-'+identity_sha256(identity)[:16]+'.so')
    cache=Path(os.environ['CLASH_NATIVE_BUILD_CACHE']) if os.getenv('CLASH_NATIVE_BUILD_CACHE') else None
    return materialize_build(output,identity,lambda destination:[compiler['path'],*flags,str(HERE/'device_branch.cu'),'-o',str(destination)],cache=cache)

def _load():
    library=build();lib=C.CDLL(str(library))
    lib.cn_branch_create.argtypes=[C.c_void_p,C.c_int,C.c_int64,C.c_int,C.POINTER(C.c_int),C.POINTER(C.c_void_p),C.c_int,C.c_void_p,C.POINTER(C.c_void_p)]
    lib.cn_branch_create.restype=C.c_int
    lib.cn_branch_launch.argtypes=[C.c_void_p,C.c_void_p];lib.cn_branch_launch.restype=C.c_int
    lib.cn_branch_destroy.argtypes=[C.c_void_p];lib.cn_branch_destroy.restype=C.c_int
    lib.cn_branch_probe.argtypes=[C.c_void_p,C.c_int,C.c_int64,C.c_int,C.POINTER(C.c_int),C.c_int,C.c_void_p,C.c_void_p];lib.cn_branch_probe.restype=C.c_int
    lib.cn_branch_error_string.argtypes=[C.c_int];lib.cn_branch_error_string.restype=C.c_char_p
    lib.cn_branch_validate_child.argtypes=[C.c_void_p,C.POINTER(C.c_int),C.c_int];lib.cn_branch_validate_child.restype=C.c_int
    lib.cn_branch_test_unsupported_children.argtypes=[];lib.cn_branch_test_unsupported_children.restype=C.c_int
    lib.cn_branch_runtime_version.restype=C.c_int
    return library,lib

class DeviceBranchGraph:
    """Select first inclusive bucket from GPU max raw entity_count, then execute.

    ``children`` are Torch CUDAGraph(keep_graph=True) objects captured without
    RNG. Retain ALL child input/model/output tensors in the owning actor/learner.
    This wrapper retains raw_obs and child graphs until close. It has one graph
    executable, must not be concurrently mutated, and is not itself capturable.
    Invalid raw counts (negative or >200) trigger a device assertion before any
    child executes. Valid 97..200 always choose the full-capacity final child.

    ``selection`` is a device int32[3]: bucket index, maximum count, error bits
    (negative=1, overflow=2). It is diagnostic output, never read on CPU in replay.
    """
    def __init__(self,raw_obs,entity_count_index,bucket_limits,children):
        import torch
        from build_receipt import backend_build_binding
        if not isinstance(raw_obs,torch.Tensor) or raw_obs.dtype!=torch.int32 or not raw_obs.is_cuda:raise ValueError('raw_obs must be a CUDA int32 tensor')
        if raw_obs.ndim!=2 or raw_obs.shape[0]<1 or not raw_obs.is_contiguous():raise ValueError('raw_obs must be a nonempty contiguous [N,stride] tensor')
        if not isinstance(entity_count_index,int) or not 0<=entity_count_index<raw_obs.shape[1]:raise ValueError('entity_count_index is outside the raw observation row')
        self.limits=_limits(bucket_limits);self.children=tuple(children)
        if len(self.children)!=len(self.limits):raise ValueError('one retained child graph per bucket is required')
        self.raw_obs=raw_obs;self.column=entity_count_index;self.device=raw_obs.device;self.handle=C.c_void_p()
        with torch.cuda.device(self.device):
            if torch.cuda.is_current_stream_capturing():raise RuntimeError('construct DeviceBranchGraph outside stream capture')
            self.library,self.lib=_load();self.build_binding=backend_build_binding(self.library)
            self.selection=torch.empty(3,dtype=torch.int32,device=self.device)
            self.selected_bucket=self.selection[0]
            self.maximum_count=self.selection[1]
            self.error_bits=self.selection[2]
            handles=[]
            for child in self.children:
                if not isinstance(child,torch.cuda.CUDAGraph):raise TypeError('children must be retained Torch CUDAGraph objects')
                try:handles.append(int(child.raw_cuda_graph()))
                except RuntimeError as exc:raise ValueError('children require CUDAGraph(keep_graph=True) and completed capture') from exc
            if not all(handles):raise ValueError('raw child graph handles must be nonzero')
            self.raw_child_handles=tuple(handles)
            self.child_node_types=[]
            for handle in handles:
                counts=(C.c_int*32)()
                self._check(self.lib.cn_branch_validate_child(C.c_void_p(handle),counts,32))
                self.child_node_types.append({str(i):n for i,n in enumerate(counts) if n})
            bounds=(C.c_int*len(self.limits))(*self.limits);graphs=(C.c_void_p*len(handles))(*handles)
            self._check(self.lib.cn_branch_create(C.c_void_p(raw_obs.data_ptr()),raw_obs.shape[0],raw_obs.stride(0),self.column,bounds,graphs,len(handles),C.c_void_p(self.selection.data_ptr()),C.byref(self.handle)))
            if not self.handle:raise RuntimeError('CUDA branch graph construction returned no executable')
    def _check(self,code):
        if code:raise RuntimeError('CUDA branch graph: '+self.lib.cn_branch_error_string(code).decode())
    def replay(self):
        import torch
        if not self.handle:raise RuntimeError('CUDA branch graph is closed')
        with torch.cuda.device(self.device):
            if torch.cuda.is_current_stream_capturing():raise RuntimeError('conditional graph cannot be captured as an outer child; disable outer rollout capture')
            self._check(self.lib.cn_branch_launch(self.handle,C.c_void_p(torch.cuda.current_stream(self.device).cuda_stream)))
        return None
    def probe(self):
        """Qualification only: same GPU selector, without poisoning on bad counts.

        Writes diagnostic selection/error bits, executes no child, and makes no
        CPU entity-count read. Production replay always uses the assertive path.
        """
        import torch
        if not self.handle:raise RuntimeError('CUDA branch graph is closed')
        bounds=(C.c_int*len(self.limits))(*self.limits)
        with torch.cuda.device(self.device):
            self._check(self.lib.cn_branch_probe(C.c_void_p(self.raw_obs.data_ptr()),self.raw_obs.shape[0],self.raw_obs.stride(0),self.column,bounds,len(self.limits),C.c_void_p(self.selection.data_ptr()),C.c_void_p(torch.cuda.current_stream(self.device).cuda_stream)))
        return self.selection
    def close(self):
        import torch
        if self.handle:
            with torch.cuda.device(self.device):
                torch.cuda.synchronize(self.device)
                self._check(self.lib.cn_branch_destroy(self.handle));self.handle=C.c_void_p()
            self.children=();self.raw_child_handles=()
    def __enter__(self):return self
    def __exit__(self,*exc):self.close()


def _toy_graphs(raw):
    import torch
    output=torch.empty(8,dtype=torch.int32,device=raw.device);graphs=[]
    stream=torch.cuda.Stream(device=raw.device);stream.wait_stream(torch.cuda.current_stream(raw.device))
    for index in range(4):
        with torch.cuda.stream(stream):output.fill_(index)
        stream.synchronize();graph=torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(graph,stream=stream):output.fill_(index)
        graphs.append(graph)
    torch.cuda.current_stream(raw.device).wait_stream(stream)
    return output,graphs

def _invalid_child(value,out):
    """Run assertive invalid-input test in an isolated CUDA process/context."""
    import torch
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    raw=torch.zeros((5,7024),dtype=torch.int32,device='cuda');output,children=_toy_graphs(raw)
    branch=DeviceBranchGraph(raw,48,DEFAULT_LIMITS,children);raw[2,48]=value
    try:
        branch.replay();torch.cuda.synchronize()
    except RuntimeError as exc:
        message=str(exc);rejected='assert' in message.lower()
        (out/'result.json').write_text(json.dumps({'value':value,'rejected_by_actual_device_assert':rejected,'error':message},indent=2)+'\n')
        os._exit(0 if rejected else 1)  # poisoned CUDA context must not clean up graphs
    (out/'result.json').write_text(json.dumps({'value':value,'rejected_by_actual_device_assert':False})+'\n')
    os._exit(1)

def qualify(out,*,invalid_asserts=True):
    import torch
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    raw=torch.zeros((5,7024),dtype=torch.int32,device='cuda');output,children=_toy_graphs(raw)
    rows=[];negative=[]
    with DeviceBranchGraph(raw,48,DEFAULT_LIMITS,children) as branch:
        for count in (0,32,33,64,65,96,97,200):
            raw.zero_();raw[3,48]=count;output.fill_(-1);branch.replay();torch.cuda.synchronize()
            want=next(i for i,limit in enumerate(DEFAULT_LIMITS) if count<=limit)
            assert branch.selection.cpu().tolist()==[want,count,0]
            assert output.cpu().tolist()==[want]*8
            rows.append({'maximum_count':count,'bucket':want,'limit':DEFAULT_LIMITS[want]})
        for bad in (-1,201):
            raw.zero_();raw[1,48]=bad;output.fill_(-1);branch.probe();torch.cuda.synchronize()
            got=branch.selection.cpu().tolist();assert got==[-1,max(bad,0),1 if bad<0 else 2]
            assert output.cpu().tolist()==[-1]*8;negative.append({'count':bad,'probe':got})
        outer=torch.cuda.CUDAGraph();rejected=False
        with torch.cuda.graph(outer):
            output.zero_()
            try:branch.replay()
            except RuntimeError:rejected=True
        assert rejected
        branch._check(branch.lib.cn_branch_test_unsupported_children())
        binding=branch.build_binding;node_types=branch.child_node_types
    actual=[]
    if invalid_asserts:
        for bad in (-1,201):
            child=out/('invalid-negative' if bad<0 else 'invalid-overflow')
            subprocess.run([sys.executable,str(Path(__file__).resolve()),'--invalid-count-child',str(bad),'--out',str(child)],check=True)
            actual.append(json.loads((child/'result.json').read_text()))
    result={'status':'passed','cuda_executed':True,'gpu':torch.cuda.get_device_name(),'boundary_selection':rows,'invalid_probes':negative,'actual_invalid_device_asserts':actual,'outer_capture_rejected':True,'retained_raw_torch_graphs':4,'raw_graph_keep_graph_required':True,'child_node_type_counts':node_types,'host_and_nested_host_negative_controls_rejected':True,'binding':binding,'runtime_cpu_entity_count_reads':False,'timing_claims':False}
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True);return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);p.add_argument('--invalid-count-child',type=int);p.add_argument('--skip-invalid-assert-subprocesses',action='store_true');args=p.parse_args()
    if args.invalid_count_child is not None:_invalid_child(args.invalid_count_child,args.out)
    else:qualify(args.out,invalid_asserts=not args.skip_invalid_assert_subprocesses)
