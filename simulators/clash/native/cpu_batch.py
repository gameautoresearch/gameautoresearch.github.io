"""Persistent C++ worker-pool baseline with one native ABI call per vector operation.

``threads=0`` and ``threads=1`` execute serially on the caller. Larger values
create exactly that many persistent native workers. The caller must choose a
value within its fixed allocated CPU budget; no automatic extra cores are used.
Game state and public observations are the same NativeWorld as NativeCPU/CUDA.
"""
from pathlib import Path
import ctypes as C
import sys
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from build import build


def pointers(lib):
    lib.cn_batch_create.argtypes=[C.c_int,C.c_int,C.c_void_p];lib.cn_batch_create.restype=C.c_void_p
    lib.cn_batch_free.argtypes=[C.c_void_p];lib.cn_batch_free.restype=None
    lib.cn_batch_count.argtypes=[C.c_void_p];lib.cn_batch_count.restype=C.c_int
    lib.cn_batch_threads.argtypes=[C.c_void_p];lib.cn_batch_threads.restype=C.c_int
    lib.cn_batch_world_bytes.argtypes=[];lib.cn_batch_world_bytes.restype=C.c_size_t
    lib.cn_batch_handles.argtypes=[C.c_void_p,C.POINTER(C.c_void_p),C.c_int];lib.cn_batch_handles.restype=C.c_int
    lib.cn_batch_reset.argtypes=[C.c_void_p]*3;lib.cn_batch_reset.restype=C.c_int
    lib.cn_batch_step.argtypes=[C.c_void_p,C.c_void_p,C.c_int];lib.cn_batch_step.restype=C.c_int
    lib.cn_batch_read.argtypes=[C.c_void_p]*5;lib.cn_batch_read.restype=C.c_int
    lib.cn_batch_step_read.argtypes=[C.c_void_p,C.c_void_p,C.c_int]+[C.c_void_p]*4;lib.cn_batch_step_read.restype=C.c_int
    lib.cn_batch_reset_read.argtypes=[C.c_void_p]*7;lib.cn_batch_reset_read.restype=C.c_int
    for name in ('cn_batch_save','cn_batch_load'):
        getattr(lib,name).argtypes=[C.c_void_p,C.c_void_p,C.c_size_t];getattr(lib,name).restype=C.c_int
    return lib


def _ptr(value):return C.c_void_p(value.ctypes.data) if value is not None else C.c_void_p()


class NativeCPUBatch:
    def __init__(self,seeds,threads=1,form='full'):
        if isinstance(threads,bool) or not isinstance(threads,int) or not 0<=threads<=256:raise ValueError('explicit threads must be an integer in [0,256], within the allocated CPU budget')
        seeds=np.asarray(seeds,dtype=np.int32)
        if seeds.ndim!=1 or not len(seeds):raise ValueError('one seed is required per batch lane')
        seeds=np.ascontiguousarray(seeds);self.n=len(seeds);self.threads=threads;self.form=form;self.library=build(form);self.lib=pointers(C.CDLL(str(self.library)))
        self.handle=self.lib.cn_batch_create(self.n,threads,_ptr(seeds))
        if not self.handle:raise RuntimeError('native CPU batch allocation/initialization failed')
        self.observations=np.zeros((self.n,2,7024),np.int32);self.masks=np.zeros((self.n,2,2307),np.uint8);self.infos=np.zeros((self.n,12),np.int32);self.rewards=np.zeros((self.n,2),np.float32)
        handles=(C.c_void_p*self.n)()
        if self.lib.cn_batch_handles(self.handle,handles,self.n):self.close();raise RuntimeError('native CPU batch handle export failed')
        self.handles=list(handles) # Debug/test compatibility, exported in ONE batch ABI call.
        self.world_bytes=int(self.lib.cn_batch_world_bytes());self._state_buffer=np.empty((self.n,self.world_bytes),np.uint8)
        try:self.read()
        except BaseException:self.close();raise
    def _check(self,code):
        if code:
            lane=np.flatnonzero(self.infos[:,5])
            raise RuntimeError(f'native CPU batch simulator error {code}'+(f' lane {int(lane[0])}' if len(lane) else ''))
    def _outputs(self):return tuple(_ptr(getattr(self,key)) for key in ('observations','masks','infos','rewards'))
    def _result(self):return self.observations,self.masks,self.infos,self.rewards
    def _open(self):
        if not self.handle:raise RuntimeError('native CPU batch is closed')
    def read(self):
        self._open();self._check(self.lib.cn_batch_read(self.handle,*self._outputs()));return self._result()
    def step(self,actions,tick_limit=0):
        self._open();actions=np.asarray(actions,dtype=np.int32)
        if actions.shape!=(self.n,2):raise ValueError(f'actions must have shape {(self.n,2)}')
        if isinstance(tick_limit,bool) or not isinstance(tick_limit,(int,np.integer)) or tick_limit<0:raise ValueError('tick_limit must be a nonnegative integer')
        actions=np.ascontiguousarray(actions);self._check(self.lib.cn_batch_step_read(self.handle,_ptr(actions),int(tick_limit),*self._outputs()));return self._result()
    def reset(self,seeds,selected=None):
        self._open();seeds=np.asarray(seeds,dtype=np.int32)
        if seeds.shape!=(self.n,):raise ValueError(f'seeds must have shape {(self.n,)}')
        seeds=np.ascontiguousarray(seeds)
        if selected is not None:
            selected=np.asarray(selected,dtype=np.int32)
            if selected.shape!=(self.n,):raise ValueError(f'selected must have shape {(self.n,)}')
            selected=np.ascontiguousarray(selected)
        self._check(self.lib.cn_batch_reset_read(self.handle,_ptr(seeds),_ptr(selected),*self._outputs()));return self._result()
    def state(self):
        self._open();self._check(self.lib.cn_batch_save(self.handle,_ptr(self._state_buffer),self._state_buffer.nbytes));return [row.tobytes() for row in self._state_buffer]
    def load(self,values):
        self._open();values=list(values)
        if len(values)!=self.n or any(len(value)!=self.world_bytes for value in values):raise ValueError('native CPU batch checkpoint shape differs')
        for target,value in zip(self._state_buffer,values):target[:]=np.frombuffer(value,dtype=np.uint8)
        self._check(self.lib.cn_batch_load(self.handle,_ptr(self._state_buffer),self._state_buffer.nbytes))
        # The caller restores saved output buffers at resume; loading is not an
        # observation event and must not advance first-sight movement bookkeeping.
    def close(self):
        if getattr(self,'handle',None):self.lib.cn_batch_free(self.handle);self.handle=None
        self.handles=[]
    def __enter__(self):return self
    def __exit__(self,*exc):self.close()


def _self_test(out_path):
    """Correctness only: serial/world/pool identity, no timing comparison."""
    import hashlib,json,shutil,subprocess
    from runtime import NativeCPU
    out=Path(out_path).resolve();out.mkdir(parents=True,exist_ok=False)
    seeds=np.array([1,27,819,81108],np.int32);serial=NativeCPU(seeds);batches=[];frames=0;max_entities=0;resets=0;rng=np.random.default_rng(8110901)
    layout_source=out/'layout.cpp';layout_binary=out/'layout'
    layout_source.write_text('#include <cstddef>\n#include <cstdio>\n#include "'+str(HERE/'world.hpp')+'"\nint main(){using namespace clash_native;std::printf("%zu %zu %zu %zu %zu %zu %zu %zu %zu %zu",sizeof(NativeWorld),sizeof(void*),offsetof(NativeWorld,spec),offsetof(NativeWorld,arenaGrid)+offsetof(GridView,tiles),offsetof(NativeWorld,arenaGrid)+offsetof(GridView,current),offsetof(NativeWorld,arenaGrid)+offsetof(GridView,previous),offsetof(NativeWorld,observationEntities),sizeof(ObservationEntity),offsetof(ObservationEntity,ground_trail)+sizeof(bool),offsetof(ObservationEntity,buff_count));}\n')
    subprocess.run([shutil.which('clang++') or shutil.which('g++'),'-std=c++20',str(layout_source),'-o',str(layout_binary)],check=True)
    layout=list(map(int,subprocess.check_output([str(layout_binary)],text=True).split()));size,pointer_bytes=layout[:2];offsets=layout[2:6];cache_offset,entity_bytes,padding_start,padding_end=layout[6:]
    assert padding_end-padding_start==2,'unexpected observation POD padding'
    (out/'pointer-padding-layout.json').write_text(json.dumps(dict(world_bytes=size,pointer_offsets=offsets,pointer_bytes=pointer_bytes,observation_cache_offset=cache_offset,observation_entity_bytes=entity_bytes,cache_padding_start=padding_start,cache_padding_end=padding_end),indent=2)+'\n')
    def normalize(value):
        assert len(value)==size
        a=bytearray(value)
        for offset in offsets:a[offset:offset+pointer_bytes]=b'\0'*pointer_bytes
        # Compiler-proven padding between the last bool and buff_count in the
        # output-only ObservationEntity cache is not a game-state field.
        for row in range(512):a[cache_offset+row*entity_bytes+padding_start:cache_offset+row*entity_bytes+padding_end]=b'\0'*(padding_end-padding_start)
        return bytes(a)
    def check(env,label):
        nonlocal frames,max_entities
        for key in ('observations','masks','infos','rewards'):np.testing.assert_array_equal(getattr(env,key),getattr(serial,key),err_msg=label+' '+key)
        frames+=1;max_entities=max(max_entities,int(serial.observations[...,48].max()))
    try:
        for threads in (0,1,2,4):
            env=NativeCPUBatch(seeds,threads);batches.append((str(threads),env));check(env,'initial '+str(threads))
        card_actions=[0]*8;data=json.loads((HERE/'generated-v2/task-data.json').read_text());vocab=data['vocabulary'];names=data['rules']['deck']
        for decision in range(256):
            actions=np.zeros((len(seeds),2),np.int32)
            for lane in range(len(seeds)):
                for side in range(2):
                    legal=np.flatnonzero(serial.masks[lane,side]);actions[lane,side]=rng.choice(legal[1:]) if len(legal)>1 else 0
                    if actions[lane,side]:
                        slot=(actions[lane,side]-1)//576;card=vocab[int(serial.observations[lane,side,7+slot])];card_actions[names.index(card)]+=1
            serial.step(actions)
            for label,env in batches:env.step(actions);check(env,'busy '+str(decision)+' pool'+label)
            if decision in (40,93,155):
                selected=np.array([True,decision==93,False,decision==155]);newseeds=rng.integers(0,1000000,len(seeds),dtype=np.int32);serial.reset(newseeds,selected)
                for label,env in batches:env.reset(newseeds,selected);check(env,'reset '+label)
                resets+=int(selected.sum())
            if decision==127:
                saved=serial.state();saved_outputs=[getattr(serial,k).copy() for k in ('observations','masks','infos','rewards')]
                for label,env in batches:
                    for lane,(a,b) in enumerate(zip(env.state(),saved)):
                        na,nb=normalize(a),normalize(b)
                        if na!=nb:
                            aa=np.frombuffer(na,dtype=np.uint8);bb=np.frombuffer(nb,dtype=np.uint8);diff=np.flatnonzero(aa!=bb);np.savez(out/'state-mismatch.npz',actual=aa,expected=bb,differences=diff)
                            (out/'state-difference.json').write_text(json.dumps(dict(pool=label,lane=lane,differences=len(diff),first_offsets=diff[:100].tolist()),indent=2)+'\n')
                            raise AssertionError(f'entire normalized state pool{label} lane{lane}: {len(diff)} bytes; first {diff[:16].tolist()}')
                clone=NativeCPUBatch(np.array([91,92,93,94],np.int32),2);clone.load(saved)
                for key,values in zip(('observations','masks','infos','rewards'),saved_outputs):getattr(clone,key)[:]=values
                batches.append(('restored2',clone));check(clone,'loaded from serial')
                # Opposite-direction load also repairs all four host pointers.
                serial.load(batches[0][1].state())
                for key,values in zip(('observations','masks','infos','rewards'),saved_outputs):getattr(serial,key)[:]=values
            if decision in (80,180):
                before=serial.state();serial.read();serial.reset(seeds,np.zeros(len(seeds),bool));assert serial.state()==before
                for label,env in batches:
                    before=env.state();env.read();env.reset(seeds,np.zeros(len(seeds),bool));assert env.state()==before;check(env,'read-cache '+label)
        final=serial.state()
        for label,env in batches:
            for lane,(a,b) in enumerate(zip(env.state(),final)):assert normalize(a)==normalize(b),f'final normalized state pool{label} lane{lane}'
        assert all(card_actions),card_actions
    finally:
        serial.close()
        for _,env in batches:env.close()
    # Natural wait/tiebreak/end; independently test time-limit handling too.
    terminal=[]
    for limit in (0,125):
        serial=NativeCPU([45]);env=NativeCPUBatch([45],4)
        try:
            count=0
            while serial.infos[0,2]<0 and not serial.infos[0,9]:
                actions=np.zeros((1,2),np.int32);serial.step(actions,limit);env.step(actions,limit);check(env,'terminal '+str(limit));count+=1
                assert count<1500
            terminal.append(dict(tick_limit=limit,decisions=count,winner=int(serial.infos[0,2]),truncated=int(serial.infos[0,9])))
            assert (serial.infos[0,2]>=0)==(limit==0)
            for a,b in zip(env.state(),serial.state()):assert normalize(a)==normalize(b),'terminal entire native state'
        finally:serial.close();env.close()
    result=dict(status='passed',frames=frames,threads=[0,1,2,4],busy_decisions=256,lanes=4,selective_reset_lanes=resets,max_entities=max_entities,card_actions=dict(zip(names,card_actions)),terminal=terminal,state_scope='every NativeWorld byte except four rebound pointer addresses and compiler-proven two-byte padding in each output-only ObservationEntity cache row',snapshot_restore_continuation_decisions=128,timing_comparison=False,library=str(env.library),cpp_sha256=hashlib.sha256((HERE/'cpu_batch.cpp').read_bytes()).hexdigest(),python_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result));return result


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--self-test',type=Path,required=True);args=parser.parse_args();_self_test(args.self_test)
