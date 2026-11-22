"""Private CUDA state/legals, public device buffers, and explicit debug exports."""
from __future__ import annotations
import ctypes as C
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from ..environment import ROOT,ROLES
from ..native_backend import LegalActions,NativeObservation,RANKS
from .reference import RawState,packed_deals

HERE=Path(__file__).parent
_LIB=None
_BUILD_SECONDS=0.


def source_hash():
    return hashlib.sha256(b''.join((HERE/p).read_bytes() for p in ('rules.h','engine.cu'))).hexdigest()


def library():
    global _LIB,_BUILD_SECONDS
    if _LIB is not None:
        if _LIB.source_hash!=source_hash():raise RuntimeError('CUDA source changed inside a live process; restart it')
        return _LIB
    started=time.perf_counter()
    capability=torch.cuda.get_device_capability()
    arch=f'sm_{capability[0]}{capability[1]}'
    key=hashlib.sha256((source_hash()+arch+'nvcc-O3-cxx17-v2').encode()).hexdigest()
    out=ROOT/'evidence/doudizhu-implementation/cuda-build'/key
    out.mkdir(parents=True,exist_ok=True)
    target=out/'engine.so'
    with (out/'build.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if not target.exists():
            for name in ('rules.h','engine.cu'):(out/name).write_bytes((HERE/name).read_bytes())
            command=['nvcc','-O3','-std=c++17','--ptxas-options=-v','-shared','-Xcompiler=-fPIC',f'-arch={arch}',str(out/'engine.cu'),'-o',str(target)]
            result=subprocess.run(command,capture_output=True,text=True)
            (out/'compile.log').write_text(result.stdout+result.stderr)
            if result.returncode:raise RuntimeError(f'CUDA compile failed: {out}/compile.log')
            (out/'receipt.json').write_text(json.dumps({'source_sha256':source_hash(),'architecture':arch,
                'compiler':subprocess.check_output(['nvcc','--version'],text=True),'command':command,
                'binary_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
                'compile_seconds':time.perf_counter()-started},indent=2)+'\n')
        receipt=json.loads((out/'receipt.json').read_text())
        if receipt['source_sha256']!=source_hash() or receipt['binary_sha256']!=hashlib.sha256(target.read_bytes()).hexdigest():
            raise RuntimeError('CUDA cached source/binary drift')
    lib=C.CDLL(str(target))
    p=C.c_void_p;i=C.c_int
    definitions=[('dg_error',[],C.c_char_p),('dg_state_bytes',[],i),('dg_action_bytes',[],i),
        ('dg_reset',[p,p,p,i,p],i),('dg_generate',[p,p,p,p,i,p],i),('dg_encode',[p,p,i,p,i,p],i),
        ('dg_compact',[p,p,p,p,i,p],i),('dg_step',[p,p,p,p,p,i,p],i),('dg_flags',[p,p,i,p],i),
        ('dg_store',[p,p,p,p,p,p,p,p,p,i,i,p],i),
        ('dg_choose',[p,p,p,p,p,p,i,C.c_uint64,p,p,i,p],i),('dg_argmax',[p,p,p,p,p,i,p],i)]
    for name,args,restype in definitions:
        fn=getattr(lib,name);fn.argtypes=args;fn.restype=restype
    _LIB=lib
    _LIB.source_hash=source_hash()
    _BUILD_SECONDS=time.perf_counter()-started
    return lib


class CudaBatch:
    def __init__(self,n,device='cuda:0'):
        if device!='cuda:0' or not torch.cuda.is_available():raise ValueError('A real single CUDA GPU is required')
        self.lib=library()
        self.n=n
        self.device=torch.device(device)
        self._state=torch.empty((n,self.lib.dg_state_bytes()),dtype=torch.uint8,device=device)
        self._actions=torch.empty((n,4096,32),dtype=torch.uint8,device=device)
        self.counts=torch.zeros(n,dtype=torch.int32,device=device)
        self.features=torch.zeros((n,1240),dtype=torch.int8,device=device)
        self.fault=torch.zeros(1,dtype=torch.int32,device=device)
        self.flags=torch.empty((n,4),dtype=torch.int32,device=device)
        self.closed=False
        self.turn=0

    @property
    def stream(self):return C.c_void_p(torch.cuda.current_stream().cuda_stream)

    def _ptr(self,t):return C.c_void_p(t.data_ptr())

    def check(self,status):
        if status:raise RuntimeError('CUDA simulator fault: '+self.lib.dg_error().decode())

    def reset(self,cards):
        if self.closed:raise RuntimeError('CUDA simulator closed')
        if len(cards)!=self.n:raise ValueError('CUDA batch size changed; create a new batch')
        hand,bottom=packed_deals(cards)
        hh=torch.as_tensor(hand,device=self.device)
        bb=torch.as_tensor(bottom,device=self.device)
        self.fault.zero_()
        self.check(self.lib.dg_reset(self._ptr(self._state),self._ptr(hh),self._ptr(bb),self.n,self.stream))
        self.turn=0

    def generate(self,encode=True):
        self.check(self.lib.dg_generate(self._ptr(self._state),self._ptr(self._actions),self._ptr(self.counts),self._ptr(self.fault),self.n,self.stream))
        if encode:
            self.check(self.lib.dg_encode(self._ptr(self._state),self._ptr(self.features),self.turn%3,self._ptr(self.fault),self.n,self.stream))
        # Four scalar control values; state, features, candidates and choices stay GPU.
        summary=torch.stack((self.counts.sum(),self.fault[0].long(),self.counts.max().long(),(self.counts>0).sum())).cpu().tolist()
        total,fault,maximum,active=map(int,summary)
        if fault:raise RuntimeError('CUDA simulator/policy fault; no scores or learner targets awarded')
        return total,maximum,active

    def compact(self,total):
        owners=torch.repeat_interleave(torch.arange(self.n,device=self.device),self.counts.long(),output_size=total)
        starts=self.counts.long().cumsum(0)-self.counts
        local=torch.arange(total,device=self.device)-starts[owners]
        actions=torch.empty((total,54),dtype=torch.int8,device=self.device)
        self.check(self.lib.dg_compact(self._ptr(self._actions),self._ptr(owners),self._ptr(local),self._ptr(actions),total,self.stream))
        return owners,local,actions,starts

    def argmax(self,q,starts):
        out=torch.empty(self.n,dtype=torch.int64,device=self.device)
        self.check(self.lib.dg_argmax(self._ptr(q),self._ptr(self.counts),self._ptr(starts),
                                     self._ptr(out),self._ptr(self.fault),self.n,self.stream))
        return out

    def store(self,choices,x,z,valid):
        self.check(self.lib.dg_store(self._ptr(self._state),self._ptr(self._actions),self._ptr(self.counts),
            self._ptr(choices),self._ptr(self.features),self._ptr(x),self._ptr(z),self._ptr(valid),
            self._ptr(self.fault),self.turn%3,self.n,self.stream))

    def step(self,choices):
        self.check(self.lib.dg_step(self._ptr(self._state),self._ptr(self._actions),self._ptr(self.counts),
            self._ptr(choices),self._ptr(self.fault),self.n,self.stream))
        self.turn+=1

    def choose(self,greedy,kinds,seeds,epsilon):
        choices=torch.empty(self.n,dtype=torch.int64,device=self.device)
        threshold=min(1<<32,int(epsilon*(1<<32)))
        self.check(self.lib.dg_choose(self._ptr(self._state),self._ptr(self._actions),self._ptr(self.counts),
            self._ptr(greedy),self._ptr(kinds),self._ptr(seeds),self.turn,threshold,self._ptr(choices),
            self._ptr(self.fault),self.n,self.stream))
        return choices

    def outcomes(self):
        self.check(self.lib.dg_flags(self._ptr(self._state),self._ptr(self.flags),self.n,self.stream))
        if int(self.fault.cpu()[0]):raise RuntimeError('CUDA simulator fault; no natural outcomes')
        return self.flags.cpu().numpy()

    def debug(self):
        # Explicit evaluator-side export. Never sent to a policy or ordinary rollout.
        return self._state.cpu().numpy(),self._actions.cpu().numpy(),self.counts.cpu().numpy(),self.features.cpu().numpy()

    def close(self):
        self.closed=True
