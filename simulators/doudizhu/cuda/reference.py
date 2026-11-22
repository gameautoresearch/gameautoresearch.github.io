"""Compiled CPU execution of the exact rule header used by CUDA."""
import ctypes as C
import hashlib
from pathlib import Path
import subprocess
import numpy as np
from ..environment import ROOT,ROLES,validate_deal
from ..native_backend import RANKS


class RawState(C.Structure):
    _fields_=[('hand',C.c_uint8*48),('played',C.c_uint8*48),('bottom',C.c_uint8*16),
              ('last',C.c_uint8*48),('history',C.c_uint8*(162*16)),
              ('role',C.c_int),('last_pid',C.c_int),('bombs',C.c_int),('steps',C.c_int),
              ('done',C.c_int),('won',C.c_int)]


def packed_deals(cards):
    hands=np.zeros((len(cards),3,16),dtype=np.int8)
    bottom=np.zeros((len(cards),16),dtype=np.int8)
    for deal in cards:validate_deal(deal)
    if not cards:return hands,bottom
    # Four bulk histograms replace per-card NumPy scalar writes and linear rank
    # lookups. Validation still runs for every complete pinned deal before upload.
    lookup=np.zeros(31,dtype=np.int64)
    lookup[np.asarray(RANKS)]=np.arange(len(RANKS))
    offsets=np.arange(len(cards),dtype=np.int64)[:,None]*16
    for r,role in enumerate((*ROLES,'three_landlord_cards')):
        ranks=np.asarray([c[role] for c in cards],dtype=np.int8)
        counts=np.bincount((offsets+lookup[ranks]).ravel(),minlength=len(cards)*16).reshape(len(cards),16)
        if r<3:hands[:,r]=counts
        else:bottom[:]=counts
    return hands,bottom


class CPUReference:
    def __init__(self,cards):
        here=Path(__file__).parent
        h=hashlib.sha256((here/'rules.h').read_bytes()+(here/'cpu_reference.cpp').read_bytes()).hexdigest()
        out=ROOT/'evidence/doudizhu-implementation/cuda-build-cpu'/h
        out.mkdir(parents=True,exist_ok=True)
        target=out/'reference.so'
        if not target.exists():
            subprocess.run(['c++','-O3','-std=c++17','-shared','-fPIC',str(here/'cpu_reference.cpp'),'-o',str(target)],check=True)
        self.lib=C.CDLL(str(target))
        for name,args,result in [('dg_state_bytes',[],C.c_int),('dg_action_bytes',[],C.c_int),
                                ('dg_cpu_reset',[C.c_void_p]*3+[C.c_int],None),
                                ('dg_cpu_generate',[C.c_void_p]*2,C.c_int),
                                ('dg_cpu_encode',[C.c_void_p]*2,None),('dg_cpu_encode_warp',[C.c_void_p]*2,None),
                                ('dg_cpu_step',[C.c_void_p]*2,C.c_int),
                                ('dg_cpu_choose',[C.c_void_p,C.c_void_p,C.c_int,C.c_int,C.c_int,C.c_uint64,C.c_int,C.c_uint64],C.c_int)]:
            fn=getattr(self.lib,name);fn.argtypes=args;fn.restype=result
        self.state=C.create_string_buffer(self.lib.dg_state_bytes())
        self.actions=C.create_string_buffer(4096*self.lib.dg_action_bytes())
        hand,bottom=packed_deals([cards])
        self.lib.dg_cpu_reset(self.state,C.c_void_p(hand.ctypes.data),C.c_void_p(bottom.ctypes.data),1)
        self.raw=RawState.from_buffer(self.state)

    def observe(self):
        n=self.lib.dg_cpu_generate(self.state,self.actions)
        if n<0:raise RuntimeError('candidate scratch overflow')
        if self.raw.done:return None
        rows=np.frombuffer(self.actions,dtype=np.uint8).reshape(4096,32)[:n,:15]
        legal=tuple(tuple(rank for rank,count in zip(RANKS,row) for _ in range(int(count))) for row in rows)
        features=np.zeros(1240,dtype=np.int8)
        self.lib.dg_cpu_encode(self.state,C.c_void_p(features.ctypes.data))
        warp=np.zeros(1240,dtype=np.int8)
        self.lib.dg_cpu_encode_warp(self.state,C.c_void_p(warp.ctypes.data))
        np.testing.assert_array_equal(features,warp)
        width=319 if self.raw.role==0 else 430
        return legal,features[:width],features[width:width+810].reshape(5,162)

    def step(self,index):
        if self.lib.dg_cpu_step(self.state,C.c_void_p(C.addressof(self.actions)+index*32)):
            raise RuntimeError('CPU shared-rule step fault')

    def choose(self,count,greedy,kind,seed,turn,epsilon):
        return self.lib.dg_cpu_choose(self.state,self.actions,count,greedy,kind,int(seed),turn,int(epsilon*(1<<32)))
