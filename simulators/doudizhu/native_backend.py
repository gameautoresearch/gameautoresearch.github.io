"""Source-hashed C++ batch engine, public packed observations and upstream audit mode."""
from __future__ import annotations
from collections import Counter
from collections.abc import Sequence
import ctypes as C
from dataclasses import dataclass
import fcntl
from functools import cached_property
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from .environment import ROOT,ROLES,Game,validate_deal

RANKS=(3,4,5,6,7,8,9,10,11,12,13,14,17,20,30)
_BUILD_SECONDS=0.
_LIB=None


def library():
    global _LIB,_BUILD_SECONDS
    started=time.perf_counter()
    source=Path(__file__).parent/'native/engine.cpp'
    content=source.read_bytes()
    source_hash=hashlib.sha256(content).hexdigest()
    if _LIB is not None:
        if _LIB.source_hash!=source_hash:
            raise RuntimeError('Native source changed inside a live process; restart it')
        return _LIB
    key=hashlib.sha256(content+b'c++17-O3-pthread-v2').hexdigest()
    out=ROOT/'evidence/doudizhu-implementation/native-build'/key
    out.mkdir(parents=True,exist_ok=True)
    target=out/('engine.dylib' if sys.platform=='darwin' else 'engine.so')
    with (out/'build.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if not target.exists():
            frozen_source=out/'source.cpp'
            frozen_source.write_bytes(content)
            temporary=target.with_suffix('.tmp'+target.suffix)
            command=['c++','-O3','-std=c++17','-shared','-fPIC','-pthread',str(frozen_source),'-o',str(temporary)]
            completed=subprocess.run(command,capture_output=True,text=True)
            (out/'compile.log').write_text(completed.stdout+completed.stderr)
            if completed.returncode:
                raise RuntimeError(f'Native compilation failed; see {out}/compile.log')
            temporary.replace(target)
            (out/'command.txt').write_text(' '.join(command)+'\n')
            (out/'compiler.txt').write_text(subprocess.check_output(['c++','--version'],text=True))
            (out/'receipt.json').write_text(json.dumps({'source_sha256':source_hash,
                'binary_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
                'build_seconds':time.perf_counter()-started,'platform':sys.platform},indent=2)+'\n')
        if hashlib.sha256((out/'source.cpp').read_bytes()).hexdigest()!=source_hash:
            raise RuntimeError('Native cached build source mismatch')
        receipt=json.loads((out/'receipt.json').read_text())
        if receipt['binary_sha256']!=hashlib.sha256(target.read_bytes()).hexdigest():
            raise RuntimeError('Native cached binary hash mismatch')
    lib=C.CDLL(str(target))
    p=C.c_void_p
    i=C.c_int
    pi=C.POINTER(C.c_int32)
    pb=C.POINTER(C.c_int8)
    for name,args,restype in [('db_new',[i],p),('db_delete',[p],None),('db_error',[],C.c_char_p),
                             ('db_reset',[p,pb,pb,i],i),('db_step',[p,pi,pi,i],i),('db_frame',[p,pi,i],i),
                             ('db_meta',[p],pi),('db_features',[p],pb),('db_hand',[p],pb),('db_cards',[p],pb),
                             ('db_lens',[p],pb),('db_actions',[p],pb),('db_heuristic',[p],i)]:
        fn=getattr(lib,name)
        fn.argtypes=args
        fn.restype=restype
    _LIB=lib
    _LIB.source_hash=source_hash
    _BUILD_SECONDS+=time.perf_counter()-started
    return lib


@dataclass(frozen=True)
class LegalActions(Sequence):
    lengths: bytes
    cards: bytes

    def __len__(self):
        return len(self.lengths)

    def __getitem__(self,index):
        if isinstance(index,slice):
            return tuple(self[i] for i in range(*index.indices(len(self))))
        if index<0:
            index+=len(self)
        if not 0<=index<len(self):
            raise IndexError(index)
        return tuple(self.cards[index*20:index*20+self.lengths[index]])


@dataclass(frozen=True)
class NativeObservation:
    role: str
    hand: tuple
    last_pid: str
    legal: LegalActions
    features: bytes
    action_features: bytes

    @cached_property
    def _encoding(self):
        width=319 if self.role=='landlord' else 430
        a=np.frombuffer(self.features,dtype=np.int8)
        return a[:width],a[width:].reshape(5,162),np.frombuffer(self.action_features,dtype=np.int8).reshape(-1,54)

    def encode(self,cache=True,backend='direct'):
        if backend!='direct':
            raise ValueError('Native observations require direct features')
        return self._encoding


class NativeVectorGames:
    def __init__(self,threads=1,record_limit=None):
        if not 1<=threads<=256:
            raise ValueError('Native thread count must be 1..256')
        self.lib=library()
        self.pointer=self.lib.db_new(threads)
        if not self.pointer:
            raise RuntimeError(self.lib.db_error().decode())
        self.closed=False
        self.audit={}
        self.objective='wp'
        self.record_limit=record_limit

    def _check(self,status):
        if status:
            raise RuntimeError('Native simulator fault: '+self.lib.db_error().decode())

    def _live(self):
        if self.closed:
            raise RuntimeError('Simulator workers are closed')

    def _frame(self,ids):
        frame=self.arrays(ids)
        return self._decode(frame)

    def arrays(self,ids):
        ids=np.asarray(ids,dtype=np.int32)
        n=len(ids)
        self._check(self.lib.db_frame(self.pointer,ids.ctypes.data_as(C.POINTER(C.c_int32)),n))
        meta=np.ctypeslib.as_array(self.lib.db_meta(self.pointer),shape=(n*9,)).reshape(n,9)
        features=np.ctypeslib.as_array(self.lib.db_features(self.pointer),shape=(n,1240))
        hands=np.ctypeslib.as_array(self.lib.db_hand(self.pointer),shape=(n,20))
        total=int(meta[:,3].sum()) if n else 0
        cards=np.ctypeslib.as_array(self.lib.db_cards(self.pointer),shape=(total,20)) if total else None
        lens=np.ctypeslib.as_array(self.lib.db_lens(self.pointer),shape=(total,)) if total else None
        actions=np.ctypeslib.as_array(self.lib.db_actions(self.pointer),shape=(total,54)) if total else None
        return {'ids':ids,'meta':meta,'features':features,'hands':hands,'cards':cards,'lens':lens,'actions':actions}

    def _decode(self,frame,only=None):
        ids,meta,features,hands,cards,lens,actions=(frame[k] for k in ('ids','meta','features','hands','cards','lens','actions'))
        rows={}
        for j,index in enumerate(ids):
            if only is not None and int(index) not in only:
                continue
            role,hand_n,offset,count,bombs,decisions,done,won,last_pid=map(int,meta[j])
            index=int(index)
            trace=None
            if done:
                sign=1 if won else -1
                unit=2**bombs
                target={'wp':1,'adp':unit,'logadp':bombs+1}[self.objective]
                obs=None
                outcome={'landlord_won':bool(won),'bombs':bombs,'decisions':decisions,
                         'returns':{r:float(sign*target*(1 if r=='landlord' else -1)) for r in ROLES},
                         'scores':{r:sign*unit*(2 if r=='landlord' else -1) for r in ROLES},
                         'terminal':'natural_empty_hand','truncated':False}
            else:
                width=319 if role==0 else 430
                obs=NativeObservation(ROLES[role],tuple(map(int,hands[j,:hand_n])),ROLES[last_pid],
                    LegalActions(lens[offset:offset+count].tobytes(),cards[offset:offset+count].tobytes()),
                    features[j,:width+810].tobytes(),actions[offset:offset+count].tobytes())
                outcome=None
            if index in self.audit:
                oracle=self.audit[index]
                if done:
                    if oracle.outcome()!=outcome:
                        raise AssertionError('Native/oracle terminal mismatch')
                    trace=oracle.trajectory()
                else:
                    expected=oracle.observe()
                    assert (expected.role,expected.hand,expected.last_pid)==(obs.role,obs.hand,obs.last_pid)
                    assert expected.legal==tuple(obs.legal),'Native legal set differs from frozen upstream'
                    for a,b in zip(expected.encode(backend='upstream'),obs.encode()):
                        np.testing.assert_array_equal(a,b)
            rows[index]=obs,outcome,trace
        return rows

    def advance_arrays(self,ids,choices):
        for i,game in self.audit.items():
            position=np.searchsorted(ids,i)
            if position<len(ids) and ids[position]==i:
                game.step(int(choices[position]))
        self._check(self.lib.db_step(self.pointer,ids.ctypes.data_as(C.POINTER(C.c_int32)),
                                   choices.ctypes.data_as(C.POINTER(C.c_int32)),len(ids)))
        frame=self.arrays(ids)
        if self.audit:
            self._decode(frame,only=self.audit)
        return frame

    def reset(self,deals,objective='wp',record=False):
        self._live()
        if objective not in ('wp','adp','logadp'):
            raise ValueError('Invalid objective')
        self.objective=objective
        self.size=len(deals)
        hands=np.zeros((len(deals),3,15),dtype=np.int8)
        bottom=np.zeros((len(deals),15),dtype=np.int8)
        for i,cards in enumerate(deals):
            validate_deal(cards)
            for r,role in enumerate(ROLES):
                c=Counter(cards[role])
                hands[i,r]=[c[rank] for rank in RANKS]
            c=Counter(cards['three_landlord_cards'])
            bottom[i]=[c[rank] for rank in RANKS]
        self._check(self.lib.db_reset(self.pointer,hands.ctypes.data_as(C.POINTER(C.c_int8)),
                                    bottom.ctypes.data_as(C.POINTER(C.c_int8)),len(deals)))
        self.audit={i:Game(cards,objective,record=True,engine_backend='upstream') for i,cards in enumerate(deals)
                    if self.record_limit is None or i<self.record_limit} if record else {}
        return self._frame(list(range(len(deals)))) if deals else {}

    def step(self,actions):
        self._live()
        for i,a in actions.items():
            if not isinstance(a,(int,np.integer)):
                raise ValueError('Illegal action index')
            if i in self.audit:
                self.audit[i].step(a)
        ids=np.array(list(actions),dtype=np.int32)
        choices=np.array(list(actions.values()),dtype=np.int32)
        self._check(self.lib.db_step(self.pointer,ids.ctypes.data_as(C.POINTER(C.c_int32)),
                                   choices.ctypes.data_as(C.POINTER(C.c_int32)),len(ids)))
        return self._frame(ids) if len(ids) else {}

    def simulate_heuristic(self):
        self._live()
        if self.audit:
            raise ValueError('Bulk simulator timing requires record=False')
        self._check(self.lib.db_heuristic(self.pointer))
        return self._frame(list(range(self.size)))

    def close(self):
        if not self.closed:
            self.lib.db_delete(self.pointer)
            self.closed=True
            self.audit={}
