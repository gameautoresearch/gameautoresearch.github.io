"""Native implementation of the ordinary persistent Vector contract.

The runtime receives public frozen tables, never pre-generated teams or oracle
states. Production construction requires admitted native code; explicit developer
scope is limited to differential/integration checks.
"""
import ctypes as C
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from ..efficient import PackedObservation
from ..env import WorkerError
from .bridge import Native
from .random_resident import task_seeds
from .reset import ResetTables, public_tables
from .teamgen import GeneratorTables


class NativeVector:
    def __init__(self,library,generators,constructors,chart,*,workers=3,per_worker=4,developer_scope=None,team_size=6):
        if workers<1 or per_worker<1:raise ValueError('Positive native workers and environments per worker required')
        if type(team_size)is not int or not 1<=team_size<=6:raise ValueError('Team size must be 1..6')
        self._size=workers*per_worker;self.arena=Native(library,self._size,workers)
        try:
            if not developer_scope:self.arena.require_full_task()
            raw,ctor,chart_raw=[Path(p).read_bytes()for p in(generators,constructors,chart)]
            if not developer_scope:
                for key,payload in(('generators',raw),('constructors',ctor),('chart',chart_raw)):
                    if hashlib.sha256(payload).hexdigest()!=self.arena.admission['runtime_data_sha256'][key]:raise ValueError(f'Admitted native public {key} data changed')
                # A certificate qualifies one format; a production vector plays only that team size.
                if self.arena.admission.get('team_size',6)!=team_size:raise ValueError('Team size differs from the admitted format')
            data=json.loads(raw);self.generator=GeneratorTables(data['species'],data['moves'],data['status_moves'],data['items'])
            self.constructor=ResetTables(self.generator,json.loads(ctor)['metadata'],data['moves'])
            self.moves=self.constructor.battle_moves;self.chart=np.asarray(json.loads(chart_raw)['chart'],np.int8)
            self.tables=public_tables(self.constructor,self.chart);self.generator.bind(self.arena);self.lib=self.constructor.bind(self.arena)
            self.rules_sha256=hashlib.sha256(raw+ctor+chart_raw+bytes(self.moves)).hexdigest()
        except BaseException:self.arena.close();raise
        self.seconds=0.;self.calls=0;self.states=None;self.seeds=[];self.swapped=[];self.limit=0
        self.developer_scope=developer_scope;self.team_size=team_size

    @property
    def size(self):return self._size

    @property
    def worker_metrics(self):return dict(native_batch_seconds=self.seconds,native_batch_calls=self.calls,node_processes=0)

    def _reset_slot(self,i,seed,swapped,limit):
        spec=task_seeds(seed,limit,swapped,self.team_size)
        if not self.lib.pkn_reset_battle(C.byref(self.generator.rules),C.byref(self.constructor.rules),C.byref(spec),C.byref(self.arena.records[i]),self.moves,len(self.moves),self.chart.ctypes.data):
            raise WorkerError(f'Native reset fault {self.arena.records[i].fault} for task seed {seed}','simulator_or_adapter_fault')
        self.seeds[i]=seed;self.swapped[i]=bool(swapped)

    def _observe(self):
        flat=self.tables.observe(self.arena)
        result=[]
        for i,s in enumerate(self.arena.records):
            if s.fault:
                category='policy_illegal_action'if s.fault==1 else 'simulator_or_adapter_fault'
                raise WorkerError(f'Native simulator fault {s.fault} for task seed {self.seeds[i]}',category)
            obs=[PackedObservation(flat[i,side].copy())for side in range(2)]
            potential=[]
            for side in range(2):
                foe=1-side;public_dead=sum(s.public_mons[foe][s.public_order[foe][j]].fainted for j in range(s.public_count[foe]))
                own_dead=sum(p.maxhp>0 and s.requests[side].hp[k]==0 for k,p in enumerate(s.mons[side]));potential.append((public_dead-own_dead)/6)
            winner=f'p{s.winner+1}'if s.winner>=0 else ''
            text=lambda value:','.join(str((value>>shift)&65535)for shift in(48,32,16,0))
            spec=task_seeds(self.seeds[i],self.limit,self.swapped[i],self.team_size)
            result.append(dict(seed=self.seeds[i],seeds=dict(battle=text(spec.battle),p1=text(spec.teams[0]),p2=text(spec.teams[1])),
                steps=s.phases,turn=min(s.turn,1000),terminated=bool(s.terminated),truncated=bool(s.truncated),
                reason='decision_limit'if s.truncated else 'natural'if s.terminated else None,winner=winner if s.terminated else None,
                returns=[1 if s.winner==side else -1 for side in range(2)]if s.terminated and s.winner>=0 else [0,0],
                ready=[not(s.terminated or s.truncated)and not bool(s.requests[side].waiting)for side in range(2)],
                observations=obs,potential=potential,rejections=list(s.unavailable_choices)))
        return result

    def reset(self,seeds,limit,swapped=None):
        if len(seeds)!=self.size or limit<1:raise ValueError('Reset requires vector.size seeds and a positive decision limit')
        swapped=[False]*self.size if swapped is None else swapped
        if len(swapped)!=self.size:raise ValueError('Seat assignment count differs')
        started=time.perf_counter();self.seeds=[None]*self.size;self.swapped=[False]*self.size;self.limit=limit
        for i,(seed,flip)in enumerate(zip(seeds,swapped,strict=True)):self._reset_slot(i,seed,flip,limit)
        self.states=self._observe();self.seconds+=time.perf_counter()-started;self.calls+=1
        return self.states

    def step(self,actions):
        if self.states is None or len(actions)!=self.size:raise ValueError('Step requires initialized vector.size actions')
        translated=[]
        for state,pair in zip(self.states,actions,strict=True):
            if len(pair)!=2:raise ValueError('Two player actions required')
            row=[]
            for side,action in enumerate(pair):
                if state['terminated']or state['truncated']:row.append(-1);continue
                if not state['ready'][side]:
                    if action is not None:raise WorkerError('Action for waiting seat','policy_illegal_action')
                    row.append(-1)
                else:
                    if type(action)is not int or not 0<=action<14 or not state['observations'][side]['mask'][action]:raise WorkerError(f'Masked action {action}','policy_illegal_action')
                    row.append(action)
            translated.append(row)
        started=time.perf_counter();self.arena.step(translated,self.moves,self.chart);self.states=self._observe()
        self.seconds+=time.perf_counter()-started;self.calls+=1
        return self.states

    def reset_slots(self,updates,limit):
        if self.states is None or len(updates)!=self.size or limit!=self.limit:raise ValueError('Slot reset shape or decision limit differs')
        started=time.perf_counter()
        for i,update in enumerate(updates):
            if update is None:continue
            if not(self.states[i]['terminated']or self.states[i]['truncated']):raise ValueError('Cannot reset a live battle')
            self._reset_slot(i,update[0],update[1],limit)
        self.states=self._observe();self.seconds+=time.perf_counter()-started;self.calls+=1
        return self.states

    def snapshot(self):
        return dict(source_signature=self.arena.source_signature,binary_sha256=self.arena.binary_sha256,rules_sha256=self.rules_sha256,
                    size=self.size,states=self.arena.snapshot(),seeds=list(self.seeds),swapped=list(self.swapped),limit=self.limit,
                    **({}if self.team_size==6 else {'team_size':self.team_size}))

    def restore(self,state):
        for key,value in dict(source_signature=self.arena.source_signature,binary_sha256=self.arena.binary_sha256,rules_sha256=self.rules_sha256,size=self.size).items():
            if state[key]!=value:raise ValueError(f'Native vector checkpoint {key} changed')
        if state.get('team_size',6)!=self.team_size:raise ValueError('Native vector checkpoint team_size changed')
        self.arena.restore(state['states']);self.seeds=list(state['seeds']);self.swapped=list(state['swapped']);self.limit=state['limit'];self.states=self._observe()

    def close(self):self.arena.close()
    def __enter__(self):return self
    def __exit__(self,*args):self.close()
