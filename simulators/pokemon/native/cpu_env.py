"""CPU-only native batched environments; complete-format admission stays gated.

Simulation and observation writes run directly in the persistent C++ workers.
Torch views share their host buffers. No CUDA runtime, device copy or Node IPC is
used by these developer-only environments.
"""
import ctypes as C
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .bridge import Native,State
from .resident import fixture_rules
from .teams_check import load_team_fixture
from .visible import PublicRules
from .reset import ResetTables,ResetRules,ResetSeeds,public_tables
from .teamgen import GeneratorTables,GenRules


class _CPUBase:
    backend='native_cpu_only'

    def _bind(self,library,n,threads,scope):
        if not scope:raise RuntimeError('Full Gen 9 admission is closed; explicit component scope required')
        if n<1 or threads<0:raise ValueError('Positive environment count and nonnegative thread count required')
        self.n=n;self.developer_scope=scope;self.arena=Native(library,n,threads)
        self.source_signature=self.arena.source_signature;self.binary_sha256=self.arena.binary_sha256
        self.integration_signature=hashlib.sha256(b''.join((Path(__file__).parent.parent/name).read_bytes()for name in ('network.py','efficient.py'))).hexdigest()
        self.obs=torch.empty((n,2,700),dtype=torch.float32,device='cpu');self.actions=torch.empty((n,2),dtype=torch.int32,device='cpu')
        self.arena.lib.pkn_observe_f32.argtypes=[C.c_void_p,C.c_void_p,C.POINTER(PublicRules)]

    def refresh_observation(self):
        self.arena.lib.pkn_observe_f32(self.arena.ptr,self.obs.data_ptr(),C.byref(self.tables.rules))

    def allocate_rollout(self,steps):
        if steps<1:raise ValueError('Positive rollout length required')
        self.flags=torch.empty((steps,self.n,3),dtype=torch.int32,device='cpu')
        self.rewards=torch.empty((steps,self.n,2),dtype=torch.float32,device='cpu');self.final_obs=torch.empty((steps,self.n,2,700),dtype=torch.float32,device='cpu')

    def _actions(self,actions,index):
        if actions.device.type!='cpu' or tuple(actions.shape)!=(self.n,2):raise ValueError('Expected CPU actions [environments,2]')
        if not 0<=index<len(self.flags):raise IndexError('Rollout output index outside allocation')
        self.actions.copy_(actions)

    def truncated_views(self):
        selected=self.flags[...,1].bool()[...,None].expand(-1,-1,2).reshape(-1)
        return self.final_obs.reshape(-1,700)[selected],selected

    def close(self):self.arena.close()


class CPUBattles(_CPUBase):
    """Constructed fixtures for kernel/collector development, not production reset."""
    def __init__(self,library,fixtures,n,*,threads=0,phase_limit=512,developer_scope=None):
        self._bind(library,n,threads,developer_scope)
        self.moves,self.chart,self.tables,ids=fixture_rules(fixtures)
        for i in range(n):
            with Native(library)as template:
                load_team_fixture(template,fixtures[i%len(fixtures)],ids,self.tables.species_ids)
                template.records[0].limit=phase_limit
                C.memmove(C.addressof(self.arena.records[i]),template.snapshot(),C.sizeof(State))
        self.initial=self.arena.snapshot();self.initial_host=np.frombuffer(self.initial,dtype=np.uint8).copy()
        self.initial_sha256=hashlib.sha256(self.initial).hexdigest()
        self.rules_sha256=hashlib.sha256(bytes(self.moves)+self.chart.tobytes()+json.dumps(self.tables.species_ids,sort_keys=True).encode()+b''.join(bytes(x)for x in self.tables.host_buffers.values())).hexdigest()
        self.arena.lib.pkn_env_step.argtypes=[C.c_void_p,C.c_void_p,C.c_void_p,C.c_int,C.c_void_p,C.c_void_p,C.POINTER(PublicRules)]+[C.c_void_p]*4+[C.c_double]
        self.refresh_observation()

    def step(self,actions,index,faint_reward=.1):
        self._actions(actions,index)
        self.arena.lib.pkn_env_step(self.arena.ptr,self.actions.data_ptr(),C.addressof(self.moves),len(self.moves),self.chart.ctypes.data,self.initial_host.ctypes.data,C.byref(self.tables.rules),self.obs.data_ptr(),self.final_obs[index].data_ptr(),self.rewards[index].data_ptr(),self.flags[index].data_ptr(),faint_reward)

    def snapshot(self):return self.arena.snapshot()

    def restore(self,state):self.arena.restore(state);self.refresh_observation()


class CPURandomBattles(_CPUBase):
    """Unfiltered finite seed schedules with explicit missing-mechanics faults."""
    def __init__(self,library,generator_path,constructor_path,chart,schedule,*,threads=0,developer_scope=None):
        if not schedule or not schedule[0]:raise ValueError('Nonempty seed schedule required')
        length=len(schedule[0]);n=len(schedule)
        if any(len(row)!=length or any(not isinstance(seed,ResetSeeds)for seed in row)for row in schedule):raise ValueError('Expected rectangular ResetSeeds schedule')
        self._bind(library,n,threads,developer_scope);self.schedule_length=length
        self.schedule=(ResetSeeds*(n*length))(*[seed for row in schedule for seed in row]);self.episodes=np.zeros(n,dtype=np.int32)
        self.schedule_sha256=hashlib.sha256(bytes(self.schedule)).hexdigest()
        generator_raw=Path(generator_path).read_bytes();constructor_raw=Path(constructor_path).read_bytes();data=json.loads(generator_raw)
        self.generator=GeneratorTables(data['species'],data['moves'],data['status_moves'],data['items'])
        self.constructor=ResetTables(self.generator,json.loads(constructor_raw)['metadata'],data['moves']);self.moves=self.constructor.battle_moves
        self.chart=np.ascontiguousarray(chart,dtype=np.int8)
        if self.chart.shape!=(18,18)or not np.all(np.isin(self.chart,[-127,-1,0,1])):raise ValueError('Invalid frozen type chart')
        self.tables=public_tables(self.constructor,self.chart);self.generator.bind(self.arena);lib=self.constructor.bind(self.arena)
        for i,row in enumerate(schedule):lib.pkn_reset_battle(C.byref(self.generator.rules),C.byref(self.constructor.rules),C.byref(row[0]),C.byref(self.arena.records[i]),self.moves,len(self.moves),self.chart.ctypes.data)
        self.initial_sha256=hashlib.sha256(bytes(self.schedule)+self.arena.snapshot()).hexdigest()
        self.rules_sha256=hashlib.sha256(generator_raw+constructor_raw+bytes(self.moves)+self.chart.tobytes()+b''.join(bytes(x)for x in self.tables.host_buffers.values())).hexdigest()
        lib.pkn_random_env_step.argtypes=[C.c_void_p,C.c_void_p,C.c_void_p,C.c_int,C.c_void_p,C.POINTER(GenRules),C.POINTER(ResetRules),C.c_void_p,C.c_int,C.c_void_p,C.POINTER(PublicRules)]+[C.c_void_p]*4+[C.c_double]
        self.refresh_observation()

    def step(self,actions,index,faint_reward=.1):
        self._actions(actions,index)
        self.arena.lib.pkn_random_env_step(self.arena.ptr,self.actions.data_ptr(),C.addressof(self.moves),len(self.moves),self.chart.ctypes.data,C.byref(self.generator.rules),C.byref(self.constructor.rules),C.addressof(self.schedule),self.schedule_length,self.episodes.ctypes.data,C.byref(self.tables.rules),self.obs.data_ptr(),self.final_obs[index].data_ptr(),self.rewards[index].data_ptr(),self.flags[index].data_ptr(),faint_reward)

    def snapshot(self):return dict(states=self.arena.snapshot(),episodes=self.episodes.copy(),schedule_sha256=self.schedule_sha256)

    def restore(self,state):
        if state['schedule_sha256']!=self.schedule_sha256:raise ValueError('Native reset schedule changed')
        episodes=np.asarray(state['episodes'])
        if episodes.shape!=(self.n,)or episodes.dtype!=np.int32 or np.any(episodes<0)or np.any(episodes>self.schedule_length):raise ValueError('Invalid native reset cursor')
        self.arena.restore(state['states']);self.episodes[:]=episodes;self.refresh_observation()
