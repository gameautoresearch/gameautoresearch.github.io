"""Native generated resets for the development resident collector.

The finite seed schedule is immutable checkpoint input. Unknown mechanics and
schedule exhaustion are faults, never reasons to filter teams or reuse seeds.
Full frozen-format admission remains closed in Native.require_full_task().
"""
import ctypes as C
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .bridge import Native,State
from .resident import ResidentBattles
from .reset import ResetTables,ResetRules,ResetSeeds,public_tables
from .teamgen import GeneratorTables,GenRules
from .validate import seed64
from .visible import PublicRules


def task_seeds(master,limit=512,swapped=False,team_size=6):
    """Exact worker.cjs pokemon-v1 domain separation for integer task seeds.

    team_size is the format's Max Team Size. Six is stored as 0, the bytes every
    schedule had before the field existed, so six-Pokémon schedules hash as before."""
    if not isinstance(master,int) or isinstance(master,bool) or limit<1:raise ValueError('Integer task seed and positive phase limit required')
    if type(team_size) is not int or not 1<=team_size<=6:raise ValueError('Team size must be 1..6')
    def derive(domain):
        raw=hashlib.sha256(f'pokemon-v1:{master}:{domain}'.encode()).digest()
        return seed64(','.join(str(int.from_bytes(raw[i:i+2],'little')) for i in (0,2,4,6)))
    teams=('team2','team1') if swapped else ('team1','team2')
    return ResetSeeds(derive('battle'),(C.c_uint64*2)(*[derive(domain) for domain in teams]),limit,0 if team_size==6 else team_size)


class RandomResidentBattles(ResidentBattles):
    def __init__(self,cpu_library,cuda_library,generator_path,constructor_path,chart,schedule,*,backend='cuda',cpu_threads=3,developer_scope=None):
        if not developer_scope:raise RuntimeError('Native full-task admission is closed; explicit component scope required')
        if backend not in ('cuda','native_cpu') or not schedule or not schedule[0]:raise ValueError('Nonempty rectangular reset schedule and native backend required')
        n=len(schedule);length=len(schedule[0])
        if any(len(row)!=length or any(not isinstance(seed,ResetSeeds) for seed in row) for row in schedule):raise ValueError('Expected rectangular ResetSeeds schedule')
        self.backend=backend;self.n=n;self.developer_scope=developer_scope;self.schedule_length=length
        self.seed_schedule=(ResetSeeds*(n*length))(*[seed for row in schedule for seed in row]);self.schedule_sha256=hashlib.sha256(bytes(self.seed_schedule)).hexdigest()
        self.first_seeds=(ResetSeeds*n)(*[row[0] for row in schedule]);self.host_episodes=np.zeros(n,dtype=np.int32)
        generator_raw=Path(generator_path).read_bytes();constructor_raw=Path(constructor_path).read_bytes()
        data=json.loads(generator_raw);metadata=json.loads(constructor_raw)['metadata']
        self.generator=GeneratorTables(data['species'],data['moves'],data['status_moves'],data['items'])
        self.constructor=ResetTables(self.generator,metadata,data['moves']);self.moves=self.constructor.battle_moves
        self.chart=np.ascontiguousarray(chart,dtype=np.int8)
        if self.chart.shape!=(18,18) or not np.all(np.isin(self.chart,[-127,-1,0,1])):raise ValueError('Invalid frozen type chart')
        self.tables=public_tables(self.constructor,self.chart);self.arena=Native(cpu_library,n,cpu_threads if backend=='native_cpu' else 0)
        self.generator.bind(self.arena);lib=self.constructor.bind(self.arena)
        for i in range(n):lib.pkn_reset_battle(C.byref(self.generator.rules),C.byref(self.constructor.rules),C.byref(self.first_seeds[i]),C.byref(self.arena.records[i]),self.moves,len(self.moves),self.chart.ctypes.data)
        self.initial_bytes=self.arena.snapshot();self.initial_sha256=hashlib.sha256(bytes(self.seed_schedule)+self.initial_bytes).hexdigest()
        self.integration_signature=hashlib.sha256(b''.join((Path(__file__).parent.parent/name).read_bytes() for name in ('network.py','efficient.py'))).hexdigest()
        self.source_signature=self.arena.source_signature
        self.binary_sha256=hashlib.sha256(Path(cuda_library if backend=='cuda' else cpu_library).read_bytes()).hexdigest()
        self.rules_sha256=hashlib.sha256(generator_raw+constructor_raw+bytes(self.moves)+self.chart.tobytes()+b''.join(bytes(x) for x in self.tables.host_buffers.values())).hexdigest()
        self.stream=torch.cuda.current_stream().cuda_stream;self.obs=torch.empty((n,2,700),device='cuda');self.actions=torch.empty((n,2),dtype=torch.int32,device='cuda')
        self.lib=C.CDLL(str(cuda_library))
        from .cuda_runtime import prepare
        self.runtime_limits=prepare(self.lib)
        self.lib.pkn_observe_device.argtypes=[C.c_void_p,C.c_void_p,C.c_int,C.POINTER(PublicRules),C.c_void_p]
        self.lib.pkn_random_reset_device.argtypes=[C.POINTER(GenRules),C.POINTER(ResetRules)]+[C.c_void_p]*3+[C.c_int,C.c_void_p,C.c_int,C.c_void_p,C.c_void_p]
        self.lib.pkn_random_env_step_device.argtypes=[C.c_void_p,C.c_void_p,C.c_int,C.c_void_p,C.c_void_p,C.c_int,C.POINTER(GenRules),C.POINTER(ResetRules),C.c_void_p,C.c_int,C.c_void_p,C.POINTER(PublicRules)]+[C.c_void_p]*4+[C.c_double,C.c_void_p]
        if backend=='cuda':
            def upload(value):return torch.from_numpy(np.frombuffer(bytes(value),dtype=np.uint8).copy()).cuda()
            self.state=torch.empty(n*C.sizeof(State),dtype=torch.uint8,device='cuda');self.schedule_data=upload(self.seed_schedule);self.episodes=torch.zeros(n,dtype=torch.int32,device='cuda')
            self.move_data=upload(self.moves);self.chart_data=torch.from_numpy(self.chart).cuda();self.device_rules=self.tables.to_cuda()
            self.device_generator=self.generator.to_cuda();self.device_constructor=self.constructor.to_cuda();initial=upload(self.first_seeds);ok=torch.empty(n,dtype=torch.int32,device='cuda')
            code=self.lib.pkn_random_reset_device(C.byref(self.device_generator),C.byref(self.device_constructor),initial.data_ptr(),self.state.data_ptr(),ok.data_ptr(),n,self.move_data.data_ptr(),len(self.moves),self.chart_data.data_ptr(),self.stream)
            if code:raise RuntimeError(f'CUDA random initial reset failed: {code}')
        else:
            self.host_actions=torch.empty((n,2),dtype=torch.int32,pin_memory=True);self.host_obs=torch.empty((n,2,700),pin_memory=True)
            lib.pkn_observe_f32.argtypes=[C.c_void_p,C.c_void_p,C.POINTER(PublicRules)]
            lib.pkn_random_env_step.argtypes=[C.c_void_p,C.c_void_p,C.c_void_p,C.c_int,C.c_void_p,C.POINTER(GenRules),C.POINTER(ResetRules),C.c_void_p,C.c_int,C.c_void_p,C.POINTER(PublicRules)]+[C.c_void_p]*4+[C.c_double]
        self.refresh_observation()

    def step(self,actions,index,faint_reward=.1):
        self.actions.copy_(actions)
        if self.backend=='cuda':
            code=self.lib.pkn_random_env_step_device(self.state.data_ptr(),self.schedule_data.data_ptr(),self.schedule_length,self.episodes.data_ptr(),self.actions.data_ptr(),self.n,C.byref(self.device_generator),C.byref(self.device_constructor),self.move_data.data_ptr(),len(self.moves),self.chart_data.data_ptr(),C.byref(self.device_rules),self.obs.data_ptr(),self.final_obs[index].data_ptr(),self.rewards[index].data_ptr(),self.flags[index].data_ptr(),faint_reward,self.stream)
            if code:raise RuntimeError(f'CUDA random environment launch failed: {code}')
        else:
            self.host_actions.copy_(self.actions,non_blocking=True);torch.cuda.current_stream().synchronize()
            self.arena.lib.pkn_random_env_step(self.arena.ptr,self.host_actions.data_ptr(),C.addressof(self.moves),len(self.moves),self.chart.ctypes.data,C.byref(self.generator.rules),C.byref(self.constructor.rules),C.addressof(self.seed_schedule),self.schedule_length,self.host_episodes.ctypes.data,C.byref(self.tables.rules),self.host_obs.data_ptr(),self.host_final[index].data_ptr(),self.host_rewards[index].data_ptr(),self.host_flags[index].data_ptr(),faint_reward)
            for target,source in ((self.obs,self.host_obs),(self.rewards[index],self.host_rewards[index]),(self.flags[index],self.host_flags[index])):target.copy_(source,non_blocking=True)

    def snapshot(self):
        return dict(states=self.state.clone() if self.backend=='cuda' else self.arena.snapshot(),episodes=self.episodes.clone() if self.backend=='cuda' else self.host_episodes.copy(),schedule_sha256=self.schedule_sha256)

    def restore(self,state):
        if state['schedule_sha256']!=self.schedule_sha256:raise ValueError('Native reset schedule changed')
        if self.backend=='cuda':self.state.copy_(state['states']);self.episodes.copy_(state['episodes'])
        else:
            episodes=np.asarray(state['episodes'])
            if episodes.shape!=(self.n,) or episodes.dtype!=np.int32 or np.any(episodes<0) or np.any(episodes>self.schedule_length):raise ValueError('Invalid native reset cursor')
            self.arena.restore(state['states']);self.host_episodes[:]=episodes
        self.refresh_observation()
