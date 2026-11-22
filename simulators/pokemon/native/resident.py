"""Native CPU and resident CUDA environment paths sharing rules and observations.

Full Gen 9 admission is still closed. Callers must explicitly identify their
component-development scope; template resets do not claim native random teams.
"""
import ctypes as C
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .bridge import Native, State
from .rules import compile_moves
from .teams_check import load_team_fixture
from .visible import PublicTables, PublicRules


def fixture_rules(fixtures):
    from .form_data import enrich_fixture
    for data in fixtures:enrich_fixture(data)
    species, moves, records = {}, {}, {}
    for data in fixtures:
        for entry in data['metadata']['species']:
            if entry['id'] in species and species[entry['id']] != entry:
                raise ValueError('Fixture species metadata changed')
            species[entry['id']] = entry
        for entry in data['metadata']['moves']:
            if entry['id'] in moves and moves[entry['id']] != entry:
                raise ValueError('Fixture move metadata changed')
            moves[entry['id']] = entry
        records.update({entry['id']: entry for entry in data['moves']})
    names = sorted(records)
    ids = {name: index for index, name in enumerate(names)}
    metadata = dict(fixtures[0]['metadata'])
    for data in fixtures:
        for name in ('ability_tokens','item_tokens','type_tokens','status_tokens'):
            incoming=data['metadata'][name];existing=metadata[name]
            # IDs are append-only. Older retained corpora need no rewriting;
            # a different token at any existing ID remains a hard failure.
            if incoming[:min(len(incoming),len(existing))]!=existing[:min(len(incoming),len(existing))]:
                raise ValueError('Mixed fixture metadata/source versions')
            if len(incoming)!=len(existing)and name not in ('ability_tokens','item_tokens'):
                raise ValueError('Mixed fixture fixed metadata sizes')
            if len(incoming)>len(existing):metadata[name]=incoming
    metadata['species'] = [species[name] for name in sorted(species)]
    metadata['moves'] = [moves[name] for name in sorted(moves)]
    if metadata['species'][0]['id'] != '':
        raise ValueError('Missing unknown-species metadata')
    chart = np.asarray(fixtures[0]['chart'], dtype=np.int8)
    if not all(np.array_equal(chart,data['chart']) for data in fixtures):
        raise ValueError('Fixture type charts changed')
    tables = PublicTables(metadata, chart, ids)
    return compile_moves([records[name] for name in names]), chart, tables, ids


class ResidentBattles:
    def __init__(self, cpu_library, cuda_library, fixtures, n, *, backend='cuda',
                 cpu_threads=3, phase_limit=512, developer_scope=None, diversify=False):
        if not developer_scope:
            raise RuntimeError('Full Gen 9 admission is closed; explicit component-development scope required')
        if backend not in ('cuda','native_cpu') or n < 1:
            raise ValueError('Invalid native environment backend/size')
        self.backend, self.n, self.developer_scope = backend, n, developer_scope
        self.moves, self.chart, self.tables, ids = fixture_rules(fixtures)
        self.arena = Native(cpu_library,n,cpu_threads if backend=='native_cpu' else 0)
        for i in range(n):
            with Native(cpu_library) as template:
                load_team_fixture(template,fixtures[i%len(fixtures)],ids,self.tables.species_ids)
                record=template.records[0];record.limit=phase_limit
                if diversify:
                    record.rng ^= ((i//len(fixtures))*0x9E3779B97F4A7C15)&((1<<64)-1)
                C.memmove(C.addressof(self.arena.records[i]),template.snapshot(),C.sizeof(State))
        self.initial_bytes=self.arena.snapshot()
        self.initial_sha256=hashlib.sha256(self.initial_bytes).hexdigest()
        self.integration_signature=hashlib.sha256(b''.join((Path(__file__).parent.parent/name).read_bytes()
                                                         for name in ('network.py','efficient.py'))).hexdigest()
        self.initial_host=np.frombuffer(self.initial_bytes,dtype=np.uint8).copy()
        self.initial_pointer=self.initial_host.ctypes.data
        self.stream=torch.cuda.current_stream().cuda_stream
        self.obs=torch.empty((n,2,700),device='cuda')
        self.actions=torch.empty((n,2),dtype=torch.int32,device='cuda')
        self.lib=C.CDLL(str(cuda_library))
        from .cuda_runtime import prepare
        self.runtime_limits=prepare(self.lib)
        self.lib.pkn_observe_device.argtypes=[C.c_void_p,C.c_void_p,C.c_int,C.POINTER(PublicRules),C.c_void_p]
        self.lib.pkn_env_step_device.argtypes=[C.c_void_p,C.c_void_p,C.c_void_p,C.c_int,C.c_void_p,C.c_int,C.c_void_p,
            C.POINTER(PublicRules),C.c_void_p,C.c_void_p,C.c_void_p,C.c_void_p,C.c_double,C.c_void_p]
        self.lib.pkn_step_device.argtypes=[C.c_void_p,C.c_void_p,C.c_int,C.c_void_p,C.c_int,C.c_void_p,C.c_void_p]
        self.lib.pkn_masks_device.argtypes=[C.c_void_p,C.c_void_p,C.c_int,C.c_void_p]
        self.source_signature=self.arena.source_signature
        self.binary_sha256=hashlib.sha256(Path(cuda_library if backend=='cuda' else cpu_library).read_bytes()).hexdigest()
        self.rules_sha256=hashlib.sha256(bytes(self.moves)+self.chart.tobytes()+json.dumps(self.tables.species_ids,sort_keys=True).encode()+b''.join(bytes(x) for x in self.tables.host_buffers.values())).hexdigest()
        if backend=='cuda':
            self.state=torch.from_numpy(self.initial_host).cuda()
            self.initial=self.state.clone()
            self.move_data=torch.from_numpy(np.frombuffer(bytes(self.moves),dtype=np.uint8).copy()).cuda()
            self.chart_data=torch.from_numpy(self.chart).cuda()
            self.device_rules=self.tables.to_cuda()
        else:
            l=self.arena.lib
            l.pkn_observe_f32.argtypes=[C.c_void_p,C.c_void_p,C.POINTER(PublicRules)]
            l.pkn_env_step.argtypes=[C.c_void_p,C.c_void_p,C.c_void_p,C.c_int,C.c_void_p,C.c_void_p,
                C.POINTER(PublicRules),C.c_void_p,C.c_void_p,C.c_void_p,C.c_void_p,C.c_double]
            self.host_obs=torch.empty((n,2,700),pin_memory=True)
            self.host_actions=torch.empty((n,2),dtype=torch.int32,pin_memory=True)
        self.refresh_observation()

    def refresh_observation(self):
        if self.backend=='cuda':
            code=self.lib.pkn_observe_device(self.state.data_ptr(),self.obs.data_ptr(),self.n,C.byref(self.device_rules),self.stream)
            if code:raise RuntimeError(f'CUDA observer launch failed: {code}')
        else:
            self.arena.lib.pkn_observe_f32(self.arena.ptr,self.host_obs.data_ptr(),C.byref(self.tables.rules))
            self.obs.copy_(self.host_obs,non_blocking=True)

    def allocate_rollout(self, steps):
        self.flags=torch.empty((steps,self.n,3),dtype=torch.int32,device='cuda')
        self.rewards=torch.empty((steps,self.n,2),device='cuda')
        self.final_obs=torch.empty((steps,self.n,2,700),device='cuda') if self.backend=='cuda' else None
        if self.backend=='native_cpu':
            self.host_flags=torch.empty((steps,self.n,3),dtype=torch.int32,pin_memory=True)
            self.host_rewards=torch.empty((steps,self.n,2),pin_memory=True)
            self.host_final=torch.empty((steps,self.n,2,700),pin_memory=True)

    def step(self, actions, index, faint_reward=.1):
        self.actions.copy_(actions)
        if self.backend=='cuda':
            code=self.lib.pkn_env_step_device(self.state.data_ptr(),self.initial.data_ptr(),self.actions.data_ptr(),self.n,
                self.move_data.data_ptr(),len(self.moves),self.chart_data.data_ptr(),C.byref(self.device_rules),
                self.obs.data_ptr(),self.final_obs[index].data_ptr(),self.rewards[index].data_ptr(),self.flags[index].data_ptr(),faint_reward,self.stream)
            if code:raise RuntimeError(f'CUDA environment launch failed: {code}')
        else:
            self.host_actions.copy_(self.actions,non_blocking=True)
            torch.cuda.current_stream().synchronize()
            self.arena.lib.pkn_env_step(self.arena.ptr,self.host_actions.data_ptr(),C.addressof(self.moves),len(self.moves),
                self.chart.ctypes.data,self.initial_pointer,C.byref(self.tables.rules),self.host_obs.data_ptr(),
                self.host_final[index].data_ptr(),self.host_rewards[index].data_ptr(),self.host_flags[index].data_ptr(),faint_reward)
            self.obs.copy_(self.host_obs,non_blocking=True)
            self.rewards[index].copy_(self.host_rewards[index],non_blocking=True)
            self.flags[index].copy_(self.host_flags[index],non_blocking=True)

    def truncated_views(self):
        if self.backend=='cuda':
            selected=self.flags[...,1].bool()[...,None].expand(-1,-1,2).reshape(-1)
            return self.final_obs.reshape(-1,700)[selected],selected
        selected_host=self.host_flags[...,1].bool()[...,None].expand(-1,-1,2).reshape(-1)
        return self.host_final.reshape(-1,700)[selected_host].cuda(),selected_host.cuda()

    def snapshot(self):
        if self.backend=='cuda':return self.state.clone()
        return self.arena.snapshot()

    def restore(self, state):
        if self.backend=='cuda':self.state.copy_(state)
        else:self.arena.restore(state)
        self.refresh_observation()

    def close(self):self.arena.close()
