"""Admitted CPU random battles with batched GPU observations and exact state.

No native source is changed. Fixed finite schedules are generated from public
training seeds; depletion faults and is never silently repeated or resampled.
The supplied default reserves 65,536 resets per slot, both seat assignments.
"""
import ctypes as C
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from ...native.cpu_env import CPURandomBattles
from ...native.random_resident import task_seeds


class GPUCPU:
    backend='admitted_native_cpu_gpu_learning'
    def __init__(self,config,device):
        manifest=Path(os.environ['POKEMON_PILOT_RUNTIME'])
        runtime=json.loads(manifest.read_bytes());paths={}
        for name in ('library','generators','constructors','chart'):
            paths[name]=manifest.parent/runtime[name]
            if hashlib.sha256(paths[name].read_bytes()).hexdigest()!=runtime['sha256'][name]:raise ValueError('Frozen runtime artifact differs')
        self.n=config['environments'];self.device=torch.device(device)
        # The admitted format's team size (runtime.json, written from the certificate; six when absent).
        self.team_size=team_size=runtime.get('team_size',6)
        length=config.get('reset_schedule_length',8192)
        start=config['training_seed_start']
        if type(length)is not int or not 2<=length<=65536 or start+((self.n+1)//2)*length>=800_000_000:
            raise ValueError('Finite unfiltered training schedule exceeds public domain')
        chart=json.loads(paths['chart'].read_bytes())['chart']
        # This existing data/step implementation is used only after full source-
        # bound production admission, never as a development-scope exemption.
        window=config.get('reset_window',64)
        if type(window)is not int or not 2<=window<=64:raise ValueError('Staged reset window2..64 only')
        self.stream_length=length;self.start=start;self.window=min(window,length);self.offsets=[0]*self.n
        self.cpu=CPURandomBattles(paths['library'],paths['generators'],paths['constructors'],chart,
            [[task_seeds(start+(i//2)*length+j,4096,bool(i%2),team_size)for j in range(self.window)]for i in range(self.n)],
            threads=3,developer_scope='Pilot production wrapper requires full evidence admission below')
        try:
            self.cpu.arena.require_full_task()
            if self.cpu.arena.admission.get('team_size',6)!=team_size:raise ValueError('Runtime team size differs from the admitted format')
            for k in ('generators','constructors','chart'):
                if hashlib.sha256(paths[k].read_bytes()).hexdigest()!=self.cpu.arena.admission['runtime_data_sha256'][k]:raise ValueError('Admitted public data differs')
        except BaseException:
            self.cpu.close();raise
        self.obs=self.cpu.obs.to(device)
        self.host_actions=torch.empty((self.n,2),dtype=torch.int32,pin_memory=self.device.type=='cuda')
        for name in ('source_signature','binary_sha256','rules_sha256','initial_sha256','integration_signature'):
            setattr(self,name,getattr(self.cpu,name))
        self.runtime_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest()

    def allocate_rollout(self,steps):
        self.cpu.allocate_rollout(steps)
        self.flags=torch.empty_like(self.cpu.flags,device=self.device)
        self.rewards=torch.empty_like(self.cpu.rewards,device=self.device)

    def step(self,actions,index,faint_reward=0):
        if faint_reward!=0:raise ValueError('Frozen terminal reward only')
        # Refill before the C++ kernel could exhaust a slot's finite window.
        # Keep its current battle untouched and make that battle local cursor0;
        # the next natural reset consumes the next exact predeclared seed.
        for i in np.flatnonzero(self.cpu.episodes>=self.window-1):
            i=int(i)
            absolute=self.offsets[i]+int(self.cpu.episodes[i])
            if absolute+self.window>self.stream_length:raise RuntimeError('Finite public training seed stream exhausted')
            self.offsets[i]=absolute
            for j in range(self.window):self.cpu.schedule[i*self.window+j]=task_seeds(self.start+(i//2)*self.stream_length+absolute+j,4096,bool(i%2),self.team_size)
            self.cpu.episodes[i]=0
        self.host_actions.copy_(actions)
        self.cpu.step(self.host_actions,index,0)
        self.obs.copy_(self.cpu.obs);self.flags[index].copy_(self.cpu.flags[index]);self.rewards[index].copy_(self.cpu.rewards[index])

    def truncated_views(self):
        selected=self.flags[...,1].bool()[...,None].expand(-1,-1,2).reshape(-1)
        source=self.cpu.flags[...,1].bool()[...,None].expand(-1,-1,2).reshape(-1)
        return self.cpu.final_obs.reshape(-1,700)[source].to(self.device),selected

    def snapshot(self):
        return dict(states=self.cpu.arena.snapshot(),episodes=self.cpu.episodes.tolist(),offsets=list(self.offsets),
            stream=dict(start=self.start,length=self.stream_length,window=self.window,n=self.n))
    def restore(self,state):
        if state['stream']!=dict(start=self.start,length=self.stream_length,window=self.window,n=self.n):raise ValueError('Seed stream binding changed')
        offsets=state['offsets'];episodes=np.asarray(state['episodes'],dtype=np.int32)
        if len(offsets)!=self.n or episodes.shape!=(self.n,)or np.any(episodes<0)or np.any(episodes>=self.window):raise ValueError('Invalid exact reset cursor')
        for i,offset in enumerate(offsets):
            if type(offset)is not int or offset<0 or offset+self.window>self.stream_length:raise ValueError('Invalid stream offset')
            for j in range(self.window):self.cpu.schedule[i*self.window+j]=task_seeds(self.start+(i//2)*self.stream_length+offset+j,4096,bool(i%2),self.team_size)
        self.offsets=list(offsets);self.cpu.episodes[:]=episodes
        self.cpu.arena.restore(state['states']);self.cpu.refresh_observation();self.obs.copy_(self.cpu.obs)
    def close(self):self.cpu.close()
