"""Synchronous diagnosis of generated CUDA collection against a CPU twin."""
import ctypes as C
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from integrations.pokemon.network import Policy
from integrations.pokemon.native.random_resident import RandomResidentBattles,task_seeds
from integrations.pokemon.native.resident_ppo import ResidentPPO


def run(output,generator_path,constructor_path,reference_path,cpu_library,cuda_library):
    output=Path(output);output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    runtime=C.CDLL('/usr/local/cuda/lib64/libcudart.so');runtime.cudaDeviceSetLimit.argtypes=[C.c_int,C.c_size_t];runtime.cudaDeviceGetLimit.argtypes=[C.POINTER(C.c_size_t),C.c_int]
    previous=C.c_size_t();runtime.cudaDeviceGetLimit(C.byref(previous),0);code=runtime.cudaDeviceSetLimit(0,65536);current=C.c_size_t();runtime.cudaDeviceGetLimit(C.byref(current),0)
    print(f'RANDOM_DEBUG stack_before={previous.value} stack_requested=65536 stack_actual={current.value} status={code}',flush=True)
    if code:raise RuntimeError(f'CUDA stack limit change failed: {code}')
    reference=json.loads(Path(reference_path).read_bytes());specs=reference['specs'];chart=np.asarray(reference['chart'],dtype=np.int8)
    schedule=[[task_seeds(specs[(i+j)%len(specs)]['master'],16,specs[(i+j)%len(specs)]['swapped'])for j in range(32)]for i in range(64)]
    torch.manual_seed(1987);torch.cuda.manual_seed_all(1987)
    cpu=RandomResidentBattles(cpu_library,cuda_library,generator_path,constructor_path,chart,schedule,backend='native_cpu',developer_scope='Synchronous CPU twin diagnosis')
    gpu=RandomResidentBattles(cpu_library,cuda_library,generator_path,constructor_path,chart,schedule,backend='cuda',developer_scope='Synchronous CUDA fault diagnosis')
    cpu.allocate_rollout(32);torch.cuda.synchronize()
    def compare(phase):
        snapshot=gpu.snapshot();states=snapshot['states'].cpu().numpy().tobytes();episodes=snapshot['episodes'].cpu().numpy()
        if states!=cpu.arena.snapshot():raise AssertionError(f'CPU/CUDA state mismatch at {phase}')
        np.testing.assert_array_equal(episodes,cpu.host_episodes);np.testing.assert_array_equal(gpu.obs.cpu().numpy(),cpu.obs.cpu().numpy())
        values=gpu.obs[..., :144]
        if not torch.isfinite(gpu.obs).all()or values.min()<0 or values.max()>=16384:raise AssertionError(f'Invalid observation at {phase}')
    compare('initial');original=gpu.step;count=0;history=[]
    def checked(actions,index,faint_reward=.1):
        nonlocal count
        before=cpu.arena.snapshot();action_copy=actions.cpu().numpy();(output/f'before-{count:03}.bin.gz').write_bytes(gzip.compress(before,mtime=0));np.save(output/f'actions-{count:03}.npy',action_copy)
        if count%8==0:print(f'RANDOM_DEBUG before={count} rollout_index={index} reset_cursors={int(cpu.host_episodes.min())}:{int(cpu.host_episodes.max())}',flush=True)
        cpu.step(actions,index,faint_reward);original(actions,index,faint_reward);torch.cuda.synchronize();compare(count)
        np.testing.assert_array_equal(gpu.flags[index].cpu().numpy(),cpu.flags[index].cpu().numpy());np.testing.assert_array_equal(gpu.rewards[index].cpu().numpy(),cpu.rewards[index].cpu().numpy())
        history.append(dict(phase=count,episodes=int(cpu.host_episodes.sum())));count+=1
    gpu.step=checked
    model=Policy(hidden=64,embedding=8).cuda();optimizer=torch.optim.Adam(model.parameters(),lr=.001,eps=1e-5);trainer=ResidentPPO(gpu,model,optimizer,steps=32,minibatch=4096,epochs=2)
    rows=[]
    try:
        for _ in range(3):rows.append(trainer.collect_update())
        return dict(exact_phases=count,rows=rows,history=history,full_gen9_admission=False)
    finally:
        (output/'history.json').write_text(json.dumps(dict(completed_phases=count,history=history),indent=2)+'\n')
