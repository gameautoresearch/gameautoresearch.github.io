"""Actual-CUDA generated-reset collector and complete checkpoint qualification."""
import copy
import json
from pathlib import Path

import numpy as np
import torch

from ..network import Policy
from .random_resident import RandomResidentBattles,task_seeds
from .resident_ppo import ResidentPPO
from .resident_suite import assert_exact_tree


def run(cpu_library,cuda_library,generator_path,constructor_path,reference_path,output):
    reference=json.loads(Path(reference_path).read_bytes());chart=np.asarray(reference['chart'],dtype=np.int8);specs=reference['specs'];n=64;length=32
    schedule=[[task_seeds(specs[(i+j)%len(specs)]['master'],16,specs[(i+j)%len(specs)]['swapped'])for j in range(length)]for i in range(n)]
    identities=[];rows=[]
    for backend in ('native_cpu','cuda'):
        torch.manual_seed(1987);torch.cuda.manual_seed_all(1987)
        env=RandomResidentBattles(cpu_library,cuda_library,generator_path,constructor_path,chart,schedule,backend=backend,developer_scope='Generated reset and full optimizer resume canary using previously qualified diagnostic pairs; no full-task admission')
        initial_gpu=env.state.cpu().numpy().tobytes()if backend=='cuda' else env.arena.snapshot()
        if initial_gpu!=env.initial_bytes:raise AssertionError('Generated initial CPU/CUDA reset changed')
        model=Policy(hidden=64,embedding=8).cuda();optimizer=torch.optim.Adam(model.parameters(),lr=.001,eps=1e-5);trainer=ResidentPPO(env,model,optimizer,steps=32,minibatch=4096,epochs=2)
        initial={name:value.clone()for name,value in model.state_dict().items()}
        first=trainer.collect_update();checkpoint=trainer.checkpoint();path=Path(output)/f'random-{backend}-checkpoint.pt';torch.save(checkpoint,path)
        second=trainer.collect_update();expected=trainer.checkpoint();trainer.restore(torch.load(path,weights_only=False));repeated=trainer.collect_update();actual=trainer.checkpoint()
        assert_exact_tree(actual,expected)
        if second!=repeated:raise AssertionError('Generated-reset rollout continuation changed')
        state=env.snapshot();raw=state['states'].cpu().numpy().tobytes()if backend=='cuda'else state['states'];cursors=state['episodes'].cpu().numpy()if backend=='cuda'else state['episodes']
        if not np.all(cursors>0):raise AssertionError('Generated canary did not actually auto-reset every game')
        change=sum(float((value-initial[name]).square().sum())for name,value in model.state_dict().items())**.5
        if change==0:raise AssertionError('Generated-reset canary did not update model weights')
        identities.append((copy.deepcopy(model.state_dict()),raw,cursors.copy(),torch.cuda.get_rng_state().clone()))
        rows.append(dict(backend=backend,first=first,second=second,parameter_change_l2=change,resets=int(cursors.sum()),exact_complete_checkpoint_resume=True))
        # Reset cursors and schedule binding are both part of checkpoint identity.
        altered=copy.deepcopy(state);altered['schedule_sha256']='invalid'
        try:env.restore(altered)
        except ValueError:pass
        else:raise AssertionError('Changed reset schedule was accepted')
        env.close();del env,trainer,model,optimizer
    a,b=identities
    if a[1]!=b[1] or not np.array_equal(a[2],b[2]) or not torch.equal(a[3],b[3])or any(not torch.equal(a[0][name],value)for name,value in b[0].items()):raise AssertionError('Generated CPU/CUDA collector or learner differs')
    return dict(full_gen9_admission=False,scope='Finite diagnostic generated-team schedule; full-task distribution is not admitted or filtered for production',exact_cpu_cuda_learning=True,exact_state_optimizer_rng_and_reset_cursor_resume=True,rows=rows)
