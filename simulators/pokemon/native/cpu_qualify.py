"""CPU-only generated collector, thread parity and complete PPO resume canary."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from ..network import Policy
from .bridge import build,Native
from .cpu_env import CPURandomBattles
from .random_resident import task_seeds
from .resident_ppo import ResidentPPO
from .resident_suite import assert_exact_tree


def run(output,generator_path,constructor_path,reference_path,library_path=None,team_size=6):
    output=Path(output);output.mkdir(parents=True,exist_ok=False);library=Path(library_path)if library_path else build(output/'build')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    reference=json.loads(Path(reference_path).read_bytes());specs=reference['specs'];n=32;length=32
    schedule=[[task_seeds(specs[(i+j)%len(specs)]['master'],16,specs[(i+j)%len(specs)]['swapped'],team_size)for j in range(length)]for i in range(n)]
    identities=[];rows=[]
    if torch.cuda.is_initialized():raise RuntimeError('Run CPU-only qualification in a fresh process')
    with patch('torch.cuda._lazy_init',side_effect=AssertionError('CUDA initialization attempted')),patch('torch.cuda.cudart',side_effect=AssertionError('CUDA runtime attempted')):
        for threads in (0,3):
            torch.manual_seed(1987)
            env=CPURandomBattles(library,generator_path,constructor_path,reference['chart'],schedule,threads=threads,developer_scope='CPU-only generated collector and optimizer checkpoint canary; no format admission')
            model=Policy(hidden=64,embedding=8);optimizer=torch.optim.Adam(model.parameters(),lr=.001,eps=1e-5)
            trainer=ResidentPPO(env,model,optimizer,steps=32,minibatch=2048,epochs=2)
            initial=copy.deepcopy(model.state_dict());first=trainer.collect_update();checkpoint=trainer.checkpoint();path=output/f'cpu-{threads}-checkpoint.pt';torch.save(checkpoint,path)
            second=trainer.collect_update();expected=trainer.checkpoint();trainer.restore(torch.load(path,weights_only=False));repeated=trainer.collect_update();actual=trainer.checkpoint()
            assert_exact_tree(actual,expected)
            if second!=repeated:raise AssertionError('CPU rollout continuation changed')
            if not np.all(env.episodes>0):raise AssertionError('Generated canary did not actually reset')
            changed=sum(float((v-initial[k]).square().sum())for k,v in model.state_dict().items())**.5
            if not changed:raise AssertionError('CPU model did not learn')
            altered=copy.deepcopy(actual);altered['environment']['schedule_sha256']='changed'
            try:trainer.restore(altered)
            except ValueError:pass
            else:raise AssertionError('Changed seed schedule accepted')
            if actual['cuda_rng']is not None:raise AssertionError('CPU checkpoint includes CUDA RNG')
            identities.append(actual);rows.append(dict(threads=threads,first=first,second=second,parameter_change_l2=changed,resets=int(env.episodes.sum()),exact_complete_resume=True));env.close()
    # Thread count can change without changing any trajectory or optimizer state.
    assert_exact_tree(identities[1],identities[0])
    with Native(library)as native:
        try:native.require_full_task()
        except RuntimeError:pass
        else:raise AssertionError('Incomplete format passed admission')
    if torch.cuda.is_initialized():raise AssertionError('CPU qualification initialized CUDA')
    result=dict(full_gen9_admission=False,library_sha256=native.binary_sha256,source_signature=native.source_signature,**({}if team_size==6 else {'team_size':team_size}),cuda_initialized=False,exact_serial_parallel_learning=True,exact_model_optimizer_rng_environment_and_cursor_resume=True,scope='Unfiltered finite diagnostic seed schedules; unsupported mechanics remain faults',rows=rows)
    (output/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--generators',required=True);p.add_argument('--constructors',required=True);p.add_argument('--reference',required=True);p.add_argument('--library');p.add_argument('--team-size',type=int,default=6);a=p.parse_args();run(a.output,a.generators,a.constructors,a.reference,a.library,a.team_size)
