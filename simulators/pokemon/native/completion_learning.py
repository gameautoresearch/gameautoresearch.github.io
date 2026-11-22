"""Retain a real CPU PPO policy on unfiltered task seeds before qualification.

Development use only. This does not enable full-format production admission.
Every accepted decision is newly collected; PPO epoch reuse is reported separately.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import torch

from ..network import Policy
from .bridge import build
from .cpu_env import CPURandomBattles
from .random_resident import task_seeds
from .resident_ppo import ResidentPPO
from .resident_suite import assert_exact_tree


def run(output,generators,constructors,chart_path,updates=64,n=96,start=1120000,team_size=6):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter();library=build(output/'build')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.manual_seed(1987)
    chart=json.loads(Path(chart_path).read_bytes())['chart']
    schedule=[[task_seeds(start+(i//2)*128+j,4096,bool(i%2),team_size)for j in range(128)]for i in range(n)]
    env=CPURandomBattles(library,generators,constructors,chart,schedule,threads=3,
        developer_scope='Unfiltered development PPO policy retention and exact resume; production admission closed')
    model=Policy(hidden=64,embedding=8);optimizer=torch.optim.Adam(model.parameters(),lr=.001,eps=1e-5)
    trainer=ResidentPPO(env,model,optimizer,steps=32,minibatch=4096,epochs=2)
    initial={k:v.clone()for k,v in model.state_dict().items()};rows=[]
    try:
        for i in range(updates):
            row=trainer.collect_update();rows.append(row)
            if row['truncations']:raise AssertionError('Natural-game retention encountered a phase cap')
            if i%8==7:print(f'PPO {i+1}/{updates} samples={sum(r["samples"]for r in rows)} natural={sum(r["natural"]for r in rows)}',flush=True)
        checkpoint=trainer.checkpoint();path=output/'checkpoint.pt';torch.save(checkpoint,path)
        expected_row=trainer.collect_update();expected=trainer.checkpoint()
        trainer.restore(torch.load(path,weights_only=False));actual_row=trainer.collect_update()
        assert_exact_tree(trainer.checkpoint(),expected)
        if actual_row!=expected_row:raise AssertionError('Complete continuation rollout changed')
        change=sum(float((v-initial[k]).square().sum())for k,v in checkpoint['model'].items())**.5
        if not change:raise AssertionError('Model parameters did not change')
        result=dict(full_gen9_admission=False,scope='Unfiltered native CPU development training; differential qualification remains separate',
            updates=updates,environments=n,seed_start=start,seed_schedule_sha256=env.schedule_sha256,
            accepted_new_rows=sum(r['samples']for r in rows),natural_games=sum(r['natural']for r in rows),
            truncations=sum(r['truncations']for r in rows),faults=sum(r['faults']for r in rows),ppo_epochs=2,
            parameter_change_l2=change,exact_full_resume=True,cuda_initialized=torch.cuda.is_initialized(),
            source_signature=checkpoint['source_signature'],integration_signature=checkpoint['integration_signature'],
            checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),process_seconds=time.perf_counter()-started,
            timing_is_performance_comparison=False,rows=rows)
        (output/'results.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps({k:v for k,v in result.items()if k!='rows'},indent=2))
    except BaseException as error:
        torch.save(trainer.checkpoint(),output/'failure-checkpoint.pt')
        (output/'failure.json').write_text(json.dumps(dict(full_gen9_admission=False,error=str(error),completed_updates=len(rows),rows=rows),indent=2)+'\n')
        raise
    finally:env.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--generators',required=True);p.add_argument('--constructors',required=True);p.add_argument('--chart',required=True);p.add_argument('--updates',type=int,default=64);p.add_argument('--team-size',type=int,default=6);a=p.parse_args();run(a.output,a.generators,a.constructors,a.chart,a.updates,team_size=a.team_size)
