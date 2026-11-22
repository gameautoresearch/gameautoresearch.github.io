"""Synchronous full-Showdown shadow for CPU-only native PPO trajectories."""
import argparse
import ctypes as C
import gzip
import json
from pathlib import Path
import subprocess

import numpy as np
import torch

from ..network import Policy
from .bridge import build
from .cpu_env import CPURandomBattles
from .cpu_learning_profile import _ShowdownCPU
from .random_resident import task_seeds
from .resident_ppo import ResidentPPO
from .oracle_stage import stage_oracle


def run(output,generator_path,constructor_path,reference_path,n=96,updates=6,limit=128,faint_reward=.1,library_path=None,team_size=6):
    output=Path(output);output.mkdir(parents=True,exist_ok=False);library=Path(library_path)if library_path else build(output/'build');staged=stage_oracle(output,Path(library).parent/'source')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);reference=json.loads(Path(reference_path).read_bytes());specs=reference['specs'];schedule=[[specs[(i+j)%len(specs)]for j in range(32)]for i in range(n)]
    seeds=[[task_seeds(s['master'],limit,s['swapped'],team_size)for s in row]for row in schedule]
    torch.manual_seed(1987);env=CPURandomBattles(library,generator_path,constructor_path,reference['chart'],seeds,threads=3,developer_scope='Synchronous Showdown shadow of native CPU PPO; no format admission');shadow=_ShowdownCPU(schedule,limit,team_size)
    model=Policy(hidden=64,embedding=8);optimizer=torch.optim.Adam(model.parameters(),lr=.001,eps=1e-5);trainer=ResidentPPO(env,model,optimizer,steps=32,minibatch=4096,epochs=2,faint_reward=faint_reward);shadow.allocate_rollout(32)
    histories=[[]for _ in range(n)];phase=0;original=env.step;rows=[]
    def checked(actions,index,faint_reward=faint_reward):
        nonlocal phase
        before=env.arena.snapshot();cursors=env.episodes.copy()
        for history,action in zip(histories,actions.tolist()):history.append(action)
        original(actions,index,faint_reward);shadow.step(actions,index,faint_reward)
        comparisons=[('observation',env.obs.numpy(),shadow.obs.numpy()),('flags',env.flags[index].numpy(),shadow.flags[index].numpy()),('reward',env.rewards[index].numpy(),shadow.rewards[index].numpy())]
        selected=env.flags[index,:,1].bool()
        comparisons.append(('final',env.final_obs[index,selected].numpy(),shadow.final_obs[index,selected].numpy()))
        for kind,actual,expected in comparisons:
            if not np.array_equal(actual,expected):
                where=np.argwhere(actual!=expected);i=int(where[0,0]);detail=dict(phase=phase,kind=kind,differences=[dict(index=pos.tolist(),actual=float(actual[tuple(pos)]),expected=float(expected[tuple(pos)]))for pos in where[:30]])
                (output/'before.bin.gz').write_bytes(gzip.compress(before,mtime=0));(output/'actions.json').write_text(json.dumps(actions.tolist()))
                if kind!='final':
                    spec=schedule[i][int(cursors[i])];seed=seeds[i][int(cursors[i])];text=lambda v:','.join(str((v>>shift)&65535)for shift in (48,32,16,0))
                    request=dict(generated=True,observe=True,**({}if team_size==6 else {'team_size':team_size}),seed=text(seed.battle),team_seeds=[text(v)for v in seed.teams],phases=len(histories[i]),schedule=histories[i]);detail.update(environment=i,master=spec['master'],swapped=spec['swapped'],episode=int(cursors[i]),spec=request)
                    (output/'oracle-fixtures.jsonl').write_text(subprocess.check_output(['node',str(staged/'team_oracle.cjs')],input=json.dumps(dict(cases=[request])),text=True).strip()[1:-1]+'\n')
                (output/'failure.json').write_text(json.dumps(detail,indent=2)+'\n');raise AssertionError(f'CPU shadow mismatch: {detail}')
        np.testing.assert_array_equal(env.episodes,shadow.episodes)
        for i in range(n):
            if cursors[i]!=env.episodes[i]:histories[i].clear()
        phase+=1
    env.step=checked
    try:
        np.testing.assert_array_equal(env.obs.numpy(),shadow.obs.numpy())
        for _ in range(updates):rows.append(trainer.collect_update())
        result=dict(full_gen9_admission=False,library_sha256=env.arena.binary_sha256,source_signature=env.arena.source_signature,**({}if team_size==6 else {'team_size':team_size}),faint_reward=faint_reward,exact_phases=phase,rows=rows,cuda_initialized=torch.cuda.is_initialized());(output/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
    finally:env.close();shadow.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--generators',required=True);p.add_argument('--constructors',required=True);p.add_argument('--reference',required=True);p.add_argument('--updates',type=int,default=6);p.add_argument('--limit',type=int,default=128);p.add_argument('--faint-reward',type=float,default=.1);p.add_argument('--library');p.add_argument('--team-size',type=int,default=6);a=p.parse_args();run(a.output,a.generators,a.constructors,a.reference,updates=a.updates,limit=a.limit,faint_reward=a.faint_reward,library_path=a.library,team_size=a.team_size)
