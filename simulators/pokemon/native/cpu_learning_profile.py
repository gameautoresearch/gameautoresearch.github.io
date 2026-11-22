"""Matched CPU-only collector+PPO comparison with full Showdown as reference.

Uses retained implemented-team diagnostic seeds. It neither admits a reduced
format for production nor claims whole-format throughput.
"""
import argparse
import copy
import fcntl
import hashlib
import json
from pathlib import Path
import statistics
import time
import traceback

import numpy as np
import torch

from ..env import Vector
from ..network import Policy
from .bridge import build
from .cpu_env import CPURandomBattles
from .random_resident import task_seeds
from .resident_ppo import ResidentPPO
from .resident_suite import assert_exact_tree


class _ShowdownCPU:
    """Benchmark adapter to the original engine, not a resumable production API."""
    def __init__(self,schedule,limit,team_size=6):
        self.n=len(schedule);self.schedule=schedule;self.limit=limit;self.episodes=np.zeros(self.n,dtype=np.int32)
        self.vector=Vector(3,self.n//3,packed=True,binary=True,shared=True,lazy=True,team_size=team_size)
        self.obs=torch.empty((self.n,2,700),dtype=torch.float32);self._load(self.vector.reset([r[0]['master']for r in schedule],limit,[r[0]['swapped']for r in schedule]))

    def _load(self,states):
        self.obs.numpy()[:]=np.asarray([[obs.flat for obs in s['observations']]for s in states],dtype=np.float32)
        self.potential=np.asarray([s['potential']for s in states],dtype=np.float64)

    def allocate_rollout(self,steps):
        self.flags=torch.empty((steps,self.n,3),dtype=torch.int32);self.rewards=torch.empty((steps,self.n,2));self.final_obs=torch.empty((steps,self.n,2,700))

    def step(self,actions,index,faint_reward=.1):
        states=self.vector.step([[int(a)if a>=0 else None for a in row]for row in actions.tolist()])
        potential=np.asarray([s['potential']for s in states],dtype=np.float64);rewards=faint_reward*(potential-self.potential)
        updates=[];self.flags[index].zero_()
        for i,s in enumerate(states):
            terminal=bool(s['terminated']);truncated=bool(s['truncated']);self.flags[index,i,0]=terminal;self.flags[index,i,1]=truncated
            if terminal:rewards[i]+=np.asarray(s['returns'])
            if truncated:self.final_obs[index,i].numpy()[:]=np.asarray([obs.flat for obs in s['observations']],dtype=np.float32)
            if terminal or truncated:
                self.episodes[i]+=1
                if self.episodes[i]>=len(self.schedule[i]):raise RuntimeError('Diagnostic reset schedule exhausted')
                spec=self.schedule[i][self.episodes[i]];updates.append((spec['master'],spec['swapped'],self.limit))
            else:updates.append(None)
        self.rewards[index].numpy()[:]=rewards.astype(np.float32)
        if any(u is not None for u in updates):states=self.vector.reset_slots(updates,self.limit)
        self._load(states)

    def truncated_views(self):
        selected=self.flags[...,1].bool()[...,None].expand(-1,-1,2).reshape(-1)
        return self.final_obs.reshape(-1,700)[selected],selected

    def close(self):self.vector.close()


def run(output,generator_path,constructor_path,reference_path,n=96,repeats=5,limit=128):
    if n<3 or n%3 or repeats<3:raise ValueError('Multiple of three environments and at least three repetitions required')
    root=Path(__file__).resolve().parents[3]
    with(root/'evidence/local-performance.lock').open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);output=Path(output);output.mkdir(parents=True,exist_ok=False)
        torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
        library=build(output/'build');raw=Path(reference_path).read_bytes();reference=json.loads(raw);specs=reference['specs'];length=32
        schedule=[[specs[(i+j)%len(specs)]for j in range(length)]for i in range(n)]
        seeds=[[task_seeds(s['master'],limit,s['swapped'])for s in row]for row in schedule]
        entries={};rows=[];audit={};setup={}
        try:
            for backend in ('showdown_cpu','native_cpp_cpu'):
                tick=time.perf_counter();torch.manual_seed(1987)
                env=_ShowdownCPU(schedule,limit)if backend=='showdown_cpu'else CPURandomBattles(library,generator_path,constructor_path,reference['chart'],seeds,threads=3,developer_scope='Matched diagnostic CPU full-loop comparison; no full-format admission')
                model=Policy(hidden=64,embedding=8);optimizer=torch.optim.Adam(model.parameters(),lr=.001,eps=1e-5);trainer=ResidentPPO(env,model,optimizer,steps=32,minibatch=4096,epochs=2)
                setup[backend]=time.perf_counter()-tick
                digest=hashlib.sha256();original=env.step
                def checked(actions,index,faint_reward=.1,_env=env,_original=original,_digest=digest):
                    _digest.update(_env.obs.numpy().tobytes());_digest.update(actions.numpy().tobytes());_original(actions,index,faint_reward)
                    _digest.update(_env.obs.numpy().tobytes());_digest.update(_env.flags[index].numpy().tobytes());_digest.update(_env.rewards[index].numpy().tobytes());selected=_env.flags[index,:,1].bool();_digest.update(_env.final_obs[index,selected].numpy().tobytes())
                env.step=checked;warm=trainer.collect_update();env.step=original
                audit[backend]=dict(digest=digest.hexdigest(),row=warm)
                entries[backend]=dict(env=env,model=model,optimizer=optimizer,trainer=trainer,rng=torch.get_rng_state(),rows=[])
            if audit['showdown_cpu']!=audit['native_cpp_cpu']:raise AssertionError(f'CPU paired trajectory audit differs: {audit}')
            for repeat in range(repeats):
                order=('showdown_cpu','native_cpp_cpu')if repeat%2==0 else('native_cpp_cpu','showdown_cpu')
                for backend in order:
                    entry=entries[backend];torch.set_rng_state(entry['rng']);started=time.perf_counter();row=entry['trainer'].collect_update();row.update(seconds=time.perf_counter()-started,backend=backend,repeat=repeat);row['samples_per_second']=row['samples']/row['seconds'];entry['rng']=torch.get_rng_state();entry['rows'].append(row);rows.append(row)
                    (output/'partial.json').write_text(json.dumps(dict(full_gen9_admission=False,audit=audit,rows=rows),indent=2)+'\n')
                    print(f'CPU_PPO {repeat} {backend} {row["samples"]} {row["seconds"]:.6f}s',flush=True)
                a,b=entries['showdown_cpu'],entries['native_cpp_cpu']
                for key in ('samples','natural','truncations','faults'):
                    if a['rows'][-1][key]!=b['rows'][-1][key]:raise AssertionError(f'Paired rollout count differs: {key}')
                assert_exact_tree(a['model'].state_dict(),b['model'].state_dict());assert_exact_tree(a['optimizer'].state_dict(),b['optimizer'].state_dict());torch.testing.assert_close(a['rng'],b['rng'],rtol=0,atol=0)
                np.testing.assert_array_equal(a['env'].episodes,b['env'].episodes);torch.testing.assert_close(a['env'].obs,b['env'].obs,rtol=0,atol=0)
            medians={backend:statistics.median(r['samples_per_second']for r in entry['rows'])for backend,entry in entries.items()}
            result=dict(full_gen9_admission=False,scope='CPU simulation/reset/host observations/inference/GAE/PPO on retained diagnostic seeds with implemented mechanics; no full-format speedup claim',environments=n,simulator_workers=3,torch_threads=1,phase_limit=limit,rollout_steps=32,ppo_epochs=2,minibatch=4096,exact_first_rollout_audit=audit,exact_all_paired_models_optimizer_rng_and_final_views=True,median_samples_per_second=medians,matched_diagnostic_learning_speedup=medians['native_cpp_cpu']/medians['showdown_cpu'],one_time_setup_seconds=setup,reference_sha256=hashlib.sha256(raw).hexdigest(),rows=rows,cuda_initialized=torch.cuda.is_initialized())
            (output/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items()if k!='rows'},indent=2))
        except BaseException as error:
            (output/'failure.json').write_text(json.dumps(dict(full_gen9_admission=False,error=str(error),traceback=traceback.format_exc(),audit=audit,rows=rows),indent=2)+'\n');raise
        finally:
            for entry in entries.values():entry['env'].close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--generators',required=True);p.add_argument('--constructors',required=True);p.add_argument('--reference',required=True);p.add_argument('--n',type=int,default=96);p.add_argument('--repeats',type=int,default=5);p.add_argument('--limit',type=int,default=128);a=p.parse_args();run(a.output,a.generators,a.constructors,a.reference,a.n,a.repeats,a.limit)
