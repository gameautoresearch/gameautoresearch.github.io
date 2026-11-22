"""Fresh-policy learning equivalence for CPU restructuring; no checkpoint migration.

Outputs are comparison artifacts, not resumable training checkpoints. Each arm
records the actually loaded native source, with identical pinned public recipes.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]


def worker(out,seed):
    import numpy as np
    import torch
    import taverngym as tg
    from model import Policy
    from rollout import Timer,collect
    from run import ppo_update
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    config=json.loads(Path(__file__).with_name('recipes').joinpath('gpu-throughput.json').read_text())
    config.update(device='cpu',n_games=32,updates=4,graph_inference=False,graph_learner=False,
        allocated_worker_hours=None)
    model=Policy(**config['network']);optimizer=torch.optim.Adam(model.parameters(),lr=config['learning_rate'],eps=1e-5)
    initial=copy.deepcopy(model.state_dict());history=[];timer=Timer('cpu');start=time.perf_counter()
    for update in range(config['updates']):
        eps,records,counts,states=collect(model,config,(seed+update*1000003)%2**64,
            torch.device('cpu'),timer,time.perf_counter()+120,copy_observations=False)
        if counts.get('invalid_lobbies',0) or counts.get('rejected',0):raise AssertionError('invalid learning cohort')
        losses=ppo_update(model,optimizer,eps,config,torch.device('cpu'))
        if not np.isfinite(losses['loss']) or losses['parameter_delta_l2']<=0:raise AssertionError('no finite neural update')
        history.append({'counts':counts,'episodes':records,'states':states,'losses':losses})
    result={'schema':'cpu-learning-equivalence-artifact-v1','not_resumable_checkpoint':True,
        'native':tg.build_info(),'config':config,'seed':seed,'initial':initial,'policy':model.state_dict(),
        'optimizer':optimizer.state_dict(),'history':history,'torch_rng':torch.get_rng_state(),
        'numpy_rng':np.random.get_state(),'python_rng':random.getstate(),
        'wall_seconds':time.perf_counter()-start,'collection_stage_seconds':dict(timer.seconds)}
    torch.save(result,out)


def equal(a,b):
    import numpy as np
    import torch
    if isinstance(a,torch.Tensor):return torch.equal(a,b)
    if isinstance(a,np.ndarray):return np.array_equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--baseline-package',type=Path);p.add_argument('--worker',action='store_true')
    p.add_argument('--seed',type=int,default=12345);a=p.parse_args()
    if a.worker:return worker(a.out,a.seed)
    a.out.mkdir(parents=True,exist_ok=False);rows=[]
    import fcntl,torch
    with (ROOT/'evidence/local-performance.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        for trial,seed in enumerate([12345,45678,78901]):
            data={}
            for arm in (['baseline','candidate'] if trial%2==0 else ['candidate','baseline']):
                dest=a.out/f'{seed}-{arm}.pt'
                package=a.baseline_package if arm=='baseline' else ROOT/'candidates/AutoTavern/python'
                command=[sys.executable,str(Path(__file__)),'--worker','--seed',str(seed),'--out',str(dest.resolve())]
                with dest.with_suffix('.log').open('x') as log:
                    subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT,timeout=300,
                        env=dict(os.environ,PYTHONPATH=str(package.resolve()),TAVERNGYM_WORKERS='pooled',TAVERNGYM_SCHEDULER='dynamic'))
                data[arm]=torch.load(dest,map_location='cpu',weights_only=False)
            checks={key:equal(data['baseline'][key],data['candidate'][key]) for key in
                ['initial','policy','optimizer','history','torch_rng','numpy_rng','python_rng']}
            row={'seed':seed,'checks':checks,'samples':sum(h['counts']['accepted_learning_samples'] for h in data['candidate']['history']),
                'native_sources':{k:v['native'] for k,v in data.items()},
                'wall_seconds':{k:v['wall_seconds'] for k,v in data.items()}}
            rows.append(row)
            (a.out/'comparison.json').write_text(json.dumps({'rows':rows,'passed':all(all(r['checks'].values()) for r in rows)},indent=2)+'\n')
            if not all(checks.values()):raise AssertionError(row)
            print(json.dumps(row),flush=True)


if __name__=='__main__':main()
