"""Serialized, counterbalanced full-loop ablation and worker-scaling measurement.

No paid execution. Every candidate uses the same trained weights, seeds, vector
size, deterministic decisions and PPO settings; all trajectories must match.
"""
from __future__ import annotations
import argparse,fcntl,json,time,statistics,resource,platform,copy
from pathlib import Path
import numpy as np
import torch
from .cuda_suite import config
from .cli import initialize,load_policy
from .env import Vector
from .efficient import EfficientInference
from .rl import Timings,rollout,update
from .provenance import ROOT,manifest,write_json,digest

VARIANTS={
 'packed':dict(binary=False,token_cache=False,flat_copy=False),
 'flat':dict(binary=False,token_cache=False,flat_copy=True),
 'binary':dict(binary=True,token_cache=False,flat_copy=True),
 'cached':dict(binary=False,token_cache=True,flat_copy=True),
 'stream':dict(binary=True,token_cache=True,flat_copy=True),
 'memo':dict(binary=True,token_cache=True,mon_cache=True,flat_copy=True),
 'engine':dict(binary=True,token_cache=True,mon_cache=True,fast_callbacks=True,flat_copy=True),
}

def case(cfg,cp,variant,workload,out,warm_batches=1):
    options=VARIANTS[variant];t=Timings();started=time.perf_counter()
    m=initialize(cfg);m=load_policy(cp,cfg)[0]
    opt=torch.optim.Adam(m.parameters(),lr=cfg['learning_rate'],eps=1e-5)
    inf=EfficientInference('cpu',t,flat_copy=options['flat_copy'])
    h=cfg['hardware'];batches=[]
    with Vector(h['workers'],h['environments_per_worker'],packed=True,
                binary=options['binary'],token_cache=options['token_cache'],mon_cache=options.get('mon_cache',False),fast_callbacks=options.get('fast_callbacks',False)) as v:
        for batch_index in range(warm_batches):
            begin=time.perf_counter();seeds=list(range(20000,20000+v.size))
            assignments=[]
            for i in range(v.size):
                if workload=='tails':pair=[m,m]
                else:
                    pair=['power' if i%2==0 else 'random']*2;pair[(i//2)%2]=m
                assignments.append(pair)
            samples,report=rollout(v,m,assignments,seeds,cfg,np.random.default_rng(1),inf,t,deterministic=True)
            # Snapshot collection time before hashes/learner/checkpoint overhead.
            collect_seconds=time.perf_counter()-begin
            hashes=v.hashes()
            stats=update(m,opt,samples,dict(cfg,optimization='packed'),'cpu',t)
            batches.append(dict(collect_seconds=collect_seconds,samples=len(samples),
                traces=[g['sha256'] for g in hashes],games=report['results'],
                late_samples=report['late_policy_samples_turn50'],ppo=stats))
        torch.save({'model':m.state_dict(),'optimizer':opt.state_dict()},out)
        metrics=v.worker_metrics
    seconds=time.perf_counter()-started
    return dict(variant=variant,workload=workload,seconds=seconds,batches=batches,
                samples=sum(b['samples'] for b in batches),timings=t.result(),worker_metrics=metrics,
                checkpoint_sha256=digest(out),rss_max_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if platform.system()=='Darwin' else 1024))

def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--output',required=True)
    p.add_argument('--repeats',type=int,default=5);p.add_argument('--scaling',action='store_true')
    p.add_argument('--batches',type=int,default=1);args=p.parse_args()
    cfg=config('cpu',2,4);cfg['hardware']['machine']='local-'+platform.machine()
    cp=ROOT/'evidence/pokemon-implementation/learning-verified/checkpoint.pt'
    torch.set_num_threads(1)
    with (ROOT/'evidence/local-performance.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        out=Path(args.output);begin=time.perf_counter()
        mh=manifest(out,cfg,'structural-profile',{'checkpoint_sha256':digest(cp),'seeds':'20000..','heldout_used':False,'repeats':args.repeats,'batches':args.batches})
        rows=[]
        plans=[(2,4)] if not args.scaling else [(1,24),(2,12),(3,8)]
        for workers,envs in plans:
            c=copy.deepcopy(cfg);c['hardware'].update(workers=workers,environments_per_worker=envs)
            for workload in ('mixed','tails'):
                for trial in range(args.repeats):
                    variants=list(VARIANTS) if not args.scaling else ['packed','stream']
                    if trial%2:variants.reverse()
                    for variant in variants:
                        label=f'w{workers}e{envs}-{workload}-{trial}-{variant}'
                        row=case(c,cp,variant,workload,out/(label+'.pt'),args.batches)
                        row.update(trial=trial,workers=workers,envs=envs)
                        rows.append(row)
                        print(json.dumps({k:row[k] for k in ('variant','workload','workers','envs','trial','seconds','samples')}),flush=True)
                        write_json(out/'partial.json',{'rows':rows})
        # Parity is per workload/vector size/batch, not across different RNG batch shapes.
        groups={}
        for row in rows:
            key=f"w{row['workers']}e{row['envs']}-{row['workload']}"
            groups.setdefault(key,[]).append(row)
        summary={}
        for key,group in groups.items():
            ref=group[0]['batches']
            for r in group:
                assert [b['traces'] for b in r['batches']]==[b['traces'] for b in ref],(key,r['variant'])
                assert [b['ppo'] for b in r['batches']]==[b['ppo'] for b in ref],(key,r['variant'],'ppo')
            summary[key]={v:statistics.median(r['seconds'] for r in group if r['variant']==v) for v in set(r['variant'] for r in group)}
        write_json(out/'results.json',dict(rows=rows,median_seconds=summary,exact_trajectories_and_ppo=True,
            manifest_sha256=mh,lock_exclusive=True,allocation_seconds=time.perf_counter()-begin,paid_cost_usd=0))
if __name__=='__main__':main()
