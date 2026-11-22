"""Matched episode-count static vs work-conserving queue full-loop profile."""
import argparse,fcntl,json,time,statistics,platform
from pathlib import Path
import numpy as np
import torch
from .cuda_suite import config
from .cli import initialize,load_policy
from .env import Vector
from .rl import rollout,rollout_queue,update,Timings,inference_for
from .provenance import ROOT,manifest,write_json,digest

def run(out,device='cpu',repeats=3,episodes=192,slots=(24,48,96,192),batches=1):
    out=Path(out);cfg=config(device,3,8);cfg.update(optimization='stream')
    cfg['hardware']['machine']='local-'+platform.machine() if device=='cpu' else 'Modal-L4-4physical-8GiB'
    cp=ROOT/'evidence/pokemon-implementation/learning-verified/checkpoint.pt'
    if not cp.exists():cp=ROOT/'input-policy.pt'
    manifest(out,cfg,'queue-profile',dict(episodes=episodes,checkpoint_sha256=digest(cp),slots=slots,repeats=repeats))
    rows=[];torch.set_num_threads(1);started=time.perf_counter()
    for workload in ('mixed','tails'):
        for trial in range(repeats):
            sizes=list(slots)
            if trial%2:sizes.reverse()
            for size in sizes:
                c=json.loads(json.dumps(cfg));c['hardware']['environments_per_worker']=size//3
                m=initialize(c);m=load_policy(cp,c)[0];t=Timings();inf=inference_for(c,t)
                opt=torch.optim.Adam(m.parameters(),lr=c['learning_rate'],eps=1e-5)
                begin=time.perf_counter();records=[]
                with Vector(3,size//3,packed=True,binary=True,token_cache=True) as v:
                    for iteration in range(batches):
                        seeds=list(range(20000,20000+episodes));assignments=[]
                        for i in range(episodes):
                            pair=[m,m] if workload=='tails' else ['power' if i%2==0 else 'random']*2
                            if workload!='tails':pair[(i//2)%2]=m
                            assignments.append(pair)
                        collect=rollout if size==episodes else rollout_queue
                        b=time.perf_counter();samples,report=collect(v,m,assignments,seeds,c,np.random.default_rng(0),inf,t,deterministic=True)
                        collected=time.perf_counter()-b
                        stats=update(m,opt,samples,c,device,t)
                        records.append(dict(collect_seconds=collected,samples=len(samples),ppo=stats,report=report))
                    torch.save(m.state_dict(),out/f'{workload}-{trial}-{size}.pt')
                    metrics=v.worker_metrics
                row=dict(workload=workload,trial=trial,slots=size,episodes=episodes,seconds=time.perf_counter()-begin,records=records,timings=t.result(),worker_metrics=metrics)
                rows.append(row);write_json(out/'partial.json',{'rows':rows})
                print(json.dumps({k:row[k] for k in ('workload','trial','slots','seconds')}),flush=True)
    # Canonical trace identity across vector shapes is a qualification check. It
    # may fail on a numerical argmax tie; retain the evidence rather than hide it.
    parity={}
    medians={}
    for w in ('mixed','tails'):
        group=[r for r in rows if r['workload']==w]
        reference=next(r for r in group if r['slots']==episodes)['records'][0]['report']['results']
        hashes=[g['trace_sha256'] for g in reference]
        parity[w]={str(s):all([g['trace_sha256'] for g in r['records'][0]['report']['results']]==hashes for r in group if r['slots']==s) for s in slots}
        medians[w]={str(s):statistics.median(r['seconds'] for r in group if r['slots']==s) for s in slots}
    result=dict(rows=rows,trace_parity=parity,median_seconds=medians,allocation_seconds=time.perf_counter()-started)
    write_json(out/'results.json',result);return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--repeats',type=int,default=3);p.add_argument('--episodes',type=int,default=192)
    args=p.parse_args()
    with (ROOT/'evidence/local-performance.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);run(args.output,repeats=args.repeats,episodes=args.episodes)
if __name__=='__main__':main()
