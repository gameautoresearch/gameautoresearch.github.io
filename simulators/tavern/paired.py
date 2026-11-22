#!/usr/bin/env python3
"""Same-hardware initial and busy trained-policy ablation with complete neural PPO updates."""
import argparse
import copy
import json
import os
from pathlib import Path
import random
import statistics
import time

from gpu_suite import run_command, HERE


def paired(out,device,checkpoint,trials=3):
    import torch
    from migrate_performance_checkpoint import migrate
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    checkpoint=migrate(checkpoint,out/'pilot-compatible.pt')
    c=json.loads((HERE/'recipes/profile-t4-copy1.json').read_text())
    c.update(device=device,n_games=32,profile_updates=2,updates=2,snapshot_every=0,torch_threads=1)
    c['machine'].update(label='fixed-four-core-comparison',attached_cpu_threads=4,attached_ram_gib=16,gpu_model='NVIDIA L4' if device.startswith('cuda') else None)
    rows=[];started=time.perf_counter()
    for workload in ['initial','busy-trained']:
        for trial in range(trials):
            modes=[False,True] if trial%2==0 else [True,False]
            for optimized in modes:
                cc=copy.deepcopy(c)
                cc.update(worker_backend='pooled' if optimized else 'scoped',fast_inference=optimized,
                          skip_finished_lobbies=optimized,defer_learner_metrics=optimized,
                          start_turn=8 if workload=='busy-trained' else 1)
                name=f'{workload}-trial{trial}-'+('optimized' if optimized else 'baseline')
                row=run_command(out/name,cc,'profile',checkpoint if workload=='busy-trained' else None)
                row.update(workload=workload,trial=trial,optimized=optimized)
                row['loop_seconds']=sum(row['profile']['stage_seconds'].values())
                counts=row['training']['counts']
                row['ready_decisions_per_second']=counts['ready_learning_decisions']/row['loop_seconds']
                row['learner_samples_per_second']=counts['accepted_learning_samples']/row['loop_seconds']
                row['updates_per_second']=2/row['loop_seconds']
                updates=[json.loads(x) for x in (out/name/'metrics.jsonl').read_text().splitlines()]
                row['update_timings']=[{k:u[k] for k in ['update','phase','iteration_wall_seconds_excluding_checkpoint','stage_seconds_this_update']} for u in updates]
                row['warmed_update_seconds']=updates[-1]['iteration_wall_seconds_excluding_checkpoint']
                row['warmed_samples_per_second']=updates[-1]['counts']['accepted_learning_samples']/row['warmed_update_seconds']
                rows.append(row)
                print(json.dumps({'completed':name,'device':device,'seconds':row['loop_seconds'],'samples':counts['accepted_learning_samples']}),flush=True)
    checks=[]
    for workload in ['initial','busy-trained']:
        for trial in range(trials):
            a=next(r for r in rows if r['workload']==workload and r['trial']==trial and not r['optimized'])
            b=next(r for r in rows if r['workload']==workload and r['trial']==trial and r['optimized'])
            ca=torch.load(out/a['run']/'latest.pt',weights_only=False,map_location='cpu')
            cb=torch.load(out/b['run']/'latest.pt',weights_only=False,map_location='cpu')
            am=[json.loads(s) for s in (out/a['run']/'metrics.jsonl').read_text().splitlines()]
            bm=[json.loads(s) for s in (out/b['run']/'metrics.jsonl').read_text().splitlines()]
            keys=['ready_learning_decisions','accepted_learning_samples','completed_learning_episodes','invalid_lobbies','rejected']
            check={'workload':workload,'trial':trial,'weights_identical':all(torch.equal(ca['policy'][k],cb['policy'][k]) for k in ca['policy']),
               'max_weight_difference':max(float((ca['policy'][k]-cb['policy'][k]).abs().max()) for k in ca['policy']),
               'episodes_identical':[r['episodes'] for r in am]==[r['episodes'] for r in bm],
               'sample_and_fault_counts_identical':all(a['training']['counts'][k]==b['training']['counts'][k] for k in keys),
               'faults':a['training']['counts']['invalid_lobbies']+b['training']['counts']['invalid_lobbies']}
            checks.append(check)
    summary={}
    for workload in ['initial','busy-trained']:
        base=[r for r in rows if r['workload']==workload and not r['optimized']]
        opt=[r for r in rows if r['workload']==workload and r['optimized']]
        summary[workload]={'baseline_median_loop_seconds':statistics.median(r['loop_seconds'] for r in base),
          'optimized_median_loop_seconds':statistics.median(r['loop_seconds'] for r in opt),
          'median_paired_speedup':statistics.median(a['loop_seconds']/b['loop_seconds'] for a,b in zip(base,opt)),
          'baseline_median_samples_per_second':statistics.median(r['learner_samples_per_second'] for r in base),
          'optimized_median_samples_per_second':statistics.median(r['learner_samples_per_second'] for r in opt)}
    result={'device':device,'rows':rows,'correctness':checks,'summary':summary,'wall_seconds':time.perf_counter()-started,
        'valid_comparison':all(c['weights_identical'] and c['episodes_identical'] and c['sample_and_fault_counts_identical'] and c['faults']==0 for c in checks)}
    (out/'paired-summary.json').write_text(json.dumps(result,indent=2)+'\n')
    if not result['valid_comparison']:raise RuntimeError('performance parity failed; inspect paired-summary.json')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',required=True,type=Path);p.add_argument('--device',default='cpu');p.add_argument('--checkpoint',required=True,type=Path);p.add_argument('--trials',default=3,type=int)
    args=p.parse_args()
    if args.device=='cpu':
        import fcntl
        with (HERE.parents[1]/'evidence/local-performance.lock').open('a+') as lock:
            print('waiting for exclusive local performance lock',flush=True);fcntl.flock(lock,fcntl.LOCK_EX)
            paired(args.out,args.device,args.checkpoint,args.trials)
    else:paired(args.out,args.device,args.checkpoint,args.trials)


if __name__=='__main__':main()
