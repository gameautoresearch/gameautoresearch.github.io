#!/usr/bin/env python3
"""Consolidate retained run receipts without changing historical source or outputs."""
import argparse
import json
from pathlib import Path
import statistics
import time

from provenance import ROOT,write_json
from structural_gpu import compare

BASE=ROOT/'evidence/tavern-implementation/structural-20261108'


def metrics(row):
    samples=row.get('training',{}).get('counts',{}).get('accepted_learning_samples',0)
    stages=row.get('profile',{}).get('stage_seconds',{})
    loop=sum(stages.values())
    process=row['full_process_seconds']
    return {'run':row['run'],'samples':samples,'loop_seconds':loop,'process_seconds':process,
        'samples_per_loop_second':samples/loop if loop else None,
        'samples_per_process_second':samples/process,
        'samples_per_process_allocated_hour':samples/process*3600,
        'stage_seconds':stages,'counts':row.get('training',{}).get('counts'),
        'peak_cuda_allocated_bytes':row.get('profile',{}).get('peak_cuda_allocated_bytes'),
        'peak_process_rss_bytes':row.get('profile',{}).get('peak_process_rss_bytes')}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True,type=Path)
    a=p.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    result={'generated_unix':time.time(),'historical_ledgers_modified':False,'runs':{},'attempts':[]}
    for run in sorted(BASE.iterdir()):
        if not run.is_dir():continue
        attempt=run/'attempt.json'
        if attempt.exists():result['attempts'].append(json.loads(attempt.read_text()))
        retained=run/'retained'/run.name
        if not retained.exists():continue
        allocation=retained/'allocation.json'
        record={'allocation':json.loads(allocation.read_text()) if allocation.exists() else None,'path':str(retained.relative_to(ROOT))}
        for f in ['paired.json','full-loops.json','summary.json','refinement-summary.json','massive-summary.json','packed-summary.json','final-summary.json','wall-summary.json']:
            path=retained/f
            if path.exists():record[f]=json.loads(path.read_text())
        result['runs'][run.name]=record
    wall=[]
    for name in ['cuda-wall-s12345-v19','cuda-wall-s45678-v20','cuda-wall-s78901-v21']:
        record=result['runs'].get(name,{})
        data=record.get('wall-summary.json')
        if not data:continue
        pair=[]
        for row in data['rows']:
            measure=metrics(row)
            measure.update(mean_placement=row['evaluation']['evaluation']['mean_placement'],
                top4_rate=row['evaluation']['evaluation']['top4_rate'],
                evaluation_process_seconds=row['evaluation']['full_process_seconds'],
                evaluation_complete=row['evaluation']['evaluation']['complete_valid_schedule'])
            pair.append(measure)
        wall.append({'seed':data['seed'],'arms':pair,'protocol':data['protocol'],'semantics':data['semantics']})
    result['fixed_wall_learning']=wall
    q=BASE/'cuda-qualify-v3/retained/cuda-qualify-v3'
    if (q/'learning-progress.json').exists():
        rows=json.loads((q/'learning-progress.json').read_text());checks=[];learn=[]
        for seed in [12345,45678,78901]:
            x=next(r for r in rows if r['seed']==seed and r['mode']=='current')
            y=next(r for r in rows if r['seed']==seed and r['mode']=='graphs')
            check=compare(q/x['run'],q/y['run']);checks.append({'seed':seed,'checks':check})
            learn.append({'seed':seed,'baseline':metrics(x),'graphs':metrics(y),
                'process_speedup':x['full_process_seconds']/y['full_process_seconds'],
                'initial_placement':x['initial_eval']['evaluation']['mean_placement'],
                'final_placement':y['final_eval']['evaluation']['mean_placement'],
                'evaluation_valid':y['final_eval']['evaluation']['complete_valid_schedule']})
        result['three_seed_learning']={'rows':learn,'parity':checks,
            'median_process_speedup':statistics.median(x['process_speedup'] for x in learn),
            'initial_mean_placement':statistics.mean(x['initial_placement'] for x in learn),
            'final_mean_placement':statistics.mean(x['final_placement'] for x in learn),
            'unique_samples':sum(x['baseline']['samples'] for x in learn),
            'function_status':'failed only after all learning/evaluation/resume completed, at optional native-profile lock path; no retraining required'}
    write_json(a.out,result)
    print(json.dumps({'written':str(a.out),'attempts':len(result['attempts']),'retained_runs':len(result['runs'])}))


if __name__=='__main__':main()
