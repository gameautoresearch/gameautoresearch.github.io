"""CUDA-only sampler ablation against the already optimized loop, identical RNG algorithm."""
import copy
import json
from pathlib import Path
import statistics
from gpu_suite import HERE,run_command


def experiment(out):
    import torch
    from migrate_performance_checkpoint import migrate
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    pilot=migrate('/workspace/pilot.pt',out/'pilot-compatible.pt')
    base=json.loads((HERE/'recipes/gpu-small.json').read_text())
    base.update(allocated_worker_hours=None,updates=2,profile_updates=2,minibatch_size=256,snapshot_every=0,save_initial=False)
    rows=[];checks=[]
    for workload in ['initial','busy-trained']:
        for trial in range(2):
            for candidate in ([False,True] if trial%2==0 else [True,False]):
                c=copy.deepcopy(base);c.update(validated_sampler=candidate,start_turn=8 if workload=='busy-trained' else 1)
                name=f'{workload}-t{trial}-'+('candidate' if candidate else 'current')
                row=run_command(out/name,c,'profile',pilot if workload=='busy-trained' else None)
                row.update(workload=workload,trial=trial,candidate=candidate,loop_seconds=sum(row['profile']['stage_seconds'].values()))
                rows.append(row);print(json.dumps({'sampling_completed':name,'seconds':row['loop_seconds']}),flush=True)
            a=next(r for r in rows if r['workload']==workload and r['trial']==trial and not r['candidate'])
            b=next(r for r in rows if r['workload']==workload and r['trial']==trial and r['candidate'])
            aa=torch.load(out/a['run']/'latest.pt',map_location='cpu',weights_only=False)
            bb=torch.load(out/b['run']/'latest.pt',map_location='cpu',weights_only=False)
            am=[json.loads(s) for s in (out/a['run']/'metrics.jsonl').read_text().splitlines()]
            bm=[json.loads(s) for s in (out/b['run']/'metrics.jsonl').read_text().splitlines()]
            check={'workload':workload,'trial':trial,'weights_identical':all(torch.equal(aa['policy'][k],bb['policy'][k]) for k in aa['policy']),
                'cuda_rng_identical':all(torch.equal(x,y) for x,y in zip(aa['cuda_rng'],bb['cuda_rng'])),
                'episodes_identical':[r['episodes'] for r in am]==[r['episodes'] for r in bm],
                'samples_identical':aa['stats']['accepted_learning_samples']==bb['stats']['accepted_learning_samples'],
                'faults':aa['stats']['invalid_lobbies']+bb['stats']['invalid_lobbies'],
                'rejected':aa['stats']['rejected']+bb['stats']['rejected']}
            checks.append(check)
    # A genuine full 64-update learning pair verifies RNG/gradient parity beyond short probes.
    long_rows=[]
    for candidate in [False,True]:
        c=copy.deepcopy(base);c.update(validated_sampler=candidate,updates=64,minibatch_size=1024,start_turn=1)
        name='learning64-'+('candidate' if candidate else 'current')
        row=run_command(out/name,c,'train');row.update(candidate=candidate,workload='learning64',trial=0,loop_seconds=sum(row['profile']['stage_seconds'].values()))
        long_rows.append(row);rows.append(row)
        print(json.dumps({'sampling_trained':name,'seconds':row['full_process_seconds'],'samples':row['training']['counts']['accepted_learning_samples']}),flush=True)
    aa=torch.load(out/long_rows[0]['run']/'latest.pt',map_location='cpu',weights_only=False)
    bb=torch.load(out/long_rows[1]['run']/'latest.pt',map_location='cpu',weights_only=False)
    am=[json.loads(x) for x in (out/long_rows[0]['run']/'metrics.jsonl').read_text().splitlines()]
    bm=[json.loads(x) for x in (out/long_rows[1]['run']/'metrics.jsonl').read_text().splitlines()]
    checks.append({'workload':'learning64','trial':0,
        'weights_identical':all(torch.equal(aa['policy'][k],bb['policy'][k]) for k in aa['policy']),
        'cuda_rng_identical':all(torch.equal(x,y) for x,y in zip(aa['cuda_rng'],bb['cuda_rng'])),
        'episodes_identical':[r['episodes'] for r in am]==[r['episodes'] for r in bm],
        'samples_identical':aa['stats']['accepted_learning_samples']==bb['stats']['accepted_learning_samples'],
        'faults':aa['stats']['invalid_lobbies']+bb['stats']['invalid_lobbies'],
        'rejected':aa['stats']['rejected']+bb['stats']['rejected']})
    summary={}
    for workload in ['initial','busy-trained']:
        a=[r for r in rows if r['workload']==workload and not r['candidate']]
        b=[r for r in rows if r['workload']==workload and r['candidate']]
        summary[workload]={'current_median_seconds':statistics.median(r['loop_seconds'] for r in a),
            'candidate_median_seconds':statistics.median(r['loop_seconds'] for r in b),
            'median_paired_speedup':statistics.median(x['loop_seconds']/y['loop_seconds'] for x,y in zip(a,b))}
    valid=all(c['weights_identical'] and c['cuda_rng_identical'] and c['episodes_identical'] and c['samples_identical'] and c['faults']==0 and c['rejected']==0 for c in checks)
    result={'learning64_full_process_speedup':long_rows[0]['full_process_seconds']/long_rows[1]['full_process_seconds'],'rows':rows,'checks':checks,'summary':summary,'valid':valid,
        'algorithm_source':'https://github.com/pytorch/pytorch/blob/v2.9.1/aten/src/ATen/native/Distributions.cpp#L545-L579',
        'invariants':'CPU ready/mask validation plus finite returned logp/value before any simulator action; softmax of finite logits supplies nonnegative normalized probabilities.'}
    (out/'sampling-summary.json').write_text(json.dumps(result,indent=2)+'\n')
    if not valid:raise RuntimeError('sampler parity failed: keep default sampler')
    return summary
