"""GPU validation for the newly authorized structural performance campaign."""
from pathlib import Path
import json,time,statistics,copy,torch
from . import cuda_suite as suite
from .provenance import write_json,manifest

def run(out,mode='transport'):
    out=Path(out);cfg=suite.config(workers=3,per_worker=8)
    manifest(out,cfg,'structural-cuda-'+mode,{'authorization':'User removed prior caps and attempt limits; retain history separately','profile_seeds':'public 20000..','private_panel_used':False})
    result={'environment':suite.environment(),'profiles':{},'learning':[],'parity':[]}
    if mode in ('transport','final'):
        for workers,envs in ([(3,8),(3,32),(3,64)] if mode=='transport' else [(3,8),(3,32)]):
            c=suite.config(workers=workers,per_worker=envs)
            c['profile_opponents']='mixed-scripted'
            groups={}
            for variant in ('packed','stream'):
                c.update(mon_cache=mode=='final' and variant=='stream',fast_callbacks=mode=='final' and variant=='stream')
                groups[variant]=suite.profile_case(out/f'w{workers}e{envs}-{variant}',c,variant=variant,repeats=5,checkpoint=Path('/workspace/input-policy.pt'))
            ref=groups['packed'][0]['traces']
            parity=all(r['traces']==ref for rows in groups.values() for r in rows)
            if not parity:raise AssertionError('Stream changed deterministic trajectories')
            result['profiles'][f'w{workers}e{envs}']={'rows':groups,'parity':parity,'median_seconds':{v:statistics.median(r['seconds'] for r in rows[1:]) for v,rows in groups.items()}}
            write_json(out/'partial.json',result);suite.PERSIST()
        for seed in (1729,2718,3141):
            pair=[]
            for variant in ('packed','stream'):
                started=time.perf_counter()
                c=copy.deepcopy(cfg);c.update(mon_cache=mode=='final' and variant=='stream',fast_callbacks=mode=='final' and variant=='stream')
                case=suite.learning_case(out/f'learn-{seed}-{variant}',c,variant,seed,iterations=100 if mode=='transport' else 80)
                case['inclusive_seconds']=time.perf_counter()-started
                result['learning'].append(case);pair.append(case)
                write_json(out/'partial.json',result);suite.PERSIST()
            a=torch.load(out/f'learn-{seed}-packed/final.pt',map_location='cpu',weights_only=False)
            b=torch.load(out/f'learn-{seed}-stream/final.pt',map_location='cpu',weights_only=False)
            delta=max(float((a['model'][k]-v).abs().max()) for k,v in b['model'].items())
            identity=delta==0 and pair[0]['training_trajectory_sha256']==pair[1]['training_trajectory_sha256']
            if not identity:raise AssertionError(f'Stream stochastic training identity failed {seed}: {delta}')
            result['parity'].append({'seed':seed,'max_parameter_error':delta,'exact':identity})
    elif mode=='graph':
        from .efficient import EfficientInference
        from .rl import Timings
        from .env import Vector
        import numpy as np
        m=suite.initialize(cfg);t=Timings()
        with Vector(1,16,packed=True) as v:
            states=v.reset(list(range(15500,15516)),512)
        obs=[g['observations'][seat] for g in states for seat in (0,1)]
        eager=EfficientInference('cuda:0',t);graph=EfficientInference('cuda:0',t,graph=True)
        checks=[]
        for n in [1,2,7,16,3,31,5,2,16,1,31,3]*3:
            state=torch.cuda.get_rng_state();a=eager(m,obs[:n]);ra=torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(state);b=graph(m,obs[:n]);rb=torch.cuda.get_rng_state()
            check=dict(n=n,actions_equal=np.array_equal(a[0],b[0]),rng_equal=torch.equal(ra,rb),
                max_logp_error=float(np.max(np.abs(a[1]-b[1]))),max_value_error=float(np.max(np.abs(a[2]-b[2]))))
            checks.append(check)
        result['stochastic_graph_checks']=checks
        exact=all(x['actions_equal'] and x['rng_equal'] and max(x['max_logp_error'],x['max_value_error'])==0 for x in checks)
        result['stochastic_graph_exact']=exact
        if exact:
            for seed in (1729,2718,3141):
                pair=[]
                for variant in ('packed','graph'):
                    case=suite.learning_case(out/f'learn-{seed}-{variant}',cfg,variant,seed,iterations=60)
                    result['learning'].append(case);pair.append(case)
                    write_json(out/'partial.json',result);suite.PERSIST()
                a=torch.load(out/f'learn-{seed}-packed/final.pt',map_location='cpu',weights_only=False)
                b=torch.load(out/f'learn-{seed}-graph/final.pt',map_location='cpu',weights_only=False)
                delta=max(float((a['model'][k]-v).abs().max()) for k,v in b['model'].items())
                same=delta==0 and pair[0]['training_trajectory_sha256']==pair[1]['training_trajectory_sha256']
                result['parity'].append(dict(seed=seed,exact=same,max_parameter_error=delta))
                if not same:break
    else:raise ValueError(mode)
    write_json(out/'results.json',result);suite.PERSIST()
    return result
