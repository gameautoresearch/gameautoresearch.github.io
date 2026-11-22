"""Two explicit historical baselines on the same fixed worker as the latest candidate."""
import copy,json,os,subprocess,time
from pathlib import Path
ROOT=Path('/workspace')


def run(command,log,**kwargs):
    begin=time.perf_counter()
    with log.open('x') as stream:
        p=subprocess.run(list(map(str,command)),stdout=stream,stderr=subprocess.STDOUT,timeout=1200,**kwargs)
    if p.returncode:raise RuntimeError(f'failed {command}; retain {log}')
    return time.perf_counter()-begin


def main(out):
    out.mkdir(parents=True,exist_ok=False);started=time.perf_counter();sim=[];train=[]
    py={m:Path('/venvs')/m/'bin/python' for m in ['upstream','earliest','candidate']}
    tool=ROOT/'candidate/integrations/tavern/original_simulation.py'
    for n in [32,128,1024]:
        trace=out/f'scripted-{n}.json'
        run([py['candidate'],tool,'record','--trace',trace,'--n-games',n,'--steps',300],out/f'record-{n}.log')
        verified=[]
        for mode in ['upstream','candidate']:
            target=out/f'n{n}-{mode}-validation.json'
            cmd=[py[mode],tool,'replay','--trace',trace,'--out',target,'--verify']
            if mode=='candidate':cmd+=['--compact']
            run(cmd,target.with_suffix('.log'));r=json.loads(target.read_text());r.update(n_games=n,mode=mode);verified.append(r)
        for key in ['decisions','natural_episodes','views_masks_results_sha256','final_state_rng_sha256']:
            if verified[0][key]!=verified[1][key]:raise AssertionError((n,key,verified[0][key],verified[1][key]))
        for trial in range(3):
            for mode in (['upstream','candidate'] if trial%2==0 else ['candidate','upstream']):
                target=out/f'n{n}-t{trial}-{mode}.json'
                cmd=[py[mode],tool,'replay','--trace',trace,'--out',target]
                if mode=='candidate':cmd+=['--compact']
                wall=run(cmd,target.with_suffix('.log'));r=json.loads(target.read_text());r.update(n_games=n,mode=mode,trial=trial,process_seconds=wall);sim.append(r)
                (out/'simulation-progress.json').write_text(json.dumps(sim,indent=2)+'\n')
    base=json.loads((ROOT/'candidate/integrations/tavern/recipes/gpu-throughput.json').read_text())
    base.update(seed=12345,allocated_worker_hours=None,save_initial=False)
    # Identical learning recipe at each size. The earliest code ignores new performance
    # options, but retains its original model/collection/optimizer implementation.
    for n,updates in [(32,64),(128,8),(512,4)]:
        for trial in range(2 if n!=32 else 1):
            for mode in (['earliest','candidate'] if trial%2==0 else ['candidate','earliest']):
                config=copy.deepcopy(base);config.update(n_games=n,updates=updates,profile_updates=updates)
                recipe=out/f'learn-n{n}-t{trial}-{mode}.recipe.json';recipe.write_text(json.dumps(config,indent=2)+'\n')
                dest=out/f'learn-n{n}-t{trial}-{mode}'
                program=ROOT/mode/'integrations/tavern/run.py'
                wall=run([py[mode],program,'train','--config',recipe,'--out',dest],dest.with_suffix('.log'))
                r={'mode':mode,'n_games':n,'trial':trial,'process_seconds':wall,'path':str(dest),
                   'training':json.loads((dest/'training.json').read_text()),'profile':json.loads((dest/'profile.json').read_text())}
                train.append(r);(out/'training-progress.json').write_text(json.dumps(train,indent=2)+'\n')
    import torch
    checks=[]
    for n in [32,128,512]:
        for t in range(2 if n!=32 else 1):
            x=torch.load(out/f'learn-n{n}-t{t}-earliest/latest.pt',map_location='cpu',weights_only=False)
            y=torch.load(out/f'learn-n{n}-t{t}-candidate/latest.pt',map_location='cpu',weights_only=False)
            mx=[json.loads(s) for s in (out/f'learn-n{n}-t{t}-earliest/metrics.jsonl').read_text().splitlines()]
            my=[json.loads(s) for s in (out/f'learn-n{n}-t{t}-candidate/metrics.jsonl').read_text().splitlines()]
            check={'policy':all(torch.equal(v,y['policy'][k]) for k,v in x['policy'].items()),
                   'cuda_rng':all(torch.equal(u,v) for u,v in zip(x['cuda_rng'],y['cuda_rng'])),
                   'optimizer':all(torch.equal(v,y['optimizer']['state'][k][f]) for k,s in x['optimizer']['state'].items() for f,v in s.items()),
                   'episodes':[r['episodes'] for r in mx]==[r['episodes'] for r in my],
                   'accepted_samples':[r['counts']['accepted_learning_samples'] for r in mx]==[r['counts']['accepted_learning_samples'] for r in my]}
            checks.append({'n_games':n,'trial':t,'checks':check})
    (out/'original-summary.json').write_text(json.dumps({'simulation':sim,'training':train,'training_parity':checks,
        'allocated_function_seconds':time.perf_counter()-started,'protocol':'Pinned original upstream simulator and earliest retained complete pre-performance training source; same worker/resources and declared rules/learner per size. Three interleaved native pairs; training as declared. No multiplied cross-campaign ratios.'},indent=2)+'\n')


if __name__=='__main__':
    import sys
    main(Path(sys.argv[1]))
