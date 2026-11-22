"""Matched-worker full simulation and real PPO comparisons after GPU qualification.

Each arm runs in a fresh process. Original source and the earliest complete trainer
are independent installs; the strong native arm uses the current native integration.
No cross-hardware or multiplied historical ratios are used.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path('/workspace')
TOOLS=ROOT/'integrations/tavern'


def run(command,log,timeout=900):
    start=time.perf_counter()
    print(json.dumps({'stage':'arm_start','log':str(log),'command':list(map(str,command))}),flush=True)
    with log.open('x') as stream:
        proc=subprocess.run(list(map(str,command)),stdout=stream,stderr=subprocess.STDOUT,
                            timeout=timeout,env=dict(os.environ,TAVERN_DEVICE_TRACE_INIT='0',
                            TAVERN_DEVICE_TRACE_KERNELS='0',CUDA_LAUNCH_BLOCKING='0'))
    seconds=time.perf_counter()-start
    if proc.returncode:raise RuntimeError(f'arm failed ({proc.returncode}); retain {log}')
    print(json.dumps({'stage':'arm_complete','log':str(log),'process_seconds':seconds}),flush=True)
    return seconds


def simulation(out,sizes,trials=2):
    rows=[];checks=[]
    for n in sizes:
        trace=out/f'scripted-{n}.json'
        run([sys.executable,TOOLS/'original_simulation.py','record','--trace',trace,
             '--n-games',n,'--steps',300],out/f'record-{n}.log')
        commands={
            'original':['/venvs/upstream/bin/python',TOOLS/'original_simulation.py','replay','--trace',trace],
            'native':[sys.executable,TOOLS/'original_simulation.py','replay','--trace',trace,'--compact'],
            'device':[sys.executable,TOOLS/'device_simulation.py','--trace',trace]}
        validation=[]
        for arm,command in commands.items():
            dest=out/f'n{n}-{arm}-validation.json'
            run([*command,'--verify','--out',dest],dest.with_suffix('.log'))
            validation.append(json.loads(dest.read_text()))
        for field in ['trace_sha256','decisions','natural_episodes','views_masks_results_sha256','final_state_rng_sha256']:
            if len({row[field] for row in validation})!=1:raise AssertionError((n,field))
        checks.append({'n_games':n,'all_ready_views_masks_results_final_state_rng_exact':True})
        for trial in range(trials):
            for arm in (list(commands) if trial%2==0 else list(reversed(commands))):
                dest=out/f'n{n}-t{trial}-{arm}.json'
                wall=run([*commands[arm],'--out',dest],dest.with_suffix('.log'))
                row=json.loads(dest.read_text());row.update(arm=arm,trial=trial,n_games=n,process_seconds=wall)
                rows.append(row)
                (out/'simulation-progress.json').write_text(json.dumps(rows,indent=2)+'\n')
    return {'rows':rows,'independent_oracle_checks':checks}


def learning(out,sizes):
    rows=[]
    base=json.loads((TOOLS/'recipes/gpu-throughput.json').read_text())
    base.update(seed=12345,allocated_worker_hours=None,save_initial=False,
                machine={'label':'Modal-B200-fixed-4core-16GiB','attached_cpu_threads':4,
                         'attached_ram_gib':16,'gpu_model':'NVIDIA B200'})
    for n in sizes:
        updates=8 if n<=128 else 4
        for trial in range(2):
            arms=['earliest','native','device']
            for arm in (arms if trial%2==0 else list(reversed(arms))):
                dest=out/f'learn-n{n}-t{trial}-{arm}'
                if arm=='device':
                    command=[sys.executable,TOOLS/'device_training.py','--out',dest,
                             '--n-games',n,'--updates',updates,'--seed',12345]
                else:
                    config=copy.deepcopy(base);config.update(n_games=n,updates=updates,profile_updates=updates)
                    recipe=dest.with_suffix('.recipe.json');recipe.write_text(json.dumps(config,indent=2)+'\n')
                    python='/venvs/earliest/bin/python' if arm=='earliest' else sys.executable
                    program=ROOT/'earliest/integrations/tavern/run.py' if arm=='earliest' else TOOLS/'run.py'
                    command=[python,program,'train','--config',recipe,'--out',dest]
                wall=run(command,dest.with_suffix('.log'))
                training=json.loads((dest/'training.json').read_text())
                row={'arm':arm,'n_games':n,'updates':updates,'trial':trial,'process_seconds':wall,
                     'training':training,'path':str(dest)}
                if arm=='device':
                    row['accepted_samples']=sum(r['collection']['samples'] for r in training['history'])
                    row['natural_episodes']=sum(r['collection']['natural_episodes'] for r in training['history'])
                    row['invalid_lobbies']=sum(r['collection']['invalid_lobbies'] for r in training['history'])
                else:
                    metrics=[json.loads(line) for line in (dest/'metrics.jsonl').read_text().splitlines()]
                    row['accepted_samples']=sum(r['counts']['accepted_learning_samples'] for r in metrics)
                    row['counts']=[r['counts'] for r in metrics]
                rows.append(row);(out/'learning-progress.json').write_text(json.dumps(rows,indent=2)+'\n')
    return {'rows':rows,'scope':'Real masked neural collection, original GAE convention and three PPO epochs; full process plus checkpoints. Device correctness was qualified separately. Earliest four-thread initialization differs numerically from the current one-thread initializer; no bitwise three-arm training claim.'}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--phase',choices=['simulation','learning','dispatch','dispatch_debug','dispatch_watch','dispatch_scaling','dispatch_sanitizer','dispatch_initcheck'],required=True)
    p.add_argument('--sizes',type=int,nargs='+',required=True)
    p.add_argument('--trials',type=int,default=2);a=p.parse_args()
    if not 0<=a.trials<=3:raise ValueError('bounded repeat count required')
    if not (a.out/'worker-start.json').is_file():raise RuntimeError('missing allocation claim')
    start=time.perf_counter()
    if a.phase in ['dispatch_sanitizer','dispatch_initcheck']:
        from device_sanitizer import qualify
        result=qualify(a.out)
    elif a.phase in ['dispatch','dispatch_debug','dispatch_watch','dispatch_scaling']:
        if a.phase=='dispatch':from device_dispatch import qualify
        elif a.phase=='dispatch_debug':from device_dispatch_debug import qualify
        elif a.phase=='dispatch_watch':from device_dispatch_debug import watch_hidden as qualify
        else:from device_scaling import qualify
        result=qualify(a.out,a.sizes) if a.phase=='dispatch_scaling' else qualify(a.out)
    else:result=simulation(a.out,a.sizes,a.trials) if a.phase=='simulation' else learning(a.out,a.sizes)
    result.update(phase=a.phase,wall_seconds=time.perf_counter()-start,
        accounting='Full fixed B200/4CPU/16GiB allocation is recorded by the outer worker. CPU source builds and artifact preparation are separately retained setup costs, never free speedup.')
    (a.out/'matched.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
