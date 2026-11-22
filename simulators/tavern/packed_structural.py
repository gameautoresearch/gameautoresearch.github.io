"""Compact native RL observation engine: paired differential full-loop qualification."""
import copy
import json
import os
import subprocess
import sys
import time
from gpu_suite import HERE,run_command
from provenance import write_json
from structural_gpu import compare


def packed(out):
    started=time.perf_counter();cores=int(os.environ.get('TAVERN_CPU_CORES','4'))
    gpu=os.environ.get('TAVERN_GPU','L4');ram=int(os.environ.get('TAVERN_RAM_GIB','16'))
    c=json.loads((HERE/'recipes/gpu-fast.json').read_text());c.update(allocated_worker_hours=None,
        updates=8,profile_updates=8,save_initial=False,native_threads=cores,native_warmstart=True)
    c['machine'].update(label=f'packed-{gpu}-{cores}core-{ram}GiB',attached_cpu_threads=cores,
        attached_ram_gib=ram,gpu_model=gpu)
    from migrate_performance_checkpoint import migrate
    pilot=migrate('/workspace/pilot.pt',out/'pilot-compatible.pt')
    rows=[];checks=[]
    for workload in ['initial','busy-trained']:
        for t in range(2):
            for compact in ([False,True] if t%2==0 else [True,False]):
                cc=copy.deepcopy(c);cc.update(sparse_observations=compact,copy_observations=not compact,
                    start_turn=8 if workload=='busy-trained' else 1,allow_profile_eliminated=workload=='busy-trained')
                name=f'{workload}-t{t}-'+('packed' if compact else 'dense')
                r=run_command(out/name,cc,'profile',pilot if workload=='busy-trained' else None)
                r.update(workload=workload,trial=t,compact=compact,n_games=32);rows.append(r)
                write_json(out/'progress.json',{'rows':rows,'checks':checks})
            checks.append({'workload':workload,'trial':t,
                'checks':compare(out/f'{workload}-t{t}-dense',out/f'{workload}-t{t}-packed')})
    # Same hardware, paired at a substantial batch size, including Python overhead.
    sizes=[512,1024] if gpu=='L4' else [1024,4096]
    errors=[]
    for n in sizes:
        for compact in [False,True]:
            cc=copy.deepcopy(c);cc.update(n_games=n,profile_updates=4,
                sparse_observations=compact,copy_observations=not compact)
            name=f'scale-{n}-'+('packed' if compact else 'dense')
            try:
                r=run_command(out/name,cc,'profile');r.update(compact=compact,n_games=n,workload='initial');rows.append(r)
            except Exception as exc:errors.append({'run':name,'error':repr(exc)})
            write_json(out/'progress.json',{'rows':rows,'checks':checks,'errors':errors})
        if not errors:checks.append({'n_games':n,'checks':compare(out/f'scale-{n}-dense',out/f'scale-{n}-packed')})
    cc=copy.deepcopy(c);cc.update(n_games=512,profile_updates=4,sparse_observations=True,
        copy_observations=False,start_turn=8,allow_profile_eliminated=True)
    r=run_command(out/'busy-scale-512-packed',cc,'profile',pilot)
    r.update(compact=True,n_games=512,workload='busy-trained');rows.append(r)
    rc=copy.deepcopy(c);rc.update(n_games=8,updates=4,minibatch_size=256,snapshot_every=2,
        sparse_observations=True,copy_observations=False)
    cfg=out/'resume-recipe.json';write_json(cfg,rc)
    result=subprocess.run([sys.executable,str(HERE/'validate_resume.py'),'--device','cuda:0',
        '--config',str(cfg),'--out',str(out/'resume-packed')],timeout=120)
    write_json(out/'packed-summary.json',{'rows':rows,'parity':checks,'errors':errors,
        'resume_returncode':result.returncode,'allocated_function_seconds':time.perf_counter()-started})
