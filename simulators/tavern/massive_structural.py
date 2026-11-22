"""Native batch/device-rollout scaling: same resources and explicitly larger allocations."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from gpu_suite import HERE, run_command
from provenance import write_json
from structural_gpu import compare


def massive(out):
    started=time.perf_counter();cores=int(os.environ.get('TAVERN_CPU_CORES','4'))
    gpu=os.environ.get('TAVERN_GPU','L4');ram=int(os.environ.get('TAVERN_RAM_GIB','16'))
    c=json.loads((HERE/'recipes/gpu-small.json').read_text())
    c.update(allocated_worker_hours=None,updates=4,profile_updates=4,save_initial=False,
        native_threads=cores,torch_threads=1)
    c['machine'].update(label=f'{gpu}-{cores}core-{ram}GiB-massive',attached_cpu_threads=cores,
        attached_ram_gib=ram,gpu_model=gpu)
    optimized={'preencode_learner':True,'native_scheduler':'dynamic','graph_inference':True,
        'graph_learner':True,'native_warmstart':True,'actor_graph_max_shapes':32}
    rows=[];errors=[];checks=[]
    # Matched same-batch identity checks on initial and realistic busy continuations.
    from migrate_performance_checkpoint import migrate
    pilot=migrate('/workspace/pilot.pt',out/'pilot-compatible.pt')
    for workload in ['initial','busy-trained']:
        for trial in range(2):
            for retained in ([False,True] if trial%2==0 else [True,False]):
                cc=copy.deepcopy(c);cc.update(optimized,device_rollouts=retained,
                    start_turn=8 if workload=='busy-trained' else 1)
                name=f'{workload}-t{trial}-'+('device' if retained else 'host')
                try:
                    row=run_command(out/name,cc,'profile',pilot if workload=='busy-trained' else None)
                    row.update(mode='device' if retained else 'host',workload=workload,trial=trial,n_games=32);rows.append(row)
                except Exception as exc:errors.append({'run':name,'error':repr(exc)})
                write_json(out/'progress.json',{'rows':rows,'errors':errors})
            checks.append({'workload':workload,'trial':trial,
                'checks':compare(out/f'{workload}-t{trial}-host',out/f'{workload}-t{trial}-device')})
    sizes=[128,512,1024] if gpu=='L4' else [512,1024,2048,4096]
    for n in sizes:
        for trial in range(2):
            cc=copy.deepcopy(c);cc.update(optimized,device_rollouts=True,n_games=n)
            name=f'scale-{n}-t{trial}'
            try:
                row=run_command(out/name,cc,'profile');row.update(mode='device',trial=trial,n_games=n,workload='initial');rows.append(row)
            except Exception as exc:
                errors.append({'run':name,'error':repr(exc)})
                # Do not repeat a capacity/OOM failure blindly.
                write_json(out/'progress.json',{'rows':rows,'errors':errors});break
            print(json.dumps({'completed':name,'process':row['full_process_seconds'],
                'samples':row['training']['counts']['accepted_learning_samples']}),flush=True)
            write_json(out/'progress.json',{'rows':rows,'errors':errors})
    # Populated continuation at a substantial batch size, independently of empty starts.
    cc=copy.deepcopy(c);cc.update(optimized,device_rollouts=True,n_games=512,start_turn=8)
    try:
        row=run_command(out/'busy-scale-512',cc,'profile',pilot);row.update(mode='device',n_games=512,workload='busy-trained');rows.append(row)
    except Exception as exc:errors.append({'run':'busy-scale-512','error':repr(exc)})
    # Exercise both graph caches plus retained inputs across a source-bound resume.
    rc=copy.deepcopy(c);rc.update(optimized,device_rollouts=True,n_games=8,minibatch_size=256,
        updates=4,snapshot_every=2)
    recipe=out/'resume-recipe.json';write_json(recipe,rc)
    result=subprocess.run([sys.executable,str(HERE/'validate_resume.py'),'--device','cuda:0',
        '--config',str(recipe),'--out',str(out/'resume-device')],timeout=120)
    write_json(out/'massive-summary.json',{'rows':rows,'errors':errors,'parity':checks,
        'resume_returncode':result.returncode,'allocation':{'gpu':gpu,'cpu_cores':cores,'ram_gib':ram},
        'allocated_function_seconds':time.perf_counter()-started,
        'scaling_semantics':'n_games changes policy cohort size/gradient schedule; report separately from identical-task same-batch optimizations.'})
