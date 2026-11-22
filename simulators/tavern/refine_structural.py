"""Matched scaling and numerical/device alternatives; preserve cold costs and semantics labels."""
import copy
import json
import subprocess
import sys
import time

from gpu_suite import HERE, run_command
from provenance import write_json
from structural_gpu import compare


def refine(out):
    started=time.perf_counter()
    c=json.loads((HERE/'recipes/gpu-small.json').read_text())
    c.update(updates=8,profile_updates=8,allocated_worker_hours=None,save_initial=False,
        preencode_learner=True,native_scheduler='dynamic')
    modes={'current':{'preencode_learner':False,'native_scheduler':'static'},
        'graphs':{'graph_inference':True,'graph_learner':True},
        'fused':{'graph_inference':True,'graph_learner':True,'fused_adam':True},
        'cpu-actor':{'actor_device':'cpu','graph_learner':True}}
    rows=[];errors=[]
    for trial in range(3):
        order=list(modes) if trial%2==0 else list(reversed(modes))
        for mode in order:
            cc=copy.deepcopy(c);cc.update(modes[mode])
            name=f'full-t{trial}-{mode}'
            try:
                row=run_command(out/name,cc,'profile');row.update(trial=trial,mode=mode,n_games=32);rows.append(row)
            except Exception as exc:errors.append({'run':name,'error':repr(exc)})
            write_json(out/'progress.json',{'rows':rows,'errors':errors})
    for n in [128,256,512]:
        for mode in ['graphs','fused','cpu-actor']:
            cc=copy.deepcopy(c);cc.update(modes[mode],n_games=n,profile_updates=4)
            name=f'scale-{n}-{mode}'
            try:
                row=run_command(out/name,cc,'profile');row.update(mode=mode,n_games=n);rows.append(row)
            except Exception as exc:errors.append({'run':name,'error':repr(exc)})
            write_json(out/'progress.json',{'rows':rows,'errors':errors})
    # Explicitly quantify batch-size / fewer-Adam-step semantics separately.
    cc=copy.deepcopy(c);cc.update(modes['fused'],n_games=512,minibatch_size=8192,profile_updates=4)
    try:
        row=run_command(out/'scale-512-fused-mb8192',cc,'profile');row.update(mode='fused-mb8192',n_games=512);rows.append(row)
    except Exception as exc:errors.append({'run':'scale-512-fused-mb8192','error':repr(exc)})
    # Both forward and learner graphs must rebuild at resume; use 256-sized minibatches.
    rc=copy.deepcopy(c);rc.update(modes['graphs'],updates=4,n_games=8,minibatch_size=256,snapshot_every=2)
    recipe=out/'resume-recipe.json';write_json(recipe,rc)
    result=subprocess.run([sys.executable,str(HERE/'validate_resume.py'),'--device','cuda:0',
        '--config',str(recipe),'--out',str(out/'resume-both-graphs')],timeout=120)
    checks=[{'trial':t,'checks':compare(out/f'full-t{t}-current',out/f'full-t{t}-graphs')} for t in range(3)]
    write_json(out/'refinement-summary.json',{'rows':rows,'errors':errors,'parity':checks,
        'resume_returncode':result.returncode,'allocated_function_seconds':time.perf_counter()-started})
