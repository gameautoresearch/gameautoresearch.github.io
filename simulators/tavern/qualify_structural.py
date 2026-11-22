"""Predeclared three-seed learning parity and separate fused-numerics qualification."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import time

from gpu_suite import HERE, run_command
from provenance import write_json
from structural_gpu import compare


def qualify(out):
    started=time.perf_counter()
    c=json.loads((HERE/'recipes/gpu-small.json').read_text())
    c.update(updates=128,profile_updates=4,save_initial=True,allocated_worker_hours=None)
    c['evaluation'].update(seeds=[10101,10102,10103,10104],games_per_seat=2)
    modes={'current':{},'graphs':{'preencode_learner':True,'native_scheduler':'dynamic',
        'graph_inference':True,'graph_learner':True},
        'fused':{'preencode_learner':True,'native_scheduler':'dynamic',
        'graph_inference':True,'graph_learner':True,'fused_adam':True}}
    rows=[];checks=[]
    for seed in [12345,45678,78901]:
        order=['current','graphs'] if seed!=45678 else ['graphs','current']
        if seed==12345:order.append('fused')
        for mode in order:
            cc=copy.deepcopy(c);cc.update(modes[mode],seed=seed)
            name=f'seed{seed}-{mode}'
            row=run_command(out/name,cc,'train')
            if mode=='current':
                row['initial_eval']=run_command(out/(name+'-initial-eval'),cc,'evaluate',out/name/'initial.pt')
            row['final_eval']=run_command(out/(name+'-final-eval'),cc,'evaluate',out/name/'latest.pt')
            row.update(seed=seed,mode=mode)
            rows.append(row)
            write_json(out/'learning-progress.json',rows)
            print(json.dumps({'completed':name,'process_seconds':row['full_process_seconds'],
                'samples':row['training']['counts']['accepted_learning_samples'],
                'placement':row['final_eval']['evaluation']['mean_placement']}),flush=True)
        checks.append({'seed':seed,'checks':compare(out/f'seed{seed}-current',out/f'seed{seed}-graphs')})
    # Reconstruct all graph/RNG caches at a checkpoint boundary. Keep the exact recipe.
    rc=copy.deepcopy(c);rc.update(modes['graphs'],updates=4,snapshot_every=2,n_games=8)
    recipe=out/'resume-recipe.json';write_json(recipe,rc)
    subprocess.run([sys.executable,str(HERE/'validate_resume.py'),'--device','cuda:0',
        '--config',str(recipe),'--out',str(out/'resume-graphs')],check=True,timeout=120)
    # Native-only evidence is separately labelled; all of its idle L4 time stays charged.
    subprocess.run([sys.executable,str(HERE/'native_profile.py'),'--out',str(out/'native-scripted'),
        '--trials','5'],check=True,timeout=120)
    write_json(out/'learning-summary.json',{'rows':rows,'parity':checks,
        'allocated_function_seconds':time.perf_counter()-started,
        'fused_numerics':'Separate fused Adam variant; identical task/optimizer hyperparameters, different floating-point kernels. No exact trajectory parity assumed.'})
