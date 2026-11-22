"""Independent standard-CLI qualification and fixed-wall-time practice learning."""
import copy
import json
import subprocess
import sys
import time
from gpu_suite import HERE, run_command
from provenance import write_json
from structural_gpu import compare


def final(out):
    c=json.loads((HERE/'recipes/gpu-fast.json').read_text());c.update(allocated_worker_hours=None,
        updates=8,profile_updates=8,save_initial=False)
    legacy={'preencode_learner':False,'native_scheduler':'static','graph_inference':False,
        'graph_learner':False,'device_rollouts':False,'vector_bookkeeping':False}
    rows=[];checks=[];started=time.perf_counter()
    for trial in range(3):
        for mode in (['current','fast'] if trial%2==0 else ['fast','current']):
            cc=copy.deepcopy(c)
            if mode=='current':cc.update(legacy)
            row=run_command(out/f't{trial}-{mode}',cc,'profile');row.update(mode=mode,trial=trial);rows.append(row)
        checks.append({'trial':trial,'checks':compare(out/f't{trial}-current',out/f't{trial}-fast')})
    write_json(out/'paired.json',{'rows':rows,'checks':checks})
    learning=[]
    for mode in ['current128','fast128','batch128-wall','batch512-wall']:
        cc=copy.deepcopy(c);cc.update(updates=128,save_initial=True)
        if mode=='current128':cc.update(legacy)
        if mode.startswith('batch'):
            n=128 if mode.startswith('batch128') else 512
            cc.update(n_games=n,updates=1000,allocated_worker_hours=205/3600)
        cc['evaluation'].update(seeds=[10101,10102,10103,10104],games_per_seat=2)
        row=run_command(out/mode,cc,'train')
        ec=copy.deepcopy(cc);ec['allocated_worker_hours']=None
        row['evaluation']=run_command(out/(mode+'-eval'),ec,'evaluate',out/mode/'latest.pt')
        row['mode']=mode;learning.append(row)
        write_json(out/'learning-progress.json',learning)
        print(json.dumps({'completed':mode,'seconds':row['full_process_seconds'],
            'updates':row['training']['completed_updates'],'samples':row['training']['counts']['accepted_learning_samples'],
            'placement':row['evaluation']['evaluation']['mean_placement']}),flush=True)
    check=compare(out/'current128',out/'fast128')
    write_json(out/'final-summary.json',{'paired':rows,'parity':checks,'learning':learning,
        'full128_parity':check,'allocated_function_seconds':time.perf_counter()-started,
        'fixed_wall_time_protocol':'Batch128 and batch512 use 205 elapsed command-worker seconds, including cold setup; incomplete final cohort is discarded and charged. Public practice only. Larger policy cohorts change the update schedule.'})
