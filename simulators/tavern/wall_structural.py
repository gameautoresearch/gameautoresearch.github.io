"""Predeclared fixed allocated-process-time learning comparison, with recipe changes explicit."""
import copy
import json
import time
from gpu_suite import HERE,run_command
from provenance import write_json


def wall(out,seed):
    started=time.perf_counter()
    c=json.loads((HERE/'recipes/gpu-fast.json').read_text())
    c.update(seed=seed,updates=10000,allocated_worker_hours=205/3600,save_initial=True)
    c['evaluation'].update(seeds=[10101,10102,10103,10104],games_per_seat=2)
    modes={'current32':{'preencode_learner':False,'native_scheduler':'static','graph_inference':False,
        'graph_learner':False,'device_rollouts':False,'vector_bookkeeping':False,'sparse_observations':False,
        'copy_observations':True,'n_games':32},
        'packed128':{'sparse_observations':True,'copy_observations':False,'n_games':128}}
    rows=[]
    for mode in (['packed128','current32'] if seed==45678 else ['current32','packed128']):
        cc=copy.deepcopy(c);cc.update(modes[mode])
        r=run_command(out/mode,cc,'train')
        ec=copy.deepcopy(cc);ec['allocated_worker_hours']=None
        r['evaluation']=run_command(out/(mode+'-eval'),ec,'evaluate',out/mode/'latest.pt')
        r.update(seed=seed,mode=mode);rows.append(r)
        write_json(out/'wall-progress.json',rows)
        print(json.dumps({'completed':mode,'seed':seed,'process':r['full_process_seconds'],
            'updates':r['training']['completed_updates'],'samples':r['training']['counts']['accepted_learning_samples'],
            'placement':r['evaluation']['evaluation']['mean_placement']}),flush=True)
    write_json(out/'wall-summary.json',{'rows':rows,'seed':seed,'allocated_function_seconds':time.perf_counter()-started,
        'protocol':'205 elapsed train-process seconds including startup, inference, transfers, simulation, learner and IO; partial final cohort discarded but charged. Evaluation also charged in outer allocation. Public predeclared practice only.',
        'semantics':'Packed128 changes cohort size and actor-refresh/gradient schedule. This is a learning-recipe comparison, not bitwise trajectory equivalence.'})
