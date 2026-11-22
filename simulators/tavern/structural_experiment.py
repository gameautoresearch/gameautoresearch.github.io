#!/usr/bin/env python3
"""Current-baseline structural ablations, matched full loops and recorded native replay.

Local timings acquire evidence/local-performance.lock. Every directory is new. Native
replay contains the actual actions/active masks of a trained, populated-board rollout;
its encoding, step, reset and state checks exclude neural inference on replay only.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import statistics
import time

from gpu_suite import HERE, run_command
from provenance import ROOT, write_json


def config():
    c = json.loads((HERE/'recipes/gpu-small.json').read_text())
    c.update(device='cpu', updates=4, profile_updates=4, save_initial=False,
             allocated_worker_hours=None)
    c['machine'].update(label='local-structural-comparison', gpu_model=None)
    return c


def full_loops(out, checkpoint, trials, mode):
    import torch
    from migrate_performance_checkpoint import migrate
    cp = migrate(checkpoint, out/'pilot-compatible.pt')
    rows = []
    for workload in ['initial', 'busy-trained']:
        for trial in range(trials):
            c = config()
            c.update(start_turn=8 if workload == 'busy-trained' else 1)
            if mode != 'baseline':
                c.update(native_scheduler=mode, preencode_learner=mode == 'encoded')
                if mode == 'encoded': c['native_scheduler'] = 'dynamic'
            name = f'{workload}-t{trial}'
            row = run_command(out/name, c, 'profile', cp if workload == 'busy-trained' else None)
            row.update(workload=workload, trial=trial, mode=mode)
            row['loop_seconds'] = sum(row['profile']['stage_seconds'].values())
            rows.append(row)
            print(json.dumps({'completed':name, 'mode':mode, 'process':row['full_process_seconds'],
                              'loop':row['loop_seconds']}), flush=True)
    write_json(out/'full-loops.json', rows)
    return rows


def record_trace(out, checkpoint):
    import numpy as np
    import torch
    import taverngym as tg
    from model import load_policy
    from rollout import Timer, collect
    from migrate_performance_checkpoint import migrate
    original = tg.BatchEnv
    trace = {'calls': [], 'initial': None, 'config': config(), 'seed':12345}
    trace['config'].update(start_turn=8)
    class RecordedEnv(original):
        def observe(self, *args, **kwargs):
            if trace['initial'] is None:
                trace['initial'] = [self.save(g) for g in range(self.n_games)]
            return super().observe(*args, **kwargs)
        def step(self, actions, active_games=None):
            trace['calls'].append({'actions':np.asarray(actions).tolist(),
                'active':np.asarray(active_games).tolist() if active_games is not None else None})
            return super().step(actions, active_games=active_games)
    torch.set_num_threads(1)
    torch.manual_seed(12345)
    cp=migrate(checkpoint,out/'trace-compatible.pt')
    policy,_=load_policy(cp,torch.device('cpu'))
    tg.BatchEnv=RecordedEnv
    try:
        eps,records,stats,states=collect(policy,trace['config'],trace['seed'],torch.device('cpu'),
            Timer(torch.device('cpu')),time.perf_counter()+120,copy_observations=True)
    finally:
        tg.BatchEnv=original
    trace.update(final=states, records=records, stats=stats, native=tg.build_info())
    write_json(out/'native-trace.json',trace)
    return out/'native-trace.json'


def replay(trace_path, trials, scheduler):
    import numpy as np
    import taverngym as tg
    trace=json.loads(Path(trace_path).read_text());c=trace['config']
    os.environ['TAVERNGYM_SCHEDULER']=scheduler
    calls=[(np.array(x['actions'],np.int64),np.array(x['active'],bool) if x['active'] is not None else None) for x in trace['calls']]
    rows=[]
    for trial in range(trials):
        init=time.perf_counter()
        env=tg.BatchEnv(c['n_games'],seed=trace['seed'],controlled=c['learning_seats'],
                        config=c['rules'],threads=c['native_threads'])
        init=time.perf_counter()-init
        for g,state in enumerate(trace['initial']): env.load(g,state)
        stages={'observation':0.,'step':0.};h=hashlib.sha256();done_count=0
        for actions,active in calls:
            t=time.perf_counter();obs=env.observe(copy=True,active_games=active)
            stages['observation']+=time.perf_counter()-t
            # Hash outside timed native phases. Includes all player-visible features/masks.
            for key in sorted(obs):h.update(obs[key].tobytes())
            t=time.perf_counter();rw,dn,info=env.step(actions,active_games=active)
            stages['step']+=time.perf_counter()-t
            done_count+=int(dn.sum())
            for x in [rw,dn,*[info[k] for k in sorted(info)]]:h.update(x.tobytes())
        exact=all(env.save(g)==state for g,state in enumerate(trace['final']))
        if not exact: raise RuntimeError('native replay diverged')
        rows.append(dict(trial=trial,scheduler=scheduler,environment_init_seconds=init,
            stage_seconds=stages,observable_digest=h.hexdigest(),final_states_identical=exact,
            completed=done_count,ready_decisions=trace['stats']['ready_learning_decisions'],
            ready_decisions_per_native_second=trace['stats']['ready_learning_decisions']/sum(stages.values())))
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,default=ROOT/'evidence/tavern-implementation/cuda/cuda-learning-v3/learning/seed12345-optimized/latest.pt')
    p.add_argument('--mode',choices=['baseline','static','dynamic','encoded','replay'],default='baseline')
    p.add_argument('--trace',type=Path)
    p.add_argument('--trials',type=int,default=3)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    import fcntl
    with (ROOT/'evidence/local-performance.lock').open('a+') as lock:
        print('Waiting for exclusive performance lock',flush=True)
        fcntl.flock(lock,fcntl.LOCK_EX)
        started=time.perf_counter()
        if a.mode!='replay':full_loops(a.out,a.checkpoint,a.trials,a.mode)
        trace=a.trace or record_trace(a.out,a.checkpoint)
        rows=[]
        modes=['static','dynamic'] if a.mode=='replay' else ['static' if a.mode=='baseline' else 'dynamic']
        for mode in modes:rows.extend(replay(trace,a.trials,mode))
        write_json(a.out/'native-replay.json',dict(trace=str(trace.resolve()),rows=rows,
            locked_wall_seconds=time.perf_counter()-started,measurement_kind='local_cpu_validation'))


if __name__=='__main__': main()
