#!/usr/bin/env python3
"""Occupied-board scripted-policy native replay, with real late combat and lobby tails."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import time

import numpy as np
import taverngym as tg
from provenance import ROOT, write_json
from structural_experiment import config, replay


def record(out):
    c=config();c.update(start_turn=1)
    env=tg.BatchEnv(c['n_games'],seed=c['seed'],controlled=[0],config=c['rules'],threads=4)
    trace={'initial':[env.save(i) for i in range(c['n_games'])], 'seed':c['seed'],
        'config':c,'calls':[],'native':tg.build_info(),'policy':'public scripted bot_action for controlled seat'}
    active=np.ones(c['n_games'],bool);decisions=turn_sum=late=occupied=board_sum=0
    results=[];faults=0
    for step in range(4000):
        if not active.any():break
        obs=env.observe(active_games=active)
        actions=np.zeros(c['n_games'],np.int64)
        ids=np.flatnonzero(obs['ready'])
        for g in ids: actions[g]=env.game(int(g)).bot_action(0)
        decisions+=len(ids)
        turn_sum+=float(obs['features'][ids,0].sum()*20)
        board_sum+=float(obs['features'][ids,33].sum()*7)
        late+=int((obs['features'][ids,0]>=.5).sum())
        occupied+=int((obs['features'][ids,33]>=5/7-1e-6).sum())
        trace['calls'].append({'actions':actions.tolist(),'active':active.tolist()})
        rw,dn,info=env.step(actions,active_games=active)
        for g in np.flatnonzero(dn & active):
            results.append({'game':int(g),'placement':int(info['placement'][g]),'stop_reason':int(info['stop_reason'][g])})
            faults+=int(info['stop_reason'][g]!=1)
        active &= ~dn
    else:raise RuntimeError('scripted trace driver bound')
    trace.update(final=[env.save(g) for g in range(c['n_games'])],records=results,
        stats={'ready_learning_decisions':decisions,'ready_turn_mean':turn_sum/decisions,
            'ready_board_count_mean':board_sum/decisions,'turn10_plus_fraction':late/decisions,
            'five_minions_plus_fraction':occupied/decisions,'faults':faults})
    write_json(out/'scripted-trace.json',trace)
    return out/'scripted-trace.json'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--trials',type=int,default=5)
    p.add_argument('--trace',type=Path)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    import fcntl
    (ROOT/'evidence').mkdir(parents=True,exist_ok=True)
    with (ROOT/'evidence/local-performance.lock').open('a+') as lock:
        print('waiting for local performance lock',flush=True);fcntl.flock(lock,fcntl.LOCK_EX)
        trace=a.trace or record(a.out)
        rows=[]
        for t in range(a.trials):
            for scheduler in (['static','dynamic'] if t%2==0 else ['dynamic','static']):
                rows.extend(replay(trace,1,scheduler))
        write_json(a.out/'native-summary.json',{'trace':str(trace.resolve()),'rows':rows,
            'workload':json.loads(trace.read_text())['stats'],'measurement_kind':'local_cpu_validation'})
    print(json.dumps({'completed':str(a.out)}),flush=True)


if __name__=='__main__':main()
