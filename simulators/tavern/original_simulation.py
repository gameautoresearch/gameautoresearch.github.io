"""Pinned-original/common-task replay, independently installed original and candidate.

Precomputed real scripted actions remove policy selection from simulator timings.
Timed simulation includes initialization, every full observe/step and final JSON IO;
untimed validation hashes every ready view/mask/outcome and exact final state/RNG.
"""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import taverngym as tg
RULES={'num_players':8,'content':'Implemented','turn_action_caps':[30],'max_actions_per_turn':30,'max_turns':100,'record_log':False}


def record(path,n_games,steps):
    env=tg.BatchEnv(n_games,seed=12345,controlled=[0],config=RULES,threads=4)
    actions=[];natural=decisions=late=occupied=0
    for step in range(steps):
        o=env.observe();a=np.zeros(n_games,np.int64);ids=np.flatnonzero(o['ready'])
        for g in ids:a[g]=env.game(int(g)).bot_action(0)
        decisions+=len(ids);late+=int((o['features'][ids,0]>=.5).sum());occupied+=int((o['features'][ids,33]>=5/7-1e-6).sum())
        r,d,i=env.step(a)
        if 'stop_reason' in i and (d&(i['stop_reason']!=1)).any():raise RuntimeError('fault in scripted workload')
        if i['rejected'].any():raise RuntimeError('rejected trace action')
        natural+=int(d.sum());actions.append(a.tolist())
    path.write_text(json.dumps({'n_games':n_games,'seed':12345,'rules':RULES,'steps':steps,'actions':actions,
        'decisions':decisions,'natural_episodes':natural,'turn10_plus_fraction':late/decisions,
        'five_minions_plus_fraction':occupied/decisions},separators=(',',':'))+'\n')


def replay(trace,compact,verify,state_out=None):
    t=json.loads(trace.read_text());actions=[np.array(a,np.int64) for a in t['actions']]
    digest=hashlib.sha256();begin=time.perf_counter();env=tg.BatchEnv(t['n_games'],seed=t['seed'],controlled=[0],config=t['rules'],threads=4)
    init=time.perf_counter()-begin;obs_time=step_time=0.;natural=0
    for a in actions:
        at=time.perf_counter()
        o=env.observe_ready(copy=False) if compact else env.observe()
        obs_time+=time.perf_counter()-at
        if verify:
            if compact:ids=o['slots'];ready=np.zeros(t['n_games'],np.uint8);ready[ids]=1;rows=np.arange(len(ids))
            else:ready=o['ready'];ids=np.flatnonzero(ready);rows=ids
            digest.update(np.asarray(ready,dtype=np.uint8).tobytes())
            for k in ['features','cards','action_mask']:digest.update(o[k][rows].tobytes())
        at=time.perf_counter();r,d,i=env.step(a);step_time+=time.perf_counter()-at
        natural+=int(d.sum())
        if verify:
            for v in [r,d,i['placement'],i['rejected']]:digest.update(v.tobytes())
            if 'stop_reason' in i and (d&(i['stop_reason']!=1)).any():raise RuntimeError('simulator fault')
    at=time.perf_counter();states=[json.loads(env.save(g)) for g in range(t['n_games'])];saved=time.perf_counter()-at
    state_hash=hashlib.sha256(json.dumps(states,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    if state_out is not None:state_out.write_text(json.dumps(states,sort_keys=True,separators=(',',':'))+'\n')
    if natural!=t['natural_episodes']:raise AssertionError('natural completion count changed')
    measured=init+obs_time+step_time+saved
    return {'trace_sha256':hashlib.sha256(trace.read_bytes()).hexdigest(),'source':tg.build_info(),
        'compact':compact,'verify':verify,'initialization_seconds':init,'observation_seconds':obs_time,'simulation_step_seconds':step_time,
        'checkpoint_seconds':saved,'complete_protocol_seconds':measured,'decisions':t['decisions'],
        'decisions_per_complete_protocol_second':t['decisions']/measured,
        'decisions_per_observe_step_second':t['decisions']/(obs_time+step_time),
        'natural_episodes':natural,'views_masks_results_sha256':digest.hexdigest() if verify else None,
        'final_state_rng_sha256':state_hash,'wall_seconds':time.perf_counter()-begin}


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['record','replay'])
    p.add_argument('--trace',type=Path,required=True);p.add_argument('--out',type=Path)
    p.add_argument('--n-games',type=int,default=32);p.add_argument('--steps',type=int,default=300)
    p.add_argument('--compact',action='store_true');p.add_argument('--verify',action='store_true');a=p.parse_args()
    if a.command=='record':record(a.trace,a.n_games,a.steps)
    else:
        if a.out.exists():raise FileExistsError(a.out)
        a.out.write_text(json.dumps(replay(a.trace,a.compact,a.verify,a.out.with_suffix('.states.json') if a.verify else None),indent=2)+'\n')


if __name__=='__main__':main()
