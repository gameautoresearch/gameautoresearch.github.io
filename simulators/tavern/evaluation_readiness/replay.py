"""Replay private recorded actions; no candidate inference or source migration."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import taverngym as tg
from .lobby import STOP

def replay(path):
    result=json.loads((path/'result.json').read_text());initial=(path/'initial.json').read_text()
    if result['native']!=tg.build_info()['source_hash']:raise ValueError('Replay native mismatch')
    cfg=json.loads(initial)['game']['cfg'];n=int(cfg.get('num_players',8))
    env=tg.BatchEnv(1,seed=result['seed'],controlled=range(n),config=cfg,threads=1);env.load(0,initial)
    if env.save(0)!=initial:raise ValueError('Initial state restore mismatch')
    terminals={};finished=np.zeros(n,bool);rejected=deferred=0
    with np.load(path/'actions.npz',allow_pickle=False) as z:actions=z['actions']
    for a in actions:
        obs=env.observe();ready=np.flatnonzero(obs['ready'] & ~finished)
        if not obs['action_mask'][ready,a[ready]].all():raise ValueError('Replay preframe legality mismatch')
        reward,done,info=env.step(a);rejected+=int(info['rejected'].sum());deferred+=int(info['deferred'].sum())
        for seat in np.flatnonzero(done & ~finished):
            reason=int(info['stop_reason'][seat]);terminals[str(seat)]=dict(placement=int(info['placement'][seat]) if reason==1 else None,stop_reason=STOP[reason],natural=reason==1,reward=float(reward[seat]));finished[seat]=True
    digest=hashlib.sha256(env.save(0).encode()).hexdigest()
    if digest!=result['final_state_sha256'] or terminals!=result['terminals'] or rejected!=result['counts'].get('rejected',0) or deferred!=result['counts'].get('deferred',0):raise ValueError('Native replay differs: '+str(path))
    return dict(path=str(path),frames=len(actions),exact_final_state=True,exact_all8_terminals=True,rejections=rejected,deferred=deferred)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--traces',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    records=[replay(f.parent) for f in sorted(a.traces.glob('*/result.json'))]
    result=dict(native=tg.build_info()['source_hash'],all_passed=bool(records),lobbies=len(records),frames=sum(r['frames'] for r in records),records=records)
    a.out.write_text(json.dumps(result,indent=2)+'\n');print({k:v for k,v in result.items() if k!='records'})
