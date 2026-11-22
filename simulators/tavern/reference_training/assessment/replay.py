"""Replay every recorded native action through complete lobbies, without inference."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import taverngym as tg
STOP={0:'ongoing',1:'natural',2:'no_progress',3:'driver_guard',4:'turn_limit',5:'combat_cutoff',6:'recruit_cutoff'}
def replay(folder):
 expected=json.loads((folder/'result.json').read_text())
 if expected['native']!=tg.build_info()['source_hash']:raise ValueError('Native replay identity mismatch')
 initial=(folder/'initial.json').read_text();state=json.loads(initial)
 env=tg.BatchEnv(1,seed=expected['seed'],controlled=range(8),config=state['game']['cfg'],threads=1);env.load(0,initial)
 if env.save(0)!=initial:raise ValueError('Initial native state failed exact restore')
 with np.load(folder/'actions.npz',allow_pickle=False) as z:actions=z['actions']
 terminal={};finished=np.zeros(8,bool);rejections=deferred=0
 for frame,a in enumerate(actions):
  obs=env.observe();ready=np.flatnonzero(obs['ready'] & ~finished)
  if not obs['action_mask'][ready,a[ready]].all():raise ValueError('Recorded action outside preframe mask')
  reward,done,info=env.step(a);rejections+=int(info['rejected'].sum());deferred+=int(info['deferred'].sum())
  for seat in np.flatnonzero(done & ~finished):
   reason=int(info['stop_reason'][seat]);terminal[str(seat)]=dict(placement=int(info['placement'][seat]) if reason==1 else None,stop_reason=STOP[reason],natural=reason==1,reward=float(reward[seat]));finished[seat]=True
 finalhash=hashlib.sha256(env.save(0).encode()).hexdigest()
 ok=finalhash==expected['final_state_sha256'] and terminal==expected['terminals'] and rejections==expected['counts'].get('rejected',0) and deferred==expected['counts'].get('deferred',0)
 if not ok:raise ValueError('Native action replay differs: '+str(folder))
 return dict(path=str(folder),frames=len(actions),exact_final_state=True,exact_terminals=True,rejections=rejections,deferred=deferred)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 if a.out.exists():raise FileExistsError(a.out)
 records=[replay(f.parent) for f in sorted(a.root.glob('*/traces/*/*/result.json'))]
 result=dict(native=tg.build_info()['source_hash'],all_passed=bool(records),lobbies=len(records),frames=sum(r['frames'] for r in records),records=records);a.out.write_text(json.dumps(result,indent=2)+'\n');print({k:v for k,v in result.items() if k!='records'})
