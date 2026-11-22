"""Reachable public wallet-state probes; no state edits or strength score."""
import argparse,fcntl,json,time
from pathlib import Path
import numpy as np
from .common import MATCH_FORM,ROOT,DATA,dump,sha,setup_native
from .evaluation_v2 import Actor
from .bundle import verify

def run(specs,out):
 setup_native()
 from cpu_batch import NativeCPUBatch
 from evaluate import scripted_bot
 out=Path(out);out.mkdir(parents=True,exist_ok=False);data=json.loads(DATA.read_text());vocab=data['vocabulary'];seeds=list(range(2100501,2100517));env=NativeCPUBatch(seeds,threads=4,form=MATCH_FORM);snapshots={};actors={};records=[]
 def snapshot():return env.observations.copy().reshape(-1,7024),env.masks.copy().reshape(-1,2307)
 try:
  snapshots['initial_wallet']=snapshot()
  for _ in range(200):
   if (env.observations[:,:,2]>=100000).all():break
   env.step(np.zeros((len(seeds),2),np.int32))
   if env.infos[:,[5,7,8,9,11]].any() or env.observations[:,:,49].any():raise ValueError('invalid reachable waiting trajectory')
  else:raise ValueError('full wallet not reached in early regulation')
  snapshots['full_wallet']=snapshot()
  for spec in specs:
   verify(spec['bundle'],spec['identity']);actors[spec['name']]=Actor(spec,data,20261109,True)
  teacher=scripted_bot(data)
  for stage,(obs,masks) in snapshots.items():
   choices={};diagnostics={}
   for name,actor in actors.items():
    choices[name]=actor.choose(obs,masks,[(seed,seat) for seed in seeds for seat in (0,1)]);diagnostics[name]=dict(actor.process.last_diagnostics)
   choices['standard-fixed-tie']=[int(teacher(o,m,np.random.default_rng(0))) for o,m in zip(obs,masks)]
   for i,(o,m) in enumerate(zip(obs,masks)):
    hand=[(vocab[int(c)] if int(c)>=0 else 'Empty') for c in o[7:11]];giant=hand.index('Giant') if 'Giant' in hand else None;giant_legal=bool(giant is not None and m[1+giant*576:1+(giant+1)*576].any())
    def describe(name,action):
     if not m[action]:raise ValueError('illegal diagnostic choice')
     return {'action':int(action),'card':'wait' if action==0 else hand[(action-1)//576],'wait_probability_in_exported_decoder':diagnostics[name]['wait_probability'][i] if name in diagnostics else None}
    records.append({'stage':stage,'seed':seeds[i//2],'seat':i%2,'tick':int(o[0]),'elixir':int(o[2]),'hand':hand,'Giant_legal':giant_legal,'choices':{name:describe(name,values[i]) for name,values in choices.items()}})
  summary={}
  for stage in snapshots:
   rows=[r for r in records if r['stage']==stage and r['Giant_legal']];summary[stage]={'reachable_rows_with_legal_Giant':len(rows),'chosen_cards':{name:{card:sum(r['choices'][name]['card']==card for r in rows) for card in sorted({r['choices'][name]['card'] for r in rows})} for name in choices}}
  np.savez_compressed(out/'public_states.npz',**{stage+'_'+kind:values[j] for stage,values in snapshots.items() for j,kind in enumerate(('observations','masks'))})
  value={'public_only':True,'source':'real native trajectories with both players waiting; no observation/world/RNG edits','strength_score':None,'seeds':seeds,'specs':specs,'summary':summary,'records':records,'interpretation':'Reachable sparse initial/full-wallet contexts only. These probes distinguish conditional choices but do not establish a causal learning defect or match strength. Exported decoder probabilities differ from original PPO sampling distribution.'};dump(out/'result.json',value);return summary
 finally:
  env.close()
  for actor in actors.values():actor.close()

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--specs',required=True);p.add_argument('--out',required=True);a=p.parse_args()
 with (ROOT/'evidence/local-performance.lock').open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);print(json.dumps(run(json.loads(Path(a.specs).read_text()),a.out),indent=2))
