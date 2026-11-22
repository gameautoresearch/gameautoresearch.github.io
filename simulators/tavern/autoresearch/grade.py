"""Trusted independent CPU evaluator; candidate code executes in another UID/process."""
import argparse,json,sys,subprocess,os,io,base64,select,time
from pathlib import Path
import numpy as np
from contract import HERE,GAME,RULES,EDITABLE,BENCHMARK_ID,PLAYERS,binding,points,write,sha,validate
class Remote:
 def __init__(self,bundle):
  self.inference_metrics={}
  def demote():
   os.setgroups([]);os.setgid(65534);os.setuid(65534)
  self.p=subprocess.Popen([sys.executable,'-I',str(HERE/'policy_worker.py'),str(bundle)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=sys.stderr,env={'PATH':'/usr/local/bin:/usr/bin:/bin','OMP_NUM_THREADS':'1'},preexec_fn=demote if os.geteuid()==0 else None,cwd='/tmp',text=True,bufsize=1)
 def remote_act(self,obs,indices):
  buf=io.BytesIO();np.savez(buf,features=obs['features'][indices],cards=obs['cards'][indices],mask=obs['action_mask'][indices]);self.p.stdin.write(base64.b64encode(buf.getvalue()).decode()+'\n');self.p.stdin.flush()
  if not select.select([self.p.stdout],[],[],15)[0]:raise ValueError('Policy deadline')
  line=self.p.stdout.readline(1000001)
  if len(line)>1000000:raise ValueError('Oversized policy output')
  values=json.loads(line)
  if not isinstance(values,list) or any(type(v) is not int for v in values):raise ValueError('Policy actions must be JSON integers')
  a=np.array(values,dtype=np.int64);mask=obs['action_mask'][indices]
  if a.shape!=(len(indices),) or ((a<0)|(a>=mask.shape[1])).any() or not mask[np.arange(len(a)),a].all():raise ValueError('Illegal policy response')
  return a,np.zeros(len(a)),np.zeros(len(a))
 def close(self):
  self.p.kill();self.p.wait()
def main():
 if os.geteuid()!=0:raise RuntimeError('Official grading requires the dedicated root container and unprivileged policy UID')
 p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);p.add_argument('--panel',type=Path,required=True);p.add_argument('--opponent',type=Path);p.add_argument('--out',type=Path,required=True);p.add_argument('--trace-dir',type=Path);a=p.parse_args();started=time.perf_counter()
 ident=json.loads((a.bundle/'identity.json').read_text())
 if ident['binding']!=binding():raise ValueError('Source/rules-bound checkpoint mismatch')
 validate(ident['config'])
 if any(sha(a.bundle/f)!=ident['source'][f] for f in EDITABLE):raise ValueError('Editable source identity mismatch')
 summary=json.loads((a.bundle/'summary.json').read_text())
 if sha(a.bundle/'identity.json')!=summary['identity_sha256'] or sha(a.bundle/'weights.npz')!=summary['weights_sha256']:raise ValueError('Weights/identity digest mismatch')
 if (a.bundle/'weights.npz').stat().st_size>32000000:raise ValueError('Oversized weights')
 panel=json.loads(a.panel.read_text());a.out.mkdir(parents=True,exist_ok=False)
 # Load only host-owned model code for fixed opponent. Candidate import lives solely in worker.
 sys.path.insert(0,str(HERE/'starter'));sys.path.insert(1,str(GAME));import policy as fixed
 sys.modules['model']=fixed
 from rollout import collect,Timer
 import torch
 torch.set_num_threads(1)
 opponent=None
 if a.opponent:
  oi=json.loads((a.opponent/'identity.json').read_text())
  if oi['binding']!=binding():raise ValueError('Opponent source mismatch')
  opponent=fixed.Policy(**oi['config']['network'])
  with np.load(a.opponent/'weights.npz',allow_pickle=False) as z:opponent.load_state_dict({k:torch.tensor(z[k]) for k in z.files})
 undo_trace=None
 if a.trace_dir:
  from diagnostic_trace import install
  undo_trace=install(a.trace_dir,6000)
 candidate=Remote(a.bundle);rows=[]
 try:
  for seed in panel['seeds']:
   for seat in range(PLAYERS):
    c=json.loads((HERE/'recipes/starter.json').read_text())
    c.update(device='cpu',rules=RULES,learning_seats=[seat],n_games=panel['games_per_seat'],opponent='frozen' if opponent else 'scripted',start_turn=1)
    _,records,counts,_=collect(candidate,c,seed,torch.device('cpu'),Timer(torch.device('cpu')),time.perf_counter()+120,opponent,deterministic=True)
    if counts.get('rejected',0) or counts.get('invalid_lobbies',0) or len(records)!=panel['games_per_seat'] or any(not r['valid'] or r['stop_reason']!='natural' or not 1<=r['placement']<=PLAYERS for r in records):
     write(a.out/'failure.json',dict(seed=seed,seat=seat,records=records,counts=counts,no_score_assigned=True))
     raise ValueError('Incomplete/faulted grading schedule; no valid score')
    rows.extend(records)
 finally:
  candidate.close()
  if undo_trace:undo_trace()
 group=[np.mean([points(r['placement']) for r in rows if r['seed']==s]) for s in panel['seeds']]
 rng=np.random.default_rng(42);means=np.mean(rng.choice(group,(2000,len(group))),axis=1)
 result=dict(benchmark_id=BENCHMARK_ID,players=PLAYERS,raw_score=float(np.mean(group)),reference_normalized_score=None,normalization_qualified=False,top_half_rate=float(np.mean([r['placement']<=PLAYERS//2 for r in rows])),mean_placement=float(np.mean([r['placement'] for r in rows])),episodes=len(rows),natural=len(rows),illegal_actions=0,simulator_faults=0,truncations=0,complete_valid_schedule=True,per_seat={str(s):float(np.mean([points(r['placement']) for r in rows if r['seat']==s])) for s in range(PLAYERS)},confidence_interval_95=np.quantile(means,[.025,.975]).tolist() if len(group)>1 else None,interval_method='seed-group bootstrap; null with fewer than two groups',opponent='frozen-small-ppo' if opponent else 'scripted',cpu_seconds=time.perf_counter()-started)
 result['per_opponent']={result['opponent']:{'raw_score':result['raw_score'],'episodes':len(rows)}}
 result['candidate_process_uid']=65534 if os.geteuid()==0 else os.geteuid()
 result['isolation']='separate unprivileged process in network-blocked container' if os.geteuid()==0 else 'local trusted-code diagnostic only'
 write(a.out/'result.json',result);write(a.out/'episodes.json',rows);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
