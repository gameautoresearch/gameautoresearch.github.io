"""Seat-balanced paired assessment with exact source-bound public mixed lineups."""
import argparse,copy,hashlib,importlib.util,json,os,random,sys,time,signal
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[4]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');t.replace(p)
def module(path,name):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def verify(artifact,runtime):
 ident=json.loads((artifact/'identity.json').read_text());summary=json.loads((artifact/'summary.json').read_text())
 if sha(artifact/'identity.json')!=summary['identity_sha256'] or sha(artifact/'weights.npz')!=summary['weights_sha256']:raise ValueError('Artifact identity/weights mutation')
 if any(sha(artifact/f)!=h for f,h in ident['source'].items()):raise ValueError('Declared inference source mutated')
 sys.path.insert(0,str(runtime/'simulators/tavern/autoresearch'))
 import contract
 # Contract resolves frozen simulator root; engine source comes from original immutable base.
 b=ident['binding']
 import taverngym as tg
 if b['native_source_hash']!=tg.build_info()['source_hash'] or b['observations']!=[3227,124,245]:raise ValueError('Native/interface mismatch')
 for name,h in b['files'].items():
  p=runtime/name if (runtime/name).is_file() else ROOT/name
  if not p.is_file() or sha(p)!=h:raise ValueError('Admission source mismatch: '+name)
 for name,h in ident['reference_lineage']['reference_binding']['files'].items():
  p=runtime/name if (runtime/name).is_file() else ROOT/name
  if not p.is_file() or sha(p)!=h:raise ValueError('Training source mismatch: '+name)
 m=module(artifact/'policy.py','assessed_policy');policy=m.Policy(**ident['config']['network'])
 with np.load(artifact/'weights.npz',allow_pickle=False) as z:state={k:torch.from_numpy(z[k].copy()) for k in z.files}
 if sum(v.numel() for v in state.values())>2000000 or not all(torch.isfinite(v).all() for v in state.values()):raise ValueError('Nonfinite/size violation')
 policy.load_state_dict(state,strict=True);policy.eval();return policy,ident

def assess(artifacts,runtime,scripts,pool_checkpoint,seeds,out,commit=None,pool_runtime=None):
 out=Path(out);out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
 sys.path.insert(0,str(runtime));sys.path.insert(0,str(runtime/'simulators/tavern'))
 from model import Policy,act as neural_act
 visible=module(scripts,'frozen_visible_scripts')
 lobby=module(scripts.parent/'lobby.py','frozen_mixed_lobby')
 class NeuralActor:
  def __init__(self,policy,identity):self.policy=policy;self.policy_id=identity
  def remote_act(self,obs,indices):
   def timeout(*args):raise TimeoutError('15-second neural inference cap')
   old=signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,15)
   try:return neural_act(self.policy,obs,indices,torch.device('cpu'),True)
   finally:signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old)
 pool_runtime=pool_runtime or runtime
 if sha(pool_runtime/'simulators/tavern/model.py')!=sha(runtime/'simulators/tavern/model.py'):raise ValueError('Historical pool needs a separately adapted inference class')
 frozen=torch.load(pool_checkpoint,map_location='cpu',weights_only=False)
 population=[];ids=[]
 current_hash=lambda state:hashlib.sha256(b''.join(k.encode()+v.cpu().contiguous().numpy().tobytes() for k,v in sorted(state.items()))).hexdigest()
 for name,digest in frozen['binding']['files'].items():
  source=pool_runtime/name if (pool_runtime/name).is_file() else ROOT/name
  if not source.is_file() or sha(source)!=digest:raise ValueError('Pool source binding mismatch: '+name)
 finalhash=current_hash(frozen['policy'])
 if any(current_hash(e['policy'])!=e['sha256'] for e in frozen['population']['entries']):raise ValueError('Mutated frozen population')
 eligible=[e for e in frozen['population']['entries'] if e['sha256']!=finalhash]
 if len(eligible)<2:raise ValueError('Need independent historical policies, excluding final clone')
 selected=[eligible[0],eligible[len(eligible)//2],eligible[-1]]
 selected=list({e['sha256']:e for e in selected}.values())
 for e in selected:
  p=Policy(**frozen['architecture']);p.load_state_dict(e['policy']);p.eval();population.append(NeuralActor(p,e['sha256']));ids.append(dict(kind=e['kind'],update=e['update'],sha256=e['sha256']))
 policies=[visible.VisibleScript(style) for style in ('tempo','economy','synergy')]+population
 h=list(range(4,4+len(population)))
 profiles={'scripts-mixed':[1,2,3,1,2,3,1], 'scripts-history-mixed':[1,2,3,1,h[0],h[-1],h[len(h)//2]],'history-heavy':[1,3,h[0],h[-1],h[len(h)//2],h[0],h[-1]]}
 plan=dict(schema='tavern-reference-assessment-v1',seeds=seeds,seats=list(range(8)),profiles=profiles,script_sha256=sha(scripts),lobby_driver_sha256=sha(scripts.parent/'lobby.py'),pool_checkpoint_sha256=sha(pool_checkpoint),historical_population=ids,artifacts={str(p):sha(p/'identity.json') for p in artifacts},deterministic_candidate_and_opponents=True,neural_inference_cap_seconds=15,natural_scoring='placement/top4; faulted schedule aggregate null')
 if commit and json.loads(Path(commit).read_text())!=plan:raise ValueError('Precommitted assessment changed')
 write(out/'commitment.json',plan)
 results={}
 for artifact_index,artifact in enumerate(artifacts):
  artifact_label=str(artifact_index)+'-'+artifact.name
  candidate,ident=verify(artifact,runtime);c=copy.deepcopy(ident['config']);c.update(n_games=1,learning_seats=list(range(8)),device='cpu',native_threads=1,graph_inference=False,graph_learner=False,sparse_observations=False,device_rollouts=False,opponent='selfplay')
  opponent_labels=['tempo','economy','synergy']+[e['kind']+'-update-'+str(e['update']) for e in ids]
  rows=[];candidate_actor=NeuralActor(candidate,'candidate-'+sha(artifact/'identity.json'))
  for profile,lineup in profiles.items():
   for seed in seeds:
    for seat in range(8):
     torch.manual_seed(seed+seat);random.seed(seed+seat);np.random.seed((seed+seat)%2**32)
     actors=[policies[i-1] for i in lineup];actors.insert(seat,candidate_actor)
     result=lobby.run_lobby(actors,seed,c['rules'],threads=1,seconds=180,trace=out/artifact_label/'traces'/profile/f'group-{seeds.index(seed)}-seat-{seat}')
     terminal=result['terminals'].get(seat,result['terminals'].get(str(seat),{}))
     opponent_ids=lineup[:seat]+[-1]+lineup[seat:]
     pairwise=[dict(policy=opponent_labels[pid-1],seat=other,placement=result['terminals'].get(other,result['terminals'].get(str(other),{})).get('placement') if result['complete_valid_lobby'] else None) for other,pid in enumerate(opponent_ids) if pid!=-1]
     rows.append(dict(opponents=pairwise,seed=seed,seat=seat,profile=profile,valid=result['complete_valid_lobby'],stop_reason=terminal.get('stop_reason','unresolved'),placement=terminal.get('placement') if result['complete_valid_lobby'] else None,failure=result['failure'],counts=result['counts'],full_lobby_natural=result['complete_valid_lobby']))
     write(out/artifact_label/'episodes.json',rows)
  valid=[r for r in rows if r['valid'] and r['stop_reason']=='natural']
  def summary(rs):
   good=[r for r in rs if r['valid'] and r['stop_reason']=='natural'];placements=[r['placement'] for r in good]
   return dict(required=len(rs),natural=len(good),faults=len(rs)-len(good),natural_first_places=sum(p==1 for p in placements),tied_placements=0,placement_counts={str(p):placements.count(p) for p in range(1,9)},failure_counts={kind:sum((r.get('failure') or {}).get('kind')==kind for r in rs) for kind in ['administrative_cap','illegal_action','visible_engine_rejection','simulator_fault','infrastructure_fault','inference_deadline']},mean_placement=float(np.mean(placements)) if placements and len(rs)==len(good) else None,top4_rate=float(np.mean(np.array(placements)<=4)) if placements and len(rs)==len(good) else None,descriptive_valid_mean_placement=float(np.mean(placements)) if placements else None,aggregate_qualified=len(rs)==len(good))
  per_opponent={}
  for label in opponent_labels:
   pairs=[(r['placement'],o['placement']) for r in valid for o in r['opponents'] if o['policy']==label]
   per_opponent[label]=dict(comparisons=len(pairs),candidate_outrank_rate=sum(a<b for a,b in pairs)/len(pairs) if pairs else None,note='Descriptive pairwise placement in mixed eight-player lobbies; not a1v1 win rate.')
  result=dict(per_opponent=per_opponent,identity_sha256=sha(artifact/'identity.json'),episodes=rows,summary=summary(rows),per_profile={p:summary([r for r in rows if r['profile']==p]) for p in profiles},per_seat={str(s):summary([r for r in rows if r['seat']==s]) for s in range(8)},counts={key:sum(r['counts'].get(key,0) for r in rows) for key in ('rejected','deferred','invalid_lobbies')})
  results[str(artifact)]=result;write(out/'results.json',results)
 # Paired bootstrap whole lobby seed groups, preserving all seats/profiles.
 if len(artifacts)==2:
  first,second=[results[str(p)] for p in artifacts]
  if first['summary']['faults']==second['summary']['faults']==0:
   a={ (r['profile'],r['seed'],r['seat']):r['placement'] for r in first['episodes']};b={ (r['profile'],r['seed'],r['seat']):r['placement'] for r in second['episodes']}
   deltas=[np.mean([b[k]-a[k] for k in a if k[1]==s]) for s in seeds];rng=np.random.default_rng(7281);boot=rng.choice(deltas,(10000,len(seeds)),replace=True).mean(1)
   write(out/'paired.json',dict(final_minus_initial_mean_placement=float(np.mean(deltas)),seed_group_bootstrap_95=np.quantile(boot,[.025,.975]).tolist(),groups=len(seeds),lower_is_better=True,uncertainty_limit='Few seed groups and one training seed; conditional on frozen lineups, not general strength.'))
 return results
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--artifacts',nargs='+',type=Path,required=True);p.add_argument('--runtime',type=Path,required=True);p.add_argument('--scripts',type=Path,required=True);p.add_argument('--pool-checkpoint',type=Path,required=True);p.add_argument('--pool-runtime',type=Path);p.add_argument('--seeds',nargs='+',type=int,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();assess(a.artifacts,a.runtime,a.scripts,a.pool_checkpoint,a.seeds,a.out,pool_runtime=a.pool_runtime)
