"""Matched incumbent continuation and on-policy script/history population PPO."""
import argparse,copy,json,os,random,time
from pathlib import Path
import numpy as np
import torch
from .common import ROOT,GAME,HERE,admission,binding,validate_config,write,sha
from .policy import Policy
from .population import Population,schedule
from .lineage import import_parent,module
from .learner import update as ppo_update
from .. import collection
from ..population import weight_hash
from model import act as neural_act

def actor_act(policy,obs,indices,device,deterministic=False):
 if hasattr(policy,'remote_act'):return policy.remote_act(obs,indices)
 return neural_act(policy,obs,indices,device,deterministic)
collection.act=actor_act

def rng_state():return dict(torch=torch.get_rng_state(),python=random.getstate(),numpy=np.random.get_state(),cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
def restore_rng(r):
 torch.set_rng_state(r['torch']);random.setstate(r['python']);np.random.set_state(r['numpy'])
 if r['cuda'] is not None and torch.cuda.is_available():torch.cuda.set_rng_state_all(r['cuda'])
def checkpoint(path,model,opt,pop,update,c,pin,history,states,lineage):
 item=dict(schema='tavern-strong-population-v1',binding=pin,config=c,architecture=model.arch,policy=model.state_dict(),optimizer=opt.state_dict(),population=pop.state_dict(),rng=rng_state(),update=update,history=history,env_states=states,lineage=lineage,next_cohort_seed=(c['seed']+update*1000003)%2**64)
 tmp=path.with_suffix('.tmp');torch.save(item,tmp);tmp.replace(path)
 return item

def run(c,out,inputs,seconds,resume=None,limit_updates=None):
 validate_config(c);out=Path(out);out.mkdir(parents=True,exist_ok=False);started=time.perf_counter();deadline=started+seconds
 os.environ['TAVERNGYM_WORKERS']='pooled';os.environ['TAVERNGYM_SCHEDULER']='dynamic';os.environ['TAVERNGYM_GUARD_DIR']=str(out.resolve())
 torch.set_num_threads(c.get('torch_threads',1));torch.use_deterministic_algorithms(True)
 device=torch.device(c['device']);pin=binding();start=0;history=[];states=[]
 if device.type=='cuda':
  torch.cuda.init();torch.cuda.reset_peak_memory_stats(device)
 model,parent,pop,lineage=import_parent(inputs,device)
 anchor=copy.deepcopy(model).eval().requires_grad_(False) if c.get('anchor_kl_coef',0) else None
 opt=torch.optim.Adam(model.parameters(),lr=c['learning_rate'],eps=1e-5);opt.load_state_dict(parent['optimizer'])
 restore_rng(dict(torch=parent['torch_rng'],python=parent['python_rng'],numpy=parent['numpy_rng'],cuda=parent['cuda_rng'] if device.type=='cuda' else None))
 if resume:
  loaded=torch.load(resume,map_location='cpu',weights_only=False)
  if loaded['schema']!='tavern-strong-population-v1' or loaded['binding']!=pin or loaded['config']!=c:raise ValueError('Exact source/config/identity required for resume')
  if loaded['lineage']['input_manifest_sha256']!=lineage['input_manifest_sha256']:raise ValueError('Warmstart/script input source changed')
  model.load_state_dict(loaded['policy']);opt.load_state_dict(loaded['optimizer']);pop=Population.restore(loaded['population']);start=loaded['update'];history=loaded['history'];states=loaded['env_states'];lineage=loaded['lineage'];restore_rng(loaded['rng'])
  import taverngym as tg
  controlled=[0] if c['mode']=='plain' else list(range(8))
  env=tg.BatchEnv(c['n_games'],seed=(c['seed']+max(0,start-1)*1000003)%2**64,controlled=controlled,config=c['rules'],threads=c['native_threads'])
  for g,state in enumerate(states):
   env.load(g,state)
   if env.save(g)!=state:raise ValueError('Native state restore differs')
 parameters=sum(p.numel() for p in model.parameters())
 if parameters>2000000:raise ValueError('Entrant parameter ceiling')
 write(out/'manifest.json',dict(binding=pin,config=c,lineage=lineage,parameters=parameters,resume_sha256=sha(resume) if resume else None,incumbent_anchor_policy_sha256=weight_hash(anchor.state_dict()) if anchor is not None else None,incumbent_anchor_scope='Frozen imported-parent policy, visible arrays and legal masks only; training regularizer, no test-time teacher' if anchor is not None else None))
 timer=collection.Timer(device,synchronized=c.get('profile_synchronized',False))
 with timer.measure('checkpoint_io'):
  checkpoint(out/'initial.pt',model,opt,pop,start,c,pin,history,states,lineage)
 scripts=module(Path(inputs)/'scripts/opponents.py','campaign_public_scripts');script_actors=[scripts.VisibleScript(style) for style in ('tempo','economy','synergy')]
 if c.get('batched_scripts',False):
  from .scripts import BatchedVisibleScript
  script_actors=[BatchedVisibleScript(scripts,style) for style in ('tempo','economy','synergy')]
 done=start;termination='update_limit'
 for i in range(start,min(c['updates'],start+limit_updates) if limit_updates else c['updates']):
  if time.perf_counter()>=deadline:termination='deadline';break
  # Identical original DeepSeek schedule in both arms; optimizer moments carried.
  frac=i/max(1,c.get('anneal_updates',c['updates']))
  for group in opt.param_groups:group['lr']=c['learning_rate']*max(.25,1-.75*frac)
  live=copy.deepcopy(c);live['entropy_coef']=c['entropy_coef']*max(.15,1-.85*frac)
  assignment=None if c['mode']=='plain' else schedule(c['seed'],i,c['n_games'],len(pop.entries))
  with timer.measure('population_setup'):
   actors=() if assignment is None else pop.actors(device)+script_actors
  try:episodes,records,counts,states=collection.collect(model,live,(c['seed']+i*1000003)%2**64,device,timer,deadline,assignment=assignment,population=actors)
  except collection.BudgetExpired:termination='deadline_partial_cohort_discarded';break
  bad={r['game'] for r in records if not r['valid']}
  if bad:write(out/'faults'/f'cohort-{i}.json',dict(records=records,counts=counts,states={str(g):states[g] for g in bad}))
  before=weight_hash(model.state_dict());metrics=ppo_update(model,opt,episodes,live,device,anchor=anchor,timer=timer);metrics['training_samples']=metrics['samples'];done=i+1
  if done%c['population_snapshot_every']==0:pop.add(model,done)
  row=dict(update=done,learning=metrics,counts=counts,episodes=records,assignment=assignment.tolist() if assignment is not None else None,population=[{k:v for k,v in e.items() if k!='policy'} for e in pop.entries],policy_before=before,policy_after=weight_hash(model.state_dict()))
  history.append(row)
  with (out/'history.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
  with timer.measure('checkpoint_io'):
   checkpoint(out/'latest.pt',model,opt,pop,done,c,pin,history,states,lineage)
  status=dict(update=done,session_elapsed=time.perf_counter()-started,lineage_samples=sum(h['learning']['samples'] for h in history),learning=metrics,counts=counts)
  write(out/'progress.json',status);print(json.dumps(status),flush=True)
 if not (out/'latest.pt').exists():checkpoint(out/'latest.pt',model,opt,pop,done,c,pin,history,states,lineage)
 elapsed=time.perf_counter()-started
 memory=(dict(peak_allocated_bytes=torch.cuda.max_memory_allocated(device),peak_reserved_bytes=torch.cuda.max_memory_reserved(device),device_bytes=torch.cuda.get_device_properties(device).total_memory) if device.type=='cuda' else None)
 write(out/'profile.json',dict(stage_seconds=dict(timer.seconds),diagnostics=dict(timer.diagnostics),counters=dict(timer.counters),synchronized=timer.synchronized,elapsed_seconds=elapsed,cuda_memory=memory,unattributed_seconds=max(0.,elapsed-sum(timer.seconds.values())),scope='Complete from-start population cohorts, real PPO and checkpoint IO; diagnostic actor times overlap inference stage'))
 write(out/'result.json',dict(initial_update=start,update=done,termination=termination,elapsed_seconds=elapsed,checkpoint_sha256=sha(out/'latest.pt'),samples=sum(h['learning']['samples'] for h in history),new_samples=sum(h['learning']['samples'] for h in history if h['update']>start),parameters=parameters,population=[{k:v for k,v in e.items() if k!='policy'} for e in pop.entries]))
 return out/'latest.pt'
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--inputs',type=Path,required=True);p.add_argument('--seconds',type=float,required=True);p.add_argument('--resume',type=Path);p.add_argument('--limit-updates',type=int);a=p.parse_args()
 from ..diagnostic_trace import install
 undo=install(a.out/'action-traces',12000)
 try:run(json.loads(a.config.read_text()),a.out,a.inputs,a.seconds,a.resume,a.limit_updates)
 finally:undo()
