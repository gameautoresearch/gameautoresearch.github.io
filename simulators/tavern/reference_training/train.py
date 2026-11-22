"""Full-cohort on-policy population PPO with strict update-boundary resume."""
import argparse,copy,json,os,random,time
from pathlib import Path
from .bootstrap import ROOT,HERE,binding,write,sha
import numpy as np
import torch
from model import Policy
from run import make_optimizer,ppo_update
from rollout import Timer
from .collection import collect,BudgetExpired
from .population import Population,schedule,weight_hash

def rng_state():return dict(torch=torch.get_rng_state(),python=random.getstate(),numpy=np.random.get_state(),cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
def restore_rng(r):
 torch.set_rng_state(r['torch']);random.setstate(r['python']);np.random.set_state(r['numpy'])
 if r['cuda'] is not None:torch.cuda.set_rng_state_all(r['cuda'])
def save(path,model,opt,pop,update,config,pin,history,states):
 obj=dict(schema='tavern-reference-population-v1',binding=pin,architecture=model.arch,policy=model.state_dict(),optimizer=opt.state_dict(),population=pop.state_dict(),update=update,config=config,rng=rng_state(),history=history,env_states=states,next_cohort_seed=(config['seed']+update*1000003)%2**64)
 t=path.with_suffix('.tmp');torch.save(obj,t);t.replace(path)
 return obj

def run(config,out,seconds,resume=None,limit_updates=None):
 os.environ['TAVERNGYM_WORKERS']=config.get('worker_backend','pooled');os.environ['TAVERNGYM_SCHEDULER']=config.get('native_scheduler','dynamic')
 out=Path(out);out.mkdir(parents=True,exist_ok=False);started=time.perf_counter();deadline=started+seconds
 torch.set_num_threads(config.get('torch_threads',1));torch.use_deterministic_algorithms(True)
 random.seed(config['seed']);np.random.seed(config['seed']%2**32);torch.manual_seed(config['seed'])
 os.environ['TAVERNGYM_GUARD_DIR']=str(out.resolve())
 device=torch.device(config['device']);pin=binding()
 model=Policy(**config['network']).to(device)
 if sum(p.numel() for p in model.parameters())>2000000:raise ValueError('Entrant-compatible parameter cap')
 opt=make_optimizer(model,config);pop=Population(model.arch);pop.add(model,0,'random-initial-anchor')
 # Explicit independent random historical bootstrap prevents identical clones.
 with torch.random.fork_rng(devices=[]):
  torch.manual_seed(config['seed']+991);bootstrap=Policy(**model.arch)
 pop.add(bootstrap,0,'independent-random-bootstrap')
 history=[];start=0;states=[]
 if resume:
  loaded=torch.load(resume,map_location='cpu',weights_only=False)
  if loaded['binding']!=pin or loaded['config']!=config:raise ValueError('Source/interface/recipe differs; no identity migration')
  import taverngym as tg
  env=tg.BatchEnv(config['n_games'],seed=(config['seed']+max(0,loaded['update']-1)*1000003)%2**64,controlled=tuple(range(8)),config=config['rules'],threads=config['native_threads'])
  for g,state in enumerate(loaded['env_states']):
   env.load(g,state)
   if env.save(g)!=state:raise ValueError('Native environment state restore mismatch')
  model.load_state_dict(loaded['policy']);opt.load_state_dict(loaded['optimizer']);pop=Population.restore(loaded['population']);start=loaded['update'];history=loaded['history'];states=loaded['env_states'];restore_rng(loaded['rng'])
 write(out/'manifest.json',dict(binding=pin,config=config,parameters=sum(p.numel() for p in model.parameters()),resume_sha256=sha(resume) if resume else None))
 save(out/'initial.pt',model,opt,pop,start,config,pin,history,states)
 timer=Timer(device);done=start;termination='update_limit'
 for i in range(start,min(config['updates'],start+limit_updates) if limit_updates else config['updates']):
  if time.perf_counter()>=deadline:termination='deadline';break
  assignment=schedule(config['seed'],i,config['n_games'],len(pop.entries))
  actors=pop.actors(device)
  try:episodes,records,counts,states=collect(model,config,(config['seed']+i*1000003)%2**64,device,timer,deadline,assignment=assignment,population=actors)
  except BudgetExpired:termination='deadline_partial_cohort_discarded';break
  before=weight_hash(model.state_dict());metrics=ppo_update(model,opt,episodes,config,device);done=i+1
  if done%config['population_snapshot_every']==0:pop.add(model,done)
  row=dict(update=done,learning=metrics,counts=counts,episodes=records,assignment=assignment.tolist(),population=[{k:v for k,v in e.items() if k!='policy'} for e in pop.entries],policy_before=before,policy_after=weight_hash(model.state_dict()))
  history.append(row)
  bad={r['game'] for r in records if not r['valid']}
  if bad:write(out/'faults'/f'update-{done}.json',dict(episodes=records,counts=counts,states={str(g):states[g] for g in bad}))
  with (out/'history.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
  save(out/'latest.pt',model,opt,pop,done,config,pin,history,states)
  write(out/'progress.json',dict(update=done,elapsed=time.perf_counter()-started,learning=metrics,counts=counts))
  print(json.dumps(dict(update=done,elapsed=time.perf_counter()-started,learning=metrics,counts=counts)),flush=True)
 if not (out/'latest.pt').exists():save(out/'latest.pt',model,opt,pop,done,config,pin,history,states)
 write(out/'result.json',dict(initial_update=start,update=done,termination=termination,elapsed_seconds=time.perf_counter()-started,checkpoint_sha256=sha(out/'latest.pt'),population=[{k:v for k,v in e.items() if k!='policy'} for e in pop.entries],stage_seconds=dict(timer.seconds)))
 return out/'latest.pt'
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--seconds',type=float,required=True);p.add_argument('--resume',type=Path);p.add_argument('--limit-updates',type=int);a=p.parse_args()
 from .diagnostic_trace import install
 undo=install(a.out/'action-traces',6000)
 try:run(json.loads(a.config.read_text()),a.out,a.seconds,a.resume,a.limit_updates)
 finally:undo()
