"""Trusted training entry; editable learner, safe finite export, exact resume binding."""
import argparse,sys,json,time,random,importlib.util,os
from pathlib import Path
import numpy as np
import torch
from contract import HERE,GAME,EDITABLE,binding,validate,sha,write

def load_modules(source):
 sys.path.insert(0,str(source));sys.path.insert(1,str(GAME))
 for name,file in [('model','policy.py'),('learner','learner.py'),('entrant_train','train.py')]:
  spec=importlib.util.spec_from_file_location(name,source/file);mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod)
 return sys.modules['model'],sys.modules['entrant_train']
def main():
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--recipe',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--seconds',type=float,default=60);p.add_argument('--device',default='cpu');p.add_argument('--resume',type=Path);p.add_argument('--checkpoint-interval',type=float,default=0);p.add_argument('--allocation-started',type=float);p.add_argument('--allocation-deadline',type=float);p.add_argument('--record-actions',action='store_true');a=p.parse_args()
 started=time.perf_counter();a.out.mkdir(parents=True,exist_ok=False);c=json.loads(a.recipe.read_text());validate(c);c['device']=a.device
 os.environ['TAVERNGYM_WORKERS']=c.get('worker_backend','pooled');os.environ['TAVERNGYM_SCHEDULER']=c.get('native_scheduler','static');os.environ['TAVERNGYM_GUARD_DIR']=str(a.out.resolve())
 torch.set_num_threads(c.get('torch_threads',1));torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
 random.seed(c['seed']);np.random.seed(c['seed']);torch.manual_seed(c['seed'])
 if a.device.startswith('cuda') and not torch.cuda.is_available():raise ValueError('CUDA unavailable; no fallback')
 b=binding();source={f:sha(a.source/f) for f in EDITABLE};ident=dict(binding=b,source=source,config=c)
 policy,trainer=load_modules(a.source);model=policy.Policy(**c['network']).to(a.device);opt=torch.optim.Adam(model.parameters(),lr=c['learning_rate'],eps=1e-5)
 from rollout import Timer
 timer=Timer(a.device);start=0;history=[];prior=0.
 if a.resume:
  ck=torch.load(a.resume,map_location=a.device,weights_only=False)
  if ck['identity']!=ident:raise ValueError('Resume source/config/rules mismatch')
  model.load_state_dict(ck['policy']);opt.load_state_dict(ck['optimizer']);torch.set_rng_state(ck['torch_rng'].cpu());np.random.set_state(ck['numpy_rng']);random.setstate(ck['python_rng'])
  if ck['cuda_rng'] is not None:torch.cuda.set_rng_state_all([x.cpu() for x in ck['cuda_rng']])
  start=ck['update'];prior=ck['accounted_seconds'];history=ck['history']
  ledger=a.resume.parent/'summary.json'
  if ledger.exists():prior=max(prior,json.loads(ledger.read_text())['accounted_seconds'])
 last_export=0.;last_state=[];last_update=start;milestones=set()
 def export(i,states):
  nonlocal last_export
  arrays={k:v.detach().cpu().numpy() for k,v in model.state_dict().items()}
  if not all(np.isfinite(v).all() for v in arrays.values()) or sum(v.size for v in arrays.values())>500000:raise ValueError('Nonfinite/oversized export')
  # Every file is replaced whole, so a trainer ended at any instant leaves its last complete export for
  # finalize_entry.py (a run the researcher stopped, or one cut off at its time limit, keeps that network).
  with (a.out/'weights.partial').open('wb') as stream:np.savez(stream,**arrays)
  torch.save(dict(identity=ident,policy=model.state_dict(),optimizer=opt.state_dict(),update=i,torch_rng=torch.get_rng_state(),numpy_rng=np.random.get_state(),python_rng=random.getstate(),cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,history=history,env_states=states,accounted_seconds=prior+time.perf_counter()-started),a.out/'resume.partial')
  os.replace(a.out/'weights.partial',a.out/'weights.npz');os.replace(a.out/'resume.partial',a.out/'resume.pt')
  write(a.out/'identity.json',ident);write(a.out/'metrics.json',history)
  write(a.out/'export.json',dict(update=i,accounted_seconds=prior+time.perf_counter()-started))
  last_export=time.perf_counter()
  elapsed=time.time()-a.allocation_started if a.allocation_started else time.perf_counter()-started
  for threshold in [300,600]:
   if elapsed>=threshold and threshold not in milestones:
    target=a.out/'checkpoints'/str(threshold);target.mkdir(parents=True,exist_ok=False)
    import shutil
    for f in ['weights.npz','identity.json']:shutil.copy2(a.out/f,target/f)
    write(target/'timing.json',dict(allocation_elapsed_seconds=elapsed,update=i,target_seconds=threshold))
    milestones.add(threshold)
  if i==2:torch.save(torch.load(a.out/'resume.pt',weights_only=False),a.out/'update2.pt')
 def save(i,row,states):
  nonlocal last_state,last_update
  history.append(dict(update=i,**row));last_state=states;last_update=i
  bad={r['game'] for r in row.get('episodes',[]) if not r.get('valid',True)}
  if bad:write(a.out/'faults'/('update-'+str(i)+'.json'),dict(update=i,counts=row['counts'],episodes=row['episodes'],states={str(j):states[j] for j in bad}))
  elapsed=time.time()-a.allocation_started if a.allocation_started else time.perf_counter()-started
  if a.checkpoint_interval<=0 or i==2 or time.perf_counter()-last_export>=a.checkpoint_interval or any(elapsed>=t and t not in milestones for t in [300,600]):export(i,states)
 remaining=min(a.seconds,a.allocation_deadline-time.time()) if a.allocation_deadline else a.seconds
 undo_trace=None
 if a.record_actions:
  from diagnostic_trace import install
  undo_trace=install(a.out/'action-traces',c['max_batch_steps'])
 try:trainer.train(model,opt,c,torch.device(a.device),timer,time.perf_counter()+max(0,remaining),start,save)
 finally:
  if undo_trace:undo_trace()
 if last_update>start:export(last_update,last_state)
 if not history:raise ValueError('No completed PPO update')
 write(a.out/'summary.json',dict(updates=history[-1]['update'],samples=sum(x['learning']['samples'] for x in history),finite_losses=all(np.isfinite(x['learning']['loss']) for x in history),changed_weights=any(x['learning']['parameter_delta_l2']>0 for x in history),accounted_seconds=prior+time.perf_counter()-started,device=a.device,identity_sha256=sha(a.out/'identity.json'),weights_sha256=sha(a.out/'weights.npz')))
 print(json.dumps(json.loads((a.out/'summary.json').read_text())),flush=True)
if __name__=='__main__':main()
