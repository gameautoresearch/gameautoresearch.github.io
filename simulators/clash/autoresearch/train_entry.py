"""Run editable RL code against the fixed native world; serialize complete resume."""
from pathlib import Path
import argparse,json,os,sys,time,shutil
from .common import setup_native,engine_binding,EDITABLE,MATCH_FORM,STOP_REQUEST,sha,dump,load_module

def run(source,out,seconds,resume=None,updates=None,deadline_at=None,worker_started_at=None,milestone_seconds=(),milestone_offset_seconds=0):
 if json.loads((Path(source)/'config.json').read_text()).get('training_route')=='native-fast-v1':
  from .fast_entry import run as fast_run
  return fast_run(source,out,seconds,resume,updates,deadline_at,worker_started_at,milestone_seconds,milestone_offset_seconds)
 setup_native()
 import torch,numpy as np
 from build_receipt import backend_build_binding
 from snapshots import encode_engine_state,decode_engine_state
 source=Path(source).resolve();out=Path(out).resolve();out.mkdir(parents=True,exist_ok=False)
 c=json.loads((source/'config.json').read_text());device=c.get('device','cuda');torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
 if c.get('max_ticks',0)!=0:raise ValueError('benchmark training uses natural match horizon')
 if c.get('match_form')!=MATCH_FORM:raise ValueError(f'benchmark training uses match_form {MATCH_FORM!r}')
 if c['envs']<1 or c['envs']>2048 or c['rollout_steps']<1 or c['rollout_steps']>256:raise ValueError('shape cap')
 sys.path.insert(0,str(source));module=load_module(source/'train.py','entrant_train')
 if device=='cuda':
  if torch.cuda.device_count()!=1 or torch.cuda.get_device_name()!='NVIDIA L40S':raise ValueError('one exact L40S required')
  from runtime import NativeCUDA
  env=NativeCUDA(list(range(c['envs'])),launch_threads=1,render_mode=4,form=MATCH_FORM)
 else:
  from cpu_batch import NativeCPUBatch
  env=NativeCPUBatch(list(range(c['envs'])),threads=min(4,c['envs']),form=MATCH_FORM)
 binding={'engine':engine_binding(),'source':{n:sha(source/n) for n in (*EDITABLE,'config.json')},'build':backend_build_binding(env.library),'runtime':{'torch':str(torch.__version__),'numpy':np.__version__,'device':device}}
 trainer=module.Trainer(c,env,device);initial={k:v.detach().cpu().clone() for k,v in trainer.model.state_dict().items()}
 if resume:
  saved=torch.load(resume,map_location='cpu',weights_only=True)
  if saved['binding']!=binding:raise ValueError('resume binding differs')
  trainer.model.load_state_dict(saved['model']);trainer.optimizer.load_state_dict(saved['optimizer'])
  state=decode_engine_state(saved['engine'],as_tensor=device=='cuda',expected_count=c['envs'],expected_world_bytes=4195384,max_total_bytes=16*1024**3,workers=1)
  env.load(state)
  for k,v in saved['outputs'].items():
   target=getattr(env,k)
   if isinstance(target,torch.Tensor):target.copy_(v.to(device))
   else:target[:]=v.numpy()
  for k in ('updates','samples','natural_games','losses','steps'):setattr(trainer,k,saved[k])
  torch.set_rng_state(saved['rng'])
  if device=='cuda':torch.cuda.set_rng_state_all(saved['cuda_rng'])
 started=time.monotonic();limit=c.get('updates',1000000) if updates is None else updates
 milestones_done=[]
 try:
  with (out/'progress.jsonl').open('w') as log:
   while trainer.updates<limit and time.monotonic()-started<seconds and (deadline_at is None or time.time()<deadline_at) and not os.path.exists(STOP_REQUEST):
    row=trainer.step();row['process_seconds']=time.monotonic()-started;log.write(json.dumps(row)+'\n');log.flush()
    if worker_started_at is not None:
     worker_elapsed=time.time()-worker_started_at
     for threshold in milestone_seconds:
      if threshold not in milestones_done and worker_elapsed+milestone_offset_seconds>=threshold:
       folder=out/'milestones'/str(int(threshold));folder.mkdir(parents=True)
       np.savez_compressed(folder/'weights.npz',**{k:v.detach().cpu().numpy().astype(np.float32) for k,v in trainer.model.state_dict().items()})
       for n in (*EDITABLE,'config.json'):shutil.copyfile(source/n,folder/n)
       dump(folder/'result.json',{'target_cumulative_worker_seconds':threshold,'observed_worker_seconds':worker_elapsed,'prior_failed_worker_seconds':milestone_offset_seconds,'observed_cumulative_worker_seconds':worker_elapsed+milestone_offset_seconds,'updates':trainer.updates,'samples':trainer.samples,'natural_games':trainer.natural_games,'kind':'weights-only intermediate; not an optimizer/world resume checkpoint'})
       milestones_done.append(threshold)
  state=env.states if device=='cuda' else env.state()
  saved={'format':'clash-ar-training-v1','binding':binding,'config':c,'model':{k:v.detach().cpu() for k,v in trainer.model.state_dict().items()},'optimizer':trainer.optimizer.state_dict(),'engine':encode_engine_state(state,workers=1),'outputs':{k:torch.as_tensor(getattr(env,k)).cpu().clone() for k in ('observations','masks','infos','rewards')},'rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if device=='cuda' else [],**{k:getattr(trainer,k) for k in ('updates','samples','natural_games','losses','steps')}}
  torch.save(saved,out/'checkpoint.pt')
  np.savez_compressed(out/'weights.npz',**{k:v.numpy().astype(np.float32) for k,v in saved['model'].items()})
  for n in (*EDITABLE,'config.json'):shutil.copyfile(source/n,out/n)
  changed=sum(float((v-initial[k]).abs().sum()) for k,v in saved['model'].items())
  result={'updates':trainer.updates,'samples':trainer.samples,'natural_games':trainer.natural_games,'optimizer_steps':trainer.steps,'finite_losses':bool(trainer.losses) and all(np.isfinite(trainer.losses)),'loss_summary':{'last':trainer.losses[-1] if trainer.losses else None,'mean':float(np.mean(trainer.losses)) if trainer.losses else None},'weight_l1_change':changed,'seconds':time.monotonic()-started,'binding':binding,'milestone_seconds':milestones_done}
  dump(out/'result.json',result);return result
 finally:env.close()

def main():
 p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--out',required=True);p.add_argument('--seconds',type=float,required=True);p.add_argument('--resume');p.add_argument('--updates',type=int);p.add_argument('--deadline-at',type=float);p.add_argument('--worker-started-at',type=float);p.add_argument('--milestone-seconds',type=float,nargs='*',default=[]);p.add_argument('--milestone-offset-seconds',type=float,default=0);a=p.parse_args();print(json.dumps(run(a.source,a.out,a.seconds,a.resume,a.updates,a.deadline_at,a.worker_started_at,a.milestone_seconds,a.milestone_offset_seconds)))
if __name__=='__main__':main()
