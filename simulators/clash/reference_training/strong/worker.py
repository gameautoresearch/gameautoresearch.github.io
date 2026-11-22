"""Finite exact-source continuation, with a fresh replay audit and milestone exports."""
from pathlib import Path
import argparse,json,os,time,subprocess
from simulators.clash.autoresearch.common import dump,sha,setup_native

def run(parent,expected_sha,out,seconds,absolute_end,variant='v4',teacher_coef=None,seed=None,mode='auto'):
 if time.time()>=absolute_end:raise ValueError('Absolute training selection cutoff reached before initialization')
 setup_native()
 import importlib,torch,numpy as np
 r=importlib.import_module('simulators.clash.reference_training.'+variant+'.run')
 Trainer=importlib.import_module('simulators.clash.reference_training.'+variant+'.trainer').Trainer
 policy=importlib.import_module('simulators.clash.reference_training.'+variant+'.policy')
 if sha(parent)!=expected_sha:raise ValueError('Parent full checkpoint bytes changed')
 saved=torch.load(parent,map_location='cpu',weights_only=True)
 if mode=='warmstart' or (mode=='auto' and variant in ('v5','v6','v7','v8','v9','v10','v11','v12','v13') and saved['format']!='clash-reference-'+variant):
  c=json.loads((Path(r.__file__).parent/'recipe.json').read_text())
  if teacher_coef is not None:c['teacher_coef']=teacher_coef
  if seed is not None:c['seed']=seed
  target=Path(out);target.mkdir(parents=True,exist_ok=True);config=target/'recipe.json';dump(config,c)
  del saved
  result=r.run(config,target/'training',min(time.time()+seconds,absolute_end),proof=True,warmstart=parent,warmstart_sha=expected_sha)
  subprocess.run(['sync',str(target)],check=True,timeout=20);return result
 from simulators.clash.reference_training.league import tensor_identity
 if sha(parent)!=expected_sha:raise ValueError('Parent full checkpoint bytes changed')
 c=saved['binding']['config'];out=Path(out);out.mkdir(parents=True,exist_ok=True)
 if seed is not None and seed!=c['seed']:raise ValueError('Changing seed requires explicit qualified warmstart, not an exact-resume label')
 if teacher_coef is not None and teacher_coef!=c['teacher_coef']:raise ValueError('Changing assistance requires an explicit warmstart/migration')
 torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 if c['max_ticks'] or c['device']!='cuda' or torch.cuda.device_count()!=1 or torch.cuda.get_device_name()!='NVIDIA H100 80GB HBM3':raise ValueError('Pinned natural-horizon single H100 task required')
 from runtime import NativeCUDA
 env=NativeCUDA(list(range(c['seed'],c['seed']+c['envs'])),launch_threads=1,render_mode=4);t=None;started=time.time();deadline=min(started+seconds,absolute_end)
 try:
  t=Trainer(c,env,'cuda');pin=r.binding(c,env)
  r.restore(t,parent,pin) # Exact old source/config/compiler/runtime; no migration waiver.
  parent_updates=t.updates;parent_samples=t.samples;parent_games=t.natural_games;initial=tensor_identity(t.model.state_dict())
  r.export(t,out/'initial','exact-resumed-parent',pin)
  r.save(t,out/'resume-probe.pt',pin);first=t.step();expected=r.exact_state(t)
  r.restore(t,out/'resume-probe.pt',pin);second=t.step();actual=r.exact_state(t)
  if not r.equal(expected,actual) or first!=second:raise ValueError('Actual GPU exact continuation replay mismatch')
  with torch.no_grad():
   o,m,_,_=t.outputs();x=o.reshape(-1,7024);mask=m.reshape(-1,2307);a=policy.act(t.model,x,mask);b=policy.act(t.model,x,mask)
   if not torch.equal(a,b) or not mask[torch.arange(len(a)),a].all():raise ValueError('Greedy inference is nondeterministic or illegal')
  proof={'exact_next_update_resume':True,'deterministic_legal_inference':True,'tested_rows':len(a),'parent_source_config_build_runtime_exact':True,'parent_checkpoint_sha256':expected_sha,'parent_updates':parent_updates,'parent_policy_identity':initial,'extra_discarded_audit_updates':1,'extra_discarded_audit_learner_rows':c['envs']*c['rollout_steps']}
  dump(out/'proof.json',proof);del expected,actual,saved
  with (out/'progress.jsonl').open('w') as log:
   log.write(json.dumps(first)+'\n');log.flush()
   while time.time()<deadline-35:
    row=t.step();row['unix']=time.time();log.write(json.dumps(row,allow_nan=False)+'\n');log.flush()
    if (t.updates-parent_updates)%64==0:r.save(t,out/'checkpoint.pt',pin)
    if (t.updates-parent_updates)%256==0:
     r.export(t,out/f'milestone-{t.updates:06d}','public-development-milestone',pin);r.save(t,out/f'milestone-{t.updates:06d}'/'checkpoint.pt',pin);dump(out/f'milestone-{t.updates:06d}'/'checkpoint-receipt.json',{'checkpoint_sha256':sha(out/f'milestone-{t.updates:06d}'/'checkpoint.pt'),'updates':t.updates});subprocess.run(['sync',str(out)],check=True,timeout=20)
  r.save(t,out/'checkpoint.pt',pin);r.export(t,out/'final','continuation-final',pin)
  value={'format':'clash-strong-exact-continuation-v1','variant':variant,'binding':pin,'proof':proof,'parent_updates':parent_updates,'updates':t.updates,'new_updates':t.updates-parent_updates,'parent_samples':parent_samples,'samples':t.samples,'new_samples':t.samples-parent_samples,'new_natural_games':t.natural_games-parent_games,'initial_policy_identity':initial,'final_policy_identity':tensor_identity(t.model.state_dict()),'finite_actual_learning':all(np.isfinite(t.losses)) and t.updates>parent_updates,'parameters':sum(p.numel() for p in t.model.parameters()),'deadline_unix':deadline,'wrapper_source_sha256':sha(__file__),'no_old_costs_reassigned':True,'external_model_calls':0}
  dump(out/'result.json',value);subprocess.run(['sync',str(out)],check=True,timeout=20);return value
 finally:
  env.close()
  if t is not None and t.demo_env is not None:t.demo_env.close()

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--parent',required=True);p.add_argument('--parent-sha',required=True);p.add_argument('--out',required=True);p.add_argument('--seconds',type=float,required=True);p.add_argument('--absolute-end',type=float,required=True);p.add_argument('--variant',default='v4');p.add_argument('--teacher-coef',type=float);p.add_argument('--seed',type=int);p.add_argument('--mode',choices=['auto','exact','warmstart'],default='auto');a=p.parse_args();print(json.dumps(run(a.parent,a.parent_sha,a.out,a.seconds,a.absolute_end,a.variant,a.teacher_coef,a.seed,a.mode)))
