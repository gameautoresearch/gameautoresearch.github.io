"""Finite worker entrypoint with strict state/lineage resume and durable exports."""
from pathlib import Path
import argparse,hashlib,json,shutil,time,copy
from simulators.clash.autoresearch.common import setup_native,engine_binding,dump,sha
HERE=Path(__file__).resolve().parent
SOURCES=('policy.py','learner.py','league.py','trainer.py','run.py')

def binding(c,env):
 from build_receipt import backend_build_binding
 import torch,numpy as np
 return {'engine':engine_binding(),'source':{n:sha(HERE/n) for n in SOURCES},'config':c,'build':backend_build_binding(env.library),'runtime':{'torch':str(torch.__version__),'numpy':np.__version__,'device':c['device'],'cuda':torch.version.cuda,'matmul_allow_tf32':torch.backends.cuda.matmul.allow_tf32,'cudnn_allow_tf32':torch.backends.cudnn.allow_tf32,'cudnn_deterministic':torch.backends.cudnn.deterministic}}

def save(t,path,pin):
 import torch
 from snapshots import encode_engine_state
 state=t.env.states if t.device=='cuda' else t.env.state()
 value={'format':'clash-reference-v11','binding':pin,'model':{k:v.detach().cpu() for k,v in t.model.state_dict().items()},'optimizer':t.optimizer.state_dict(),'engine':encode_engine_state(state,workers=1),'outputs':{k:torch.as_tensor(getattr(t.env,k)).cpu().clone() for k in ('observations','masks','infos','rewards')},'rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if t.device=='cuda' else [],'league':[{k:v.cpu() for k,v in m.state_dict().items()} for m in t.league],'identities':t.identities}
 for k in ('updates','samples','natural_games','steps','losses','neural_opponent_decisions','script_opponent_decisions','learned_neural_opponent_decisions','visible_refusal_nonzero_decisions','visible_run_refusal_nonzero_advancing_decisions','actual_learner_publicly_placed_cards','learner_nonwait_attempts','learner_advancing_decisions','demo_natural_games','demo_decisions','demo_nonwait'):value[k]=getattr(t,k)
 for k in ('wins','games','assignment','seat'):value[k]=getattr(t,k).cpu()
 demo=t.demo_env
 value['demo_rng']=t.demo_rng.get_state()
 value['demo_engine']=encode_engine_state(demo.states if t.device=='cuda' else demo.state(),workers=1) if demo is not None else None
 value['demo_outputs']={k:torch.as_tensor(getattr(demo,k)).cpu().clone() for k in ('observations','masks','infos','rewards')} if demo is not None else {}
 torch.save(value,path)

def restore(t,path,pin):
 import torch
 from snapshots import decode_engine_state
 from . import policy
 from .league import tensor_identity
 v=torch.load(path,map_location='cpu',weights_only=True)
 if v['format']!='clash-reference-v11' or v['binding']!=pin:raise ValueError('source/config/runtime binding differs')
 if [tensor_identity(s) for s in v['league']]!=v['identities']:raise ValueError('frozen league lineage differs')
 t.model.load_state_dict(v['model']);t.optimizer.load_state_dict(v['optimizer']);t.league=[];t.identities=v['identities']
 for state in v['league']:
  model=policy.create(t.c).to(t.device);model.load_state_dict(state);model.eval()
  for p in model.parameters():p.requires_grad_(False)
  t.league.append(model)
 state=decode_engine_state(v['engine'],as_tensor=t.device=='cuda',expected_count=t.c['envs'],expected_world_bytes=4195384,max_total_bytes=16*1024**3,workers=1);t.env.load(state)
 for k,x in v['outputs'].items():
  target=getattr(t.env,k)
  if isinstance(target,torch.Tensor):target.copy_(x.to(t.device))
  else:target[:]=x.numpy()
 for k in ('updates','samples','natural_games','steps','losses','neural_opponent_decisions','script_opponent_decisions','learned_neural_opponent_decisions','visible_refusal_nonzero_decisions','visible_run_refusal_nonzero_advancing_decisions','actual_learner_publicly_placed_cards','learner_nonwait_attempts','learner_advancing_decisions','demo_natural_games','demo_decisions','demo_nonwait'):setattr(t,k,v[k])
 for k in ('wins','games','assignment','seat'):setattr(t,k,v[k].to(t.device))
 if t.demo_env is not None:
  demo=decode_engine_state(v['demo_engine'],as_tensor=t.device=='cuda',expected_count=t.c['envs'],expected_world_bytes=4195384,max_total_bytes=16*1024**3,workers=1);t.demo_env.load(demo)
  for k,x in v['demo_outputs'].items():
   target=getattr(t.demo_env,k)
   if isinstance(target,torch.Tensor):target.copy_(x.to(t.device))
   else:target[:]=x.numpy()
 elif v['demo_engine'] is not None:raise ValueError('unexpected control demonstration state')
 t.demo_rng.set_state(v['demo_rng'])
 torch.set_rng_state(v['rng'])
 if t.device=='cuda':torch.cuda.set_rng_state_all(v['cuda_rng'])
 return v

def export(t,path,label,pin):
 import numpy as np
 from .league import tensor_identity
 path=Path(path);path.mkdir(parents=True,exist_ok=False)
 for name in ('policy.py','learner.py'):shutil.copyfile(HERE/name,path/name)
 # Export-only train.py is never used for training; inference imports only policy.py.
 (path/'train.py').write_text('"""Reference inference bundle; training recipe in reference_training/."""\n')
 dump(path/'config.json',t.c)
 np.savez_compressed(path/'weights.npz',**{k:v.detach().cpu().numpy().astype(np.float32) for k,v in t.model.state_dict().items()})
 dump(path/'lineage.json',{'label':label,'binding':pin,'updates':t.updates,'samples':t.samples,'teacher_updates':(min(t.updates,t.c['teacher_updates']) if t.c['teacher_coef'] else 0),'ppo_only_updates':(max(0,t.updates-t.c['teacher_updates']) if t.c['teacher_coef'] else t.updates),'policy_identity':tensor_identity(t.model.state_dict()),'frozen_league_identities':t.identities})
 return str(path)

def exact_state(t):
 import torch
 return {'model':{k:v.detach().cpu().clone() for k,v in t.model.state_dict().items()},'optimizer':copy.deepcopy(t.optimizer.state_dict()),'world':t.env.states.cpu().clone() if t.device=='cuda' else t.env.state(),'rng':torch.get_rng_state().clone(),'cuda_rng':torch.cuda.get_rng_state_all() if t.device=='cuda' else [],'seat':t.seat.clone(),'assignment':t.assignment.clone(),'wins':t.wins.clone(),'games':t.games.clone(),'identities':list(t.identities),'demo_rng':t.demo_rng.get_state().clone(),'demo_world':(t.demo_env.states.cpu().clone() if t.device=='cuda' else t.demo_env.state()) if t.demo_env is not None else None}

def equal(a,b):
 import torch
 if isinstance(a,torch.Tensor):return torch.equal(a,b)
 if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
 if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
 return a==b

def run(config,out,deadline,resume=None,proof=False,warmstart=None,warmstart_sha=None):
 setup_native()
 import torch,numpy as np
 from .trainer import Trainer
 from . import policy
 from .league import tensor_identity
 c=json.loads(Path(config).read_text());out=Path(out);out.mkdir(parents=True,exist_ok=False)
 torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 if c['max_ticks']!=0:raise ValueError('natural fixed8 horizon required')
 if c['device']=='cuda':
  if torch.cuda.device_count()!=1 or torch.cuda.get_device_name()!='NVIDIA H100 80GB HBM3':raise ValueError('exact one H100 required')
  from runtime import NativeCUDA
  env=NativeCUDA(list(range(c['seed'],c['seed']+c['envs'])),launch_threads=1,render_mode=4)
 else:
  from cpu_batch import NativeCPUBatch
  env=NativeCPUBatch(list(range(c['seed'],c['seed']+c['envs'])),threads=min(4,c['envs']))
 try:
  t=Trainer(c,env,c['device']);pin=binding(c,env);initial=tensor_identity(t.model.state_dict())
  parameters=sum(p.numel() for p in t.model.parameters())
  if parameters>1000000:raise ValueError('entrant-compatible parameter cap exceeded')
  migration=None
  if resume and warmstart:raise ValueError('Choose exact resume or explicit weights-only warm start')
  if warmstart:
   if sha(warmstart)!=warmstart_sha:raise ValueError('Warm-start parent checkpoint bytes differ')
   parent=torch.load(warmstart,map_location='cpu',weights_only=True)
   formats={'clash-reference-v4':'v4','clash-reference-v5':'v5','clash-reference-v7':'v7','clash-reference-v9':'v9','clash-reference-v10':'v10','clash-reference-v11':'v11'}
   if parent['format'] not in formats:raise ValueError('Unqualified source format for conditioned input transfer')
   old=HERE.parent/formats[parent['format']]
   if parent['binding']['engine']!=pin['engine'] or parent['binding']['build']!=pin['build'] or parent['binding']['runtime']!=pin['runtime']:raise ValueError('Parent engine/build/runtime differs')
   if any(sha(old/n)!=v for n,v in parent['binding']['source'].items()):raise ValueError('Parent source differs from immutable declared source')
   parent_identity=tensor_identity(parent['model']);converted={k:v.clone() for k,v in parent['model'].items()}
   changed=parent['format'] not in ('clash-reference-v10','clash-reference-v11')
   if changed:
    if parent['binding']['source']['policy.py']!=sha(HERE.parent/'v4/policy.py'):raise ValueError('Only the qualified original spatial architecture can be reparameterized')
    converted['body.0.weight'][:,:224]/=t.model.header_gain.cpu()[None,:]
    converted['query.0.weight'][:,-1]/=16384
    converted['header_gain']=t.model.header_gain.cpu().clone()
   elif not torch.equal(converted['header_gain'],t.model.header_gain.cpu()):raise ValueError('Immutable conditioning buffer changed')
   t.model.load_state_dict(converted);initial=tensor_identity(t.model.state_dict())
   # Compensating binary rescalings preserve products exactly. Test actual
   # public combat observations, both seats, before accepting any imported state.
   rng_before=torch.get_rng_state();gpu_before=torch.cuda.get_rng_state_all() if c['device']=='cuda' else []
   from simulators.clash.autoresearch.common import load_module
   old_policy=load_module(old/'policy.py','conditioned_parent_policy');baseline=old_policy.create(parent['binding']['config']).to(c['device']);baseline.load_state_dict(parent['model']);baseline.eval()
   observations=[torch.as_tensor(env.observations).reshape(-1,7024),parent['outputs']['observations'].reshape(-1,7024)]
   masks=[torch.as_tensor(env.masks).reshape(-1,2307),parent['outputs']['masks'].reshape(-1,2307)]
   if parent.get('demo_outputs'):
    observations.append(parent['demo_outputs']['observations'].reshape(-1,7024));masks.append(parent['demo_outputs']['masks'].reshape(-1,2307))
   x=torch.cat([v.to(c['device']) for v in observations]);mask=torch.cat([v.to(c['device']) for v in masks])
   with torch.no_grad():
    left=baseline(x,mask);right=t.model(x,mask)
    if not torch.equal(left[0],right[0]) or not torch.equal(left[1],right[1]) or not torch.equal(old_policy.act(baseline,x,mask),policy.act(t.model,x,mask)):raise ValueError('Conditioned input transfer failed exact output/action parity')
   parity_rows=len(x);del baseline,x,mask,left,right;torch.set_rng_state(rng_before)
   if gpu_before:torch.cuda.set_rng_state_all(gpu_before)
   rng=torch.get_rng_state();gpu=torch.cuda.get_rng_state_all() if c['device']=='cuda' else []
   t.league=[];t.identities=[];t.add_history();torch.set_rng_state(rng)
   if gpu:torch.cuda.set_rng_state_all(gpu)
   migration={'kind':'explicit-model-only-warmstart','parent_checkpoint_sha256':warmstart_sha,'parent_format':parent['format'],'parent_binding':parent['binding'],'parent_updates':parent['updates'],'parent_policy_identity':parent_identity,'new_policy_identity_at_import':initial,'architecture_decoder_source_identical':not changed,'function_preserving_binary_input_reparameterization':changed,'exact_logit_value_and_action_parity_rows':parity_rows,'transformed_parameters':['body.0.weight first224columns','query.0.weight finalcolumn'] if changed else [],'constant_buffer_added':'header_gain' if changed else None,'optimizer_reset':True,'learner_worlds_reset':True,'demonstration_worlds_reset':True,'new_rng_seed':c['seed'],'initial_neural_pool_member':'trained parent incumbent'}
   dump(out/'warmstart.json',migration);del parent
  if resume:restore(t,resume,pin);initial=tensor_identity(t.model.state_dict())
  export(t,out/'initial','pretrained-incumbent-warmstart' if warmstart else ('untrained-control' if not resume else 'resumed-parent'),pin)
  audit={}
  with (out/'progress.jsonl').open('w') as log:
   while t.updates<c['updates'] and time.time()<deadline-30:
    row=t.step();row['unix']=time.time();log.write(json.dumps(row,allow_nan=False)+'\n');log.flush()
    if t.updates==c['teacher_updates']:export(t,out/'post-teacher','teacher-assisted-ppo-stage',pin)
    if proof and t.updates==2:
     save(t,out/'resume-probe.pt',pin)
     row1=t.step();expected=exact_state(t)
     restore(t,out/'resume-probe.pt',pin);row2=t.step();actual=exact_state(t)
     if not equal(expected,actual) or row1!=row2:raise ValueError('exact next-update resume mismatch')
     o,m,_,_=t.outputs();x=o.reshape(-1,7024);mask=m.reshape(-1,2307)
     with torch.no_grad():a=policy.act(t.model,x,mask);b=policy.act(t.model,x,mask)
     if not torch.equal(a,b) or not mask[torch.arange(len(a)),a].all():raise ValueError('deterministic greedy legality failed')
     audit={'exact_next_update_resume':True,'deterministic_legal_inference':True,'tested_rows':len(a),'league_identity_verified':True}
     dump(out/'proof.json',audit)
    if t.updates%128==0:export(t,out/f'milestone-{t.updates:06d}','public-development-milestone',pin)
    if t.updates%64==0:
     save(t,out/'checkpoint.pt',pin)
  save(t,out/'checkpoint.pt',pin);export(t,out/'final','selected-final',pin)
  result={'format':'clash-reference-result-v11','warmstart':migration,'demonstrator':'standard-fixed-tie','updates':t.updates,'samples':t.samples,'natural_games':t.natural_games,'optimizer_steps':t.steps,'parameters':parameters,'static_opponent_names':['standard-fixed-tie','pressure-fixed-tie','intermediate-public-v2','teacher-public-v2'],'initial_policy_identity':initial,'final_policy_identity':tensor_identity(t.model.state_dict()),'finite_losses':bool(t.losses) and all(np.isfinite(t.losses)),'teacher_updates':(min(t.updates,c['teacher_updates']) if c['teacher_coef'] else 0),'ppo_only_updates':(max(0,t.updates-c['teacher_updates']) if c['teacher_coef'] else t.updates),'neural_opponent_decisions':t.neural_opponent_decisions,'script_opponent_decisions':t.script_opponent_decisions,'learned_neural_opponent_decisions':t.learned_neural_opponent_decisions,'visible_refusal_nonzero_decisions':t.visible_refusal_nonzero_decisions,'visible_run_refusal_nonzero_advancing_decisions':t.visible_run_refusal_nonzero_advancing_decisions,'actual_learner_publicly_placed_cards':t.actual_learner_publicly_placed_cards,'learner_nonwait_attempts':t.learner_nonwait_attempts,'learner_advancing_decisions':t.learner_advancing_decisions,'demo_natural_games':t.demo_natural_games,'demo_decisions':t.demo_decisions,'demo_nonwait':t.demo_nonwait,'league_identities':t.identities,'league_games':t.games.tolist(),'league_points':t.wins.tolist(),'proof':audit,'binding':pin}
  dump(out/'result.json',result);return result
 finally:
  env.close()
  if 't' in locals() and t.demo_env is not None:t.demo_env.close()

def main():
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--out',required=True);p.add_argument('--deadline',type=float,required=True);p.add_argument('--resume');p.add_argument('--proof',action='store_true');a=p.parse_args();print(json.dumps(run(a.config,a.out,a.deadline,a.resume,a.proof)))
if __name__=='__main__':main()
