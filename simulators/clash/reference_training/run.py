"""Finite worker entrypoint with strict state/lineage resume and durable exports."""
from pathlib import Path
import argparse,hashlib,json,shutil,time,copy
from simulators.clash.autoresearch.common import setup_native,engine_binding,dump,sha
HERE=Path(__file__).resolve().parent
SOURCES=('policy.py','learner.py','league.py','trainer.py','run.py')

def binding(c,env):
 from build_receipt import backend_build_binding
 import torch,numpy as np
 return {'engine':engine_binding(),'source':{n:sha(HERE/n) for n in SOURCES},'config':c,'build':backend_build_binding(env.library),'runtime':{'torch':str(torch.__version__),'numpy':np.__version__,'device':c['device']}}

def save(t,path,pin):
 import torch
 from snapshots import encode_engine_state
 state=t.env.states if t.device=='cuda' else t.env.state()
 value={'format':'clash-reference-v1','binding':pin,'model':{k:v.detach().cpu() for k,v in t.model.state_dict().items()},'optimizer':t.optimizer.state_dict(),'engine':encode_engine_state(state,workers=1),'outputs':{k:torch.as_tensor(getattr(t.env,k)).cpu().clone() for k in ('observations','masks','infos','rewards')},'rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if t.device=='cuda' else [],'league':[{k:v.cpu() for k,v in m.state_dict().items()} for m in t.league],'identities':t.identities}
 for k in ('updates','samples','natural_games','steps','losses','neural_opponent_decisions','script_opponent_decisions'):value[k]=getattr(t,k)
 for k in ('wins','games','assignment','seat'):value[k]=getattr(t,k).cpu()
 torch.save(value,path)

def restore(t,path,pin):
 import torch
 from snapshots import decode_engine_state
 from . import policy
 from .league import tensor_identity
 v=torch.load(path,map_location='cpu',weights_only=True)
 if v['format']!='clash-reference-v1' or v['binding']!=pin:raise ValueError('source/config/runtime binding differs')
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
 for k in ('updates','samples','natural_games','steps','losses','neural_opponent_decisions','script_opponent_decisions'):setattr(t,k,v[k])
 for k in ('wins','games','assignment','seat'):setattr(t,k,v[k].to(t.device))
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
 dump(path/'lineage.json',{'label':label,'binding':pin,'updates':t.updates,'samples':t.samples,'teacher_updates':min(t.updates,t.c['teacher_updates']),'ppo_only_updates':max(0,t.updates-t.c['teacher_updates']),'policy_identity':tensor_identity(t.model.state_dict()),'frozen_league_identities':t.identities})
 return str(path)

def exact_state(t):
 import torch
 return {'model':{k:v.detach().cpu().clone() for k,v in t.model.state_dict().items()},'optimizer':copy.deepcopy(t.optimizer.state_dict()),'world':t.env.states.cpu().clone() if t.device=='cuda' else t.env.state(),'rng':torch.get_rng_state().clone(),'cuda_rng':torch.cuda.get_rng_state_all() if t.device=='cuda' else [],'seat':t.seat.clone(),'assignment':t.assignment.clone(),'wins':t.wins.clone(),'games':t.games.clone(),'identities':list(t.identities)}

def equal(a,b):
 import torch
 if isinstance(a,torch.Tensor):return torch.equal(a,b)
 if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
 if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
 return a==b

def run(config,out,deadline,resume=None,proof=False):
 setup_native()
 import torch,numpy as np
 from .trainer import Trainer
 from . import policy
 from .league import tensor_identity
 c=json.loads(Path(config).read_text());out=Path(out);out.mkdir(parents=True,exist_ok=False)
 torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
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
  if resume:restore(t,resume,pin);initial=tensor_identity(t.model.state_dict())
  export(t,out/'initial','untrained-control' if not resume else 'resumed-parent',pin)
  audit={}
  with (out/'progress.jsonl').open('w') as log:
   while t.updates<c['updates'] and time.time()<deadline-30:
    row=t.step();row['unix']=time.time();log.write(json.dumps(row,allow_nan=False)+'\n');log.flush()
    if t.updates==c['teacher_updates']:export(t,out/'post-teacher','teacher-assisted-ppo-stage',pin)
    if proof and t.updates==max(c['teacher_updates']+1,2):
     save(t,out/'resume-probe.pt',pin)
     row1=t.step();expected=exact_state(t)
     restore(t,out/'resume-probe.pt',pin);row2=t.step();actual=exact_state(t)
     if not equal(expected,actual) or row1!=row2:raise ValueError('exact next-update resume mismatch')
     o,m,_,_=t.outputs();x=o.reshape(-1,7024);mask=m.reshape(-1,2307)
     with torch.no_grad():a=policy.act(t.model,x,mask);b=policy.act(t.model,x,mask)
     if not torch.equal(a,b) or not mask[torch.arange(len(a)),a].all():raise ValueError('deterministic greedy legality failed')
     audit={'exact_next_update_resume':True,'deterministic_legal_inference':True,'tested_rows':len(a),'league_identity_verified':True}
     dump(out/'proof.json',audit)
    if t.updates%16==0:
     save(t,out/'checkpoint.pt',pin)
  save(t,out/'checkpoint.pt',pin);export(t,out/'final','selected-final',pin)
  result={'format':'clash-reference-result-v1','updates':t.updates,'samples':t.samples,'natural_games':t.natural_games,'optimizer_steps':t.steps,'parameters':parameters,'initial_policy_identity':initial,'final_policy_identity':tensor_identity(t.model.state_dict()),'finite_losses':bool(t.losses) and all(np.isfinite(t.losses)),'teacher_updates':min(t.updates,c['teacher_updates']),'ppo_only_updates':max(0,t.updates-c['teacher_updates']),'neural_opponent_decisions':t.neural_opponent_decisions,'script_opponent_decisions':t.script_opponent_decisions,'league_identities':t.identities,'league_games':t.games.tolist(),'league_points':t.wins.tolist(),'proof':audit,'binding':pin}
  dump(out/'result.json',result);return result
 finally:env.close()

def main():
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--out',required=True);p.add_argument('--deadline',type=float,required=True);p.add_argument('--resume');p.add_argument('--proof',action='store_true');a=p.parse_args();print(json.dumps(run(a.config,a.out,a.deadline,a.resume,a.proof)))
if __name__=='__main__':main()
