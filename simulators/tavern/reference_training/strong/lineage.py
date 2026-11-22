"""Explicit old full-checkpoint import with unchanged inference/optimizer tensors."""
import importlib.util,json,copy,sys
from pathlib import Path
import numpy as np
import torch
from .common import sha,admission,write
from .policy import Policy
from .population import Population
from ..population import weight_hash

def module(path,name):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
 previous=sys.dont_write_bytecode;sys.dont_write_bytecode=True
 try:spec.loader.exec_module(m)
 finally:sys.dont_write_bytecode=previous
 return m

def import_parent(inputs,device):
 inputs=Path(inputs);bundle=inputs/'incumbent';ident=json.loads((bundle/'identity.json').read_text());summary=json.loads((bundle/'summary.json').read_text())
 if ident['binding']!=admission():raise ValueError('Old checkpoint admission binding differs')
 if sha(bundle/'identity.json')!=summary['identity_sha256'] or sha(bundle/'weights.npz')!=summary['weights_sha256']:raise ValueError('Old inference identity/weights mutation')
 if any(sha(bundle/f)!=h for f,h in ident['source'].items()):raise ValueError('Old training source mutation')
 ck=torch.load(inputs/'parent-resume.pt',map_location='cpu',weights_only=False)
 if ck.get('schema')=='tavern-strong-population-v1':
  return import_strong_parent(inputs,device,ident,summary,ck)
 if ck['identity']!=ident:raise ValueError('Old full resume does not match admitted incumbent')
 with np.load(bundle/'weights.npz',allow_pickle=False) as z:
  if set(z.files)!=set(ck['policy']) or any(not np.array_equal(z[k],v.numpy()) for k,v in ck['policy'].items()):raise ValueError('NPZ is not full checkpoint policy')
 model=Policy(**ident['config']['network']).to(device);model.load_state_dict(ck['policy'],strict=True)
 pop=Population(model.arch);pop.add_state(ck['policy'],0,'retained-deepseek-r3-incumbent',dict(identity_sha256=sha(bundle/'identity.json'),weights_sha256=sha(bundle/'weights.npz')))
 for name in ('r2','r1'):
  op=inputs/name;oi=json.loads((op/'identity.json').read_text());os=json.loads((op/'summary.json').read_text())
  if oi['binding']!=ident['binding'] or oi['source']['policy.py']!=ident['source']['policy.py'] or sha(op/'weights.npz')!=os['weights_sha256'] or sha(op/'identity.json')!=os['identity_sha256']:raise ValueError('Historical source/interface/weights mismatch')
  with np.load(op/'weights.npz',allow_pickle=False) as z:state={k:torch.from_numpy(z[k].copy()) for k in z.files}
  pop.add_state(state,-1,name,dict(identity_sha256=sha(op/'identity.json'),weights_sha256=sha(op/'weights.npz')))
 receipt=dict(input_manifest_sha256=sha(inputs/'manifest.json'),input_files=json.loads((inputs/'manifest.json').read_text()),kind='explicit-new-campaign-warmstart',parent_resume_sha256=sha(inputs/'parent-resume.pt'),parent_identity_sha256=sha(bundle/'identity.json'),parent_policy_sha256=weight_hash(ck['policy']),parent_update=ck['update'],parent_samples=sum(r['learning']['samples'] for r in ck['history']),copied=['policy','optimizer','Python/NumPy/Torch/CUDA RNG'],reset=['new campaign cohort seed/domain','new logical update starts0','new frozen source/config'],old_environment_states_retained=len(ck['env_states']),not_an_exact_old_source_resume=True)
 return model,ck,pop,receipt

def import_strong_parent(inputs,device,ident,summary,ck):
 """Explicit source migration; the archived checkpoint itself is never edited."""
 bundle=inputs/'incumbent';request=json.loads((inputs/'fork-request.json').read_text())
 if request.get('kind')!='explicit-strong-parent-import-v1':raise ValueError('Explicit own-checkpoint import receipt required')
 if request.get('parent_checkpoint_sha256')!=sha(inputs/'parent-resume.pt') or request.get('parent_identity_sha256')!=sha(bundle/'identity.json'):raise ValueError('Explicit fork selection identity differs')
 manifest=json.loads((inputs/'manifest.json').read_text())
 changed=[f for f,h in manifest.items() if sha(inputs/f)!=h]
 if changed:raise ValueError('Frozen fork inputs mutated: '+str(changed[:8]))
 ref=ident.get('reference_lineage',{})
 if (ref.get('checkpoint_sha256')!=sha(inputs/'parent-resume.pt') or ref.get('reference_binding')!=ck['binding'] or ref.get('update')!=ck['update'] or ref.get('parent')!=ck['lineage'] or ident['config']!=ck['config']):raise ValueError('Strong parent checkpoint/export/source lineage mismatch')
 if ck['binding']['admission']!=admission():raise ValueError('Strong parent native/rules admission differs')
 for f,h in ck['binding']['files'].items():
  if sha(inputs/'parent-source'/f)!=h:raise ValueError('Archived strong parent source mutated: '+f)
 for f,h in ident['source'].items():
  if sha(inputs/'parent-source/simulators/tavern/reference_training/strong'/f)!=h:raise ValueError('Parent inference source is not the archived training source')
 with np.load(bundle/'weights.npz',allow_pickle=False) as z:
  if set(z.files)!=set(ck['policy']) or any(not np.array_equal(z[k],v.numpy()) for k,v in ck['policy'].items()):raise ValueError('Strong parent NPZ is not the complete checkpoint policy')
 model=Policy(**ident['config']['network']).to(device);model.load_state_dict(ck['policy'],strict=True)
 pop=Population.restore(ck['population'])
 if pop.arch!=model.arch:raise ValueError('Parent population architecture differs')
 pop.add_state(ck['policy'],0,'explicit-imported-strong-parent',dict(identity_sha256=sha(bundle/'identity.json'),original_update=ck['update']))
 parent_samples=sum(r['learning']['samples'] for r in ck['history'])
 prior=ck['lineage'].get('parent_total_known_samples',ck['lineage']['parent_samples'])
 receipt=dict(input_manifest_sha256=sha(inputs/'manifest.json'),input_files=manifest,kind='explicit-strong-parent-warmstart',parent_resume_sha256=sha(inputs/'parent-resume.pt'),parent_identity_sha256=sha(bundle/'identity.json'),parent_policy_sha256=weight_hash(ck['policy']),parent_update=ck['update'],parent_samples=parent_samples,parent_total_known_samples=prior+parent_samples,parent_source_binding=ck['binding'],parent_lineage=ck['lineage'],copied=['policy','optimizer','Python/NumPy/Torch/CUDA RNG','parent historical population; current parent added as a labeled imported snapshot'],reset=['new cohort seed/domain','new logical update starts0','new frozen source/config','new native cohorts'],old_environment_states_retained=len(ck['env_states']),not_an_exact_old_source_resume=True)
 # Normalize the importer return interface only. No checkpoint is relabeled,
 # rewritten, or admitted by the ordinary exact-resume path.
 parent=dict(ck,identity=ident,torch_rng=ck['rng']['torch'],python_rng=ck['rng']['python'],numpy_rng=ck['rng']['numpy'],cuda_rng=ck['rng']['cuda'])
 return model,parent,pop,receipt

def parity(inputs,out,device="cpu"):
 import taverngym as tg
 model,ck,pop,receipt=import_parent(inputs,torch.device(device));old=module(Path(inputs)/'incumbent/policy.py','old_incumbent_parity').Policy(**model.arch).to(device);old.load_state_dict(ck['policy']);old.eval();model.eval()
 env=tg.BatchEnv(4,seed=531301,controlled=range(8),config=ck["identity"]["config"]["rules"],threads=1);rows=0
 with torch.no_grad():
  for _ in range(12):
   obs=env.observe();ids=np.flatnonzero(obs['ready']);f=torch.from_numpy(obs['features'][ids].copy()).to(device);c=torch.from_numpy(obs['cards'][ids].copy()).long().to(device);m=torch.from_numpy(obs['action_mask'][ids].copy()).to(device)
   a,v=old(f,c);b,w=model(f,c)
   if not torch.equal(a,b) or not torch.equal(v,w):raise ValueError('Old/new forward parity failed')
   da,_=old.distribution(f,c,m);db,_=model.distribution(f,c,m)
   if not torch.equal(da.logits,db.logits) or not torch.equal(da.probs,db.probs):raise ValueError('Masked distribution parity failed')
   acts=np.zeros(32,np.int64);acts[ids]=a.masked_fill(~m.bool(),-1e9).argmax(-1).cpu().numpy();env.step(acts);rows+=len(ids)
 receipt.update(parity_device=str(device),exact_forward_and_value=True,exact_masked_distribution=True,public_visible_rows=rows,parameters=sum(p.numel() for p in model.parameters()))
 write(Path(out),receipt);return receipt
