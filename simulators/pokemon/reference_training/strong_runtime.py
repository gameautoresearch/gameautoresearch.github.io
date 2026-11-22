"""Admitted warm-start + visible expert BC + sparse current/history/search PPO.

All actual transition/PPO mathematics are the canonical supported CPU Showdown stack.
Engine hashes are verified at stage admission/export, then reused from immutable source
while checkpointing. No engine source or checkpoint signature is bypassed.
"""
from __future__ import annotations
import contextlib,copy,hashlib,json,os,random,subprocess,tarfile,time,concurrent.futures
from pathlib import Path
import numpy as np
import torch
from .. import cli
from ..network import Policy,batch
from ..rl import Population,Timings,rollout,update,inference_for
from ..provenance import Ledger,canonical,digest
from ..autoresearch.evaluation_v2.env import EvaluationWorker
from ..env import Vector,Worker
from ..autoresearch.evaluation_v2.foul_process import FoulPlayProcess
from ..autoresearch.evaluation_v2.qualification import prepare_search
from . import trainer
from .strong_campaign import write,sha

KEYS=('tokens','mons','global','global_tokens','action_tokens','actions','mask')

def extra_bindings():
 root=Path(__file__).resolve().parents[1]
 files={str(p.relative_to(root)):sha(p) for p in (root/'autoresearch').rglob('*') if p.is_file() and p.suffix in ('.py','.cjs') and '__pycache__' not in p.parts}
 tools=root.parents[1]/'tools'
 for name in ('open_credit_campaign.py','overnight_budget.py'):
  p=tools/name;files['../../tools/'+name]=sha(p)
 return files

def validate_bindings(cfg):
 if not cfg.get('extra_source_sha256') or extra_bindings()!=cfg['extra_source_sha256']:raise ValueError('Extended evaluator/search/guard source binding differs')
 pins=cfg.get('search_pins')
 if not pins:raise ValueError('Pinned public search dependencies required')
 fixed=cfg.get('frozen_neural_opponent')
 if any(x['kind']=='retained' for x in cfg['opponents']) and not fixed:raise ValueError('Retained opposition requires an immutable admitted neural dependency')
 if fixed:
  from ..autoresearch.bundle import verify
  from ..autoresearch.contract import runtime_signature
  root=Path(fixed['path']);verify(root,fixed['bundle_sha256'])
  if fixed['kind']!='plain-starter-bundle' or sha(root/'policy.py')!='638d949c10a5f57d235a3080f12ce3a304c88849495ecfbeaed485568cb42003' or sha(root/'weights.pt')!=fixed['weights_sha256']:raise ValueError('Frozen neural opponent source/tensor identity differs')
  if json.loads((root/'binding.json').read_text())['runtime_sha256']!=runtime_signature():raise ValueError('Frozen neural opponent original runtime binding differs')
 return True


class ReferenceWorker(Worker):
 def __init__(self):
  self.metrics={};self.decode_seconds=0.;self.pipe_wait_seconds=0.;self.binary=False;self.timeout=60;self.lazy=False;self.pending=bytearray();self.shared=None;self.shared_file=None;self.shared_dir=None
  self.process=subprocess.Popen(['node',str(Path(__file__).with_name('strong_vector_worker.cjs'))],stdin=subprocess.PIPE,stdout=subprocess.PIPE,bufsize=0)

class ReferenceVector(Vector):
 def __init__(self,workers,per_worker):
  self.workers=[ReferenceWorker() for _ in range(workers)];self.per_worker=per_worker;self.batched=True;self.optimized=True;self.packed=False;self.token_cache=False;self.mon_cache=False;self.fast_callbacks=False;self.states=None;self.seconds=0.;self.calls=0
 def visible(self,seats,offsets):
  n=self.per_worker;return self.call([{'op':'visible','seats':seats[i:i+n],'offsets':offsets[i:i+n]} for i in range(0,self.size,n)])

class SearchActor:
 def __init__(self,vector,index,seat,source,sets,request):
  self.vector=vector;self.index=index;self.seat=seat;self.offset=0;self.seconds=0.;self.decisions=0;self.request_path=Path(request);self.public_input_history=[]
  self.bot=FoulPlayProcess(source,sets,mode='fixed-work',iterations=1000,time_ms=10,samples=2,timeout=1.,deny_file=request)
 def choose_visible(self,visible):
  self.offset=visible['protocol_offset'];before=time.perf_counter();entry={'visible_own_player_input':copy.deepcopy(visible)};self.public_input_history.append(entry)
  try:a,metadata=self.bot.choose(visible)
  except BaseException as exc:
   write(self.request_path.parent/'search-faults'/f'{self.index}-{self.seat}-{self.decisions}-{time.time_ns()}.json',{'error_type':type(exc).__name__,'error':str(exc),'game_index':self.index,'seat':self.seat,'completed_search_decisions':self.decisions,'visible_own_player_input':visible,'own_public_input_and_action_history':self.public_input_history,'no_fallback':True});raise
  entry.update(action=a,metadata=metadata);self.seconds+=time.perf_counter()-before;self.decisions+=1;return a
 def choose(self):
  seats=[None]*self.vector.size;offsets=[0]*self.vector.size;seats[self.index]=self.seat;offsets[self.index]=self.offset
  visible=self.vector.visible(seats,offsets)[self.index];return self.choose_visible(visible)
 def close(self):self.bot.close()

class SearchInference:
 def __init__(self,base,vector=None,parallel=False,greedy_policy=None):self.base=base;self.device=base.device;self.vector=vector;self.parallel=parallel;self.greedy_policy=greedy_policy;self.actors=[];self.cached_states=None;self.cached_actions={}
 def begin_rollout(self):
  self.cached_states=None;self.cached_actions={}
  if hasattr(self.base,'begin_rollout'):self.base.begin_rollout()
 def __call__(self,model,observations,deterministic=False):
  if isinstance(model,SearchActor):
   if len(observations)!=1:raise ValueError('Search actor must own exactly one game/seat')
   if self.parallel:
    if self.cached_states is not self.vector.states:
     ready=[actor for actor in self.actors if self.vector.states[actor.index]['ready'][actor.seat]];seats=[None]*self.vector.size;offsets=[0]*self.vector.size
     for actor in ready:seats[actor.index]=actor.seat;offsets[actor.index]=actor.offset
     visible=self.vector.visible(seats,offsets)
     with concurrent.futures.ThreadPoolExecutor(max_workers=min(3,len(ready))) as pool:
      futures={actor:pool.submit(actor.choose_visible,visible[actor.index]) for actor in ready};self.cached_actions={actor:future.result() for actor,future in futures.items()}
     self.cached_states=self.vector.states
    action=self.cached_actions[model]
   else:action=model.choose()
   return np.asarray([action]),np.zeros(1),np.zeros(1)
  return self.base(model,observations,deterministic or model is self.greedy_policy)

def load_data(directory):
 index=json.loads((Path(directory)/'index.json').read_text());parts={'train':[],'validation':[]};manifest=[]
 for r in index:
  if not r.get('natural') or r.get('fault') or r.get('truncated') or not r.get('samples'):continue
  p=Path(directory)/r['shard']
  if sha(p)!=r['sha256']:raise ValueError('Teacher shard changed')
  part={k:v for k,v in np.load(p,allow_pickle=False).items()};actions=part['action'].astype(np.int64)
  if not part['mask'][np.arange(len(actions)),actions].all():raise ValueError('Masked teacher training label')
  parts['validation' if r['index']%5==0 else 'train'].append(part);manifest.append({'shard':r['shard'],'sha256':r['sha256'],'natural':True,'samples':r['samples'],'split':'validation' if r['index']%5==0 else 'train'})
 data={s:{k:np.concatenate([x[k] for x in values]) for k in (*KEYS,'action','seat','value')} for s,values in parts.items() if values}
 if set(data)!=set(parts):raise ValueError('Need disjoint natural training/validation teacher games')
 return data,manifest

def tensors(data,indices,device):
 return {k:torch.as_tensor(data[k][indices],dtype=torch.long if k in ('tokens','global_tokens','action_tokens') else torch.bool if k=='mask' else torch.float32,device=device) for k in KEYS}

def teacher_step(model,optimizer,data,indices,device,coef=1.,value_coef=0.,parent_anchor=None,parent_kl_coef=0.):
 x=tensors(data,indices,device);actions=torch.as_tensor(data['action'][indices],dtype=torch.long,device=device);labels=torch.as_tensor(data['value'][indices],device=device)
 logits,value=model(x);ce=torch.nn.functional.cross_entropy(logits,actions);vl=.5*(value-labels).square().mean();kl=logits.new_zeros(())
 if parent_anchor is not None and parent_kl_coef:
  with torch.no_grad():parent_logits,_=parent_anchor(x)
  kl=torch.nn.functional.kl_div(torch.log_softmax(logits,dim=-1),torch.softmax(parent_logits,dim=-1),reduction='batchmean')
 loss=coef*(ce+value_coef*vl+parent_kl_coef*kl)
 optimizer.zero_grad(set_to_none=True);loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),.5)
 if not torch.isfinite(norm):raise FloatingPointError('Teacher gradients nonfinite')
 optimizer.step();return {'ce':float(ce.detach()),'value_loss':float(vl.detach()),'parent_kl':float(kl.detach()),'grad_norm':float(norm.detach())}

@torch.no_grad()
def validate_teacher(model,data,device,parent_anchor=None,parent_kl_coef=0.):
 loss=correct=total=0;kl_sum=0.
 for start in range(0,len(data['action']),1024):
  indices=np.arange(start,min(start+1024,len(data['action'])));logits,_=model(tensors(data,indices,device));actions=torch.as_tensor(data['action'][indices],dtype=torch.long,device=device)
  loss+=float(torch.nn.functional.cross_entropy(logits,actions,reduction='sum'));correct+=int((logits.argmax(-1)==actions).sum());total+=len(indices)
  if parent_anchor is not None and parent_kl_coef:
   parent_logits,_=parent_anchor(tensors(data,indices,device));kl_sum+=float(torch.nn.functional.kl_div(torch.log_softmax(logits,dim=-1),torch.softmax(parent_logits,dim=-1),reduction='sum'))
 return {'cross_entropy':loss/total,'accuracy':correct/total,'samples':total,'parent_kl':kl_sum/total,'selection_loss':(loss+parent_kl_coef*kl_sum)/total}

def admit_original(job,spec):
 """Run strict parent admission under its exact source, before explicit new lineage."""
 if spec.get('parent_kind')=='random-canonical':
  cfg=spec['config'];seed=cfg['seed'];random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);rng=np.random.default_rng(seed)
  model=Policy(**cfg['network']);state=model.state_dict();identity=trainer.state_hash(state);torch.save(state,job/'random-parent-weights.pt')
  cp={'network':cfg['network'],'model':state,'model_sha256':identity,'optimizer':None,'numpy_rng':rng.bit_generator.state,'torch_rng':torch.get_rng_state(),'python_rng':random.getstate(),'cuda_rng':torch.cuda.get_rng_state_all(),'history':[state],'population_lineage':[{'iteration':0,'policy_sha256':identity}],'next_battle_index':0,'engine_runtime_signature':trainer.engine_runtime_signature()[0],'executable_signature':cli.executable_signature()}
  lineage={'admitted':True,'initialization':'Fresh seeded canonical Policy; no imported weights, optimizer, RNG or private observations','network':cfg['network'],'seed':seed,'model_sha256':identity,'trusted_reference_only':True,'entrant_architecture_unchanged':True,'training_seed_domain':cfg['training_seed_start']}
  write(job/'parent-admission.json',lineage);return cp,lineage
 if spec.get('parent_kind')=='sealed-plain':
  from ..autoresearch.bundle import verify
  from ..autoresearch.contract import runtime_signature
  bundle=Path(spec['parent_bundle']);meta=verify(bundle,spec['parent_bundle_sha256'])
  original_binding=json.loads((bundle/'binding.json').read_text())
  if original_binding['runtime_sha256']!=runtime_signature():raise ValueError('Original sealed parent runtime binding differs')
  if sha(bundle/'policy.py')!='638d949c10a5f57d235a3080f12ce3a304c88849495ecfbeaed485568cb42003':raise ValueError('Unaudited sealed policy source')
  if sha(bundle/'weights.pt')!=spec['parent_weights_sha256']:raise ValueError('Retained neural tensor bytes changed')
  config=json.loads((bundle/'config.json').read_text());network=config['network']
  model_state=torch.load(bundle/'weights.pt',map_location='cpu',weights_only=True);model=Policy(**network);model.load_state_dict(model_state,strict=True)
  if not all(torch.isfinite(x).all() for x in model_state.values()):raise ValueError('Nonfinite retained neural weight')
  random.seed(spec['config']['seed']);torch.manual_seed(spec['config']['seed']);torch.cuda.manual_seed_all(spec['config']['seed']);rng=np.random.default_rng(spec['config']['seed']);identity=trainer.state_hash(model_state)
  cp={'network':network,'model':model_state,'model_sha256':identity,'optimizer':None,'numpy_rng':rng.bit_generator.state,'torch_rng':torch.get_rng_state(),'python_rng':random.getstate(),'cuda_rng':torch.cuda.get_rng_state_all(),'history':[model_state],'population_lineage':[{'iteration':0,'policy_sha256':identity}],'next_battle_index':0,'engine_runtime_signature':trainer.engine_runtime_signature()[0],'executable_signature':cli.executable_signature()}
  lineage={'admitted':True,'original_bundle_sha256':spec['parent_bundle_sha256'],'weights_sha256':spec['parent_weights_sha256'],'model_sha256':identity,'policy_source_sha256':sha(bundle/'policy.py'),'original_binding':original_binding,'sealed_original_runtime_admitted':True,'original_config':config,'warm_start':'Audited canonical128/16 Policy tensor-only; no optimizer/RNG resume claim; original LLM-trained recipe lineage retained','optimizer_reset':True,'training_seed_domain':spec['config']['training_seed_start']}
  write(job/'parent-admission.json',lineage);return cp,lineage
 parent=Path(spec['parent_checkpoint']);archive=Path(spec['parent_source_archive']);expected=spec['parent_sha256']
 if sha(parent)!=expected:raise ValueError('Parent checkpoint byte identity changed')
 if spec.get('parent_source_archive_sha256') and sha(archive)!=spec['parent_source_archive_sha256']:raise ValueError('Parent source archive identity changed')
 import tempfile
 dest=Path(tempfile.mkdtemp(prefix='strict-parent-source-'))
 with tarfile.open(archive) as tar:tar.extractall(dest,filter='data')
 os.symlink('/opt/pk/candidates',dest/'source/candidates')
 code="import sys,json;from pathlib import Path;sys.path.insert(0,sys.argv[1]);import torch;from integrations.pokemon.reference_training import trainer;cp=torch.load(sys.argv[2],map_location='cpu',weights_only=False);trainer.admit(sys.argv[2],cp['config']);Path(sys.argv[3]).write_text(json.dumps({'admitted':True,'model_sha256':cp['model_sha256'],'executable_signature':cp['executable_signature'],'reference_source_signature':cp['reference_source_signature'],'engine_runtime_signature':cp['engine_runtime_signature'],'config_signature':cp['config_signature'],'checkpoint_iteration':cp['iteration']}))"
 subprocess.run(['python','-c',code,str(dest/'source'),str(parent),str(job/'parent-admission.json')],check=True,timeout=45)
 attest=json.loads((job/'parent-admission.json').read_text());cp=torch.load(parent,map_location='cpu',weights_only=False)
 if attest['model_sha256']!=trainer.state_hash(cp['model']):raise ValueError('Strictly admitted parent model changed')
 return cp,dict(attest,checkpoint_sha256=expected,original_source_archive_sha256=sha(archive),migration='Explicit new source/curriculum warm-start, not pretending old checkpoint is new-source resume')

def checkpoint(path,model,optimizer,cfg,iteration,cursor,rng,bc_rng,population,lineage,anchor,ledger,mh,engine_signature,parent_lineage,bc_report):
 data={'version':1,'reference_schema':1,'strong_schema':1,'executable_signature':cli.executable_signature(),'reference_source_signature':trainer.source_signature(),'engine_runtime_signature':engine_signature,'model':model.state_dict(),'optimizer':optimizer.state_dict(),'network':cfg['network'],'config_signature':cli.signature(cfg),'config':cfg,'iteration':iteration,'next_battle_index':cursor,'numpy_rng':rng.bit_generator.state,'torch_rng':torch.get_rng_state(),'python_rng':random.getstate(),'cuda_rng':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,'history':population.states(),'population_lineage':copy.deepcopy(lineage),'bc_rng':bc_rng.bit_generator.state,'bc_anchor':{k:v.detach().cpu() for k,v in anchor.state_dict().items()} if anchor else None,'model_sha256':trainer.state_hash(model.state_dict()),'allocated_worker_seconds':ledger.elapsed,'manifest_sha256':mh,'parent_lineage':parent_lineage,'bc_report':bc_report,'resume_boundary':'Between completed vector batches; all simulator states reset; full optimizer/RNG/population/cursor included','initialization':parent_lineage.get('initialization','Admitted existing learned checkpoint; optional public-visible teacher BC explicitly labeled')}
 temp=Path(path).with_suffix('.tmp');torch.save(data,temp);temp.replace(path)


def run(spec,job):
 torch.set_num_threads(1);cfg=copy.deepcopy(spec['config']);validate_bindings(cfg)
 for key,value in cfg['search_pins'].items():
  if spec.get(key)!=value:raise ValueError('Search dependency/configuration pin differs: '+key)
 device=cfg['hardware']['device'];start=time.perf_counter();remaining=spec['deadline_unix']-time.time();deadline=start+remaining
 if not torch.cuda.is_available():raise RuntimeError('Actual L4 CUDA learner required')
 torch.use_deterministic_algorithms(True)
 cp,parent_lineage=admit_original(job,spec)
 if cp['network']!=cfg['network']:raise ValueError('Warm-start architecture mismatch')
 engine_signature=trainer.engine_runtime_signature()[0]
 if engine_signature!=cp['engine_runtime_signature']:raise ValueError('Canonical executed engine changed during source migration')
 # Original model/code outputs and tensor identity are exact; all canonical source files stay pinned.
 if cli.executable_signature()!=cp['executable_signature']:raise ValueError('Canonical executable changed; explicit separate parity needed')
 model=Policy(**cfg['network']).to(device);model.load_state_dict(cp['model'],strict=True)
 if trainer.state_hash(model.state_dict())!=cp['model_sha256']:raise ValueError('Warm-start tensor parity failed')
 rng=np.random.default_rng();rng.bit_generator.state=cp['numpy_rng'];bc_rng=np.random.default_rng(cfg['seed']^0xBCA11);torch.set_rng_state(cp['torch_rng']);random.setstate(cp['python_rng'])
 if cp['cuda_rng'] is not None:torch.cuda.set_rng_state_all([x.cpu() for x in cp['cuda_rng']])
 population=Population(cfg,model,device,cp['history']);lineage=copy.deepcopy(cp['population_lineage']);cursor=cp['next_battle_index'];iteration=0
 frozen_peer=None;frozen_peer_identity=None
 if cfg.get('frozen_neural_opponent'):
  from ..autoresearch.evaluation_v2.arena import trusted_opponent
  frozen_peer=trusted_opponent(cfg['frozen_neural_opponent']).to(device);frozen_peer.requires_grad_(False);frozen_peer_identity=trainer.state_hash(frozen_peer.state_dict())
  if frozen_peer_identity!=cfg['frozen_neural_opponent']['model_sha256']:raise ValueError('Frozen neural opponent admitted tensor identity differs')
 output=job/'training';output.mkdir();ledger=Ledger(start,cp['allocated_worker_seconds'] if cp.get('strong_schema') and (spec.get('resume_strict') or spec.get('migrate_strong')) else 0.);mh=trainer.manifest(output/'source-manifest',cfg,'strong-reference-warmstart',parent_lineage);timings=Timings();initial_hash=trainer.state_hash(model.state_dict());bc_report=[];data=None
 if spec.get('teacher_data'):
  if sha(Path(spec['teacher_data'])/'index.json')!=cfg['teacher_index_sha256']:raise ValueError('Source-bound teacher index changed')
  data,manifest=load_data(spec['teacher_data']);write(output/'teacher-data-manifest.json',manifest)
 if cp.get('strong_schema') and (spec.get('resume_strict') or spec.get('migrate_strong')):
  if spec.get('resume_strict'):trainer.admit(spec['parent_checkpoint'],cfg)
  else:
   parent_lineage['migration_details']='Original strict source admits full state; explicit new source/curriculum import copies model, Adam, all RNGs, population, cursor and BCanchor without rerunning initialization'
   parent_lineage['changed_config_keys']=[k for k in sorted(set(cfg)|set(cp['config'])) if k not in ('iterations','allocation_seconds') and canonical(cfg.get(k))!=canonical(cp['config'].get(k))]
  iteration=cp['iteration'];bc_rng.bit_generator.state=cp['bc_rng'];optimizer=torch.optim.Adam(model.parameters(),lr=cfg['learning_rate'],eps=1e-5);optimizer.load_state_dict(cp['optimizer']);bc_report=cp['bc_report'];anchor=None
  if cp['bc_anchor']:
   anchor=Policy(**cfg['network']).to(device);anchor.load_state_dict(cp['bc_anchor']);anchor.eval();anchor.requires_grad_(False)
 else:
  if cfg.get('bc_epochs',0):
   if data is None:raise ValueError('BC requires a source-hashed public teacher corpus')
   parent_kl_coef=cfg.get('bc_parent_kl_coef',0.);parent_anchor=None
   if parent_kl_coef:
    if spec.get('parent_kind')=='random-canonical' or parent_kl_coef<0:raise ValueError('BC retention requires an admitted learned parent and nonnegative coefficient')
    parent_anchor=Policy(**cfg['network']).to(device);parent_anchor.load_state_dict(cp['model'],strict=True);parent_anchor.eval();parent_anchor.requires_grad_(False)
    if trainer.state_hash(parent_anchor.state_dict())!=initial_hash:raise ValueError('BC retention anchor differs from admitted parent')
   optimizer=torch.optim.Adam(model.parameters(),lr=cfg.get('bc_learning_rate',.001),eps=1e-5);best=None;best_loss=float('inf')
   bc_report.append(dict(epoch=0,validation=validate_teacher(model,data['validation'],device,parent_anchor,parent_kl_coef)))
   for epoch in range(cfg['bc_epochs']):
    losses=[]
    for indices in np.array_split(bc_rng.permutation(len(data['train']['action'])),max(1,int(np.ceil(len(data['train']['action'])/512)))):
     if time.perf_counter()>deadline-45:break
     losses.append(teacher_step(model,optimizer,data['train'],indices,device,1.,cfg.get('bc_value_coef',.1),parent_anchor,parent_kl_coef))
    validation=validate_teacher(model,data['validation'],device,parent_anchor,parent_kl_coef);bc_report.append({'epoch':epoch+1,'validation':validation,'training_ce':float(np.mean([x['ce'] for x in losses])) if losses else None})
    if validation['selection_loss']<best_loss:best_loss=validation['selection_loss'];best={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    write(output/'bc-metrics.json',bc_report)
    if time.perf_counter()>deadline-45:break
   if best is None:raise RuntimeError('No completed finite BC epoch')
   model.load_state_dict(best);parent_lineage['teacher_contribution']={'initialization':'public-channel FoulPlay behavior cloning','teacher_data':spec['teacher_data'],'selected_validation_objective':best_loss,'selection_objective':'teacherCE + coefficient * KL(parent || student)' if parent_kl_coef else 'teacherCE','bc_parent_kl_coef':parent_kl_coef,'optimizer_reset_after_BC':True,'parent_model_retained':initial_hash};parent_anchor=None
  anchor=Policy(**cfg['network']).to(device);anchor.load_state_dict(model.state_dict());anchor.eval();anchor.requires_grad_(False)
  optimizer=torch.optim.Adam(model.parameters(),lr=cfg['learning_rate'],eps=1e-5)
  if not cfg.get('bc_epochs',0) and cfg.get('carry_optimizer',True) and cp['optimizer'] is not None:optimizer.load_state_dict(cp['optimizer'])
  population.add(model);lineage.append({'iteration':iteration,'policy_sha256':trainer.state_hash(model.state_dict())});lineage=lineage[-cfg['history_size']:]
 if cp.get('strong_schema') and (spec.get('resume_strict') or spec.get('migrate_strong')):
  rng.bit_generator.state=cp['numpy_rng'];bc_rng.bit_generator.state=cp['bc_rng'];torch.set_rng_state(cp['torch_rng']);random.setstate(cp['python_rng'])
  if cp['cuda_rng'] is not None:torch.cuda.set_rng_state_all([x.cpu() for x in cp['cuda_rng']])
 checkpoint(output/'initial.pt',model,optimizer,cfg,iteration,cursor,rng,bc_rng,population,lineage,anchor,ledger,mh,engine_signature,parent_lineage,bc_report);subprocess.run(['sync','/runs'],check=True,timeout=10)
 source=None
 if any(o['kind']=='search' and o['weight']>0 for o in cfg['opponents']):
  (job/'search-setup').mkdir();source=prepare_search(spec,job/'search-setup',str(job/'request.json'))
 reports=[];last_save=time.perf_counter()
 with ReferenceVector(cfg['hardware']['workers'],cfg['hardware']['environments_per_worker']) as vector:
  inference=SearchInference(inference_for(cfg,timings),vector,cfg.get('search_parallel',False),frozen_peer);options=cfg['opponents'];weights=np.asarray([o['weight'] for o in options]);weights=weights/weights.sum()
  while iteration<cfg['iterations'] and time.perf_counter()<deadline-15:
   seeds=list(range(cfg['training_seed_start']+cursor,cfg['training_seed_start']+cursor+vector.size));assignments=[];identities=[];search=[]
   try:
    for i in range(vector.size):
     kind=options[int(rng.choice(len(options),p=weights))]['kind'];seat=(cursor+i)%2
     if kind=='selfplay':pair=[model,model];identity=['latest','latest']
     else:
      if kind=='historical':peer=population.history[int(rng.integers(len(population.history)))];name=trainer.state_hash(peer.state_dict())
      elif kind=='bc_anchor':peer=anchor;name='bc-anchor'
      elif kind=='retained':peer=frozen_peer;name='retained:'+frozen_peer_identity
      elif kind=='power':peer='power';name='power'
      elif kind=='search':peer=SearchActor(vector,i,1-seat,source,spec['public_sets'],str(job/'request.json'));search.append(peer);name='FoulPlay-fixed1000x2'
      else:raise ValueError('Unrecognized training opponent')
      pair=[peer,peer];pair[seat]=model;identity=[name,name];identity[seat]='latest'
     assignments.append(pair);identities.append(identity)
    inference.actors=search
    samples,report=rollout(vector,model,assignments,seeds,cfg,rng,inference,timings,deadline-10)
    stats=update(model,optimizer,samples,cfg,device,timings) if time.perf_counter()<deadline-10 else {'samples':0,'skipped':'allocation_expired'}
    auxiliary=[]
    if data and cfg.get('aux_bc_coef',0)>0 and stats['samples']:
     for _ in range(cfg.get('aux_bc_steps',1)):
      idx=bc_rng.integers(len(data['train']['action']),size=512);auxiliary.append(teacher_step(model,optimizer,data['train'],idx,device,cfg['aux_bc_coef']))
    iteration+=1;cursor+=vector.size
    if iteration%cfg['snapshot_every']==0:
     population.add(model);lineage.append({'iteration':iteration,'policy_sha256':trainer.state_hash(model.state_dict())});lineage=lineage[-cfg['history_size']:]
    report.update(iteration=iteration,ppo=stats,auxiliary_bc=auxiliary,opponent_identities=identities,seeds=seeds,search_decisions=sum(x.decisions for x in search),search_seconds=sum(x.seconds for x in search),allocated_worker_seconds=ledger.elapsed)
    reports.append(report)
    with (output/'metrics.jsonl').open('a') as f:f.write(canonical(report)+'\n')
    if iteration%cfg.get('checkpoint_every',10)==0 or time.perf_counter()-last_save>45:
     checkpoint(output/'checkpoint.pt',model,optimizer,cfg,iteration,cursor,rng,bc_rng,population,lineage,anchor,ledger,mh,engine_signature,parent_lineage,bc_report);subprocess.run(['sync','/runs'],check=True,timeout=10);last_save=time.perf_counter()
    print(canonical({'iteration':iteration,'samples':stats['samples'],'natural':report['natural_terminals'],'caps':report['truncations'],'search_decisions':report['search_decisions']}),flush=True)
   finally:
    for actor in search:actor.close()
 checkpoint(output/'checkpoint.pt',model,optimizer,cfg,iteration,cursor,rng,bc_rng,population,lineage,anchor,ledger,mh,engine_signature,parent_lineage,bc_report)
 trainer.admit(output/'checkpoint.pt',cfg)
 result={'iterations':iteration,'iterations_this_process':len(reports),'ppo_batches':sum(x['ppo']['samples']>0 for x in reports),'trained_samples':sum(x['ppo']['samples'] for x in reports),'natural_games':sum(x['natural_terminals'] for x in reports),'truncations':sum(x['truncations'] for x in reports),'visible_rejections':sum(x['rejected_visible_choices'] for x in reports),'search_decisions':sum(x['search_decisions'] for x in reports),'initial_parent_model_sha256':initial_hash,'final_model_sha256':trainer.state_hash(model.state_dict()),'checkpoint_sha256':sha(output/'checkpoint.pt'),'source_admitted':True,'parent_lineage':parent_lineage,'bc_report':bc_report,'seconds':ledger.elapsed,'process_seconds':time.perf_counter()-start}
 write(output/'results.json',result);return result
