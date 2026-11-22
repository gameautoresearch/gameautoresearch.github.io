"""Untimed real-public packed-head CPU/CUDA qualifier; never launches cloud jobs.

CPU default tests dense/compact, all head boundaries, N256/48/16 tails, overflow
fallback, full-N value/PG/KL domains, every gradient/Adam tensor, and forced-after-
momentum. --device cuda additionally qualifies complete update capture, selected
caps, constructor restoration and same-mode graph continuation. --flatten-entities
adds the Cartesian entity/head selector; no silent sixteen-graph memory claim.
"""
from pathlib import Path
import argparse,copy,hashlib,json,sys,traceback
from types import SimpleNamespace
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path[:0]=[str(HERE),str(HERE.parent)]
import numpy as np
import torch
from packed_head import active_rows,capacity_for_cpu,head_bounds,packed_action_terms,packed_minibatch,ppo_loss,workload,HEAD_PARAMETER_NAMES
from gpu_features import DeviceBatchEncoder
from model import Policy
from train import schema

ATOL=2e-6;RTOL=2e-5

def write(path,data):Path(path).write_text(json.dumps(data,indent=2)+'\n')
def maxdiff(a,b):return float((a-b).abs().max().detach().cpu()) if a.numel() else 0.
def values(t):
 return {'model':[p.detach().clone() for p in (*t.model.parameters(),*t.model.buffers())],'grad':[None if p.grad is None else p.grad.detach().clone() for p in t.model.parameters()],'adam':[{k:v.detach().clone() for k,v in t.optimizer.state.get(p,{}).items()} for p in t.model.parameters()]}
def check_values(t,want,atol=ATOL,rtol=RTOL):
 maximum=0.
 for actual,expected in zip((*t.model.parameters(),*t.model.buffers()),want['model']):
  torch.testing.assert_close(actual,expected,atol=atol,rtol=rtol)
  if actual.is_floating_point():maximum=max(maximum,maxdiff(actual,expected))
 for p,g,state in zip(t.model.parameters(),want['grad'],want['adam']):
  if g is None:assert p.grad is None
  else:assert p.grad is not None;torch.testing.assert_close(p.grad,g,atol=atol,rtol=rtol)
  for k,v in state.items():torch.testing.assert_close(t.optimizer.state[p][k],v,atol=atol,rtol=rtol)
 return maximum

def fixtures(fixture,s):
 z=np.load(fixture);obs=z['observations'].reshape(-1,s.seat_ints);mask=z['masks'].reshape(-1,s.actions).astype(bool)
 a=np.flatnonzero(mask[:,1:].any(1)|~mask[:,0]);f=np.flatnonzero(~(mask[:,1:].any(1)|~mask[:,0]));assert len(a) and len(f)
 def make(n,count,population=None,requested=None,label=None):
  ix=np.resize(f,n).copy()
  if count:ix[np.linspace(0,n-1,count,dtype=int)]=np.resize(a,count)
  o=obs[ix].copy();m=mask[ix].copy()
  if population is not None:
   rows=o[:,s.header_ints:].reshape(-1,200,s.entity_ints);template=rows[:,:1].copy();rows[:]=0
   if population:rows[:,:population]=template
   o[:,s.header['entity_count'][0]]=population
  action=np.array([np.flatnonzero(row)[-1] for row in m],np.int64)
  return {'label':label or f'N{n}-active{count}-population{population}','obs':o,'mask':m,'action':action,'requested':requested,'population':population,'active_count':count}
 return obs,mask,make

def trainer(model,encoder,device):
 return SimpleNamespace(model=model,encoder=encoder,schema=model.s,c={'clip':.2,'value_coef':.5,'entropy_coef':.001,'max_grad_norm':.5,'compile_loss':False,'rollout_graph':False},optimizer=torch.optim.Adam(model.parameters(),lr=.0003,eps=1e-5,fused=device=='cuda',capturable=device=='cuda'))

def reference_terms(model,batch,actions,mode):
 if mode=='distribution':
  distribution,value=model.distribution(batch)
  return distribution.log_prob(actions),distribution.entropy(),value
 if mode=='action_terms':return model.action_terms(batch,actions)
 raise ValueError('unknown reference mode')

def fields_for(t,case,active_weight=True,reference_mode='action_terms'):
 d=next(t.model.parameters()).device
 obs=torch.tensor(case['obs'],dtype=torch.int32,device=d);mask=torch.tensor(case['mask'],dtype=torch.bool,device=d);actions=torch.tensor(case['action'],dtype=torch.int64,device=d)
 with torch.no_grad():old=reference_terms(t.model,t.encoder(obs,mask),actions,reference_mode)[0]
 active=active_rows(mask);old=old+torch.linspace(-.35,.35,len(mask),device=d)*active
 weight=active.float() if active_weight else torch.ones(len(mask),device=d)
 adv=torch.linspace(-1.3,1.1,len(mask),device=d);norm=torch.where(weight.sum()>0,weight,torch.ones_like(weight));mean=(adv*norm).sum()/norm.sum();std=torch.sqrt(((adv-mean).square()*norm).sum()/norm.sum());adv=(adv-mean)/(std+1e-8)
 return {'obs':obs,'mask':mask,'action':actions,'logp':old,'advantage':adv,'return':torch.linspace(.2,.8,len(mask),device=d),'weight':weight}

def reference_step(t,fields,reference_mode='action_terms'):
 batch=t.encoder(fields['obs'],fields['mask']);terms=reference_terms(t.model,batch,fields['action'],reference_mode)
 # Independent expression; KL remains a FULL N mean including every forced row.
 new,entropy,value=terms;old=fields['logp'];ratio=(new-old).exp();w=fields['weight'];a=fields['advantage'];c=t.c
 pg=-(torch.minimum(ratio*a,ratio.clamp(1-c['clip'],1+c['clip'])*a)*w).sum()/w.sum().clamp_min(1)
 vl=(value-fields['return']).square().mean();ent=(entropy*w).sum()/w.sum().clamp_min(1);kl=((ratio-1)-(new-old)).mean();loss=pg+c['value_coef']*vl-c['entropy_coef']*ent
 torch._assert_async(torch.isfinite(loss),'nonfinite reference')
 t.optimizer.zero_grad(set_to_none=True);loss.backward();grad=torch.nn.utils.clip_grad_norm_(t.model.parameters(),c['max_grad_norm']);t.optimizer.step()
 return torch.stack((pg.detach(),vl.detach(),ent.detach(),kl.detach(),grad.detach()))

def qualify_semantics(out,device,compact,args):
 data=json.loads((HERE/'generated-v2/task-data.json').read_text());s=schema(data);obs,masks,make=fixtures(args.fixture,s)
 torch.manual_seed(8110828);base=Policy(s,len(data['vocabulary']),len(data['buff_names']),args.width,12);base.device_indices=True
 torch.nn.init.normal_(base.slot.weight,std=.01)
 if compact:
  from compact_embeddings import install
  install(base,data)
 base.to(device);original_ids=tuple((name,id(p),p.data_ptr()) for name,p in base.named_parameters());original_state_names=tuple(base.state_dict())
 cases=[make(256,count,label=f'boundary{count}') for count in (1,32,33,64,65,128,129,256,0)]
 cases+=[make(48,7,64,label='entity-max-ties64'),make(48,7,200,label='entity-max-ties200'),make(16,3,0,label='zero-entities'),make(48,7,label='tail48'),make(48,33,requested=32,label='overflow-request32-full48'),make(16,1,label='tail16'),make(16,0,label='forced-tail16')]
 cases.insert(0,{'label':'real-public','obs':obs,'mask':masks,'action':np.array([np.flatnonzero(row)[-1] for row in masks],np.int64),'requested':None,'population':None,'active_count':int((masks[:,1:].any(1)|~masks[:,0]).sum())})
 reports=[];maximum=0.;gradient_error=metric_error=adam_error=0.
 for reference_mode,active_weight in [(mode,weight) for mode in ('action_terms','distribution') for weight in (True,False)]:
  ref=trainer(copy.deepcopy(base),DeviceBatchEncoder(s,entity_rows=200,copy=False),device);cand=trainer(copy.deepcopy(base),DeviceBatchEncoder(s,entity_rows=200,copy=False),device)
  ids=tuple((n,id(p),p.data_ptr()) for n,p in cand.model.named_parameters());names=tuple(cand.model.state_dict())
  for case in cases:
   fields=fields_for(ref,case,active_weight,reference_mode);cap,selection=capacity_for_cpu(torch.from_numpy(case['mask']),case['requested'])
   batch=cand.encoder(fields['obs'],fields['mask']);terms,selected=packed_action_terms(cand.model,batch,fields['action'],cap,return_selection=True)
   expected=reference_terms(ref.model,batch,fields['action'],reference_mode)
   for actual,want in zip(terms,expected):torch.testing.assert_close(actual,want,atol=ATOL,rtol=RTOL)
   chosen=selected.detach().cpu().tolist();activeix=np.flatnonzero(case['mask'][:,1:].any(1)|~case['mask'][:,0]).tolist();assert chosen[:len(activeix)]==activeix and len(set(chosen))==cap
   assert torch.equal(terms[0][~active_rows(fields['mask'])],torch.zeros_like(terms[0][~active_rows(fields['mask'])]))
   assert torch.equal(terms[1][~active_rows(fields['mask'])],torch.zeros_like(terms[1][~active_rows(fields['mask'])]))
   ratio=(terms[0]-fields['logp']).exp();rowkl=(ratio-1)-(terms[0]-fields['logp']);wrongkl=maxdiff(rowkl.index_select(0,selected).mean().reshape(1),rowkl.mean().reshape(1))
   if case['label']=='real-public':assert wrongkl>1e-5,'fixture must distinguish wrong selected-cap KL'
   head=dict(cand.model.named_parameters());prior={n:{k:v.clone() for k,v in cand.optimizer.state.get(head[n],{}).items()} for n in HEAD_PARAMETER_NAMES}
   before={n:head[n].detach().clone() for n in HEAD_PARAMETER_NAMES}
   if case['label']=='boundary0':assert all('exp_avg' in prior[n] and prior[n]['exp_avg'].abs().max()>0 for n in HEAD_PARAMETER_NAMES),'forced test requires nonzero real momentum for every head-exclusive parameter'
   want=reference_step(ref,fields,reference_mode);actual=packed_minibatch(cand,fields,cap);torch.testing.assert_close(actual,want,atol=ATOL,rtol=RTOL);expected_values=values(ref);maximum=max(maximum,check_values(cand,expected_values));metric_error=max(metric_error,maxdiff(actual,want))
   for p,g,state in zip(cand.model.parameters(),expected_values['grad'],expected_values['adam']):
    if g is not None:gradient_error=max(gradient_error,maxdiff(p.grad,g))
    for key,value in state.items():adam_error=max(adam_error,maxdiff(cand.optimizer.state[p][key],value))
   forced=not selection['active_count'];moved=False
   if forced:
    for name in HEAD_PARAMETER_NAMES:
     p=head[name];assert p.grad is not None and torch.equal(p.grad,torch.zeros_like(p.grad)),name
     if prior[name]:
      state=cand.optimizer.state[p];assert state['step']==prior[name]['step']+1;torch.testing.assert_close(state['exp_avg'],prior[name]['exp_avg']*.9,atol=ATOL,rtol=RTOL);torch.testing.assert_close(state['exp_avg_sq'],prior[name]['exp_avg_sq']*.999,atol=ATOL,rtol=RTOL);moved|=not torch.equal(p,before[name])
    if case['label']=='boundary0':assert moved,'real existing Adam momentum must still move forced-head weights'
   assert ids==tuple((n,id(p),p.data_ptr()) for n,p in cand.model.named_parameters()) and names==tuple(cand.model.state_dict())
   reports.append({'case':case['label'],'reference_mode':reference_mode,'active_policy_loss':active_weight,'N':len(case['obs']),'selection':selection,'wrong_selected_cap_KL_difference':wrongkl,'all_named_gradients_Adam_and_model_compared':True,'forced_head_zero_nonNone':forced,'head_momentum_moved_when_forced':moved if forced else None,'workload':workload(cand.model,len(case['obs']),cap)})
  # Bare static branch rejects overflow; CPU convenience selector took full48.
  bad=make(48,33);fields=fields_for(cand,bad)
  if device=='cpu':
   try:packed_action_terms(cand.model,cand.encoder(fields['obs'],fields['mask']),fields['action'],32)
   except RuntimeError:pass
   else:raise AssertionError('undersized static branch silently dropped active states')
  torch.save({'model':cand.model.state_dict(),'adam':cand.optimizer.state_dict()},out/f'semantic-{reference_mode}-activeweight-{active_weight}.pt')
 assert original_ids==tuple((name,id(p),p.data_ptr()) for name,p in base.named_parameters()) and original_state_names==tuple(base.state_dict())
 return {'status':'passed','device':device,'compact_embedding_grad':compact,'cases':reports,'reference_modes':['action_terms','distribution'],'maximum_post_update_parameter_error':maximum,'maximum_clipped_gradient_error':gradient_error,'maximum_metric_error':metric_error,'maximum_Adam_state_error':adam_error,'atol':ATOL,'rtol':RTOL,'entity_rows':200,'full_N_fallback_and_denominators':True,'same_parameter_objects_and_state_dict':True,'timing_claim':False}


def qualify_graphs(out,args,compact):
 from packed_head_graph import PackedHeadLearnerGraph,_encoder
 from learner_graph import _TrainingSnapshot
 data=json.loads((HERE/'generated-v2/task-data.json').read_text());s=schema(data);_,_,make=fixtures(args.fixture,s)
 torch.manual_seed(8110828);model=Policy(s,len(data['vocabulary']),len(data['buff_names']),args.width,12);model.device_indices=True;torch.nn.init.normal_(model.slot.weight,std=.01)
 if compact:
  from compact_embeddings import install
  install(model,data)
 model=model.cuda();t=trainer(model,DeviceBatchEncoder(s,entity_rows=200,copy=False),'cuda');bounds=(32,64,96,200) if args.flatten_entities else (200,)
 cases=[]
 for population in ((0,32,33,64,65,96,97,200) if args.flatten_entities else (None,)):
  for count in (0,32,33,64,65,128,129,256):cases.append(make(256,count,population))
 cases+=[make(48,33,label='tail48'),make(48,0,label='forced-tail48'),make(16,1,label='tail16'),make(16,0,label='forced-tail16'),make(17,1,label='partial-warp-tail17')]
 fields=[fields_for(t,c) for c in cases];graphs={};records=[];snapshot=_TrainingSnapshot(t);parameter_ids=tuple(id(p) for p in model.parameters());state_names=tuple(model.state_dict())
 try:
  # Two real updates establish nonzero optimizer history before constructor tests.
  packed_minibatch(t,fields[2],256);packed_minibatch(t,fields[3],256)
  saved=_TrainingSnapshot(t);expected=[]
  for f in fields:
   count=int(active_rows(f['mask']).sum().cpu());cap=next(c for c in head_bounds(len(f['action'])) if count<=c);pop=int(f['obs'][:,s.header['entity_count'][0]].max().cpu());bound=next(b for b in bounds if pop<=b)
   with _encoder(t,DeviceBatchEncoder(s,entity_rows=bound,copy=False)):metrics=packed_minibatch(t,f,cap).clone()
   expected.append((metrics,values(t),count,cap,pop,bound))
  saved.restore();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
  for f,(want,v,count,cap,pop,bound) in zip(fields,expected):
   n=len(f['action'])
   if n not in graphs:
    before=values(t);rng=torch.cuda.get_rng_state_all();cpu_rng=torch.get_rng_state().clone();graphs[n]=PackedHeadLearnerGraph(t,f,entity_bounds=bounds)
    check_values(t,before,atol=0.,rtol=0.);assert torch.equal(cpu_rng,torch.get_rng_state());assert all(torch.equal(a,b) for a,b in zip(rng,torch.cuda.get_rng_state_all()))
   graph=graphs[n];actual=graph(f).clone();torch.testing.assert_close(actual,want,atol=0.,rtol=0.);check_values(t,v,atol=0.,rtol=0.)
   ei=bounds.index(bound);hi=head_bounds(n).index(cap);selection=graph.selection.cpu().tolist();assert selection==[ei*len(head_bounds(n))+hi,count,pop,ei,hi,0],selection
   records.append({'N':n,'selection':selection,'entity_bound':bound,'head_cap':cap,'same_mode_graph_model_Adam_gradients_metrics_exact':True})
  # Shader-only controls mutate PRIVATE graph inputs and execute NO learner.
  # Include final action2306, warp-column boundaries and the final partial row.
  probes=[]
  for n,graph in graphs.items():
   saved_mask=graph.inputs['mask'].clone();saved_counts=graph.inputs['obs'][:,s.header['entity_count'][0]].clone();before=values(t)
   try:
    for column in (1,31,32,33,s.actions-1):
     for count in sorted({0,min(1,n),min(32,n),min(33,n),min(64,n),min(65,n),min(128,n),min(129,n),n}):
      graph.inputs['mask'].zero_();graph.inputs['mask'][:,0]=1
      if count>1:graph.inputs['mask'][:count-1,column]=1
      if count:graph.inputs['mask'][-1,column]=1
      graph.inputs['obs'][:,s.header['entity_count'][0]]=200
      got=graph.branch.probe().cpu().tolist();hi=next(i for i,c in enumerate(head_bounds(n)) if count<=c);ei=len(bounds)-1
      assert got==[ei*len(head_bounds(n))+hi,count,200,ei,hi,0],got
      probes.append({'N':n,'column':column,'active_count':count,'selection':got})
    graph.inputs['mask'].zero_();graph.inputs['mask'][:,0]=1;graph.inputs['mask'][-1,0]=0
    got=graph.branch.probe().cpu().tolist();assert got[1]==1,'wait-illegal row retains original active predicate'
    for bad,error in ((-1,1),(201,2)):
     graph.inputs['obs'][:,s.header['entity_count'][0]]=0;graph.inputs['obs'][-1,s.header['entity_count'][0]]=bad
     got=graph.branch.probe().cpu().tolist();assert got[0]==-1 and got[5]==error,(bad,got)
    check_values(t,before,atol=0.,rtol=0.)
   finally:
    graph.inputs['mask'].copy_(saved_mask);graph.inputs['obs'][:,s.header['entity_count'][0]].copy_(saved_counts)
  # Snapshot/restore in a NEW branch bank, with the same existing state dicts and
  # Adam objects; constructor captures are not real updates and consume no RNG.
  checkpoint={'model':copy.deepcopy(model.state_dict()),'adam':copy.deepcopy(t.optimizer.state_dict()),'cpu_rng':torch.get_rng_state().clone(),'cuda_rng':torch.cuda.get_rng_state_all()}
  torch.save(checkpoint,out/'graph-checkpoint.pt')
  g=graphs[256];f=fields[2];continuation=[]
  for _ in range(3):continuation.append((g(f).clone(),values(t)))
  for graph in graphs.values():graph.close()
  graphs={};checkpoint=torch.load(out/'graph-checkpoint.pt',weights_only=False);model.load_state_dict(checkpoint['model']);t.optimizer.load_state_dict(checkpoint['adam']);torch.set_rng_state(checkpoint['cpu_rng']);torch.cuda.set_rng_state_all(checkpoint['cuda_rng'])
  restored=PackedHeadLearnerGraph(t,f,entity_bounds=bounds);graphs[256]=restored
  for metrics,v in continuation:torch.testing.assert_close(restored(f),metrics,atol=0.,rtol=0.);check_values(t,v,atol=0.,rtol=0.)
  assert parameter_ids==tuple(id(p) for p in model.parameters()) and state_names==tuple(model.state_dict())
  rng=torch.cuda.get_rng_state_all();restored(f);torch.cuda.synchronize();assert all(torch.equal(a,b) for a,b in zip(rng,torch.cuda.get_rng_state_all()))
  restored.branch._check(restored.branch.lib.cn_head_branch_test_unsupported_children())
  outer=torch.cuda.CUDAGraph();dummy=torch.zeros(1,device='cuda');rejected=False
  with torch.cuda.graph(outer):
   dummy.zero_()
   try:restored(f)
   except RuntimeError:rejected=True
  assert rejected,'conditional parent must reject nested outer capture'
  return {'status':'passed','actual_CUDA':True,'entity_bounds':bounds,'head_bounds256':head_bounds(256),'records':records,'graph_shapes':[16,17,48,256],'maximum_children':len(bounds)*len(head_bounds(256)),'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'peak_memory_scope':'Torch allocator; native CUDA graph metadata excluded, selector scratch separately reported','selector_peak_scratch_bytes':4*(256+48+16+17),'constructor_model_Adam_RNG_restored':True,'same_mode_in_process_checkpoint_reconstruction_exact':True,'independent_process_full_trainer_resume_still_required':True,'helper_binding':restored.build_binding,'child_node_types':restored.branch.child_node_types,'selector_scratch_bytes':restored.branch.selector_scratch_bytes,'selector_probes':probes,'selector_prepare':'warp-per-row coalesced OR, retained4N flags, dependent count/dispatch kernel','unsupported_host_child_negative_control':True,'outer_capture_rejected':True,'real_replay_RNG_unchanged':True,'timing_claim':False}
 finally:
  for graph in graphs.values():graph.close()
  snapshot.restore();torch.cuda.synchronize()

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--device',choices=['cpu','cuda'],default='cpu');ap.add_argument('--out',type=Path,required=True);ap.add_argument('--fixture',type=Path,default=ROOT/'evidence/clash-implementation/native-training-20261108/compact-embedding-cpu-3/real-observations.npz');ap.add_argument('--width',type=int,default=64);ap.add_argument('--compact',action='store_true');ap.add_argument('--flatten-entities',action='store_true');ap.add_argument('--skip-graphs',action='store_true');args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=False)
 torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 result={'status':'running','torch':torch.__version__,'cuda_executed':args.device=='cuda','timing_claim':False}
 try:
  if args.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('actual CUDA required; no cloud launcher is provided')
  # Invalid bare static-cap assert is a CPU-only check; CUDA contexts must not
  # deliberately assert during normal qualification. CPU proof covers rejection.
  result['semantic']=qualify_semantics(args.out,args.device,args.compact,args)
  if args.device=='cuda' and not args.skip_graphs:result['graphs']=qualify_graphs(args.out,args,args.compact)
  result['status']='passed'
 except BaseException as exc:result.update(status='failed',failure=repr(exc),traceback=traceback.format_exc());raise
 finally:
  result['source_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob('packed_head*') if p.is_file()};result['fixture_sha256']=hashlib.sha256(args.fixture.read_bytes()).hexdigest();write(args.out/'result.json',result)
if __name__=='__main__':main()
