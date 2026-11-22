"""Bounded actual-neural CUDA trials. All durations, failures and evaluations remain charged."""
from pathlib import Path
import sys,json,time,subprocess,os,statistics,hashlib
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent

def main():
 import torch
 stage=sys.argv[1];out=Path(sys.argv[2]);device=torch.cuda.get_device_name();begun=time.perf_counter()
 c=json.loads((HERE/'configs/smoke.json').read_text())
 c.update(device='cuda',envs=32,threads=4,torch_threads=1,rollout_steps=64,minibatch=1024,updates=24,budget_seconds=650,warmstart_steps=0,opponent_mode='selfplay',checkpoint_interval=1000,snapshot_interval=1000)
 c['hardware_allocation']={'gpu_model':device,'gpu_count':1,'cpu_threads':4,'ram_gib':16,'qualification':'actual bounded Modal staging CUDA pilot'}
 def run(name,config,extra=(),command='profile'):
  path=out/(name+'.json');path.write_text(json.dumps(config,indent=2)+'\n')
  args=[sys.executable,str(HERE/'runner.py'),command,'--config',str(path),'--out',str(out/name),*extra]
  start=time.perf_counter()
  import threading
  with (out/(name+'.log')).open('w') as log:
   proc=subprocess.Popen(args,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
   deadline=threading.Timer(max(1,min(790,820-(time.perf_counter()-begun))),proc.kill);deadline.start()
   try:
    for line in proc.stdout:
     log.write(line);log.flush()
     if line.startswith('{'):
      try:
       event=json.loads(line)
       if event.get('update',-1)%50==0:print(json.dumps({'name':name,'progress_update':event['update'],'worker_seconds':event['allocated_worker_seconds'],'wins':event['counts'].get('wins',0),'losses':event['counts'].get('losses',0),'active_policy_samples':event.get('active_policy_samples'),'value_loss':event.get('value_loss')}),flush=True)
      except (ValueError,KeyError):pass
    proc.wait()
   finally:deadline.cancel()
  if proc.returncode:
   print((out/(name+'.log')).read_text()[-9000:],flush=True);raise RuntimeError(name+' failed')
  result=json.loads((out/name/'result.json').read_text());result['subprocess_wall_seconds']=time.perf_counter()-start
  if (out/name/'metrics.jsonl').exists():
   metrics=[json.loads(line) for line in (out/name/'metrics.jsonl').read_text().splitlines()]
   if metrics and 'iteration_seconds' in metrics[0]:
    warm=metrics[2:];result['cold_first_update_seconds']=metrics[0]['iteration_seconds']
    result['warm_median_update_seconds']=statistics.median(x['iteration_seconds'] for x in warm) if warm else None
    result['warm_phase_median_seconds']={k:statistics.median(x['iteration_timing'].get(k,0.) for x in warm) for k in result['timing']} if warm else {}
  (out/(name+'-receipt.json')).write_text(json.dumps(result,indent=2)+'\n')
  print(json.dumps({'name':name,'seconds':result['subprocess_wall_seconds'],'warm_update_seconds':result.get('warm_median_update_seconds'),'counts':result['counts'],'timing':result['timing']}),flush=True)
  return result
 results={}
 if stage=='probe':
  for mode in ('dense','trimmed'):
   results[mode]=run(mode,dict(c,updates=12,trim_entities=mode=='trimmed'))
 elif stage=='paired':
  checkpoint=ROOT/'evidence/clash-implementation/final-validation/training/checkpoint.pt'
  for trial in range(2):
   modes=('dense','trimmed','buffers','optimized') if trial==0 else ('optimized','buffers','trimmed','dense')
   for mode in modes:
    if time.perf_counter()-begun>650:raise RuntimeError('bounded paired stage deadline; preserve completed runs')
    config=dict(c,trim_entities=mode!='dense',entity_bucket=32 if mode=='optimized' else 0,compile_forward=mode=='optimized',pinned_transfer=mode in ('buffers','optimized'),combined_readback=mode in ('buffers','optimized'),cache_rollout_device=mode in ('buffers','optimized'))
    results[f'{mode}-{trial}']=run(f'{mode}-{trial}',config,('--init-checkpoint',str(checkpoint)))
 elif stage=='audit':
  # Same-model real-observation CUDA parity, with saved mid-game histories reconstructed.
  import numpy as np
  from game import make_vector,Session
  from runner import network,setup
  checkpoint=ROOT/'evidence/clash-implementation/cuda-campaign/paired/trimmed-0/checkpoint.pt'
  saved=torch.load(checkpoint,map_location='cpu',weights_only=False);config=dict(saved['config']);setup(config)
  with make_vector(config) as v:
   session=Session(v,config,np.random.default_rng(3));session.restore(saved['session'])
   dense=network(v,dict(config,compile_forward=False,trim_entities=False));dense.load_state_dict(saved['model'])
   fused=network(v,dict(config,compile_forward=True,entity_bucket=200,pinned_transfer=True,device_indices=True));fused.load_state_dict(saved['model'])
   with torch.no_grad():
    d1,val1=dense.distribution(dense.pack(v.observations,v.masks,'cuda',False))
    d2,val2=fused.distribution(fused.pack(v.observations,v.masks,'cuda',True))
    parity={'max_policy_kl':float(torch.distributions.kl_divergence(d1,d2).abs().max()),'max_probability_error':float((d1.probs-d2.probs).abs().max()),'max_value_error':float((val1-val2).abs().max()),
     'states':v.count*2,'active_decision_states':int((v.masks.sum(-1)>1).sum()),'max_entities':int(v.observations[:,:,v.schema.header['entity_count'][0]].max())}
   assert parity['max_policy_kl']<1e-5 and parity['max_value_error']<1e-3 and parity['max_probability_error']<1e-4,parity
   graphed=network(v,dict(config,compile_forward=False,entity_bucket=200,pinned_transfer=True,device_indices=True,graph_inference=True));graphed.load_state_dict(saved['model'])
   with torch.no_grad():
    dg,vg=graphed.distribution(graphed.pack(v.observations,v.masks,'cuda',True))
    parity['cuda_graph_probability_error']=float((dg.probs-d1.probs).abs().max());parity['cuda_graph_value_error']=float((vg-val1).abs().max())
    assert parity['cuda_graph_probability_error']<1e-5 and parity['cuda_graph_value_error']<1e-4
    dense.slot.bias[0].add_(.125);graphed.slot.bias[0].add_(.125)
    du,vu=dense.distribution(dense.pack(v.observations,v.masks,'cuda',False))
    dg,vg=graphed.distribution(graphed.pack(v.observations,v.masks,'cuda',True))
    assert float((dg.probs-du.probs).abs().max())<1e-5 and float((vg-vu).abs().max())<1e-4
    parity['cuda_graph_live_parameter_updates']='passed'
   (out/'cuda-parity.json').write_text(json.dumps(parity,indent=2)+'\n');print('PARITY',json.dumps(parity),flush=True)
  del dense,fused,graphed;torch.cuda.empty_cache()
  init=ROOT/'evidence/clash-implementation/final-validation/training/checkpoint.pt'
  for mode in ('dense','trimmed'):
   config=dict(c,device='cpu',trim_entities=mode=='trimmed',updates=16)
   results['cpu-'+mode]=run('cpu-'+mode,config,('--init-checkpoint',str(init)))
  for name,threads,compiled in (('threads-1',1,False),('threads-4',4,False),('indexed',4,False),('graphed',4,False),('static-compiled',4,True),('batch-128',4,False)):
   config=dict(c,envs=128 if name=='batch-128' else c['envs'],threads=threads,trim_entities=True,pinned_transfer=True,combined_readback=True,cache_rollout_device=True,updates=16,compile_forward=compiled,device_indices=name in ('indexed','graphed','static-compiled','batch-128'),graph_inference=name in ('graphed','batch-128'),entity_bucket=200 if compiled or name in ('graphed','batch-128') else 0)
   results[name]=run(name,config,('--init-checkpoint',str(init)))
 elif stage in ('learning1','learning2','learning-entropy'):

  seed=2718 if stage=='learning2' else 1701
  if stage=='learning-entropy':
   resume_log=out/'cuda-resume.log'
   with resume_log.open('w') as log:subprocess.run([sys.executable,str(HERE/'tests/resume_check.py'),str(out/'cuda-resume'),'--cuda'],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=150)
   init=ROOT/'evidence/clash-implementation/final-validation/training/checkpoint.pt'
   for trial in range(2):
    for mode in (('indexed','graph') if trial==0 else ('graph','indexed')):
     profile=dict(c,updates=8,device_indices=True,pinned_transfer=True,combined_readback=True,cache_rollout_device=True,graph_inference=mode=='graph',inference_entity_bucket=200 if mode=='graph' else None)
     results[f'inference-{mode}-{trial}']=run(f'inference-{mode}-{trial}',profile,('--init-checkpoint',str(init)))
  config=dict(c,seed=seed,envs=32,opponent_mode='scripted',updates=700,rollout_steps=64,minibatch=1024,
   budget_seconds=550 if stage=='learning-entropy' else 680,warmstart_steps=512,warmstart_epochs=10,warmstart_lr=.003,active_policy_loss=True,
   entity_bucket=0,compile_forward=False,pinned_transfer=True,combined_readback=True,cache_rollout_device=True,learning_rate=.001,entropy_coef=0. if stage=='learning-entropy' else .02,reward_shaping=.5,gae_lambda=.995,retain_teacher_dataset=True)
  results['train']=run('train',config,command='train')
  # Evaluation uses eager execution for variable active-batch shapes; same weights and seeded policy.
  evaluation=dict(config,compile_forward=False,budget_seconds=800)
  schedule=HERE/'configs/eval-practice.json'
  for name,checkpoint in (('initial',out/'train/initial.pt'),('final',out/'train/checkpoint.pt'),('warmstart',out/'train/post-warmstart.pt')):
   if time.perf_counter()-begun>760:raise RuntimeError('bounded evaluation deadline')
   results[name]=run(name,evaluation,('--checkpoint',str(checkpoint),'--schedule',str(schedule)),command='evaluate')
  # Diagnose teacher forgetting separately from RL return/held-out strength.
  import numpy as np
  from game import make_vector
  from runner import network,setup,Timer
  dataset=np.load(out/'train/teacher_dataset.npz');teacher_obs=dataset['obs'];teacher_mask=dataset['mask'];teacher_action=dataset['action'];setup(evaluation)
  losses={}
  with make_vector(evaluation) as v:
   for label in ('post-warmstart','checkpoint'):
    saved=torch.load(out/('train/'+label+'.pt'),map_location='cpu',weights_only=False)
    model=network(v,evaluation);model.load_state_dict(saved['model']);total=0.
    with torch.no_grad():
     for start in range(0,len(teacher_action),1024):
      batch=model.pack(teacher_obs[start:start+1024],teacher_mask[start:start+1024],'cuda',True)
      target=torch.tensor(teacher_action[start:start+1024],device='cuda')
      total+=float(-model.distribution(batch)[0].log_prob(target).sum())
    losses[label]=total/len(teacher_action)
  (out/'teacher-forgetting.json').write_text(json.dumps({'nll':losses,'examples':len(teacher_action),'qualification':'training-data forgetting diagnostic, not held-out policy strength'},indent=2)+'\n')
 else:raise ValueError('unknown bounded stage')
 (out/'summary.json').write_text(json.dumps(results,indent=2)+'\n')
if __name__=='__main__':main()
