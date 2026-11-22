import copy,json,unittest
from pathlib import Path
import numpy as np
import torch
from integrations.pokemon.env import Vector,HERE
from integrations.pokemon.network import batch
from integrations.pokemon.efficient import EfficientInference
from integrations.pokemon.rl import Timings,Inference,rollout,update
from integrations.pokemon.cli import initialize

class EfficiencyTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):torch.set_num_threads(1)
 def test_lossless_wire(self):
  with Vector(1,2,packed=False) as a,Vector(1,2,packed=True) as b:
   aa=a.reset([34,35],15);bb=b.reset([34,35],15)
   for _ in range(10):
    for x,y in zip(aa,bb):
     self.assertEqual(x['ready'],y['ready'])
     for p,q in zip(x['observations'],y['observations']):
      for key in p:np.testing.assert_array_equal(p[key],q[key])
    actions=[[int(np.flatnonzero(g['observations'][s]['mask'])[0]) if g['ready'][s] else None for s in range(2)] for g in aa]
    aa=a.step(actions);bb=b.step(actions)
   self.assertEqual([r['sha256'] for r in a.trace()],[r['sha256'] for r in b.trace()])
 def test_stochastic_rollout_and_learner_parity(self):
  cfg=json.loads((HERE/'recipes/smoke.json').read_text());cfg.update(max_decision_phases=64,ppo_epochs=2)
  def run(efficient):
   c=dict(cfg,optimization='packed' if efficient else 'baseline');model=initialize(c);t=Timings()
   with Vector(1,2,packed=efficient) as v:
    samples,report=rollout(v,model,[[model,'power'],['random',model]],[512,513],c,np.random.default_rng(0),EfficientInference('cpu',t) if efficient else Inference('cpu',t),t)
    traces=[r['sha256'] for r in v.trace()]
   optimizer=torch.optim.Adam(model.parameters(),lr=c['learning_rate'],eps=1e-5)
   update(model,optimizer,samples,c,'cpu',t)
   return model.state_dict(),samples,traces
  a,aa,ta=run(False);b,bb,tb=run(True)
  self.assertEqual(ta,tb);self.assertEqual([s['action'] for s in aa],[s['action'] for s in bb])
  for key in a:torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)
 def test_nonfinite_update_is_not_accepted(self):
  cfg=json.loads((HERE/'recipes/smoke.json').read_text());cfg.update(optimization='packed',ppo_epochs=1)
  model=initialize(cfg)
  with Vector(1,1) as v:obs=v.reset([42],5)[0]['observations'][0]
  samples=[{'obs':obs,'action':int(np.flatnonzero(obs['mask'])[0]),'logp':0.,'advantage':1.,'target':float('nan')}]
  with self.assertRaises(FloatingPointError):
   update(model,torch.optim.Adam(model.parameters(),lr=.001),samples,cfg,'cpu',Timings())

class StreamTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):torch.set_num_threads(1)
 def test_binary_cache_observation_and_trace_identity(self):
  from integrations.pokemon.network import KEYS,scripted
  with Vector(1,4,packed=True) as old,Vector(1,4,packed=True,binary=True,token_cache=True,mon_cache=True,fast_callbacks=True) as new:
   a=old.reset([20000,20001,20002,20003],160);b=new.reset([20000,20001,20002,20003],160)
   rng=np.random.default_rng(93)
   for _ in range(160):
    for x,y in zip(a,b):
     self.assertEqual({k:v for k,v in x.items() if k!='observations'}, {k:v for k,v in y.items() if k!='observations'})
     for p,q in zip(x['observations'],y['observations']):
      for key in KEYS:np.testing.assert_array_equal(p[key],q[key])
      self.assertFalse(q.flat.flags.writeable)
    if all(g['terminated'] or g['truncated'] for g in a):break
    actions=[[scripted(g['observations'][s],rng,'power') if g['ready'][s] else None for s in range(2)] for g in a]
    a=old.step(actions);b=new.step(actions)
   self.assertEqual(old.hashes(),new.hashes())
   # The same persistent workers can reset and deliver a fresh binary frame.
   a=old.reset([81,82,83,84],1);b=new.reset([81,82,83,84],1)
   self.assertEqual([g['seed'] for g in a],[g['seed'] for g in b])
 def test_stream_stochastic_update_identity(self):
  from integrations.pokemon.rl import inference_for
  cfg=json.loads((HERE/'recipes/smoke.json').read_text());cfg.update(max_decision_phases=64,ppo_epochs=2)
  def run(mode):
   c=dict(cfg,optimization=mode);m=initialize(c);t=Timings()
   with Vector(1,2,packed=True,binary=mode=='stream',token_cache=mode=='stream') as v:
    samples,report=rollout(v,m,[[m,'power'],['random',m]],[171,172],c,np.random.default_rng(0),inference_for(c,t),t)
    hashes=v.hashes()
   opt=torch.optim.Adam(m.parameters(),lr=c['learning_rate'],eps=1e-5)
   update(m,opt,samples,c,'cpu',t)
   return m.state_dict(),torch.get_rng_state(),samples,hashes
  a,ra,sa,ta=run('packed');b,rb,sb,tb=run('stream')
  self.assertEqual(ta,tb);self.assertEqual([s['action'] for s in sa],[s['action'] for s in sb])
  torch.testing.assert_close(ra,rb,rtol=0,atol=0)
  for k in a:torch.testing.assert_close(a[k],b[k],rtol=0,atol=0)
 def test_timeout_exit_and_frame_bounds(self):
  from integrations.pokemon.env import Worker,WorkerError
  import struct
  for binary in (False,True):
   w=Worker(binary=binary,timeout=.02)
   try:
    with self.assertRaisesRegex(WorkerError,'timed out'):w.receive()
   finally:w.close()
   self.assertIsNotNone(w.process.poll())
  w=Worker(binary=True)
  try:
   w.pending.extend(struct.pack('<II',w.MAX_FRAME_BYTES+1,0))
   with self.assertRaisesRegex(WorkerError,'header exceeds'):w.receive()
  finally:w.close()
  w=Worker(binary=True)
  w.process.kill();w.process.wait()
  try:
   with self.assertRaisesRegex(WorkerError,'worker exited'):w.receive()
  finally:w.close()

class QueueTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):torch.set_num_threads(1)
 def test_schedule_identity_and_reset_boundaries(self):
  from integrations.pokemon.rl import rollout_queue,inference_for
  cfg=json.loads((HERE/'recipes/smoke.json').read_text());cfg.update(max_decision_phases=80,optimization='stream')
  m=initialize(cfg);seeds=list(range(4000,4012))
  assignments=[[m,'power'] if i%2==0 else ['random',m] for i in range(12)]
  def run(size,collect):
   t=Timings()
   with Vector(1,size,packed=True,binary=True,token_cache=True) as v:
    samples,report=collect(v,m,assignments,seeds,cfg,np.random.default_rng(3),inference_for(cfg,t),t,deterministic=True)
   return samples,report
  a,ra=run(12,rollout);b,rb=run(3,rollout_queue)
  self.assertEqual([g['trace_sha256'] for g in ra['results']],[g['trace_sha256'] for g in rb['results']])
  self.assertEqual([g['reason'] for g in ra['results']],[g['reason'] for g in rb['results']])
  self.assertEqual([s['action'] for s in a],[s['action'] for s in b])
  np.testing.assert_allclose([s['target'] for s in a],[s['target'] for s in b],rtol=1e-5,atol=1e-6)
  self.assertEqual(rb['battles'],12);self.assertEqual(rb['unstarted_battles'],0)
 def test_live_reset_refused_and_deadline_separates_unstarted(self):
  from integrations.pokemon.env import WorkerError
  from integrations.pokemon.rl import rollout_queue,inference_for
  import time
  cfg=json.loads((HERE/'recipes/smoke.json').read_text());cfg.update(optimization='stream')
  m=initialize(cfg);t=Timings()
  with Vector(1,2,packed=True,binary=True) as v:
   v.reset([1,2],10)
   with self.assertRaisesRegex(WorkerError,'live battle'):v.reset_slots([(3,False),None],10)
   samples,r=rollout_queue(v,m,[[m,'power']]*6,list(range(1,7)),cfg,np.random.default_rng(0),inference_for(cfg,t),t,deadline=time.perf_counter())
   self.assertEqual(r['unstarted_battles'],4);self.assertEqual(r['truncations'],2)
   self.assertEqual(r['natural_terminals'],0);self.assertEqual(len(samples),0)

class SharedTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):torch.set_num_threads(1)
 def test_shared_snapshot_lifetime_and_learner_identity(self):
  from integrations.pokemon.rl import inference_for
  cfg=json.loads((HERE/'recipes/smoke.json').read_text());cfg.update(max_decision_phases=80,ppo_epochs=2)
  def run(mode):
   c=dict(cfg,optimization=mode,batched_scripted=mode=='shared');m=initialize(c);t=Timings()
   with Vector(1,4,packed=True,binary=True,shared=mode=='shared',lazy=mode=='shared') as v:
    paths=[w.shared_path for w in v.workers] if mode=='shared' else []
    samples,r=rollout(v,m,[[m,'power'],['random',m],[m,'power'],['random',m]],[171,172,173,174],c,np.random.default_rng(0),inference_for(c,t),t)
    first=samples[0]['obs'];snapshot={k:v.copy() for k,v in first.items()}
    traces=v.hashes();v.reset([81,82,83,84],1)
    for k in first:np.testing.assert_array_equal(first[k],snapshot[k])
   for path in paths:self.assertFalse(Path(path).exists())
   opt=torch.optim.Adam(m.parameters(),lr=c['learning_rate'],eps=1e-5)
   update(m,opt,samples,c,'cpu',t)
   return m.state_dict(),torch.get_rng_state(),traces
  a,ra,ta=run('stream');b,rb,tb=run('shared')
  self.assertEqual(ta,tb);self.assertTrue(torch.equal(ra,rb))
  for k in a:torch.testing.assert_close(a[k],b[k],rtol=0,atol=0)

class ScriptBatchTests(unittest.TestCase):
 def test_actions_and_battle_rng_states_match(self):
  from integrations.pokemon.network import scripted,scripted_batch
  with Vector(1,24,packed=True,binary=True,shared=True,lazy=True) as v:
   states=v.reset(list(range(9000,9024)),100)
   for phase in range(80):
    rows=[(i,s,g['observations'][s]) for i,g in enumerate(states) for s in (0,1) if g['ready'][s]]
    if not rows:break
    kinds=['random' if (i+s)%3==0 else 'power' for i,s,o in rows]
    a=[np.random.default_rng(np.random.SeedSequence([i,s,phase])) for i,s,o in rows]
    b=[np.random.default_rng(np.random.SeedSequence([i,s,phase])) for i,s,o in rows]
    expected=[scripted(r[2],rng,k) for r,rng,k in zip(rows,a,kinds)]
    actual=scripted_batch([r[2] for r in rows],b,kinds)
    self.assertEqual(expected,actual)
    self.assertEqual([r.bit_generator.state for r in a],[r.bit_generator.state for r in b])
    actions=[[None,None] for _ in states]
    for (i,s,o),action in zip(rows,actual):actions[i][s]=action
    states=v.step(actions)

if __name__=='__main__':unittest.main()
