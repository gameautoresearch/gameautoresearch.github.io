import json,time
from pathlib import Path
import torch,pytest,importlib
from simulators.clash.autoresearch.common import sha

@pytest.mark.parametrize('variant',['v5','v6','v7','v8','v9','v10','v11','v12','v13'])
def test_explicit_warmstart_and_live_demo_resume(tmp_path,variant):
 from simulators.clash.reference_training.v4.run import run as parent_run
 run=importlib.import_module('simulators.clash.reference_training.'+variant+'.run').run
 c=json.loads(Path('simulators/clash/reference_training/v4/recipe.json').read_text());c.update(device='cpu',envs=5,rollout_steps=2,minibatch=10,epochs=1,teacher_updates=1,updates=4)
 p=tmp_path/'parent.json';p.write_text(json.dumps(c));parent_run(p,tmp_path/'parent',time.time()+120,proof=True);checkpoint=tmp_path/'parent/checkpoint.pt'
 c.update(seed=79194,teacher_coef=.25,teacher_updates=8,history_limit=40)
 if variant=='v12':c.update(ppo_coef=0.,value_coef=0.,entropy_coef=0.,learner_state_teacher_mix=.5)
 p=tmp_path/'new.json';p.write_text(json.dumps(c));result=run(p,tmp_path/'new',time.time()+120,proof=True,warmstart=checkpoint,warmstart_sha=sha(checkpoint))
 receipt=result['warmstart'];assert receipt['optimizer_reset'] and receipt['learner_worlds_reset']
 if variant in ('v10','v11','v12','v13'):
  assert receipt['function_preserving_binary_input_reparameterization'] and receipt['exact_logit_value_and_action_parity_rows']>=30
  assert not receipt['architecture_decoder_source_identical'] and receipt['parent_policy_identity']!=result['initial_policy_identity']
  from simulators.clash.reference_training.export_adapter import derive
  from simulators.clash.autoresearch.bundle import seal
  from simulators.clash.autoresearch.readiness_contract import admit
  derive(tmp_path/'new/final',tmp_path/'derived');seal(tmp_path/'derived',tmp_path/'bundle',{'scope':'conditioning test'})
  assert admit(tmp_path/'bundle')['source_review']['greedy_entry_checked']
 else:assert receipt['architecture_decoder_source_identical']
 assert (receipt['parent_policy_identity']==result['initial_policy_identity'] or receipt.get('exact_logit_value_and_action_parity_rows',0)>0) and result['proof']['exact_next_update_resume']
 state=torch.load(tmp_path/'new/checkpoint.pt',map_location='cpu',weights_only=True);assert state['format']=='clash-reference-'+variant and state['demo_engine'] is not None
 if variant=='v13':
  assert receipt['function_preserving_zero_residual_features'] and bool(state['model']['status_projection.weight'].abs().max()>0)
 if variant=='v12':
  import numpy as np
  assert result['ppo_optimizer_updates']==0 and result['supervised_optimizer_updates']==4
  with np.load(tmp_path/'new/initial/weights.npz') as w:
   assert torch.equal(torch.from_numpy(w['value.weight']),state['model']['value.weight']) and torch.equal(torch.from_numpy(w['value.bias']),state['model']['value.bias'])

def test_history_sampling_preserves_scripts_incumbent_recent():
 from simulators.clash.reference_training.v5.league import pfsp
 p=pfsp(torch.zeros(25),torch.zeros(25));assert torch.allclose(p[:4].sum(),torch.tensor(.5)) and torch.allclose(p[4:].sum(),torch.tensor(.5))
 assert p[4]>p[5] and p[-1]>p[5] and bool((p>0).all())

def test_deployment_gate_fits_true_prior_without_losing_conditional_labels():
 from simulators.clash.reference_training.v5.learner import update as balanced
 from simulators.clash.reference_training.v6.learner import update as calibrated
 class Gate(torch.nn.Module):
  def __init__(self):super().__init__();self.bias=torch.nn.Parameter(torch.tensor(4.).log())
  def forward(self,o,m):
   logits=self.bias.new_full((len(o),2307),-1e9);logits[:,[1,577,1153,1729]]=0.;logits[:,0]=self.bias
   return logits,logits.new_zeros(len(o))
 mask=torch.zeros((1,20,2307),dtype=torch.uint8);mask[:,:,[0,1,577,1153,1729]]=1
 obs=torch.zeros((1,20,7024),dtype=torch.int32);target=torch.zeros((1,20),dtype=torch.long);target[0,:2]=1
 sample={'obs':obs,'mask':mask,'action':torch.zeros_like(target),'logp':torch.zeros((1,20)),'value':torch.zeros((1,20)),'reward':torch.zeros((1,20)),'done':torch.ones((1,20),dtype=torch.bool),'bootstrap':torch.zeros((1,20)),'demo_obs':obs,'demo_mask':mask,'demo_teacher':target}
 c={'gamma':.999,'gae_lambda':.97,'epochs':1,'minibatch':20,'clip':.2,'value_coef':0.,'entropy_coef':0.,'max_grad_norm':10.}
 old,new=Gate(),Gate()
 balanced(old,torch.optim.SGD(old.parameters(),lr=.1),sample,torch.zeros(20),c,1.)
 calibrated(new,torch.optim.SGD(new.parameters(),lr=.1),sample,torch.zeros(20),c,1.)
 assert new.bias>old.bias+.02 # True10% deployment prior reduces predicted deployment.

def test_standard_demonstrator_has_identical_teacher_off_learning(tmp_path):
 from simulators.clash.reference_training.v4.run import run as parent_run
 from simulators.clash.reference_training.v5.run import run as baseline
 from simulators.clash.reference_training.v7.run import run as standard
 from simulators.clash.reference_training.v4.run import equal
 c=json.loads(Path('simulators/clash/reference_training/v4/recipe.json').read_text());c.update(device='cpu',envs=5,rollout_steps=2,minibatch=10,epochs=1,teacher_updates=1,updates=4)
 p=tmp_path/'parent.json';p.write_text(json.dumps(c));parent_run(p,tmp_path/'parent',time.time()+120,proof=True);checkpoint=tmp_path/'parent/checkpoint.pt'
 c.update(seed=79194,teacher_coef=0.,teacher_updates=1024,history_limit=40,learning_rate=.0001,entropy_coef=.005)
 p=tmp_path/'new.json';p.write_text(json.dumps(c))
 for name,run in [('baseline',baseline),('standard',standard)]:run(p,tmp_path/name,time.time()+120,proof=True,warmstart=checkpoint,warmstart_sha=sha(checkpoint))
 a,b=[torch.load(tmp_path/name/'checkpoint.pt',map_location='cpu',weights_only=True) for name in ('baseline','standard')]
 for key in ('model','optimizer','outputs','rng','demo_rng','league','identities','wins','games','assignment','seat','samples','steps','losses'):assert equal(a[key],b[key]),key

@pytest.mark.parametrize('variant',['v8','v11'])
def test_learner_visited_state_supervision_actually_updates_its_branch(variant):
 from simulators.clash.reference_training.v7.learner import update as separate
 combined=importlib.import_module('simulators.clash.reference_training.'+variant+'.learner').update
 class Groups(torch.nn.Module):
  def __init__(self):super().__init__();self.bias=torch.nn.Parameter(torch.zeros(2))
  def forward(self,o,m):
   logits=self.bias.new_full((len(o),2307),-1e9);logits[:,[1,577,1153,1729]]=0.;logits[:,0]=self.bias[o[:,0].long()]
   return logits,logits.new_zeros(len(o))
 mask=torch.zeros((1,20,2307),dtype=torch.uint8);mask[:,:,[0,1,577,1153,1729]]=1
 obs=torch.zeros((1,20,7024),dtype=torch.int32);demo=obs.clone();demo[:,:,0]=1;target=torch.zeros((1,20),dtype=torch.long)
 sample={'obs':obs,'mask':mask,'action':target,'logp':torch.zeros((1,20)),'value':torch.zeros((1,20)),'reward':torch.zeros((1,20)),'done':torch.ones((1,20),dtype=torch.bool),'bootstrap':torch.zeros((1,20)),'teacher':target,'demo_obs':demo,'demo_mask':mask,'demo_teacher':torch.ones_like(target)}
 c={'gamma':.999,'gae_lambda':.97,'epochs':1,'minibatch':20,'clip':.2,'value_coef':0.,'entropy_coef':0.,'max_grad_norm':10.,'learner_state_teacher_mix':.5}
 a,b=Groups(),Groups()
 separate(a,torch.optim.SGD(a.parameters(),lr=.1),sample,torch.zeros(20),c,1.)
 combined(b,torch.optim.SGD(b.parameters(),lr=.1),sample,torch.zeros(20),c,1.)
 assert a.bias[0]==0 and b.bias[0]>.02

def test_entropy_does_not_reward_a_card_merely_for_many_placement_tiles():
 from simulators.clash.reference_training.v7.learner import update as joint
 from simulators.clash.reference_training.v9.learner import update as macro
 class UniformMacros(torch.nn.Module):
  def __init__(self):super().__init__();self.wait=torch.nn.Parameter(torch.tensor(0.))
  def forward(self,o,m):
   logits=self.wait.new_full((len(o),2307),-1e9);logits[:,1:2305]=-torch.tensor(576.).log();logits[:,0]=self.wait
   return logits,logits.new_zeros(len(o))
 obs=torch.zeros((1,20,7024),dtype=torch.int32);mask=torch.zeros((1,20,2307),dtype=torch.uint8);mask[:,:,:2305]=1;zero=torch.zeros((1,20))
 data={'obs':obs,'mask':mask,'action':zero.long(),'logp':zero,'value':zero,'reward':zero,'done':torch.ones_like(zero,dtype=torch.bool),'bootstrap':zero}
 c={'gamma':.999,'gae_lambda':.97,'epochs':1,'minibatch':20,'clip':.2,'value_coef':0.,'entropy_coef':.1,'max_grad_norm':10.}
 a,b=UniformMacros(),UniformMacros()
 joint(a,torch.optim.SGD(a.parameters(),lr=.1),data,torch.zeros(20),c,0.)
 macro(b,torch.optim.SGD(b.parameters(),lr=.1),data,torch.zeros(20),c,0.)
 assert a.wait<-.001 and b.wait.abs()<1e-6


def test_supervised_gate_uses_true_prior_without_tile_count_pressure():
 from simulators.clash.reference_training.v12.learner import calibrated_loss
 bias=torch.nn.Parameter(torch.tensor(4.).log());logits=bias.new_full((20,2307),-1e9);logits[:,[1,577,1153,1729]]=0.;logits[:,0]=bias
 mask=logits>-1e8;target=torch.zeros(20,dtype=torch.long);target[:2]=1
 loss=calibrated_loss(logits,mask,target);loss.backward()
 assert torch.isfinite(loss) and bias.grad<-.3 # gradient descent raises wait logit for true10%deployment prior
