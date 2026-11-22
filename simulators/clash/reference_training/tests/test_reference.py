import json,time
from pathlib import Path
import pytest,torch
from simulators.clash.reference_training import policy
from simulators.clash.reference_training.league import pfsp,visible_teacher
from simulators.clash.reference_training.run import run,restore,binding
from simulators.clash.autoresearch.common import setup_native

def test_pfsp_keeps_easy_history():
 weights=pfsp(torch.tensor([0.,100.,50.]),torch.tensor([100.,100.,100.]))
 assert weights[0]>weights[2]>weights[1]>0
 assert torch.allclose(weights.sum(),torch.tensor(1.))

def test_real_learning_resume_and_greedy_legality(tmp_path):
 c=json.loads(Path('simulators/clash/reference_training/recipe.json').read_text())
 c.update(device='cpu',envs=3,rollout_steps=2,minibatch=6,epochs=1,teacher_updates=1,history_every=1,updates=4)
 path=tmp_path/'config.json';path.write_text(json.dumps(c))
 result=run(path,tmp_path/'run',time.time()+180,proof=True)
 assert result['finite_losses'] and result['initial_policy_identity']!=result['final_policy_identity']
 assert result['proof']['exact_next_update_resume'] and result['parameters']<1000000
 assert result['neural_opponent_decisions']>0 and result['ppo_only_updates']>0
 # Neither artifact nor recipe can be relabeled across changed source/config pins.
 setup_native();from cpu_batch import NativeCPUBatch
 from simulators.clash.reference_training.trainer import Trainer
 env=NativeCPUBatch([1,2,3],threads=1)
 try:
  t=Trainer(c,env,'cpu');pin=binding(c,env);pin['source']['policy.py']='changed'
  with pytest.raises(ValueError,match='binding'):restore(t,tmp_path/'run/checkpoint.pt',pin)
 finally:env.close()

def test_teacher_masks_and_full_entity_rows():
 torch.manual_seed(7);model=policy.create({'width':32})
 o=torch.zeros(2,7024,dtype=torch.int32);m=torch.zeros(2,2307,dtype=torch.uint8);m[:,0]=1
 o[:,48]=200;o[:,7]=527;o[:,11]=50000;o[:,2]=100000;m[:,1:577]=1
 a=visible_teacher(o,m);assert m[torch.arange(2),a].all()
 x=policy.act(model,o,m);assert torch.equal(x,policy.act(model,o,m)) and m[torch.arange(2),x].all()
 before=model(o,m)[0];o[:,224+199*34+6]=4000
 assert not torch.equal(before,model(o,m)[0])

def test_derived_head_meets_direct_argmax_admission(tmp_path):
 from simulators.clash.reference_training.export_adapter import derive
 from simulators.clash.autoresearch.bundle import seal
 from simulators.clash.autoresearch.readiness_contract import admit
 import numpy as np
 c={'width':32};original=tmp_path/'original';original.mkdir();model=policy.create(c)
 source=Path('simulators/clash/reference_training')
 for name in ('policy.py','learner.py'):__import__('shutil').copyfile(source/name,original/name)
 (original/'train.py').write_text('"""Inference export fixture."""\n');(original/'config.json').write_text(json.dumps(c));(original/'lineage.json').write_text('{}')
 np.savez_compressed(original/'weights.npz',**{k:v.detach().numpy().astype(np.float32) for k,v in model.state_dict().items()})
 receipt=derive(original,tmp_path/'derived');assert receipt['weights_unchanged'] and receipt['public_fixture_rows_exact']==512
 manifest=seal(tmp_path/'derived',tmp_path/'bundle',receipt);admission=admit(tmp_path/'bundle')
 assert admission['source_review']['greedy_entry_checked'] and manifest['parameters']<1000000

def test_public_teacher_variant_exact_resume_and_control_labels(tmp_path):
 from simulators.clash.reference_training.v2.run import run as run_v2
 c=json.loads(Path('simulators/clash/reference_training/v2/recipe.json').read_text())
 c.update(device='cpu',envs=4,rollout_steps=2,minibatch=8,epochs=1,teacher_updates=1,history_every=1,updates=4,teacher_coef=0.)
 config=tmp_path/'c.json';config.write_text(json.dumps(c));result=run_v2(config,tmp_path/'out',time.time()+120,proof=True)
 assert result['proof']['exact_next_update_resume'] and result['teacher_updates']==0 and result['ppo_only_updates']==4
 assert result['parameters']<1000000 and result['neural_opponent_decisions']>0
 assert result['visible_refusal_nonzero_decisions']==0

@pytest.mark.parametrize('coefficient',[1.,0.])
def test_v3_demo_world_resume_and_separate_learner_rows(tmp_path,coefficient):
 from simulators.clash.reference_training.v3.run import run as run_v3
 c=json.loads(Path('simulators/clash/reference_training/v3/recipe.json').read_text())
 c.update(device='cpu',envs=5,rollout_steps=2,minibatch=10,epochs=1,teacher_updates=1,history_every=1,updates=4,teacher_coef=coefficient)
 config=tmp_path/'c.json';config.write_text(json.dumps(c));r=run_v3(config,tmp_path/'out',time.time()+120,proof=True)
 assert r['proof']['exact_next_update_resume'] and r['samples']==40 and r['demo_decisions']==(10 if coefficient else 0)
 assert r['parameters']<1000000 and r['finite_losses']
 state=torch.load(tmp_path/'out/checkpoint.pt',map_location='cpu',weights_only=True)
 assert 'demo_engine' in state and 'demo_outputs' in state
 assert (state['demo_engine'] is None)==(coefficient==0)
 assert r['static_opponent_names']==['standard-fixed-tie','pressure-fixed-tie','intermediate-public-v2','teacher-public-v2']

def test_selection_close_blocks_launch_before_provider_import(tmp_path,monkeypatch):
 from simulators.clash.reference_training import launch
 closed=tmp_path/'clash/rl/selection-close.json';closed.parent.mkdir(parents=True);closed.write_text('{}')
 monkeypatch.setattr(launch,'CAMPAIGN',tmp_path)
 with pytest.raises(ValueError,match='Selection'):
  launch.launch('must-not-reserve-or-create')
 with pytest.raises(ValueError,match='Selection'):
  launch.recover_prelaunch('must-not-recover','app','image')

def test_resume_while_expert_trajectory_is_still_advancing(tmp_path):
 from simulators.clash.reference_training.v3.trainer import Trainer
 from simulators.clash.reference_training.v3.run import save,restore,binding,exact_state,equal
 setup_native();from cpu_batch import NativeCPUBatch
 c=json.loads(Path('simulators/clash/reference_training/v3/recipe.json').read_text());c.update(device='cpu',envs=5,rollout_steps=2,minibatch=10,epochs=1,teacher_updates=8)
 env=NativeCPUBatch([1,2,3,4,5],threads=1);t=Trainer(c,env,'cpu');pin=binding(c,env)
 try:
  t.step();save(t,tmp_path/'mid-teacher.pt',pin);row=t.step();expected=exact_state(t)
  restore(t,tmp_path/'mid-teacher.pt',pin);resumed=t.step();actual=exact_state(t)
  assert equal(expected,actual) and row==resumed and t.demo_decisions==20
 finally:env.close();t.demo_env.close()

def test_spatial_candidate_learning_resume_and_admission(tmp_path):
 from simulators.clash.reference_training.v4.run import run as run_v4
 from simulators.clash.reference_training.export_adapter import derive
 from simulators.clash.autoresearch.bundle import seal
 from simulators.clash.autoresearch.readiness_contract import admit
 c=json.loads(Path('simulators/clash/reference_training/v4/recipe.json').read_text());c.update(device='cpu',envs=5,rollout_steps=2,minibatch=10,epochs=1,teacher_updates=1,history_every=1,updates=4)
 p=tmp_path/'c.json';p.write_text(json.dumps(c));r=run_v4(p,tmp_path/'out',time.time()+180,proof=True)
 assert r['parameters']<1000000 and r['proof']['exact_next_update_resume'] and r['finite_losses']
 receipt=derive(tmp_path/'out/final',tmp_path/'derived');seal(tmp_path/'derived',tmp_path/'bundle',receipt)
 assert admit(tmp_path/'bundle')['source_review']['greedy_entry_checked']
 assert r['binding']['runtime']['cudnn_allow_tf32'] is False

def test_refusal_observer_preserves_exact_engine_transitions():
 import numpy as np
 from simulators.clash.reference_training.assessment_worker import install_public_refusal_observer
 setup_native();import cpu_batch
 original,recorders=install_public_refusal_observer();observed=cpu_batch.NativeCPUBatch([1000001,1000002],threads=1)
 # Replay on the same allocation: load() rebinds arena scratch pointers, so
 # snapshots from different world allocations cannot be compared byte-for-byte.
 initial=observed.state();outputs={key:getattr(observed,key).copy() for key in ('observations','masks','infos','rewards')}
 try:
  expected=[]
  for _ in range(12):
   original.step(observed,np.zeros((2,2),np.int32))
   expected.append(({key:getattr(observed,key).copy() for key in outputs},observed.state()))
  observed.load(initial)
  for key,value in outputs.items():getattr(observed,key)[:]=value
  for output,state in expected:
   observed.step(np.zeros((2,2),np.int32))
   for key in outputs:assert np.array_equal(getattr(observed,key),output[key])
   assert observed.state()==state
  assert not observed.refusal_counts.any()
 finally:observed.close();cpu_batch.NativeCPUBatch=original
