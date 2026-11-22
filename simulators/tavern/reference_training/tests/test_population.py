import sys,time
from pathlib import Path
import numpy as np
import pytest
import torch
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT))
from simulators.tavern.reference_training.population import Population,schedule,weight_hash
from simulators.tavern.reference_training import collection
from simulators.tavern.reference_training.train import rng_state,restore_rng
from model import Policy

def test_schedule_is_repeatable_rotates_every_seat_and_has_history():
 a=schedule(17,0,8,3)
 assert np.array_equal(a,schedule(17,0,8,3))
 assert ((a==-1).sum(0)==4).all()
 assert ((a==-1).sum(1)==4).all()
 assert (a>0).any(1).all() and (a==0).any(1).all()
 assert not np.array_equal(a,schedule(17,1,8,3))

def test_population_frozen_hash_and_rng_restore():
 torch.set_num_threads(1);p=Policy(hidden=8,embedding=1);pop=Population(p.arch);pop.add(p,0)
 before=pop.entries[0]['sha256']
 with torch.no_grad():next(p.parameters()).add_(1)
 pop.add(p,1);assert pop.entries[0]['sha256']==before
 r=rng_state();expected=torch.rand(4);restore_rng(r);assert torch.equal(expected,torch.rand(4))
 restored=Population.restore(pop.state_dict());assert [e['sha256'] for e in restored.entries]==[e['sha256'] for e in pop.entries]
 restored.entries[0]['policy']['actor.bias'][0]+=1
 with pytest.raises(ValueError):Population.restore(restored.state_dict())

@pytest.mark.parametrize('deferred_terminal',[False,True])
def test_population_never_trains_history_or_unexecuted_rows(monkeypatch,deferred_terminal):
 class Env:
  n_slots=8
  def __init__(self,*a,**kw):self.index=0
  def observe(self,*a,**kw):
   f=np.zeros((8,3227),np.float32);f[:,0]=self.index
   m=np.zeros((8,245),bool);m[:,self.index]=True
   return dict(features=f,cards=np.zeros((8,124),np.int32),action_mask=m,ready=np.ones(8,bool))
  def step(self,actions,**kw):
   i=self.index;self.index+=1;deferred=i==1;done=deferred if deferred_terminal else i==2
   reward=1. if done else 0.
   return np.full(8,reward,np.float32),np.full(8,done),dict(placement=np.full(8,1 if done else 0),stop_reason=np.full(8,1 if done else 0),rejected=np.zeros(8,bool),applied=np.full(8,not deferred),deferred=np.full(8,deferred))
  def save(self,*a):return 'credit-fixture'
 class Actor:pass
 current=Actor();historical=Actor();calls=[]
 def act(p,obs,indices,*a):
  calls.append((p is current,len(indices)));action=obs['action_mask'][indices].argmax(-1)
  return action,np.zeros(len(indices)),np.zeros(len(indices))
 monkeypatch.setattr(collection.tg,'BatchEnv',Env);monkeypatch.setattr(collection,'act',act)
 c=dict(n_games=1,learning_seats=list(range(8)),opponent='selfplay',native_threads=1,rules={},max_batch_steps=10)
 assignment=np.array([[-1,0,-1,1,-1,1,-1,1]])
 eps,records,counts,_=collection.collect(current,c,17,torch.device('cpu'),collection.Timer('cpu'),time.perf_counter()+10,assignment=assignment,population=[historical])
 assert len(eps)==4 and {r['seat'] for r in records}=={0,2,4,6}
 assert all([r['action'] for r in ep]==([0] if deferred_terminal else [0,2]) for ep in eps)
 assert all(sum(r['reward'] for r in ep)==1 for ep in eps)
 assert counts['unapplied_learning_decisions']==4
 assert any(not is_current for is_current,n in calls)

def test_deadline_discards_partial_cohort_and_exports_last_completed(tmp_path,monkeypatch):
 import json
 from simulators.tavern.reference_training import train
 config=json.loads((ROOT/'simulators/tavern/reference_training/recipe.json').read_text())
 config.update(device='cpu',n_games=1,native_threads=1,network=dict(hidden=8,embedding=1))
 def expired(*args,**kw):raise collection.BudgetExpired('synthetic deadline during cohort')
 monkeypatch.setattr(train,'collect',expired)
 ck=train.run(config,tmp_path/'deadline',10)
 result=json.loads((tmp_path/'deadline/result.json').read_text())
 assert result['termination']=='deadline_partial_cohort_discarded' and result['update']==0
 saved=torch.load(ck,map_location='cpu',weights_only=False)
 assert saved['history']==[] and saved['update']==0
