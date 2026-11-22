"""Unexecuted responses never become PPO actions; delayed rewards keep real credit."""
import sys,time
from pathlib import Path
import numpy as np
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import rollout

@pytest.mark.parametrize('device_rows',[False,True])
@pytest.mark.parametrize('deferred_terminal',[False,True])
def test_deferred_response_removed_before_reward_credit(monkeypatch,device_rows,deferred_terminal):
 class Env:
  n_slots=1
  def __init__(self,*a,**kw):self.index=0
  def observe(self,*a,**kw):
   features=np.zeros((1,3227),np.float32);features[:,0]=self.index
   mask=np.zeros((1,245),bool);mask[:,self.index]=True
   return dict(features=features,cards=np.zeros((1,124),np.int32),action_mask=mask,ready=np.ones(1,bool))
  def step(self,actions,**kw):
   index=self.index;self.index+=1;deferred=index==1;done=deferred if deferred_terminal else index==2
   reward=(1. if done else 0.) if deferred_terminal else (.25 if deferred else .75 if done else 0.)
   return np.array([reward],np.float32),np.array([done]),dict(placement=np.array([1 if done else 0],np.uint8),stop_reason=np.array([1 if done else 0],np.uint8),rejected=np.array([False]),applied=np.array([not deferred]),deferred=np.array([deferred]))
  def save(self,*a):return 'synthetic-credit-fixture'
 class Policy:pass
 policy=Policy()
 def act(policy,obs,indices,device,deterministic):
  policy.last_actor_inputs=(torch.tensor(obs['features'][indices]),torch.tensor(obs['cards'][indices]),torch.tensor(obs['action_mask'][indices]))
  action=obs['action_mask'][indices].argmax(-1)
  return action,-.1*(action+1),np.zeros(len(indices))
 monkeypatch.setattr(rollout.tg,'BatchEnv',Env);monkeypatch.setattr(rollout,'act',act)
 config=dict(n_games=1,learning_seats=[0],opponent='scripted',native_threads=1,rules={},max_batch_steps=10,device_rollouts=device_rows)
 episodes,records,counts,_=rollout.collect(policy,config,42,torch.device('cpu'),rollout.Timer(torch.device('cpu')),time.perf_counter()+10)
 rows=episodes[0];expected_actions=[0] if deferred_terminal else [0,2]
 assert [r['action'] for r in rows]==expected_actions
 assert [r['logp'] for r in rows]==pytest.approx([-.1*(a+1) for a in expected_actions])
 assert [r['reward'] for r in rows]==pytest.approx([1.] if deferred_terminal else [.25,.75])
 assert counts['accepted_learning_samples']==len(expected_actions)
 assert counts['applied_learning_decisions']==len(expected_actions)
 assert counts['unapplied_learning_decisions']==counts['deferred']==1
 if device_rows:assert episodes.pack(rows,torch.device('cpu'))[0][:,0].tolist()==expected_actions
