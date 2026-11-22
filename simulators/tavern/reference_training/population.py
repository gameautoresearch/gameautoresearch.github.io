"""Deterministic rotating learner seats and uniform historical matchmaking.

-1 is an on-policy learner; 0 is a current-policy nonlearner; positive IDs
refer to immutable historical snapshots. Frozen decisions never enter PPO.
"""
import copy,hashlib
import numpy as np
import torch
from .bootstrap import GAME
from model import Policy

def weight_hash(state):
 h=hashlib.sha256()
 for k,v in sorted(state.items()):h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()

def schedule(seed,update,n_games,pool_size):
 if pool_size<1:raise ValueError('Historical diversity is required')
 rng=np.random.default_rng(np.random.SeedSequence([seed,update,731]))
 a=np.empty((n_games,8),dtype=np.int64)
 for g in range(n_games):
  rotation=(update+g)%8
  for seat in range(8):
   relative=(seat-rotation)%8
   a[g,seat]=-1 if relative%2==0 else (0 if relative==1 else int(rng.integers(1,pool_size+1)))
 return a

class Population:
 def __init__(self,architecture,capacity=8):self.arch=architecture;self.capacity=capacity;self.entries=[]
 def add(self,model,update,kind='historical'):
  state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
  item=dict(update=update,kind=kind,sha256=weight_hash(state),policy=state)
  if self.entries and item['sha256']==self.entries[-1]['sha256']:return
  self.entries.append(item)
  if len(self.entries)>self.capacity:self.entries.pop(1) # preserve initial anchor
 def actors(self,device):
  result=[]
  for item in self.entries:
   with torch.random.fork_rng(devices=[]):model=Policy(**self.arch).to(device)
   model.load_state_dict(item['policy']);model.eval();result.append(model)
  return result
 def state_dict(self):return dict(architecture=self.arch,capacity=self.capacity,entries=copy.deepcopy(self.entries))
 @classmethod
 def restore(cls,state):
  p=cls(state['architecture'],state['capacity']);p.entries=state['entries']
  if any(weight_hash(e['policy'])!=e['sha256'] for e in p.entries):raise ValueError('Mutated population snapshot')
  return p
