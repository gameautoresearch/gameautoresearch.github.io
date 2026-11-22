"""Competent incumbent/history population and deterministic script curriculum."""
import copy
import numpy as np
import torch
from .policy import Policy
from ..population import weight_hash
class Population:
 def __init__(self,arch,capacity=8):self.arch=arch;self.capacity=capacity;self.entries=[];self.cache={}
 def add_state(self,state,step,kind,parent=None):
  state={k:v.detach().cpu().clone() for k,v in state.items()};digest=weight_hash(state)
  if any(e['sha256']==digest for e in self.entries):return
  self.entries.append(dict(policy=state,update=step,kind=kind,sha256=digest,parent=parent))
  if len(self.entries)>self.capacity:
   removed=self.entries.pop(1);self.cache.pop(removed['sha256'],None)
 def add(self,model,step):self.add_state(model.state_dict(),step,'campaign-history')
 def actors(self,device):
  actors=[]
  for item in self.entries:
   if item['sha256'] not in self.cache:
    with torch.random.fork_rng(devices=[]):p=Policy(**self.arch).to(device)
    p.load_state_dict(item['policy']);p.eval();self.cache[item['sha256']]=p
   actors.append(self.cache[item['sha256']])
  return actors
 def state_dict(self):return dict(architecture=self.arch,capacity=self.capacity,entries=copy.deepcopy(self.entries))
 @classmethod
 def restore(cls,s):
  p=cls(s['architecture'],s['capacity']);p.entries=s['entries']
  if any(weight_hash(e['policy'])!=e['sha256'] for e in p.entries):raise ValueError('Population hash mismatch')
  return p

def schedule(seed,update,games,pool_size):
 rng=np.random.default_rng(np.random.SeedSequence([seed,update,5317]));a=np.empty((games,8),np.int64)
 for g in range(games):
  rotation=(update+g)%8
  for seat in range(8):
   rel=(seat-rotation)%8
   if rel%2==0:a[g,seat]=-1
   elif rel==7:a[g,seat]=1 if rng.random()<.5 else int(rng.integers(1,pool_size+1))
   else:a[g,seat]=pool_size+1+((rel//2+g+update)%3)
 return a
