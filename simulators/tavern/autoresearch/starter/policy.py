"""Small position-aware masked MLP; random initialization, visible inputs only."""
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
import taverngym as tg
class Policy(nn.Module):
 def __init__(self,hidden=64,embedding=4):
  super().__init__();self.arch=dict(hidden=hidden,embedding=embedding)
  self.cards=nn.Embedding(tg.N_CARDS,embedding,padding_idx=0)
  self.body=nn.Sequential(nn.Linear(tg.OBS_FLOATS+tg.OBS_CARDS*embedding,hidden),nn.Tanh())
  self.actor=nn.Linear(hidden,tg.N_ACTIONS);self.critic=nn.Linear(hidden,1)
 def forward(self,features,cards):
  f=torch.sign(features)*torch.log1p(features.abs())
  h=self.body(torch.cat((f,self.cards(cards.long()).flatten(1)),1))
  return self.actor(h),self.critic(h).squeeze(-1)
 def distribution(self,features,cards,mask):
  logits,value=self(features,cards)
  return Categorical(logits=logits.masked_fill(~mask.bool(),-1e9)),value
@torch.no_grad()
def act(policy,obs,indices,device,deterministic=False):
 if hasattr(policy,'remote_act'):return policy.remote_act(obs,indices)
 f=torch.tensor(obs['features'][indices],device=device)
 c=torch.tensor(obs['cards'][indices],device=device,dtype=torch.long)
 m=torch.tensor(obs['action_mask'][indices],device=device)
 dist,value=policy.distribution(f,c,m)
 a=dist.logits.argmax(-1) if deterministic else dist.sample()
 return a.cpu().numpy(),dist.log_prob(a).cpu().numpy(),value.cpu().numpy()
