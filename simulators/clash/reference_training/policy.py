"""Stateless factorized neural policy. Only public observations and masks enter."""
import torch
from torch import nn

class Policy(nn.Module):
 def __init__(self,c):
  super().__init__();w=c.get('width',96)
  self.body=nn.Sequential(nn.Linear(292,w),nn.Tanh(),nn.Linear(w,w),nn.Tanh())
  self.slots=nn.Linear(w,5);self.tiles=nn.Linear(w,2304);self.value=nn.Linear(w,1)
 def forward(self,o,mask):
  o=o.float();h=o[:,:224];e=o[:,224:].reshape(-1,200,34)
  alive=torch.arange(200,device=o.device)[None,:]<h[:,48,None]
  # Unknown enemy fields remain as supplied; never reconstruct hidden hand/RNG.
  scales=h.new_full((224,),1000.);scales[2:4]=100000.;scales[11:15]=100000.
  scales[36:48]=5000.;scales[0]=6000.;scales[55]=300000.
  x=torch.tanh(e/4000.)*alive[:,:,None]
  f=torch.cat((torch.tanh(h/scales),x.sum(1)/alive.sum(1).clamp_min(1)[:,None],x.amax(1)),1)
  z=self.body(f);slot=self.slots(z);legal=mask[:,1:2305].bool().reshape(-1,4,576)
  tile=self.tiles(z).reshape(-1,4,576).masked_fill(~legal,-1e9).log_softmax(-1)
  logits=torch.cat((slot[:,:1],(slot[:,1:,None]+tile).flatten(1),slot.new_full((len(z),2),-1e9)),1)
  return logits.masked_fill(~mask.bool(),-1e9),self.value(z).squeeze(-1)
def create(c):return Policy(c)
def act(model,observation,mask):
 logits,_=model(observation,mask);tiles=logits[:,1:2305].reshape(-1,4,576)
 marginal=torch.cat((logits[:,:1],tiles.logsumexp(-1)),1);slot=marginal.argmax(-1)
 tile=tiles.argmax(-1).gather(1,(slot-1).clamp_min(0)[:,None]).squeeze(1)
 return torch.where(slot==0,0,1+(slot-1)*576+tile)
