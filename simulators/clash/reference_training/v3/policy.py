"""Entity-aware stateless neural policy over the unchanged 200-row observation."""
import torch
from torch import nn
class Policy(nn.Module):
 def __init__(self,c):
  super().__init__();w=c.get('width',128)
  self.cards=nn.Embedding(2049,8,padding_idx=0)
  self.entities=nn.Sequential(nn.Linear(20,32),nn.Tanh(),nn.Linear(32,32),nn.Tanh())
  self.body=nn.Sequential(nn.Linear(224+32+128,w),nn.Tanh(),nn.Linear(w,w),nn.Tanh())
  self.slots=nn.Linear(w,5);self.tiles=nn.Linear(w,2304);self.value=nn.Linear(w,1)
 def forward(self,o,mask):
  h=o[:,:224].float();e=o[:,224:].reshape(-1,200,34);f=e.float()
  valid=torch.arange(200,device=o.device)[None,:]<o[:,48,None]
  numeric=torch.stack((f[:,:,4]/18000,f[:,:,5]/32000,f[:,:,6]/5000,f[:,:,7]/5000,f[:,:,6]/f[:,:,7].clamp_min(1),f[:,:,8]/5000,f[:,:,1]/4,f[:,:,2],f[:,:,9]/8,f[:,:,12]/18000,f[:,:,13]/32000,f[:,:,17]/1000),-1)
  z=self.entities(torch.cat((self.cards((e[:,:,3].long()+1).clamp(0,2048)),torch.tanh(numeric)),-1))
  pools=[]
  for side in (0,1):
   alive=valid&(e[:,:,2]==side);x=z*alive[:,:,None]
   pools.extend((x.sum(1)/alive.sum(1).clamp_min(1)[:,None],x.amax(1)))
  scales=h.new_full((224,),1000.);scales[2:4]=100000.;scales[11:15]=100000.;scales[36:48]=5000.;scales[0]=6000.;scales[55]=300000.
  hand=self.cards((o[:,7:11].long()+1).clamp(0,2048)).flatten(1)
  body=self.body(torch.cat((torch.tanh(h/scales),hand,*pools),1));slot=self.slots(body)
  legal=mask[:,1:2305].bool().reshape(-1,4,576)
  tile=self.tiles(body).reshape(-1,4,576).masked_fill(~legal,-1e9).log_softmax(-1)
  logits=torch.cat((slot[:,:1],(slot[:,1:,None]+tile).flatten(1),slot.new_full((len(body),2),-1e9)),1)
  return logits.masked_fill(~mask.bool(),-1e9),self.value(body).squeeze(-1)
def create(c):return Policy(c)
def act(model,observation,mask):
 logits,_=model(observation,mask);tiles=logits[:,1:2305].reshape(-1,4,576)
 marginal=torch.cat((logits[:,:1],tiles.logsumexp(-1)),1);slot=marginal.argmax(-1)
 tile=tiles.argmax(-1).gather(1,(slot-1).clamp_min(0)[:,None]).squeeze(1)
 return torch.where(slot==0,0,1+(slot-1)*576+tile)
