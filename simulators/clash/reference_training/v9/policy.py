"""Stateless spatial policy. Gaussian public-entity features feed a small CNN.

The grid is the pinned public action grid: 18x32,1000 observed units per tile.
Every valid entity is encoded; unknown positions remain in the global encoder.
There is no engine fork, search, private hand or recurrent state.
"""
import torch
from torch import nn
class Policy(nn.Module):
 def __init__(self,c):
  super().__init__();w=c.get('width',128)
  self.cards=nn.Embedding(2049,12,padding_idx=0)
  self.entities=nn.Sequential(nn.Linear(24,16),nn.Tanh(),nn.Linear(16,16),nn.Tanh())
  self.body=nn.Sequential(nn.Linear(224+48+96,w),nn.Tanh(),nn.Linear(w,w),nn.Tanh())
  self.cnn=nn.Sequential(nn.Conv2d(48,32,3,padding=1),nn.GroupNorm(8,32),nn.Tanh(),nn.Conv2d(32,32,3,padding=2,dilation=2),nn.GroupNorm(8,32),nn.Tanh(),nn.Conv2d(32,32,3,padding=4,dilation=4),nn.GroupNorm(8,32),nn.Tanh())
  self.position=nn.Embedding(576,32)
  self.query=nn.Sequential(nn.Linear(w+12+1,64),nn.Tanh(),nn.Linear(64,32),nn.Tanh())
  self.slot=nn.Linear(32,1);self.wait=nn.Linear(w,1);self.value=nn.Linear(w,1)
 def forward(self,o,mask):
  h=o[:,:224].float();e=o[:,224:].reshape(-1,200,34);f=e.float();n=len(o)
  valid=torch.arange(200,device=o.device)[None,:]<o[:,48,None]
  numeric=torch.stack((f[:,:,4]/18000,f[:,:,5]/32000,f[:,:,6]/5000,f[:,:,7]/5000,f[:,:,6]/f[:,:,7].clamp_min(1),f[:,:,8]/5000,f[:,:,1]/4,f[:,:,2],f[:,:,9]/8,f[:,:,12]/18000,f[:,:,13]/32000,f[:,:,17]/1000),-1)
  z=self.entities(torch.cat((self.cards((e[:,:,3].long()+1).clamp(0,2048)),torch.tanh(numeric)),-1))
  tiles=torch.arange(576,device=o.device);gx=(tiles%18).float()+.5;gy=(tiles//18).float()+.5
  dx=f[:,:,4,None]/1000-gx[None,None,:];dy=f[:,:,5,None]/1000-gy[None,None,:]
  kernel=torch.exp(-(dx.square()+dy.square())/(2*1.25**2))
  positioned=valid&(e[:,:,4]>=0)&(e[:,:,5]>=0);kernel=kernel*positioned[:,:,None]
  pools=[];boards=[]
  for side in (0,1,2):
   belongs=(e[:,:,2]==side) if side<2 else ((e[:,:,2]!=0)&(e[:,:,2]!=1))
   alive=valid&belongs;x=z*alive[:,:,None]
   pools.extend((x.sum(1)/alive.sum(1).clamp_min(1)[:,None],x.amax(1)))
   boards.append(torch.einsum('net,nec->nct',kernel,x).reshape(n,16,32,18))
  scales=h.new_full((224,),1000.);scales[2:4]=100000.;scales[11:15]=100000.;scales[36:48]=5000.;scales[0]=6000.;scales[55]=300000.
  hand=self.cards((o[:,7:11].long()+1).clamp(0,2048));context=self.body(torch.cat((torch.tanh(h/scales),hand.flatten(1),*pools),1))
  query=self.query(torch.cat((context[:,None,:].expand(-1,4,-1),hand,h[:,11:15,None]/100000),-1))
  board=self.cnn(torch.cat(boards,1)).flatten(2)+self.position(tiles).T[None,:,:]
  placement=torch.einsum('nhc,nct->nht',query,board)/(32**.5)
  legal=mask[:,1:2305].bool().reshape(-1,4,576);placement=placement.masked_fill(~legal,-1e9).log_softmax(-1)
  slots=self.slot(query).squeeze(-1)
  logits=torch.cat((self.wait(context),(slots[:,:,None]+placement).flatten(1),context.new_full((n,2),-1e9)),1)
  return logits.masked_fill(~mask.bool(),-1e9),self.value(context).squeeze(-1)
def create(c):return Policy(c)
def act(model,observation,mask):
 logits,_=model(observation,mask);tiles=logits[:,1:2305].reshape(-1,4,576)
 marginal=torch.cat((logits[:,:1],tiles.logsumexp(-1)),1);slot=marginal.argmax(-1)
 tile=tiles.argmax(-1).gather(1,(slot-1).clamp_min(0)[:,None]).squeeze(1)
 return torch.where(slot==0,0,1+(slot-1)*576+tile)
