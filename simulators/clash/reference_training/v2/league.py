"""Visible curriculum and episode-fixed smoothed PFSP; no engine snapshots read."""
import hashlib
import torch

def tensor_identity(state):
 h=hashlib.sha256()
 for k,v in sorted(state.items()):h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()

def pfsp(wins,games):
 # Beta(1,1) regularization and a floor preserve easy/historical coverage.
 p=(wins+1)/(games+2);w=.1+(1-p).square();return w/w.sum()

def visible_teacher(o,mask,pressure=False):
 """Simple curriculum bot: defend visible intruders; build Giant pushes.
 This is an intermediate teaching policy, not the independently pinned script.
 """
 n=len(o);device=o.device;legal=mask[:,1:2305].bool().reshape(n,4,576)
 hand=o[:,7:11];cost=o[:,11:15];troop=(hand==810)|(hand==15)|(hand==959)|(hand==527)|(hand==901)
 e=o[:,224:].reshape(n,200,34);alive=torch.arange(200,device=device)[None,:]<o[:,48,None]
 enemy=alive&(e[:,:,1]==1)&(e[:,:,2]==1)&(e[:,:,6]>0)
 threat=enemy&(e[:,:,5]<32000);depth=e[:,:,5].masked_fill(~threat,100000).argmin(-1)
 tx=e[:,:,4].gather(1,depth[:,None]).squeeze(1);danger=threat.any(-1)
 lane=torch.where(o[:,44]<=o[:,46],3,14);lane=torch.where(danger,torch.where(tx<18000,3,14),lane)
 row=torch.where(danger,4,torch.where((hand==527).any(-1),1,14))
 available=troop&legal.any(-1);rank=cost.float()/10000
 rank=rank-torch.where((hand==527)&~danger[:,None],100.,0.)
 slot=rank.masked_fill(~available,1e9).argmin(-1)
 target=row*18+lane;tile_index=torch.arange(576,device=device)
 distance=((tile_index%18)[None,:]-lane[:,None]).square()+((tile_index//18)[None,:]-row[:,None]).square()
 slotlegal=legal[torch.arange(n,device=device),slot]
 tile=distance.masked_fill(~slotlegal,1000000).argmin(-1)
 action=1+slot*576+tile
 ready=(o[:,2]-o[:,3]>= (50000 if pressure else 80000))|danger
 return torch.where(available.any(-1)&ready&(o[:,53]==0),action,0)
