"""Visible curriculum and episode-fixed smoothed PFSP; no engine snapshots read."""
import hashlib
import torch

def tensor_identity(state):
 h=hashlib.sha256()
 for k,v in sorted(state.items()):h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()

def pfsp(wins,games):
 # Four pinned scripts and neural opposition each retain half the mass.
 # The initial neural member is the trained incumbent, not random initialization.
 p=(wins+1)/(games+2);w=.1+(1-p).square()
 if len(w)>5:
  age=torch.full_like(w[4:],.2);age[0]=1.;age[-8:]=1.;w[4:]*=age
 w[:4]*=.5/w[:4].sum();w[4:]*=.5/w[4:].sum()
 return w/w.sum()
