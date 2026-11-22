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
