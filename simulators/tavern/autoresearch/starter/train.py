"""Editable training loop; frozen harness controls rules, exports and wall deadline."""
import time
from rollout import collect,BudgetExpired
from learner import update

def train(model,optimizer,config,device,timer,deadline,start_update=0,on_update=None):
 for i in range(start_update,config['updates']):
  if time.perf_counter()>=deadline:break
  try:episodes,records,counts,states=collect(model,config,(config['seed']+i*1000003)%2**64,device,timer,deadline)
  except BudgetExpired:break
  metrics=update(model,optimizer,episodes,config,device)
  if on_update:on_update(i+1,dict(learning=metrics,episodes=records,counts=counts),states)
