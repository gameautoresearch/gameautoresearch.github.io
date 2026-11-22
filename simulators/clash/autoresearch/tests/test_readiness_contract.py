from pathlib import Path
import json
import numpy as np
import pytest
from simulators.clash.autoresearch.readiness_contract import review_policy

def test_direct_greedy_entry_and_no_IO(tmp_path):
 p=tmp_path/'policy.py';p.write_text('import torch\ndef act(model,o,m):\n return model(o,m)[0].argmax(-1)\n')
 assert review_policy(p)['greedy_entry_checked']
 p.write_text('import os\ndef act(model,o,m):\n return model(o,m)[0].argmax(-1)\n')
 with pytest.raises(ValueError,match='import'):review_policy(p)
 p.write_text('def act(model,o,m):\n return 0\n')
 with pytest.raises(ValueError,match='argmax'):review_policy(p)

def test_scripted_games_are_independent_of_batch_order():
 from simulators.clash.autoresearch.common import DATA,setup_native
 from simulators.clash.autoresearch.evaluation_v2 import Actor
 setup_native()
 from cpu_batch import NativeCPUBatch
 data=json.loads(DATA.read_text());env=NativeCPUBatch(list(range(8)),threads=1)
 try:
  for _ in range(5):env.step(np.zeros((8,2),np.int32))
  x=env.observations[:,0].copy();m=env.masks[:,0].copy();contexts=[(i,0) for i in range(8)]
  a=Actor({'name':'standard','kind':'scripted'},data,1,True);b=Actor({'name':'standard','kind':'scripted'},data,1,True)
  chosen=a.choose(x,m,contexts);reverse=b.choose(x[::-1],m[::-1],contexts[::-1]);assert chosen==reverse[::-1]
  chosen=a.choose(x,m,contexts);reverse=b.choose(x[::-1],m[::-1],contexts[::-1]);assert chosen==reverse[::-1]
 finally:env.close()

@pytest.mark.parametrize('message,kind',[('policy inference read deadline','policy_timeout'),('illegal policy response','illegal_action'),('simulator fault','simulator_fault'),('bundle manifest drift','entrant_admission_failure'),('natural horizon watchdog','administrative_cap'),('provider connection reset','infrastructure_fault')])
def test_failure_categories_never_become_natural_losses(message,kind):
 from simulators.clash.autoresearch.evaluation_v2 import classify
 assert classify(RuntimeError(message))==kind

def test_invalid_arena_writes_sidecar_not_cacheable_grade(tmp_path,monkeypatch):
 from simulators.clash.autoresearch import arena
 from simulators.clash.autoresearch.bundle import seal
 from simulators.clash.autoresearch.common import dump
 source=tmp_path/'source';source.mkdir()
 for name in ('policy.py','learner.py','train.py'):(source/name).write_text('# synthetic fixture; never imported\n')
 dump(source/'config.json',{});np.savez(source/'weights.npz',test=np.zeros(1,np.float32));bundle=tmp_path/'bundle';seal(source,bundle)
 class Hung:
  def __init__(self,*args,**kwargs):pass
  def choose(self,*args):raise RuntimeError('policy inference read deadline')
  def close(self):pass
 monkeypatch.setattr(arena,'PolicyProcess',Hung);schedule=tmp_path/'schedule.json';dump(schedule,{'engine_seeds':[1],'policy_seed':1,'opponents':[{'name':'wait','kind':'wait'}]});target=tmp_path/'grade.json'
 with pytest.raises(RuntimeError,match='deadline'):arena.evaluate(bundle,schedule,target,development=True)
 assert not target.exists();failure=json.loads(target.with_suffix('.failure.json').read_text());assert failure['raw_score_0_100'] is None and failure['failure_kind']=='policy_timeout' and failure['completed_natural_games']==0
