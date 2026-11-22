"""Game-side research substrate: ledger, contract binding, training entry, policy worker, v1 evaluator pipe
and action traces. The session backend is tested in harness/tests/test_tavern.py."""
from pathlib import Path
import sys,json,subprocess
import pytest
HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE))
from ledger import Ledger
from contract import validate,binding,sha

def test_ledger_holds_failures_and_closes(tmp_path):
 l=Ledger(tmp_path/'ledger.sqlite');l.reserve('failed',300)
 with pytest.raises(ValueError):l.reserve('second',10)
 with pytest.raises(ValueError):l.close()
 l.settle('failed',dict(provider_id='sb-test',started_at=100,finished_at=220))
 assert l.snapshot()['allocated_seconds']==120
 with pytest.raises(ValueError):l.reserve('too-big',481)
 l.close()
 with pytest.raises(ValueError):l.reserve('after',1)
 with pytest.raises(ValueError):Ledger(tmp_path/'ledger.sqlite',601)

def test_rules_and_visibility_binding():
 c=json.loads((HERE/'recipes/smoke.json').read_text());validate(c)
 c['rules']['num_players']=8
 with pytest.raises(ValueError):validate(c)
 b=binding();assert b['observations']==[3227,124,245] and b['benchmark_id']=='tavern-four-small-v1'
 assert b['rules']['num_players']==4 and b['rules']['max_regular_tier']==4
 assert b['rules']['lobby_races']==3 and b['rules']['hero_choices']==2
 c=json.loads((HERE/'recipes/smoke.json').read_text());c['learning_seats']=[4]
 with pytest.raises(ValueError,match='learning configuration'):validate(c)

def test_actual_learning_and_exact_resume(tmp_path):
 import torch,numpy as np
 c=json.loads((HERE/'recipes/smoke.json').read_text());c.update(n_games=2,updates=4,native_threads=1)
 recipe=tmp_path/'recipe.json';recipe.write_text(json.dumps(c))
 def run(out,resume=None,trace=False):
  cmd=[sys.executable,str(HERE/'train_entry.py'),'--source',str(HERE/'starter'),'--recipe',str(recipe),'--out',str(out),'--seconds','90']
  if resume:cmd+=['--resume',str(resume)]
  if trace:cmd+=['--record-actions']
  subprocess.run(cmd,check=True,capture_output=True,timeout=120)
 run(tmp_path/'whole');run(tmp_path/'resumed',tmp_path/'whole/update2.pt')
 a=torch.load(tmp_path/'whole/resume.pt',weights_only=False);b=torch.load(tmp_path/'resumed/resume.pt',weights_only=False)
 def same(x,y):
  if isinstance(x,torch.Tensor):return torch.equal(x,y)
  if isinstance(x,np.ndarray):return np.array_equal(x,y)
  if isinstance(x,dict):return x.keys()==y.keys() and all(same(x[k],y[k]) for k in x)
  if isinstance(x,(tuple,list)):return len(x)==len(y) and all(same(u,v) for u,v in zip(x,y))
  return x==y
 for k in ('policy','optimizer','history','env_states','torch_rng','numpy_rng','python_rng'):assert same(a[k],b[k]),k
 run(tmp_path/'traced',trace=True)
 traced=torch.load(tmp_path/'traced/resume.pt',weights_only=False)
 for k in ('policy','optimizer','history','env_states','torch_rng','numpy_rng','python_rng'):assert same(a[k],traced[k]),('trace changed learning',k)
 s=json.loads((tmp_path/'whole/summary.json').read_text());assert s['finite_losses'] and s['changed_weights'] and s['samples']>0
 recipe.write_text(json.dumps(dict(c,seed=22)))
 with pytest.raises(subprocess.CalledProcessError):run(tmp_path/'bad',tmp_path/'whole/update2.pt')

def test_a_trainer_ended_mid_run_is_packaged_from_its_last_whole_export(tmp_path):
 import signal,time
 c=json.loads((HERE/'recipes/smoke.json').read_text());c.update(n_games=2,updates=100000,native_threads=1)
 recipe=tmp_path/'recipe.json';recipe.write_text(json.dumps(c));out=tmp_path/'run'
 finalize=lambda:subprocess.run([sys.executable,str(HERE/'finalize_entry.py'),'--out',str(out)],capture_output=True,text=True,timeout=60)
 trainer=subprocess.Popen([sys.executable,str(HERE/'train_entry.py'),'--source',str(HERE/'starter'),'--recipe',str(recipe),'--out',str(out),'--seconds','90'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
 try:
  deadline=time.time()+80
  while not (out/'export.json').exists() or json.loads((out/'export.json').read_text())['update']<3:
   assert trainer.poll() is None and time.time()<deadline,'the trainer never exported'
   time.sleep(.05)
 finally:trainer.send_signal(signal.SIGKILL);trainer.wait()
 assert not (out/'summary.json').exists()  # ended as the host ends it: no summary of its own
 done=finalize();assert done.returncode==0,done.stderr
 s=json.loads((out/'summary.json').read_text());assert s==json.loads(done.stdout)
 assert s['ended_early'] and s['updates']>=3 and s['finite_losses'] and s['changed_weights'] and s['samples']>0 and s['accounted_seconds']>0
 assert s['weights_sha256']==sha(out/'weights.npz') and s['identity_sha256']==sha(out/'identity.json') and s['device']=='cpu'
 import numpy as np
 with np.load(out/'weights.npz') as arrays:assert all(np.isfinite(arrays[k]).all() for k in arrays.files)
 assert finalize().returncode!=0  # a run with a summary is never repackaged
 empty=tmp_path/'empty';empty.mkdir()
 assert subprocess.run([sys.executable,str(HERE/'finalize_entry.py'),'--out',str(empty)],capture_output=True,timeout=60).returncode!=0

def test_manifest_paths_and_calibration_gate():
 b=json.loads((HERE/'benchmark.json').read_text());assert b['budget']['research_worker_seconds']==7200
 for p in b['starter']['editable_files']:assert (HERE/p).is_file()
 assert b['evaluation']['score_scale']==[0,100]
 assert b['evaluation']['faults']=='invalidate schedule; no placement score'


def test_full_panel_commitment_and_missing_opponent_fail_closed(tmp_path):
 from contract import write,sha,verify_private_release
 private=tmp_path/'private';private.mkdir();items=[]
 for i in range(3):
  path=private/f'op{i}';path.mkdir();(path/'weights.npz').write_bytes(b'committed development fixture')
  items.append(dict(name=f'op{i}',path=f'op{i}',files={'weights.npz':sha(path/'weights.npz')}))
 write(private/'panel-full.json',dict(seeds=list(range(16)),games_per_seat=4))
 write(private/'opponents.json',dict(full=[x['name'] for x in items],smoke=['op0'],opponents=items))
 write(tmp_path/'release.json',dict(opponent_inventory_sha256=sha(private/'opponents.json'),full_private_panel_sha256=sha(private/'panel-full.json')))
 assert verify_private_release(tmp_path,True)['episodes']==3*16*4*4
 (private/'op1/weights.npz').unlink()
 with pytest.raises(ValueError,match='missing or mutated'):verify_private_release(tmp_path,True)

@pytest.mark.parametrize('reply',['not-json','[]','[245]','[1]','["bad"]','[0.1]','[true]'])
def test_policy_pipe_rejects_malformed_or_illegal_output(reply,monkeypatch):
 import grade,io,numpy as np
 remote=grade.Remote.__new__(grade.Remote)
 class Pipe:
  stdin=io.StringIO();stdout=io.StringIO(reply+'\n')
 remote.p=Pipe()
 monkeypatch.setattr(grade.select,'select',lambda *a:([remote.p.stdout],[],[]))
 obs={'features':np.zeros((1,3227),np.float32),'cards':np.zeros((1,124),np.int32),'action_mask':np.zeros((1,245),bool)};obs['action_mask'][0,0]=True
 with pytest.raises((ValueError,IndexError,TypeError)):remote.remote_act(obs,np.array([0]))

def test_actual_policy_worker_uses_eval_mode_and_keeps_debug_off_protocol(tmp_path):
 import numpy as np,io,base64
 (tmp_path/'identity.json').write_text(json.dumps({'config':{'network':{}}}))
 (tmp_path/'policy.py').write_text('''import torch
print("debug-import")
class Policy(torch.nn.Module):
 def __init__(self):
  super().__init__();self.p=torch.nn.Parameter(torch.zeros(1))
 def forward(self,features,cards):
  if self.training:raise RuntimeError("inference must use eval mode")
  return torch.zeros((len(features),245)),torch.zeros(len(features))
''')
 np.savez(tmp_path/'weights.npz',p=np.zeros(1,np.float32));buf=io.BytesIO();mask=np.zeros((1,245),bool);mask[0,0]=True
 np.savez(buf,features=np.zeros((1,3227),np.float32),cards=np.zeros((1,124),np.int32),mask=mask)
 result=subprocess.run([sys.executable,'-I',str(HERE/'policy_worker.py'),str(tmp_path)],input=base64.b64encode(buf.getvalue()).decode()+'\n',text=True,capture_output=True,check=True,timeout=20)
 assert json.loads(result.stdout)==[0] and 'debug-import' in result.stderr

def test_bounded_action_trace_preserves_transitions_and_reproduces_guard(tmp_path):
 import taverngym as tg,numpy as np,json
 from diagnostic_trace import install,replay
 config=dict(num_players=8,content='Implemented',max_turns=1,turn_action_caps=[30],max_actions_per_turn=30)
 original=tg.BatchEnv;plain=original(2,seed=42,controlled=[0],config=config,threads=1);undo=install(tmp_path/'traces',100);wrapped=tg.BatchEnv(2,seed=42,controlled=[0],config=config,threads=1)
 try:
  for _ in range(20):
   a=plain.observe();b=wrapped.observe();assert all(np.array_equal(a[k],b[k]) for k in ['features','cards','action_mask','ready'])
   actions=a['action_mask'].argmax(-1).astype(np.int64);ra,da,ia=plain.step(actions);rb,db,ib=wrapped.step(actions)
   assert np.array_equal(ra,rb) and np.array_equal(da,db) and all(np.array_equal(ia[k],ib[k]) for k in ia)
   assert all(plain.save(g)==wrapped.save(g) for g in range(2))
   if list((tmp_path/'traces').glob('*/trace.json')):break
 finally:undo()
 traces=list((tmp_path/'traces').glob('*/trace.json'));assert len(traces)==2
 for i,p in enumerate(traces):assert replay(p.parent,tmp_path/('replay'+str(i)))['reproduced']

def test_action_trace_captures_rejection_without_terminal_guard(tmp_path):
 import taverngym as tg,numpy as np,json
 from diagnostic_trace import install,replay
 undo=install(tmp_path/'traces',100)
 try:
  env=tg.BatchEnv(1,seed=42,controlled=[0],config=dict(num_players=8,content='Implemented'),threads=1);env.observe();env.step(np.array([245],np.int64))
 finally:undo()
 trace=next((tmp_path/'traces').glob('*/trace.json'));meta=json.loads(trace.read_text())
 assert any(meta['rejected']) and not any(meta['done'])
 assert replay(trace.parent,tmp_path/'replay')['reproduced']
