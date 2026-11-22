"""Real native starter resume, editable hooks, exported ABI and source refusals."""
import json,shutil
from pathlib import Path
import numpy as np
import pytest,torch
from simulators.clash.autoresearch.common import HERE,dump,load_module
from simulators.clash.autoresearch.train_entry import run
from simulators.clash.autoresearch.bundle import seal,verify

NATIVE_CHECKPOINT_TRAIN='from simulators.clash.autoresearch.fast_adapter import Trainer as NativeTrainer\n\nclass Trainer(NativeTrainer):\n    pass\n'

def source_at(path,native_checkpoint=False):
 """The starter; `native_checkpoint` drops its lightweight checkpoint override for exact-resume checks."""
 path.mkdir()
 for n in ('policy.py','train.py','learner.py'):shutil.copyfile(HERE/'starter'/n,path/n)
 if native_checkpoint:(path/'train.py').write_text(NATIVE_CHECKPOINT_TRAIN)
 c=json.loads((HERE/'recipes/starter.json').read_text())
 c.update(device='cpu',backend='native-cpu',envs=2,threads=1,rollout_steps=4,minibatch=16,epochs=1,updates=3,bucket_actor=False,bucket_learner=False,learner_graph=False,compact_embedding_grad=False,gae_kernel='torch')
 dump(path/'config.json',c);return c

def test_fast_starter_resume_export_and_tamper(tmp_path):
 p=tmp_path/'source';source_at(p,native_checkpoint=True)
 a=run(p,tmp_path/'full',30,updates=3)
 run(p,tmp_path/'first',30,updates=1)
 b=run(p,tmp_path/'resume',30,resume=tmp_path/'first/checkpoint.pt',updates=3)
 assert a['samples']==48 and b['new_learner_decisions']==32 and a['finite_gradients'] and a['weight_l1_change']>0
 full=torch.load(tmp_path/'full/checkpoint.pt',weights_only=False);resumed=torch.load(tmp_path/'resume/checkpoint.pt',weights_only=False)
 for key in ('model','optimizer','torch_rng','cuda_rng','engine_outputs','counters','bucket_usage','entrant_state'):
  from simulators.clash.native.test_training import equal
  equal(full[key],resumed[key],key)
 # Complete byte images may differ only in rebound pointers; future outputs,
 # model, Adam and all RNG states were compared above.
 from simulators.clash.native.snapshots import decode_engine_state
 assert len(decode_engine_state(full['engine_state']))==2
 model=load_module(p/'policy.py','exported_fast_policy').create(full['config'])
 with np.load(tmp_path/'full/weights.npz',allow_pickle=False) as data:model.load_state_dict({k:torch.from_numpy(data[k]) for k in data.files})
 model2=load_module(p/'policy.py','restored_fast_policy').create(full['config']);model2.load_state_dict(full['model'])
 o=full['engine_outputs']['observations'].flatten(0,1);m=full['engine_outputs']['masks'].flatten(0,1)
 with torch.no_grad():
  for x,y in zip(model(o,m),model2(o,m)):assert torch.equal(x,y)
 seal(tmp_path/'full',tmp_path/'bundle');verify(tmp_path/'bundle')
 from simulators.clash.autoresearch.readiness_contract import admit
 admit(tmp_path/'bundle')
 (p/'learner.py').write_text((p/'learner.py').read_text()+'\n# drift\n')
 with pytest.raises(ValueError,match='binding'):run(p,tmp_path/'bad',30,resume=tmp_path/'first/checkpoint.pt')


def test_a_stop_request_ends_the_loop_at_the_next_update_with_a_complete_export(tmp_path,monkeypatch):
 # The host writes the request file into a running worker (`ar stop`); here the first update writes it.
 request=tmp_path/'stop-requested'
 monkeypatch.setattr('simulators.clash.autoresearch.fast_entry.STOP_REQUEST',str(request))
 p=tmp_path/'source';source_at(p)
 (p/'train.py').write_text((p/'train.py').read_text()+'\n_learn=Trainer.learn\ndef learn(self,data):\n    result=_learn(self,data);open('+repr(str(request))+",'w').close();return result\nTrainer.learn=learn\n")
 result=run(p,tmp_path/'stopped',30,updates=3)
 assert result['updates']==1 and result['finite_losses'] and result['weight_l1_change']>0  # one update, not three
 assert json.loads((tmp_path/'stopped/result.json').read_text())['updates']==1
 with np.load(tmp_path/'stopped/weights.npz') as arrays:assert all(np.isfinite(arrays[k]).all() for k in arrays.files)
 assert (tmp_path/'stopped/checkpoint.pt').is_file() and len((tmp_path/'stopped/progress.jsonl').read_text().splitlines())==1
 request.unlink();plain=tmp_path/'plain';source_at(plain)
 assert run(plain,tmp_path/'whole',30,updates=3)['updates']==3  # without a request the starter runs to its limit

def test_default_real_route_and_resources():
 c=json.loads((HERE/'recipes/starter.json').read_text())
 assert c['training_route']=='native-fast-v1' and c['backend']=='native-cuda'
 assert c['envs']==1024 and c['hardware_allocation']['cpu_threads']==4 and c['hardware_allocation']['ram_gib']==16
 assert c['max_ticks']==0 and c['warmstart_steps']==0 and c['reward_shaping']==0
 assert c['match_form']=='short-v1' and c['rules']['decision_ticks']==10  # the short benchmark form (common.MATCH_FORM)
 assert all(c[n] for n in ('bucket_actor','bucket_learner','learner_graph','compact_embedding_grad','optimizer_fused'))


def test_fast_v4_constructor_refuses_the_short_runtime(tmp_path):
 """fast-v4 is the full-length form (git tag full-size-20261113); this runtime plays short-v1."""
 from simulators.clash.autoresearch.tests.fixtures import development_base
 from simulators.clash.autoresearch.fast_release import prepare,verify as verify_release
 base=development_base(tmp_path/'v31')
 with pytest.raises(ValueError,match='full-size-20261113'):prepare(tmp_path/'v4',base)
 with pytest.raises(ValueError,match='full-size-20261113'):verify_release(base)
 assert not (tmp_path/'v4').exists()


def test_checkpoint_single_serialization_and_native_payload(tmp_path,monkeypatch):
 from simulators.clash.autoresearch import checkpoint_writer
 source=tmp_path/'source';source_at(source,native_checkpoint=True)
 original=torch.save;calls=[]
 def save(payload,*args,**kwargs):
  calls.append(set(payload));return original(payload,*args,**kwargs)
 monkeypatch.setattr(torch,'save',save)
 r=run(source,tmp_path/'run',30,updates=2)
 assert len(calls)==1 and {'entrant_binding','entrant_state','engine_state','optimizer','python_rng','numpy_rng','torch_rng','cuda_rng','engine_outputs','counters','bucket_usage'}<=calls[0]
 assert r['checkpoint_diagnostic']['serialization_passes']==1
 assert r['checkpoint_diagnostic']['file_fsynced']
 assert not (tmp_path/'run/checkpoint.tmp').exists()
 saved=torch.load(tmp_path/'run/checkpoint.pt',weights_only=False)
 assert saved['entrant_state']['optimizer_steps']==2 and saved['samples']==32


def test_starter_checkpoint_is_lightweight_and_still_sealed(tmp_path):
 source=tmp_path/'source';source_at(source)
 r=run(source,tmp_path/'run',30,updates=2)
 d=r['checkpoint_diagnostic']
 assert d['format']=='clash-fast-model-optimizer-v1' and not d['engine_state_saved'] and d['file_fsynced']
 saved=torch.load(tmp_path/'run/checkpoint.pt',weights_only=False)
 assert not {'engine_state','engine_outputs'}&set(saved) and {'model','optimizer','entrant_binding','entrant_state'}<=set(saved)
 assert saved['entrant_state']['optimizer_steps']==2 and not (tmp_path/'run/checkpoint.tmp').exists()
 seal(tmp_path/'run',tmp_path/'bundle',{'checkpoint_sha256':__import__('hashlib').sha256((tmp_path/'run/checkpoint.pt').read_bytes()).hexdigest()});verify(tmp_path/'bundle')


def test_checkpoint_interrupted_write_preserves_target(tmp_path,monkeypatch):
 from simulators.clash.autoresearch.checkpoint_writer import checkpoint
 from simulators.clash.native.train import Trainer as NativeTrainer
 from types import SimpleNamespace
 target=tmp_path/'checkpoint.pt';target.write_bytes(b'existing complete checkpoint')
 trainer=SimpleNamespace(torch=torch)
 def native(self,path,elapsed):
  self.torch.save({'model':torch.ones(2)},Path(path).with_suffix('.tmp'))
  Path(path).with_suffix('.tmp').replace(path)
 def interrupted(payload,stream):
  stream.write(b'incomplete');raise OSError('injected partial write')
 monkeypatch.setattr(NativeTrainer,'checkpoint',native);monkeypatch.setattr(torch,'save',interrupted)
 with pytest.raises(OSError,match='partial write'):checkpoint(trainer,target,0,entrant_binding={},entrant_state={})
 assert target.read_bytes()==b'existing complete checkpoint' and not target.with_suffix('.tmp').exists() and trainer.torch is torch


def test_checkpoint_native_metadata_cannot_be_replaced(tmp_path):
 from simulators.clash.autoresearch.checkpoint_writer import CheckpointWriter
 writer=CheckpointWriter(torch,{'entrant_binding':{},'entrant_state':{}})
 with pytest.raises(ValueError,match='collision'):writer.save({'entrant_state':{'bad':True}},tmp_path/'bad.pt')
 assert not (tmp_path/'bad.pt').exists()
 with pytest.raises(ValueError,match='exact entrant'):CheckpointWriter(torch,{'model':{}})



def test_incomplete_checkpoint_refused_and_partial_export_preserved(tmp_path,monkeypatch):
 from types import SimpleNamespace
 from simulators.clash.autoresearch.fast_entry import export
 source=tmp_path/'source';source_at(source)
 partial=tmp_path/'partial.pt';partial.write_bytes(b'PK\x03\x04incomplete')
 with pytest.raises((RuntimeError,EOFError,ValueError,IndexError)):
  run(source,tmp_path/'rejected',30,resume=partial)
 assert not (tmp_path/'rejected/checkpoint.pt').exists()
 target=tmp_path/'export';target.mkdir();(target/'weights.npz').write_bytes(b'previous complete weights')
 model=SimpleNamespace(state_dict=lambda:{'w':torch.ones(1)})
 def interrupted(stream,**arrays):
  stream.write(b'partial archive');raise OSError('injected export interruption')
 monkeypatch.setattr(np,'savez_compressed',interrupted)
 with pytest.raises(OSError,match='export interruption'):export(SimpleNamespace(model=model),source,target)
 assert (target/'weights.npz').read_bytes()==b'previous complete weights'
 assert not (target/'weights.tmp.npz').exists()


def test_enrichment_preserves_complete_native_checkpoint_payload(tmp_path):
 import sys
 from simulators.clash.autoresearch.common import setup_native,DATA
 from simulators.clash.native.train import Trainer as NativeTrainer
 from simulators.clash.native.test_training import equal
 source=tmp_path/'source';c=source_at(source,native_checkpoint=True);setup_native();sys.path.insert(0,str(source))
 for n in ('policy','learner'):sys.modules.pop(n,None)
 trainer=load_module(source/'train.py','native_payload_fixture').Trainer(c,DATA)
 try:
  trainer.learn(trainer.collect())
  NativeTrainer.checkpoint(trainer,tmp_path/'native.pt',0.)
  trainer.checkpoint(tmp_path/'enriched.pt',0.,entrant_binding={'fixture':True},entrant_state={'optimizer_steps':1,'losses':[]})
  native=torch.load(tmp_path/'native.pt',weights_only=False);enriched=torch.load(tmp_path/'enriched.pt',weights_only=False)
  assert set(enriched)==set(native)|{'entrant_binding','entrant_state'}
  for key in native:equal(native[key],enriched[key],key)
 finally:trainer.close()
