import json,shutil
import numpy as np
import pytest
from simulators.clash.autoresearch.common import HERE,dump,load_module,digest
from simulators.clash.autoresearch.bundle import seal
from simulators.clash.autoresearch.release_v3 import resolve_schedule,selected_interface,PUBLIC_SEEDS

def tiny_bundle(tmp_path):
 source=tmp_path/'source';source.mkdir()
 for n in ('policy.py','learner.py','train.py'):shutil.copyfile(HERE/'starter'/n,source/n)
 dump(source/'config.json',{'width':8});model=load_module(source/'policy.py','v3fixture').create({'width':8});np.savez_compressed(source/'weights.npz',**{k:v.detach().numpy().astype(np.float32) for k,v in model.state_dict().items()});receipt=seal(source,tmp_path/'bundle')
 return [{'name':'retained-test','bundle':str(tmp_path/'bundle'),'identity':receipt['identity']}]

def test_public_schedule_resolves_only_matching_protected_identities(tmp_path):
 from simulators.clash.autoresearch.tests.fixtures import development_base
 out=development_base(tmp_path/'release')
 practice=json.loads((out/'public/practice.json').read_text());assert practice['engine_seeds']==PUBLIC_SEEDS
 assert all('bundle' not in p for p in practice['opponents'])
 assert 'bundle' not in (out/'public/opponent-registry.json').read_text()
 resolved=resolve_schedule(out,practice);assert resolved['opponents'][-1]['bundle']==str(out/'private/opponents/retained-test')
 assert all('bundle' not in p for p in practice['opponents']),'the public schedule itself is never rewritten'
 practice['opponents'][-1]['identity']='0'*64
 with pytest.raises(ValueError,match='identity differs'):resolve_schedule(out,practice)

def test_interfaces_are_explicit_and_versioned():
 assert selected_interface('sequential')['version']=='clash-evaluation-v3' and 'grading_pair_workers' not in selected_interface('sequential')
 assert selected_interface('parallel3')['version']=='clash-evaluation-v3.1' and selected_interface('parallel3')['grading_pair_workers']==3
 assert digest(selected_interface('parallel3'))!=digest(selected_interface('sequential'))
 with pytest.raises(ValueError,match='explicit'):selected_interface('fastest')

def test_v3_invalid_never_writes_cacheable_grade(tmp_path,monkeypatch):
 from simulators.clash.autoresearch import evaluation_v3
 registry=tiny_bundle(tmp_path);schedule=tmp_path/'schedule.json';dump(schedule,{'engine_seeds':[1],'policy_seed':1,'opponents':[{'name':'wait','kind':'wait'}]});target=tmp_path/'grade.json'
 monkeypatch.setattr(evaluation_v3,'screen',lambda *a,**k:{'complete':False,'failures':[{'kind':'policy_timeout'}],'natural_games':0})
 with pytest.raises(ValueError,match='incomplete'):evaluation_v3.evaluate(registry[0]['bundle'],schedule,target,development=True)
 assert not target.exists() and json.loads(target.with_suffix('.failure.json').read_text())['raw_score_0_100'] is None
