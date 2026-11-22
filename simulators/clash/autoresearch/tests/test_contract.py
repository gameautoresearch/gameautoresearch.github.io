from pathlib import Path
import json,shutil,sys
import numpy as np
import pytest,torch
from simulators.clash.autoresearch.common import HERE,load_module,dump
from simulators.clash.autoresearch.ledger import Ledger
from simulators.clash.autoresearch.bundle import seal,verify
from simulators.clash.autoresearch.wire import encode,decode
from simulators.clash.autoresearch.arena import summarize

def test_policy_mask_and_all_entity_rows():
 p=load_module(HERE/'starter/policy.py','test_policy').create({'width':16});o=torch.zeros(2,7024,dtype=torch.int32);o[:,48]=200;m=torch.zeros(2,2307,dtype=torch.uint8);m[:,0]=1
 logits,v=p(o,m);assert logits.argmax(-1).tolist()==[0,0];assert torch.isfinite(v).all()
 o[1,224+199*34+6]=32000;assert not torch.equal(p(o,m)[1][0],p(o,m)[1][1])

def test_gae_natural_and_truncation():
 l=load_module(HERE/'starter/learner.py','test_learner')
 reward=torch.tensor([[1.,0.]]);value=torch.zeros_like(reward);done=torch.ones_like(reward,dtype=torch.bool)
 a,_=l.gae(reward,value,done,torch.tensor([[0.,2.]]),torch.ones(2)*999,.9,.97)
 assert torch.allclose(a,torch.tensor([[1.,1.8]]))

def test_wire_only_public():
 o=np.zeros((2,7024),np.int32);m=np.zeros((2,2307),np.uint8);m[:,0]=1;raw=encode(o,m);x,mask=decode(raw[4:]);assert np.array_equal(o,x) and np.array_equal(m,mask)
 with pytest.raises(ValueError):decode(encode(np.zeros((1,7000),np.int32),m[:1])[4:])

def test_budget_preserves_failed_attempts(tmp_path):
 l=Ledger(tmp_path/'ledger.json');l.reserve('first',400)
 with pytest.raises(ValueError):l.reserve('second',1)
 l.record('first',status='failed',charged_seconds=350)
 with pytest.raises(ValueError):l.reserve('second',251)
 l.reserve('second',250);l.record('second',status='uncertain');assert l.used()==600
 with pytest.raises(ValueError):l.finalize()
 with pytest.raises(ValueError):Ledger(tmp_path/'ledger.json',601)

def test_finalize_and_immutable_selection(tmp_path):
 l=Ledger(tmp_path/'ledger.json');l.reserve('x',500);l.record('x',status='complete',charged_seconds=400);l.submit('x','abc');s=l.finalize();assert s['milestones']['final']['identity']=='abc'
 with pytest.raises(ValueError):l.submit('y','def')
 with pytest.raises(ValueError):l.reserve('z',1)

def test_paired_score_and_invalids():
 rows=[{'opponent':p,'seat':seat,'seed_index':seed,'points':float(seat==0),'natural':True} for p in ('a','b') for seat in (0,1) for seed in (0,1)]
 r=summarize(rows);assert r['raw_score_0_100']==50 and r['normalized_reference_score'] is None
 rows[0]['truncated']=True
 with pytest.raises(ValueError):summarize(rows)

def test_bundle_source_tamper(tmp_path):
 source=tmp_path/'source';source.mkdir()
 for p in (HERE/'starter').glob('*.py'):shutil.copy(p,source/p.name)
 dump(source/'config.json',{'width':16});p=load_module(source/'policy.py','bundlepolicy').create({'width':16});np.savez_compressed(source/'weights.npz',**{k:v.detach().numpy().astype(np.float32) for k,v in p.state_dict().items()})
 seal(source,tmp_path/'bundle');verify(tmp_path/'bundle');f=tmp_path/'bundle/policy.py';f.chmod(0o644);f.write_text(f.read_text()+'\n# drift\n')
 with pytest.raises(ValueError):verify(tmp_path/'bundle')
