"""The game-side research substrate: ledger, policy packages, wire, scoring, evaluator and release audit.

The session backend built on these is tested in harness/tests/test_doudizhu.py.
"""
from pathlib import Path
import json
import time
import numpy as np
import pytest
import torch

from integrations.doudizhu.autoresearch.ledger import Ledger
from integrations.doudizhu.autoresearch.contract import DEFAULT,EDITABLE,write_json
from integrations.doudizhu.autoresearch.bundle import seal,verify,source_payload
from integrations.doudizhu.autoresearch.wire import encode,decode
from integrations.doudizhu.autoresearch.grading import summary,grade,calibration_gate


def ledger(tmp):return Ledger(tmp,dict(DEFAULT,gpu_minutes=3,min_run_seconds=1))

def package(tmp):
    raw=tmp/'raw';raw.mkdir()
    code='''import torch
class P(torch.nn.Module):
 def __init__(self):
  super().__init__();self.a=torch.nn.Parameter(torch.zeros(1))
def create(config):return P()
def act(policy,role,static,history,actions,counts):return [0]*len(counts)
'''
    (raw/'policy.py').write_text(code)
    for n in ('learner.py','train.py'):(raw/n).write_text('# self-contained fixture\n')
    write_json(raw/'model.json',{})
    torch.save({'a':torch.zeros(1)},raw/'weights.pt')
    seal(raw,tmp/'bundle')
    return tmp/'bundle'

def test_allocation_authority_and_failure_charge(tmp_path):
    d=ledger(tmp_path);d.queue('a',60,{'seed':1});d.start('a','provider',1000)
    assert d.used(1030)==30
    d.finish('a',1043,'failed',error='engine failure')
    assert d.used()==43
    with pytest.raises(ValueError):d.queue('b',138,{})
    d.queue('b',130,{})
    with pytest.raises(ValueError):d.queue('c',1,{})

def test_uncertain_lease_blocks_renewal_and_session_is_immutable(tmp_path):
    d=ledger(tmp_path);d.queue('a',60,{});d.uncertain('a','disconnected')
    with pytest.raises(ValueError):d.queue('b',1,{})
    with pytest.raises(ValueError):Ledger(tmp_path,dict(DEFAULT,gpu_minutes=4,min_run_seconds=1))
    with pytest.raises(ValueError):d.finalize()

def test_bad_grants_and_names(tmp_path):
    d=ledger(tmp_path)
    for value in (float('nan'),float('inf'),0,-1):
        with pytest.raises(ValueError):d.queue('a',value,{})
    for name in ('../escape','/path','a/b',''):
        with pytest.raises(ValueError):d.queue(name,2,{})

def test_latest_submission_and_milestone_are_immutable(tmp_path):
    d=ledger(tmp_path)
    d.queue('a',61,{});d.start('a','p',100);d.finish('a',161,'complete','a','sha-a')
    first=d.submit('a')
    # The earlier one-third point had no submission; later submissions do not
    # retroactively replace it. Final always uses the latest explicit submission.
    assert d.public()['points'][0]['submission'] is None
    d.queue('b',61,{});d.start('b','p2',200);d.finish('b',261,'complete','b','sha-b')
    second=d.submit('b');state=d.finalize()
    assert second>first
    assert next(x for x in state['points'] if float(x['fraction'])==1)['submission']==second

def test_bundle_sealing_drift_and_no_execution(tmp_path):
    p=package(tmp_path);m=verify(p)
    assert m['tensors']==1
    with pytest.raises(FileExistsError):seal(tmp_path/'raw',p)
    (p/'policy.py').write_text('raise RuntimeError("should never execute in the host")')
    with pytest.raises(ValueError,match='drift'):verify(p)

def test_tensor_only_and_nonfinite_weights(tmp_path):
    p=package(tmp_path)
    from integrations.doudizhu.autoresearch.bundle import validate_weights
    torch.save({'a':torch.tensor([float('nan')])},tmp_path/'bad.pt')
    with pytest.raises(ValueError,match='Non-finite'):validate_weights(tmp_path/'bad.pt')
    torch.save({'a':'not a tensor'},tmp_path/'bad.pt')
    with pytest.raises(ValueError,match='tensors only'):validate_weights(tmp_path/'bad.pt')

def test_public_wire_and_shape_validation():
    raw=encode('landlord',np.zeros((2,319),np.int8),np.zeros((2,5,162),np.int8),np.zeros((3,54),np.int8),[1,2])
    role,x,z,a,n=decode(raw[4:]);assert role=='landlord' and n.tolist()==[1,2]
    with pytest.raises(ValueError):decode(encode('landlord',np.zeros((2,319)),np.zeros((2,5,162)),np.zeros((3,54)),[1,1])[4:])

def rows(win):
    return [{'opponent':p,'deal_index':i,'role':r,'won':win,'natural':True,'truncated':False}
        for p in ('a','b') for i in range(4) for r in ('landlord','landlord_down','landlord_up')]

def test_complete_schedules_and_calibration_gate():
    a=rows(False);b=rows(True)
    gate=calibration_gate(a,b,replicates=20);assert gate['qualified']
    assert grade(b,gate)['normalized_quality']==1
    with pytest.raises(ValueError):summary(a[:-1])
    with pytest.raises(ValueError):summary(a+[a[0]])
    with pytest.raises(ValueError):grade(a,{'qualified':False})
    assert grade(a,{},development=True)['reward'] is None

def test_policy_receives_only_public_arrays_and_finishes_games(tmp_path):
    from integrations.doudizhu.autoresearch.arena import evaluate
    p=package(tmp_path)
    result=evaluate(p,[{'name':'random','kind':'random'}],deal_seed=71,policy_seed=83,deals=1,development=True,batch_size=3)
    assert result['games']==3 and all(x['natural'] for x in result['rows'])
    assert set(x['role'] for x in result['rows'])=={'landlord','landlord_down','landlord_up'}

def test_official_evaluation_refuses_unisolated_host(tmp_path):
    from integrations.doudizhu.autoresearch.arena import PolicyProcess
    import sys,os
    if sys.platform=='linux' and os.geteuid()==0:pytest.skip('Already a privileged Linux evaluator')
    p=package(tmp_path)
    with pytest.raises(RuntimeError,match='Official grading'):PolicyProcess(p)

def test_reconciliation_releases_only_unallocated_reservation(tmp_path):
    d=ledger(tmp_path);d.queue('a',60,{});d.uncertain('a','lost provider connection')
    d.reconcile('a','provider',100,123,'provider recovered')
    assert d.used()==23 and d.jobs()[0]['status']=='failed'
    with pytest.raises(ValueError):d.reconcile('a','provider',100,101,'fake repeated refund')
    d.queue('b',100,{})

def test_three_seed_anchor_gate_accounts_for_training_variance():
    from integrations.doudizhu.autoresearch.grading import multi_seed_gate
    assert multi_seed_gate([rows(False)]*3,[rows(True)]*3,replicates=30)['qualified']
    with pytest.raises(ValueError):multi_seed_gate([rows(False)],[rows(True)])
    assert not multi_seed_gate([rows(False),rows(True),rows(False)],[rows(True),rows(False),rows(True)],replicates=100)['qualified']

def test_root_export_refuses_symlink(tmp_path):
    from integrations.doudizhu.autoresearch.export_entry import safe_read
    (tmp_path/'target').write_text('private fixture')
    (tmp_path/'alias').symlink_to(tmp_path/'target')
    with pytest.raises(OSError):safe_read(tmp_path/'alias',100)
    with pytest.raises(ValueError):safe_read(tmp_path/'target',1)

def test_grading_refuses_nonbinary_or_nonfinite_outcomes():
    for invalid in (float('nan'),float('inf'),.5,2):
        values=rows(True);values[0]['won']=invalid
        with pytest.raises(ValueError):summary(values)

def test_release_audit_checks_extra_files_seeds_and_private_material(tmp_path):
    from integrations.doudizhu.autoresearch.prepare import audit_release
    from integrations.doudizhu.autoresearch.contract import sha
    for directory in ('public','frozen','private'):
        (tmp_path/directory).mkdir(parents=True,exist_ok=True)
    public=tmp_path/'public/hello.txt';public.write_text('public fixture')
    seed=tmp_path/'private/grade-seeds.json';seed.write_text('private fixture')
    write_json(tmp_path/'release.json',{'public_files':{'hello.txt':sha(public)},'frozen_files':{},
        'private_seed_commitments':{'grade-seeds.json':sha(seed)},'official_grading_ready':False})
    assert audit_release(tmp_path)['hidden_files_staged']==0
    seed.write_text('changed')
    with pytest.raises(ValueError,match='seed commitment'):audit_release(tmp_path)
    seed.write_text('private fixture')
    extra=tmp_path/'frozen/extra.py';extra.write_text('unmanifested')
    with pytest.raises(ValueError,match='Unmanifested'):audit_release(tmp_path)
    extra.unlink();public.write_text('changed')
    with pytest.raises(ValueError,match='Public context drift'):audit_release(tmp_path)
    public.write_text('public fixture');(tmp_path/'public/alias').symlink_to(public)
    with pytest.raises(ValueError):audit_release(tmp_path)

def test_bundle_archive_roundtrip_and_link_rejection(tmp_path):
    import tarfile
    from integrations.doudizhu.autoresearch.bundle import archive_bundle,unpack_bundle,FILES
    bundle=package(tmp_path)
    archive=tmp_path/'bundle.tar';archive_bundle(bundle,archive)
    unpack_bundle(archive,tmp_path/'downloaded')
    assert verify(tmp_path/'downloaded')['bundle_sha256']==verify(bundle)['bundle_sha256']
    bad=tmp_path/'bad.tar'
    with tarfile.open(bad,'w') as output:
        for name in (*FILES,'manifest.json'):
            if name=='weights.pt':
                entry=tarfile.TarInfo(name);entry.type=tarfile.SYMTYPE;entry.linkname='/private/fixture'
                output.addfile(entry)
            else:output.add(bundle/name,arcname=name)
    with pytest.raises(ValueError,match='Invalid archive'):unpack_bundle(bad,tmp_path/'rejected')

def test_transport_guard_rejects_closed_or_missing_connection():
    from types import SimpleNamespace
    from integrations.doudizhu.autoresearch.transport import connection_usable
    assert not connection_usable(None)
    connection=SimpleNamespace(is_closing=lambda:False)
    protocol=SimpleNamespace(processor=SimpleNamespace(connection=connection))
    assert connection_usable(protocol)
    connection.is_closing=lambda:True
    assert not connection_usable(protocol)

def test_smoke_contract_bounds_and_reopened_ledger(tmp_path):
    c=dict(DEFAULT,gpu_minutes=10,min_run_seconds=30,practice_deals=8,teardown_reserve_seconds=20,max_run_seconds=360)
    d=Ledger(tmp_path,c)
    with pytest.raises(ValueError):d.queue('too_big',581,{})
    d.queue('failed',300,{})
    d.start('failed','external',1000);d.finish('failed',1305,'failed')
    restored=Ledger(tmp_path,c)
    assert restored.used()==305
    with pytest.raises(ValueError):restored.queue('over',276,{})
    restored.queue('remaining',275,{})
    restored.uncertain('remaining','disconnect')
    assert restored.used()==580
    with pytest.raises(ValueError):restored.queue('renewal',30,{})


def test_raw_game_points_are_independent_of_reference_gate():
    result=grade(rows(True),{},development=True)
    assert result['raw_score']==100 and result['raw_score_ci95']==[100,100]
    assert result['per_opponent']=={'a':1,'b':1}
    assert result['normalized_quality'] is None and result['reward'] is None
    assert result['natural_games']==24


def test_development_private_prefix_does_not_mutate_release(tmp_path):
    from integrations.doudizhu.autoresearch.manage import grade_session
    root=tmp_path/'release';private=root/'private';private.mkdir(parents=True)
    seeds={'split':'heldout','deal_seed':7,'policy_seed':11,'deals':256}
    write_json(private/'grade-seeds.json',seeds)
    write_json(private/'anchors.json',{'qualified':False})
    write_json(private/'panel.json',{'development':[{'name':'random','kind':'random'}]})
    bundle=package(tmp_path);identity=verify(bundle)['bundle_sha256']
    session=tmp_path/'session';d=Ledger(session,dict(DEFAULT,gpu_minutes=10,min_run_seconds=1))
    d.queue('trained',30,{});d.start('trained','provider',100);d.finish('trained',130,'complete',bundle,identity);d.submit('trained')
    class Provider:
        def evaluate(self,bundle,panel,schedule,anchors,development):
            assert development and schedule['deals']==16
            return {'public':grade(rows(True),{},development=True),'private':{'rows':rows(True)},'allocation':{'gpu_count':0}}
    result=grade_session(root,session,Provider(),True,private_deals=16)
    assert result['1.0']['raw_score']==100
    assert json.loads((private/'grade-seeds.json').read_text())==seeds
    with pytest.raises(ValueError):grade_session(root,session,Provider(),True,private_deals=0)


def test_late_submission_cannot_retroactively_fill_crossed_point(tmp_path,monkeypatch):
    import integrations.doudizhu.autoresearch.ledger as module
    d=Ledger(tmp_path,dict(DEFAULT,gpu_minutes=3,min_run_seconds=1))
    d.queue('a',30,{});d.start('a','provider-a',100);d.finish('a',130,'complete','bundle-a','sha-a')
    d.queue('active',100,{});d.start('active','provider-b',200)
    monkeypatch.setattr(module.time,'time',lambda:235)
    d.submit('a')
    point=next(x for x in d.public()['points'] if float(x['fraction'])==1/3)
    assert point['submission'] is None  # 65 seconds spent before this submit.


@pytest.mark.parametrize('response', ['[-1]*len(counts)', '[0]*(len(counts)+1)'])
def test_invalid_policy_output_is_not_a_natural_loss(tmp_path,response):
    from integrations.doudizhu.autoresearch.arena import evaluate,PolicyFailure
    package(tmp_path)
    raw=tmp_path/'raw';code=(raw/'policy.py').read_text().replace('[0]*len(counts)',response)
    (raw/'policy.py').write_text(code);seal(raw,tmp_path/'invalid')
    with pytest.raises(PolicyFailure):evaluate(tmp_path/'invalid',[{'name':'random','kind':'random'}],deal_seed=71,policy_seed=83,deals=1,development=True,batch_size=3)


def test_teardown_crossing_records_prior_submission(tmp_path):
    d=Ledger(tmp_path,dict(DEFAULT,gpu_minutes=3,min_run_seconds=1))
    d.queue('a',30,{});d.start('a','a',100);d.finish('a',130,'complete','bundle-a','sha-a');sub=d.submit('a')
    d.queue('b',30,{});d.start('b','b',200)
    # Body ends below the boundary, provider teardown crosses it; all39s count.
    d.finish('b',239,'complete','bundle-b','sha-b')
    second=d.submit('b');state=d.finalize()
    assert next(x for x in state['points'] if float(x['fraction'])==1/3)['submission']==sub
    assert next(x for x in state['points'] if float(x['fraction'])==1)['submission']==second


