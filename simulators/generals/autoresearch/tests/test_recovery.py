import json
import time
import pytest
from integrations.generals_autoresearch.contract import DEFAULT,TRAIN_SHAPE,rate
from integrations.generals_autoresearch.ledger import Ledger
from harness.games.generals.controller import Controller

R=rate(TRAIN_SHAPE)  # dollars per second of the training worker: budgets below are written in its seconds


def test_restart_reconciles_same_provider_and_retains_failed_time(tmp_path):
    config=dict(DEFAULT,budget_usd=10*60*R,min_run_seconds=1)
    ledger=Ledger(tmp_path/'session',config)
    ledger.queue('crashed',120,{'kind':'train'});ledger.start('crashed','sb-owned',100)
    ledger.uncertain('crashed','controller interruption')
    class Provider:
        created=0
        def execute(self,*args):self.created+=1;pytest.fail('Recovery must not create a worker')
        def recover(self,ledger,job,stopped=False):
            ledger.reconcile(job['name'],dict(provider_id='sb-owned',started_at=100,finished_at=190),'failed',error='no exported artifact')
    provider=Provider();controller=Controller(tmp_path/'session',tmp_path/'release',provider)
    controller.tick();controller.join(3)
    assert controller.ledger.used()==pytest.approx(90*R) and provider.created==0
    assert controller.ledger.jobs()[0]['status']=='failed'
    assert controller.ledger.finalize()['lifecycle']=='finalized'


def test_unknown_recovery_identity_cannot_release_hold(tmp_path):
    ledger=Ledger(tmp_path,dict(DEFAULT,budget_usd=10*60*R,min_run_seconds=1))
    ledger.queue('orphan',120,{});ledger.start('orphan','sb-owned',100);ledger.uncertain('orphan','unknown')
    for receipt in ({},dict(provider_id='sb-other',started_at=100,finished_at=110)):
        with pytest.raises(ValueError):ledger.reconcile('orphan',receipt)
    assert ledger.used()==pytest.approx(120*R)
    # The unresolved worker keeps its GPU and its grant: one more GPU job fits beside it, a third does not.
    ledger.queue('second',60,{})
    with pytest.raises(ValueError,match='At most 2 GPUs'):ledger.queue('third',60,{})
    with pytest.raises(ValueError,match='exceeds remaining'):ledger.queue('big',500,{'shape':{'gpu':0,'cpu_cores':1,'memory_mib':2048},'rate':1.})


def test_a_running_job_holds_its_whole_grant_so_parallel_grants_fit_the_budget(tmp_path):
    ledger=Ledger(tmp_path,dict(DEFAULT,budget_usd=10*60*R,min_run_seconds=1))
    ledger.queue('a',400,{});ledger.start('a','sb-a',time.time())  # just started: almost no time spent yet
    assert ledger.used()==pytest.approx(400*R) and ledger.public()['remaining_usd']==pytest.approx(200*R)
    with pytest.raises(ValueError,match='exceeds remaining'):ledger.queue('b',400,{})
    ledger.queue('b',200,{})
    # Milestones still measure worker time actually allocated, not holds.
    assert ledger.allocated_elapsed()<5*R
    ledger.finish('a',ledger.jobs()[0]['started']+100);assert ledger.used()==pytest.approx(300*R)  # a's 100 s + b's hold


def test_a_session_opened_under_the_seconds_budget_is_refused(tmp_path):
    with pytest.raises(ValueError,match='budgeted in seconds'):Ledger(tmp_path,dict(DEFAULT,gpu_minutes=10))


def test_worker_seconds_are_charged_at_the_price_of_the_shape(tmp_path):
    from integrations.generals_autoresearch.contract import lease_shape,PRICE
    assert R==pytest.approx(PRICE['gpu']+4*PRICE['cpu_core']+16*PRICE['memory_gib'])  # about $2.90 an hour
    cpu=lease_shape({},0,8,16);big=lease_shape({},1,16,32)
    assert rate(cpu)<R<rate(big)
    ledger=Ledger(tmp_path,dict(DEFAULT,budget_usd=10*60*R,min_run_seconds=1))
    ledger.queue('cpu',100,{'kind':'lease','shape':cpu,'rate':rate(cpu)})
    assert ledger.used()==pytest.approx(100*rate(cpu))
    ledger.start('cpu','sb-a',100);ledger.finish('cpu',150,'stopped')
    job=ledger.public()['jobs'][0]
    assert job['charged']==pytest.approx(50*rate(cpu)) and job['rate']==rate(cpu)
    assert ledger.allocated_elapsed()==pytest.approx(50*rate(cpu))
    ledger.queue('plain',60,{});assert ledger.used()==pytest.approx(50*rate(cpu)+60*R)  # no shape: the training worker
    ledger.register('net',tmp_path/'bundle','hash-n',{'kind':'package','lease':'cpu','path':'/scratch/a'})
    assert ledger.used()==pytest.approx(50*rate(cpu)+60*R)
    for bad in ((2,4,16),(1,33,16),(0,4,1),(0,2.5,16)):
        with pytest.raises(ValueError):lease_shape({},*bad)


def test_submission_after_outage_cannot_retroactively_fill_milestone(tmp_path,monkeypatch):
    ledger=Ledger(tmp_path,dict(DEFAULT,budget_usd=3*60*R,min_run_seconds=1))
    ledger.queue('old',20,{});ledger.start('old','sb-a',100);ledger.finish('old',110,'complete','/a','hash-a')
    ledger.submit('old')
    ledger.queue('new',120,{});ledger.start('new','sb-b',200)
    # Simulate a missed polling interval: first threshold crossed during a
    # running allocation, before the next explicit submission.
    monkeypatch.setattr('integrations.generals_autoresearch.ledger.time.time',lambda:260.)
    ledger.db.execute("UPDATE jobs SET artifact='/b',artifact_sha='hash-b' WHERE name='new'")
    # Completion crosses 2/3 during provider teardown; old policy must remain.
    ledger.finish('new',320,'complete','/b','hash-b');ledger.submit('new')
    points={p['fraction']:p['submission'] for p in ledger.finalize()['points']}
    assert points[str(1/3)]==1 and points[str(2/3)]==1 and points['1.0']==2


def test_submit_itself_records_threshold_before_new_submission(tmp_path,monkeypatch):
    ledger=Ledger(tmp_path,dict(DEFAULT,budget_usd=3*60*R,min_run_seconds=1))
    ledger.queue('old',20,{});ledger.start('old','sb-a',100);ledger.finish('old',110,'complete','/a','hash-a')
    ledger.submit('old')
    ledger.queue('active',100,{});ledger.start('active','sb-b',200)
    monkeypatch.setattr('integrations.generals_autoresearch.ledger.time.time',lambda:260.)
    ledger.submit('old')
    points={p['fraction']:p['submission'] for p in ledger.public()['points']}
    assert points[str(1/3)]==1


def test_queued_reservations_never_advance_milestones(tmp_path,monkeypatch):
    ledger=Ledger(tmp_path,dict(DEFAULT,budget_usd=3*60*R,min_run_seconds=1))
    ledger.queue('old',20,{});ledger.start('old','sb-a',100);ledger.finish('old',110,'complete','/a','hash-a')
    ledger.submit('old');ledger.queue('future',150,{})
    assert ledger.used()==pytest.approx(160*R) and ledger.allocated_elapsed()==pytest.approx(10*R)
    ledger.submit('old');assert ledger.public()['points']==[]
    ledger.start('future','sb-b',200)
    monkeypatch.setattr('integrations.generals_autoresearch.ledger.time.time',lambda:260.)
    ledger.submit('old')
    assert ledger.public()['points'][0]['submission']==2


def test_unknown_provider_freezes_submission_until_actual_time_reconciles(tmp_path):
    ledger=Ledger(tmp_path,dict(DEFAULT,budget_usd=3*60*R,min_run_seconds=1))
    ledger.queue('old',20,{});ledger.start('old','sb-a',100);ledger.finish('old',110,'complete','/a','hash-a');ledger.submit('old')
    ledger.queue('orphan',150,{});ledger.start('orphan','sb-b',200);ledger.uncertain('orphan','host loss')
    with pytest.raises(ValueError,match='Reconcile unknown'):ledger.submit('old')
    ledger.reconcile('orphan',dict(provider_id='sb-b',started_at=200,finished_at=230),'failed')
    assert ledger.used()==pytest.approx(40*R) and ledger.public()['points']==[]
    assert ledger.submit('old')==2
