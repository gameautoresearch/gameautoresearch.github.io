import io
import json
import shutil
from pathlib import Path
import numpy as np
import pytest
from integrations.generals_autoresearch.contract import HERE,DEFAULT,TRAIN_SHAPE,rate,write_json
R=rate(TRAIN_SHAPE)  # dollars per second of the training worker: budgets below are written in its seconds
from integrations.generals_autoresearch.ledger import Ledger
from integrations.generals_autoresearch.bundle import seal,verify,save_weights,validate_weights
from integrations.generals_autoresearch.grading import grade,calibration_gate,by_map
from integrations.generals_autoresearch.wire import encode,decode
from integrations.generals.environment import RULESET,BOARD_SHAPE,ACTION_COUNT,PAD_TO

CAP=RULESET['truncation']
FAR=(PAD_TO-2,PAD_TO-2)  # the second general's corner in hand-built boards

def test_allocation_failure_immutable_budget_and_one_job(tmp_path):
 l=Ledger(tmp_path,dict(DEFAULT,budget_usd=3*60*R,min_run_seconds=10,max_concurrent_jobs=1))
 l.queue('a',60,{'kind':'train'})
 with pytest.raises(ValueError,match='One job'):l.queue('b',60,{})
 l.start('a','provider-a',100);l.finish('a',160,'failed',error='compile fault')
 assert l.used()==pytest.approx(60*R)
 with pytest.raises(ValueError,match='immutable'):Ledger(tmp_path,dict(DEFAULT,budget_usd=4*60*R,min_run_seconds=10))
 l.queue('b',60,{});l.uncertain('b','end not yet known')
 assert l.used()==pytest.approx(120*R)
 with pytest.raises(ValueError,match='Finish or reconcile'):l.finalize()

def test_latest_submission_only_at_explicit_finalization(tmp_path):
 l=Ledger(tmp_path,dict(DEFAULT,budget_usd=1*60*R,min_run_seconds=10))
 for name,start in [('a',100),('b',200)]:
  l.queue(name,30,{});l.start(name,'provider-'+name,start);l.finish(name,start+20,'complete','/owned/'+name,name+'-hash');l.submit(name)
 state=l.finalize();points={p['fraction']:p['submission'] for p in state['points']}
 assert points[str(1/3)] is None and points[str(2/3)]==1 and points['1.0']==2
 with pytest.raises(ValueError,match='finalized'):l.queue('c',10,{})
 with pytest.raises(ValueError,match='finalized'):l.submit('a')

@pytest.fixture
def bundle(tmp_path):
 import jax
 import jax.numpy as jnp
 from integrations.generals_autoresearch.starter.policy import create
 source=tmp_path/'source';source.mkdir()
 for n in ('policy.py','learner.py','train.py'):shutil.copy2(HERE/'starter'/n,source/n)
 cfg={'width':4,'compute_dtype':'float32'};write_json(source/'model.json',cfg)
 params=create(cfg).init(jax.random.PRNGKey(1),jnp.zeros((1,*BOARD_SHAPE,14)),jnp.ones((1,ACTION_COUNT),bool))['params']
 save_weights(params,source/'weights.npz');seal(source,tmp_path/'bundle')
 return tmp_path/'bundle'

def test_bundle_drift_and_parameter_caps_fail_closed(bundle,tmp_path):
 m=verify(bundle);assert m['tensors']>0
 (bundle/'policy.py').write_text('# changed\n')
 with pytest.raises(ValueError,match='drift'):verify(bundle)
 p=tmp_path/'oversized.npz';np.savez(p,x=np.zeros(DEFAULT['policy_tensor_cap']+1,np.float32))
 with pytest.raises(ValueError,match='cap'):validate_weights(p)
 p=tmp_path/'object.npz';np.savez(p,x=np.asarray([object()],object))
 with pytest.raises(ValueError):validate_weights(p)

def test_weight_header_rejected_before_array_allocation(tmp_path,monkeypatch):
 import zipfile
 header=io.BytesIO()
 np.lib.format.write_array_header_1_0(header,{'descr':'<f4','fortran_order':False,'shape':(2**40,)})
 p=tmp_path/'forged.npz'
 with zipfile.ZipFile(p,'w') as z:z.writestr('x.npy',header.getvalue())
 monkeypatch.setattr(np,'load',lambda *a,**k:pytest.fail('Array allocation must not be reached'))
 with pytest.raises(ValueError,match='cap'):validate_weights(p)

def test_only_observed_read_rpc_race_is_retried(monkeypatch):
 from harness.games.generals.provider import read_rpc
 monkeypatch.setattr('harness.games.generals.provider.time.sleep',lambda _:None)
 monkeypatch.setattr('harness.games.generals.provider.reset_rpc_channels',lambda:None)
 calls=[]
 def race():
  calls.append(1)
  if len(calls)<3:raise AttributeError("'Connection' object has no attribute '_transport'")
  return 7
 assert read_rpc(race)==7 and len(calls)==3
 with pytest.raises(AttributeError,match='different'):
  read_rpc(lambda:(_ for _ in ()).throw(AttributeError('different')))
 from grpclib.exceptions import StreamTerminatedError
 attempts=[]
 def disconnected():
  attempts.append(1)
  if len(attempts)==1:raise StreamTerminatedError('Connection lost')
  return 'same immutable upload'
 assert read_rpc(disconnected)=='same immutable upload' and len(attempts)==2

def test_sdk_wait_adapter_polls_one_existing_exec():
 import asyncio
 from simulators.generals.autoresearch.provider import _poll_wait
 class Process:
  calls=0
  async def poll(self):
   self.calls+=1
   return 0 if self.calls==3 else None
 p=Process();assert asyncio.run(_poll_wait(p))==0 and p.calls==3

def test_visible_wire_rejects_extra_hidden_fields():
 x=np.zeros((2,*BOARD_SHAPE,14),np.float32);mask=np.zeros((2,ACTION_COUNT),bool);mask[:,0]=True;mask[1,-1]=True
 xx,mm=decode(encode(x,mask)[4:]);np.testing.assert_array_equal(x,xx);np.testing.assert_array_equal(mask,mm)
 f=io.BytesIO();np.savez(f,x=x,mask=np.packbits(mask,axis=-1,bitorder='little'),private_seed=np.asarray(2))
 with pytest.raises(ValueError,match='Unexpected'):decode(f.getvalue())
 # Another board size (the 8x8 small form, 513 actions) is not this ruleset's frame.
 with pytest.raises(ValueError,match='Wrong public observation shape'):decode(encode(np.zeros((1,8,8,14),np.float32),np.ones((1,513),bool))[4:])
 with pytest.raises(ValueError,match='Wrong legal mask'):decode(encode(x,np.ones((2,ACTION_COUNT+8),bool))[4:])

def rows(value=.5):
 return [dict(opponent=p,map_index=i,seat=s,score=value,natural=value!=.5,truncated=value==.5,
  capture_win=value==1,capture_loss=value==0,length=CAP if value==.5 else 100)
  for p in ('a','b') for i in (0,1,2) for s in (0,1)]

def test_paired_score_and_anchor_gate_do_not_invent_strength():
 r=grade(rows(),{},development=True)
 assert r['reward']==.5 and r['panel_points']==50 and r['cap_draws']==12 and r['normalized_quality'] is None
 assert calibration_gate(rows(),rows())['qualified'] is False
 assert calibration_gate(rows(0),rows(1))['qualified'] is True
 with pytest.raises(ValueError,match='Incomplete'):by_map(rows()[:-1])
 with pytest.raises(ValueError,match='duplicate'):by_map(rows()+rows()[:1])
 late=rows();late[0]=dict(late[0],length=CAP+1)
 with pytest.raises(ValueError,match='boundary'):by_map(late)

def test_minimal_rollout_bootstraps_actual_full_cap_visible_value():
 import jax
 import jax.numpy as jnp
 from generals.core import game
 from integrations.generals.learning import Carry as OldCarry,make_collect as old_collect
 from integrations.generals_autoresearch.rollout import Carry,make_collect
 class PassValue:
  def apply(self,variables,x,mask):
   logits=jnp.full(mask.shape,-1e9).at[:,0].set(0.)
   return logits,x[:,0,0,13]
 grid=jnp.zeros(BOARD_SHAPE,jnp.int32).at[0,0].set(1).at[FAR].set(2)
 pool=jax.tree.map(lambda a:a[None],game.create_initial_state(grid))
 states=jax.tree.map(lambda a:jnp.repeat(a,2,axis=0),pool)._replace(time=jnp.full(2,CAP-1,jnp.int32))
 key=jax.random.PRNGKey(4);seats=jnp.arange(2);eps=jnp.zeros(2,jnp.int32)
 cfg=dict(num_envs=2,rollout_steps=4,gamma=.995,shaping_scale=0,opponent_mix=[1,0,0,0],history_size=1,packed_rollout=True)
 old=OldCarry(states,key,seats,jnp.zeros(2,jnp.int32),jnp.zeros(2,jnp.int32),{},eps)
 a,reference=old_collect(PassValue(),cfg)({}, {},old,pool)
 b,actual=make_collect(PassValue(),cfg)({},Carry(states,key,seats,eps),pool)
 for name in ('action','old_logp','value','reward','terminated','truncated','next_value','invalid'):
  np.testing.assert_array_equal(actual[name],reference[name])
 assert np.asarray(actual['truncated'][0]).all()
 np.testing.assert_array_equal(actual['next_value'][0],[1.,1.])
 for x,y in zip(jax.tree.leaves(a.states),jax.tree.leaves(b.states)):np.testing.assert_array_equal(x,y)

def test_local_policy_process_obeys_shape_and_legal_mask(bundle):
 from integrations.generals_autoresearch.arena import PolicyProcess
 x=np.zeros((2,*BOARD_SHAPE,14),np.float32);mask=np.zeros((2,ACTION_COUNT),bool);mask[:,0]=True
 with PolicyProcess(bundle,development=True,timeout=30) as p:
  for _ in range(20):ids=p.choose(x,mask)
 np.testing.assert_array_equal(ids,[0,0])

def test_fused_grader_matches_trusted_actions_and_native_boundary(bundle):
 import jax
 import jax.numpy as jnp
 from generals.core import game
 from generals.agents import HunterAgent
 from integrations.generals.environment import observations,encode,action_mask,encode_action,decode_action,scripted_logits,transition
 from integrations.generals_autoresearch.arena import batch_steps,TrustedPanelPolicy
 grid=jnp.zeros(BOARD_SHAPE,jnp.int32).at[0,0].set(1).at[FAR].set(2)
 state=game.create_initial_state(grid);state=state._replace(armies=state.armies+10*state.generals)
 states=jax.tree.map(lambda a:jnp.repeat(a[None],4,axis=0),state)._replace(time=jnp.array([0,123,CAP-1,200]))
 panel=[{'name':n,'kind':n} for n in ('random','expansion','hunter')]+[{'name':'net','kind':'checkpoint'}]
 part=[(p,i,i%2) for i,p in enumerate(panel)];seats=jnp.arange(4)%2;rr=jnp.arange(4)
 trusted={'net':TrustedPanelPolicy(bundle)};hunter=HunterAgent();keys=jax.random.split(jax.random.PRNGKey(9),4)
 prepare,advance=batch_steps(panel,trusted,hunter,part);x,m,opp,legal,views=prepare(states,keys,jnp.int32(37))
 assert views is None  # no `policy` opponent in this part: nothing leaves the fused step for one
 obs=observations(states);xx=encode(obs);mm=action_mask(obs);expected=[]
 for i,p in enumerate(panel):
  oo=jax.tree.map(lambda a:a[i:i+1,1-seats[i]],obs)
  if p['kind']=='checkpoint':chosen=trusted['net'].choose(xx[i:i+1,1-seats[i]],mm[i:i+1,1-seats[i]])[0]
  elif p['kind']=='hunter':chosen=encode_action(hunter.act(jax.tree.map(lambda a:a[0],oo),jax.random.PRNGKey(0)))
  else:chosen=jax.random.categorical(jax.random.fold_in(keys[i],37),scripted_logits(oo,int(p['kind']=='expansion'))[0])
  expected.append(chosen)
 np.testing.assert_array_equal(opp,expected);assert np.asarray(legal).all()
 np.testing.assert_allclose(x,xx[rr,seats],atol=1e-7);np.testing.assert_array_equal(m,mm[rr,seats])
 ids=jnp.zeros(4,jnp.int32);active=jnp.array([True,False,True,True])
 new,done,reward,natural,cap=advance(states,ids,opp,active)
 joint=jnp.zeros((4,2),jnp.int32).at[rr,1-seats].set(opp);reference=transition(states,decode_action(joint))
 frozen=jax.tree.map(lambda n,o:jnp.where(active.reshape((4,)+(1,)*(n.ndim-1)),n,o),reference.final_state,states)
 for a,b in zip(jax.tree.leaves(new),jax.tree.leaves(frozen)):np.testing.assert_array_equal(a,b)
 np.testing.assert_array_equal(reward,reference.reward[rr,seats]);np.testing.assert_array_equal(cap,reference.truncated)
 assert bool(cap[2]) and not bool(natural[2]) and bool(done[2])

def test_practice_plays_its_whole_schedule_in_one_policy_process_and_private_parts_stay_separate(bundle,tmp_path,monkeypatch):
 from integrations.generals.environment import build_pool
 from integrations.generals_autoresearch import arena
 from integrations.generals_autoresearch.prepare import save_pool
 # Games that begin a few ticks before the cap: a whole schedule of full-length games would take minutes here.
 bank=tmp_path/'maps.npz';pool=build_pool([31,32,33,34]);save_pool(pool._replace(time=pool.time+CAP-24),bank);started=[]
 class Counted(arena.PolicyProcess):
  def __init__(self,*args,**kwargs):started.append(1);super().__init__(*args,**kwargs)
 monkeypatch.setattr(arena,'PolicyProcess',Counted)
 panel=[{'name':'random','kind':'random'},{'name':'expansion','kind':'expansion'}]
 practice=arena.evaluate(bundle,panel,pool_path=bank,split='practice',batch_size=4,development=True)
 assert practice['games']==16 and len(started)==1  # four parts, one process
 assert all(1<=r['length']<=24 and (r['truncated'] or r['natural']) for r in practice['rows'])
 del started[:]
 private=arena.evaluate(bundle,panel,pool_path=bank,split='heldout',batch_size=4,development=True)
 assert private['games']==16 and len(started)==4  # one per opponent and seat: never both seats of a map
 key=lambda r:(r['opponent'],r['map_index'],r['seat'])
 assert sorted(practice['rows'],key=key)==sorted(private['rows'],key=key)  # the same games either way

def test_bundle_archive_round_trip_and_nothing_partial_or_extra_is_accepted(bundle,tmp_path):
 import tarfile
 from integrations.generals_autoresearch.bundle import archive_bundle,unpack_bundle,FILES
 archive=tmp_path/'export.tar.gz';archive_bundle(bundle,archive)
 identity=unpack_bundle(archive,tmp_path/'host/bundle')
 assert identity==verify(bundle)['bundle_sha256']==verify(tmp_path/'host/bundle')['bundle_sha256']
 assert sorted(p.name for p in (tmp_path/'host/bundle').iterdir())==sorted((*FILES,'manifest.json'))
 with pytest.raises(FileExistsError):unpack_bundle(archive,tmp_path/'host/bundle')
 cut=tmp_path/'cut.tar.gz';cut.write_bytes(archive.read_bytes()[:len(archive.read_bytes())//2])  # a transfer cut off
 extra=tmp_path/'extra.tar.gz'
 with tarfile.open(extra,'w:gz') as t:
  for n in (*FILES,'manifest.json'):t.add(bundle/n,arcname=n)
  t.add(bundle/'policy.py',arcname='private-seeds.json')
 missing=tmp_path/'missing.tar.gz'
 with tarfile.open(missing,'w:gz') as t:
  for n in FILES:t.add(bundle/n,arcname=n)
 drift=tmp_path/'drift.tar.gz';changed=tmp_path/'changed.py';changed.write_text('# not the sealed source\n')
 with tarfile.open(drift,'w:gz') as t:
  for n in (*FILES,'manifest.json'):t.add(changed if n=='policy.py' else bundle/n,arcname=n)
 link=tmp_path/'link.tar.gz'
 with tarfile.open(link,'w:gz') as t:
  for n in (*FILES,'manifest.json'):
   if n=='weights.npz':
    info=tarfile.TarInfo(n);info.type=tarfile.SYMTYPE;info.linkname='/etc/passwd';t.addfile(info)
   else:t.add(bundle/n,arcname=n)
 for number,bad in enumerate((cut,extra,missing,drift,link)):
  target=tmp_path/f'refused-{number}/bundle'
  with pytest.raises(ValueError):unpack_bundle(bad,target)
  assert not target.exists() and not target.with_name('bundle.partial').exists(),bad.name

def test_practice_and_submission_need_a_finished_training_run_with_a_policy(bundle,tmp_path):
 from harness.games.generals.controller import Controller
 release=tmp_path/'release';(release/'private').mkdir(parents=True)
 write_json(release/'private/panel.json',{'practice':[{'name':'random','kind':'random'}]})
 c=Controller(tmp_path/'session',release,object(),dict(DEFAULT,budget_usd=2.0,min_run_seconds=10))
 c.ledger.queue('good',20,{'kind':'train'});c.ledger.start('good','p1',100);c.ledger.finish('good',120,'complete',str(bundle),verify(bundle)['bundle_sha256'])
 c.ledger.queue('lost',20,{'kind':'train'});c.ledger.start('lost','p2',200)
 c.ledger.finish('lost',220,'failed',str(tmp_path/'session/jobs/lost/bundle'),None,'NotFoundError: The Sandbox is unavailable.')
 c.ledger.queue('halted',20,{'kind':'train'});c.ledger.start('halted','p3',300);c.ledger.finish('halted',320,'stopped')
 for action in ('eval','submit'):
  for name,status in (('lost','failed'),('halted','stopped')):
   with pytest.raises(ValueError,match=f'Run {name} did not finish with a policy: status {status}'):c.handle({'action':action,'name':name,'seconds':120})
  with pytest.raises(ValueError,match='Unknown run: nowhere'):c.handle({'action':action,'name':'nowhere','seconds':120})
 practice=c.handle({'action':'eval','name':'good','seconds':120})['queued']
 with pytest.raises(ValueError,match='is not a training run'):c.handle({'action':'submit','name':practice})
 c.ledger.cancel_queued(practice)
 assert c.handle({'action':'submit','name':'good'})['submission']==1
 c.ledger.queue('busy',20,{'kind':'train'})
 with pytest.raises(ValueError,match='Run busy has not finished: status queued'):c.handle({'action':'eval','name':'busy','seconds':120})

def test_hidden_grade_requires_finalization_and_is_cached(bundle,tmp_path):
 from harness.games.generals.controller import Controller
 from integrations.generals_autoresearch.grading import grade
 release=tmp_path/'release';private=release/'private';private.mkdir(parents=True)
 write_json(private/'panel.json',{'official':[{'name':'a','kind':'random'}]})
 write_json(private/'grade-seeds.json',{'seeds':[1,2]});write_json(private/'anchors.json',{'qualified':False})
 class Provider:
  calls=0
  def evaluate(self,*args,**kwargs):
   self.calls+=1
   return {'public':grade(rows(),{},development=True),'private':{'rows':rows()}}
 provider=Provider();c=Controller(tmp_path/'session',release,provider,dict(DEFAULT,budget_usd=1*60*R,min_run_seconds=10,grade_maps=2))
 with pytest.raises(ValueError,match='Finalize'):c.handle({'action':'grade'})
 c.ledger.queue('a',20,{});c.ledger.start('a','provider',100);c.ledger.finish('a',120,'complete',str(bundle),verify(bundle)['bundle_sha256'])
 c.handle({'action':'submit','name':'a'});c.handle({'action':'finalize'});c.handle({'action':'grade'})
 c.grade_thread.join(5)
 assert c.handle({'action':'grade'})['status']=='complete'
 assert provider.calls==1


def test_paths_support_canonical_and_prepared_sibling_layouts(tmp_path):
 import runpy
 from integrations.generals_autoresearch.contract import ROOT,GAME
 assert ROOT/'simulators/generals' == GAME
 assert ROOT/'simulators/generals/autoresearch' == HERE
 # Prepared container contexts intentionally retain the original package layout.
 control=tmp_path/'frozen/integrations/generals_autoresearch'
 control.mkdir(parents=True)
 shutil.copy2(HERE/'contract.py',control/'contract.py')
 staged=runpy.run_path(str(control/'contract.py'))
 assert staged['ROOT']==tmp_path/'frozen'
 assert staged['GAME']==tmp_path/'frozen/integrations/generals'
