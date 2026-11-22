"""Learned panel opponents with their own network code, and the Elo ladder that rates a panel."""
import json
import shutil
from pathlib import Path
import numpy as np
import pytest
from integrations.generals_autoresearch.contract import HERE,DEFAULT,PANEL_TENSOR_CAP,write_json
from integrations.generals_autoresearch.bundle import seal,verify,save_weights
from integrations.generals_autoresearch.grading import grade,panel_elo
from integrations.generals_autoresearch import ladder
from integrations.generals.environment import RULESET,BOARD_SHAPE,ACTION_COUNT

CAP=RULESET['truncation']

def package(folder,seed,width=4,source=None):
 """A sealed package of the starter architecture; `source` replaces policy.py (another architecture's code)."""
 import jax
 import jax.numpy as jnp
 from integrations.generals_autoresearch.starter.policy import create
 work=folder.with_name(folder.name+'-source');work.mkdir()
 for n in ('policy.py','learner.py','train.py'):shutil.copy2(HERE/'starter'/n,work/n)
 if source:(work/'policy.py').write_text(source)
 cfg={'width':width,'compute_dtype':'float32'};write_json(work/'model.json',cfg)
 params=create(cfg).init(jax.random.PRNGKey(seed),jnp.zeros((1,*BOARD_SHAPE,14)),jnp.ones((1,ACTION_COUNT),bool))['params']
 save_weights(params,work/'weights.npz',cap=PANEL_TENSOR_CAP);seal(work,folder)
 return folder

@pytest.fixture
def bank(tmp_path):
 from integrations.generals.environment import build_pool
 from integrations.generals_autoresearch.prepare import save_pool
 # Games that begin a few ticks before the cap, so every schedule here ends within seconds.
 pool=build_pool([41,42,43,44]);path=tmp_path/'maps.npz';save_pool(pool._replace(time=pool.time+CAP-24),path)
 return path

KEY=lambda r:(r['map_index'],r['seat'])

def test_an_isolated_policy_opponent_plays_the_games_the_in_process_one_plays(tmp_path,bank,monkeypatch):
 import jax
 from integrations.generals.environment import observations,encode,action_mask
 from integrations.generals_autoresearch import arena
 from integrations.generals_autoresearch.rollout import load_pool
 candidate=package(tmp_path/'candidate',1);rival=package(tmp_path/'rival',2)
 # The two ways of running one network choose the same action in every position.
 obs=observations(load_pool(bank));x=np.asarray(encode(obs)[:,0]);mask=np.asarray(action_mask(obs)[:,0])
 with arena.PolicyProcess(rival,development=True,timeout=60,tensor_cap=PANEL_TENSOR_CAP) as isolated:
  np.testing.assert_array_equal(isolated.choose(x,mask),np.asarray(arena.TrustedPanelPolicy(rival).choose(x,mask)))
 # The same network once as a trusted in-process checkpoint and once as a `policy` in its own process.
 inside=[{'name':'net','kind':'checkpoint','bundle':str(rival)}];outside=[{'name':'net','kind':'policy','bundle':str(rival)}]
 expected=arena.evaluate(candidate,inside,pool_path=bank,split='heldout',batch_size=4,development=True)
 started=[]
 class Counted(arena.PolicyProcess):
  def __init__(self,bundle,*args,**kwargs):started.append(Path(bundle).name);super().__init__(bundle,*args,**kwargs)
 monkeypatch.setattr(arena,'PolicyProcess',Counted)
 private=arena.evaluate(candidate,outside,pool_path=bank,split='heldout',batch_size=4,development=True)
 assert sorted(private['rows'],key=KEY)==sorted(expected['rows'],key=KEY) and private['games']==8
 assert started==['candidate','rival']*2  # fresh processes for each seat: neither sees both seats of a private map
 del started[:]
 mixed=[{'name':'random','kind':'random'}]+outside
 practice=arena.evaluate(candidate,mixed,pool_path=bank,split='practice',batch_size=4,development=True)
 assert started==['candidate','rival'] and practice['games']==16  # one process each for the whole practice schedule
 assert sorted((r for r in practice['rows'] if r['opponent']=='net'),key=KEY)==sorted(expected['rows'],key=KEY)

def test_a_policy_opponent_may_have_its_own_architecture_and_a_checkpoint_may_not(tmp_path,bank):
 from integrations.generals_autoresearch import arena
 own=(HERE/'starter/policy.py').read_text()+'\n# a submitted architecture, not the starter\n'
 candidate=package(tmp_path/'candidate',1);rival=package(tmp_path/'rival',2,source=own)
 with pytest.raises(ValueError,match='trusted built-in architecture'):
  arena.evaluate(candidate,[{'name':'net','kind':'checkpoint','bundle':str(rival)}],pool_path=bank,split='heldout',batch_size=4,development=True)
 result=arena.evaluate(candidate,[{'name':'net','kind':'policy','bundle':str(rival)}],pool_path=bank,split='heldout',batch_size=4,development=True)
 assert result['games']==8 and all(r['opponent']=='net' and 1<=r['length']<=24 for r in result['rows'])

def test_a_failing_panel_policy_aborts_grading_and_is_not_a_candidate_fault(tmp_path,bank):
 from integrations.generals_autoresearch import arena
 broken=(HERE/'starter/policy.py').read_text().replace("return jnp.where(mask,logits,-1e9),value","return jnp.where(mask,logits,-1e9)*jnp.nan,value")
 candidate=package(tmp_path/'candidate',1);rival=package(tmp_path/'rival',2,source=broken)
 with pytest.raises(RuntimeError,match='Panel policy net failed') as caught:
  arena.evaluate(candidate,[{'name':'net','kind':'policy','bundle':str(rival)}],pool_path=bank,split='heldout',batch_size=4,development=True)
 assert not isinstance(caught.value,arena.PolicyFailure)

def test_a_panel_opponent_may_be_larger_than_an_entrant(tmp_path,monkeypatch):
 from integrations.generals_autoresearch import bundle
 big=package(tmp_path/'big',3,width=8);count=verify(big)['tensors']
 monkeypatch.setitem(bundle.DEFAULT,'policy_tensor_cap',count-1)
 with pytest.raises(ValueError,match='cap'):verify(big)
 assert verify(big,cap=PANEL_TENSOR_CAP)['tensors']==count and PANEL_TENSOR_CAP>DEFAULT['policy_tensor_cap']

def rows(scores):
 """Grader rows: {opponent: score in every game}, three maps, both seats."""
 return [dict(opponent=p,map_index=i,seat=s,score=v,natural=v!=.5,truncated=v==.5,capture_win=v==1,capture_loss=v==0,
  length=CAP if v==.5 else 100) for p,v in scores.items() for i in (0,1,2) for s in (0,1)]

def test_panel_elo_is_fitted_against_the_rated_opponents_only():
 assert panel_elo(rows({'a':.5}),{}) is None and 'panel_elo' not in grade(rows({'a':.5}),development=True)
 level=grade(rows({'a':.5,'b':1.}),development=True,ratings={'a':120.})['panel_elo']  # b is unrated: left out
 assert level['opponents']==['a'] and abs(level['elo']-120)<1e-6 and not level['at_bound']
 between=panel_elo(rows({'a':1.,'b':0.}),{'a':-100.,'b':100.})
 assert abs(between['elo'])<1e-6 and between['ci95']==[between['elo']]*2
 assert panel_elo(rows({'a':1.}),{'a':0.})['at_bound'] and panel_elo(rows({'a':0.}),{'a':0.})['elo']<-2000

def games(a,b,points,maps=8):
 return [dict(a=a,b=b,map_index=m,a_seat=s,a_points=points[(m+s)%len(points)]) for m in range(maps) for s in (0,1)]

def test_ladder_orders_players_anchors_one_and_refuses_disconnected_games():
 played=games('strong','middle',[1,1,1,0])+games('middle','weak',[1,1,.5,0])+games('strong','weak',[1])
 rated=ladder.ladder(played,('middle',0.),replicates=50)
 assert rated['strong']['elo']>0>rated['weak']['elo'] and rated['middle']['elo']==0 and rated['middle']['ci95']==[0,0]
 assert rated['strong']['games']==32 and rated['strong']['ci95'][0]<=rated['strong']['elo']<=rated['strong']['ci95'][1]
 # 75% over a rated opponent is 191 Elo above it (the fit's small ridge matters only in very few games).
 pair=ladder.ladder(games('x','y',[1,1,1,0],maps=128),('y',0.),replicates=10)
 assert abs(pair['x']['elo']-400*np.log10(3))<1
 with pytest.raises(ValueError,match='do not connect'):ladder.ladder(played+games('p','q',[1]),None,replicates=1)

def test_a_sealed_panel_becomes_a_release_panel_only_when_bound_to_its_engine(tmp_path):
 from integrations.generals_autoresearch.prepare import prepare,audit_release
 own=(HERE/'starter/policy.py').read_text()+'\n# a submitted architecture\n'
 one=package(tmp_path/'one',5,source=own);two=package(tmp_path/'two',6,source=own)
 graded=tmp_path/'graded.json'
 write_json(graded,{'private':{'rows':[dict(r,opponent=p) for p in ('random','expansion','hunter') for r in rows({'x':1. if p!='hunter' else .5})]}})
 tournament=tmp_path/'tournament';(tournament/'pairs/one-vs-two').mkdir(parents=True);write_json(tournament/'result.json',{})
 lines=[dict(g,outcome='natural') for g in games('one','two',[1,1,0],maps=3)]+[dict(a='one',b='two',map_index=9,a_seat=0,a_points=1.,outcome='fault')]
 (tournament/'pairs/one-vs-two/games.jsonl').write_text('\n'.join(map(json.dumps,lines))+'\n')
 record=ladder.fit_record([tournament],[('one',graded)],('hunter',0.),replicates=20)
 assert record['forfeits_left_out']==1 and record['games']==6+18 and record['ratings']['hunter']['elo']==0
 panel=tmp_path/'panel'
 with pytest.raises(ValueError,match='lower-case'):ladder.seal(panel,record,[('Net_A','one',one)])
 with pytest.raises(ValueError,match='scripted'):ladder.seal(panel,record,[('hunter','one',one)])
 bots=ladder.seal(panel,record,[('learned-1','two',two),('learned-2','one',one)],practice=['learned-1'])
 assert [b['origin'] for b in bots]==['two','one'] and bots[1]['elo']>bots[0]['elo']
 with pytest.raises(FileExistsError):ladder.seal(panel,record,[('learned-1','two',two)])
 release=tmp_path/'release';prepare(release,private_maps=2,panel=panel)
 catalog=json.loads((release/'private/panel.json').read_text())
 assert [p['name'] for p in catalog['official']]==['random','expansion','hunter','learned-1','learned-2']
 assert [p['name'] for p in catalog['practice']]==['random','expansion','learned-1']
 assert all('elo' in p for p in catalog['official']) and catalog['official'][2]['elo']==0
 assert [p['kind'] for p in catalog['official'][3:]]==['policy','policy'] and audit_release(release)['public_files']
 public=json.loads((release/'public/practice-opponents.json').read_text())
 assert public[-1]=={'name':'learned-1','kind':'policy'}  # no rating, origin or package identity is public
 assert (release/'private/ladder.json').is_file() and not list((release/'public').rglob('weights.npz'))
 from integrations.generals_autoresearch.manage import panel_paths
 assert Path(panel_paths(release,'official')[3]['bundle'])==(release/'private/bots/learned-1').resolve()
 # A package sealed against other rules (here: an edited engine digest) is refused, as is a changed ladder.
 other=tmp_path/'other-panel';shutil.copytree(panel,other)
 manifest=json.loads((other/'learned-1/manifest.json').read_text());manifest['engine']='0'*64
 from integrations.generals_autoresearch.contract import digest
 manifest['bundle_sha256']=digest({k:v for k,v in manifest.items() if k!='bundle_sha256'});write_json(other/'learned-1/manifest.json',manifest)
 catalog=json.loads((other/'bots.json').read_text());catalog['bots'][0]['bundle_sha256']=manifest['bundle_sha256'];write_json(other/'bots.json',catalog)
 with pytest.raises(ValueError,match='Frozen rules/source mismatch'):prepare(tmp_path/'release-2',private_maps=2,panel=other)
 (panel/'ladder.json').write_text('{}')
 with pytest.raises(ValueError,match='calibration record drift'):prepare(tmp_path/'release-3',private_maps=2,panel=panel)
