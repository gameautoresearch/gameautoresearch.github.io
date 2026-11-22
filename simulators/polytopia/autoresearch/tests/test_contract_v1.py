"""The contract: the ledger in both budget modes, sealed manifests, frozen task fields, grading schedule and provider pins.

Run from the repository root with the game interpreter: .venv-polytopia/bin/python -m pytest -q simulators/polytopia/autoresearch/tests
"""
import json,sys,types
from pathlib import Path
import pytest
HERE=Path(__file__).resolve().parents[1]
ROOT=HERE.parents[2]
BACKEND=ROOT/'harness/games/polytopia'
from simulators.polytopia.autoresearch.v1.core import BENCHMARK,MODE,seal,verify,validate_config,write
from simulators.polytopia.autoresearch.contract import DEFAULT,PRACTICE_SHAPE,PRICE,SECONDS_DEFAULT,TRAIN_SHAPE,lease_shape,rate,validate_session
from simulators.polytopia.autoresearch.ledger import Ledger
from simulators.polytopia.autoresearch import build_context
from harness.games.polytopia.controller import Controller,grade_submission

R=rate(TRAIN_SHAPE)
def session(tmp_path,seconds=600,**config):
 """A ledger whose budget is written in seconds of the training worker."""
 return Ledger(tmp_path/'session',dict(DEFAULT,budget_usd=seconds*R,**config))
def run(seconds,**payload):return (float(seconds),dict({'kind':'train','shape':dict(TRAIN_SHAPE),'rate':R},**payload))
def receipt(start,end,name='sb-1'):return {'provider_id':name,'started_at':start,'finished_at':end}

def test_prices_and_shapes_are_the_contracts():
 # One L40S, 4 cores and 16 GiB at the frozen list prices: about $2.90 an hour, so $23 is about 8 hours of it.
 assert TRAIN_SHAPE=={'gpu':1,'cpu_cores':4,'memory_mib':16384} and PRACTICE_SHAPE=={'gpu':0,'cpu_cores':4,'memory_mib':16384}
 assert R==pytest.approx(0.000542+4*0.00003942+16*0.00000667) and 2.9<3600*R<2.91
 assert DEFAULT['budget_usd']==23.0 and 28500<DEFAULT['budget_usd']/R<28600
 assert lease_shape({},0,8,16)=={'gpu':0,'cpu_cores':8,'memory_mib':16384} and rate(lease_shape({},0,1,2))<R/10
 for bad in ((2,4,16),(1,0,16),(1,33,16),(1,4,1),(1,4,65),(True,4,16),(1,4.0,16)):
  with pytest.raises(ValueError):lease_shape({},*bad)
 with pytest.raises(ValueError):validate_session({'cpu_cores':16})
 with pytest.raises(ValueError):validate_session({'budget_usd':0})

def test_full_allocation_failure_and_hold(tmp_path):
 l=session(tmp_path);l.queue('failed',*run(300),minimum=300)
 with pytest.raises(ValueError,match='at least 300'):l.queue('short',*run(100),minimum=300)
 assert l.public()['remaining_usd']==pytest.approx(300*R)   # a queued grant is held whole, in dollars
 l.start('failed','sb-1',10.);assert l.used()==pytest.approx(300*R)   # and so is a running one
 with pytest.raises(ValueError,match='exceeds the remaining'):l.queue('over',*run(301))
 l.settle('failed',receipt(10,95),'failed','trainer failed')
 row=l.job('failed');assert (row['status'],row['error'])==('failed','trainer failed') and row['charged']==pytest.approx(85*R)
 l.queue('unknown',*run(200));l.uncertain('unknown','no provider record')
 assert l.public()['remaining_usd']==pytest.approx((600-85-200)*R)   # an unknown end holds the whole grant
 with pytest.raises(ValueError):l.submit('failed')
 with pytest.raises(ValueError):l.finalize()
 assert Ledger(l.root).used()==pytest.approx(285*R)   # reopened: the same ledger, no new allowance

def test_seconds_mode_charges_training_worker_seconds_one_job_at_a_time(tmp_path):
 from harness import budget
 assert budget.PRICE==PRICE==DEFAULT['price'] and DEFAULT['budget_mode']=='usd'
 l=Ledger(tmp_path/'s',dict(SECONDS_DEFAULT,budget_seconds=600.))
 assert (l.mode,l.unit,l.budget(),l.limit('max_concurrent_jobs'),l.limit('max_concurrent_cpu_cores'))==('seconds','seconds',600.,1,None)
 assert 'budget_usd' not in l.config and l.config['train_shape']==TRAIN_SHAPE
 lease=dict(kind='lease',shape={'gpu':0,'cpu_cores':2,'memory_mib':4096})
 weight=budget.seconds_weight(lease['shape'],TRAIN_SHAPE)
 l.queue('a',300.,{'kind':'train','shape':dict(TRAIN_SHAPE),'rate':1.0})
 assert l.public()['remaining_seconds']==300 and 'remaining_usd' not in l.public()
 with pytest.raises(ValueError,match='One training run or lease at a time.*: a'):l.queue('b',60.,dict(lease,rate=weight))
 l.queue('eval-a-1',0.,{'kind':'practice','shape':dict(PRACTICE_SHAPE),'rate':0.})   # practice is not counted
 with pytest.raises(ValueError,match='One training run'):l.queue('b',60.,dict(lease,rate=weight))
 l.settle('a',receipt(10,210),'complete');l.settle('eval-a-1',receipt(10,20),'complete')
 assert l.used()==200.   # exactly the training worker's seconds
 l.queue('b',100.,dict(lease,rate=weight));l.settle('b',receipt(5,85,'sb-2'),'stopped')
 assert l.used()==pytest.approx(200+80*weight) and weight==pytest.approx(rate(lease['shape'])/R)
 with pytest.raises(ValueError,match=r'Allocation of 600 s exceeds the remaining \d+ s'):l.queue('c',600.,{'kind':'train','rate':1.0})
 with pytest.raises(ValueError,match='opened in seconds budget mode'):Ledger(l.root,dict(DEFAULT))
 with pytest.raises(ValueError,match='opened in usd budget mode'):Ledger(session(tmp_path).root,dict(SECONDS_DEFAULT))
 with pytest.raises(ValueError):validate_session({'budget_mode':'seconds','budget_seconds':0})
 with pytest.raises(ValueError):validate_session({'budget_mode':'seconds','budget_seconds':600,'budget_usd':1})
 with pytest.raises(ValueError):validate_session({'budget_mode':'minutes'})

def test_budget_is_immutable_and_closes(tmp_path):
 l=session(tmp_path);l.queue('a',*run(590));l.settle('a',receipt(1,591),'complete')
 with pytest.raises(ValueError,match='exceeds the remaining'):l.queue('b',*run(30))
 with pytest.raises(ValueError,match='immutable'):Ledger(l.root,dict(DEFAULT,budget_usd=1200*R))
 with pytest.raises(ValueError,match='needs an explicit session contract'):Ledger(tmp_path/'other')
 l.finalize()
 with pytest.raises(ValueError,match='finalized'):l.queue('c',*run(1))
 assert l.lifecycle()=='finalized'

def test_concurrency_limits_and_shape_prices(tmp_path):
 l=session(tmp_path,100000);cpu=lambda cores:dict(kind='lease',shape={'gpu':0,'cpu_cores':cores,'memory_mib':2048})
 l.queue('g1',*run(300));l.queue('g2',*run(300))
 with pytest.raises(ValueError,match='At most 2 GPUs'):l.queue('g3',*run(300))
 l.queue('c1',60.,cpu(32))   # two training workers hold 8 cores
 with pytest.raises(ValueError,match='At most 64 CPU cores'):l.queue('c2',60.,cpu(25))
 l.cancel_queued('c1');l.queue('c2',60.,cpu(16));l.queue('c3',60.,cpu(16))
 with pytest.raises(ValueError,match='At most 4 jobs'):l.queue('c4',60.,cpu(1))
 # a lease is charged its own shape's price; practice rows carry rate 0
 l.settle('c2',receipt(100,140,'sb-c2'),'stopped');assert l.job('c2')['charged']==pytest.approx(40*rate(cpu(16)['shape']))
 l.queue('eval-g1-1',0.,dict(kind='practice',run='g1',shape=dict(PRACTICE_SHAPE),rate=0.));l.start('eval-g1-1','cpu',5.)
 l.settle('eval-g1-1',{'started_at':5.,'finished_at':5000.},'complete');assert l.job('eval-g1-1')['charged']==0

def test_unstarted_workers_are_never_charged(tmp_path):
 l=session(tmp_path);l.queue('a',*run(300))
 with pytest.raises(ValueError):l.abandon_unstarted('a',{'provider_id':'sb-1','started_at':5.},'x')   # it started: charge it
 with pytest.raises(ValueError):l.abandon_unstarted('a',{},'x')   # no provider record: nothing proves it never ran
 l.abandon_unstarted('a',{'provider_id':'sb-1','started_at':0},'never started',stopped=True)
 assert (l.job('a')['status'],l.job('a')['charged'],l.used())==('stopped',0,0)
 l.queue('b',*run(300));l.cancel_queued('b','refused','failed');assert (l.job('b')['status'],l.job('b')['error'])==('failed','refused')
 with pytest.raises(ValueError):l.cancel_queued('b')

def test_seal_detects_checkpoint_or_source_tamper(tmp_path):
 (tmp_path/'final.npz').write_bytes(b'checkpoint');(tmp_path/'network.py').write_text('source')
 identity=seal(tmp_path);assert verify(tmp_path)==identity
 (tmp_path/'network.py').write_text('changed')
 with pytest.raises(ValueError):verify(tmp_path)

@pytest.mark.parametrize('key,value',[('size',14),('size',18),('size',11.0),('players',4),('max_turns',200),('max_turns',MODE.ROUND_LIMIT+1),('max_turns',None),
 ('maps','proposed-v1'),('maps','classic'),('map_seed_low',0),('map_seed_high',300001)])
def test_frozen_dimensions_and_horizon(key,value):
 cfg=json.loads((HERE/'recipes/smoke.json').read_text());cfg[key]=value
 with pytest.raises(ValueError,match='Frozen task field '+key):validate_config(cfg)

def test_the_small_mode_is_the_frozen_configuration():
 # 11x11, both Bardur, drylands only, training seeds 200000-300000; the round limit is whatever mode.py says (30-50).
 assert (MODE.SIZE,MODE.PLAYERS,MODE.MAPS,MODE.MAP_SEED_LOW,MODE.MAP_SEED_HIGH)==(11,2,'drylands',200000,300000)
 assert type(MODE.ROUND_LIMIT) is int and 30<=MODE.ROUND_LIMIT<=50
 assert MODE.FROZEN=={'size':11,'players':2,'maps':'drylands','max_turns':MODE.ROUND_LIMIT,'map_seed_low':200000,'map_seed_high':300000}
 assert BENCHMARK=='polytopia-bardur-1v1-11-drylands-v1' and MODE.MAP_TABLE=='table-drylands-11-2p-200000-300000.npz'
 for recipe in ('starter.json','smoke.json'):
  cfg=json.loads((HERE/'recipes'/recipe).read_text());assert {k:cfg[k] for k in MODE.FROZEN}==MODE.FROZEN,recipe
  assert validate_config(cfg) is cfg
 # A recipe may leave a frozen field out (the trainer's defaults are the same module), never change one.
 assert validate_config({'learning_rate':.001,'entropy_coef':.01})

def test_the_round_limit_and_the_mode_are_defined_once():
 """Code reads rl/mode.py; the written copies that cannot import it must agree with it."""
 import re
 limit=MODE.ROUND_LIMIT;backend=sorted(BACKEND.glob('*.py'))
 code=[HERE/'v1'/name for name in ('core.py','grade.py','policy.py','preflight.py','train.py')]+[HERE/'build_context.py',HERE/'prepare.py',*backend,
  *(HERE/'starter'/name for name in ('src/polytopia_bot/rl/gpu_resident.py','src/polytopia_bot/rl/infra.py','src/polytopia_bot/rl/network.py','src/polytopia_bot/rl/ppo.py','experiments/gpu_resident_train.py'))]
 stale=re.compile(r"training_round_limit|TRAINING_ROUND_LIMITS|proposed-v1|max_turns\s*=\s*\d|RustGames\(\d|MapSource\('|size\s*=\s*\d|'size'\s*:\s*\d|view_layout\(\d|bardur-1v1-14")
 for path in code:
  for number,line in enumerate(path.read_text().splitlines(),1):assert not stale.search(line),f'{path.name}:{number}: {line.strip()[:120]}'
 for name in ('grade.py','policy.py','preflight.py'):assert 'mode import' in (HERE/'v1'/name).read_text() or 'from grade import' in (HERE/'v1'/name).read_text()
 # the one definition, and nothing else in the package assigning it
 assert len(re.findall(r'^ROUND_LIMIT\s*=',(HERE/'starter/src/polytopia_bot/rl/mode.py').read_text(),re.M))==1
 contract=json.loads((HERE/'benchmark.json').read_text())
 assert contract['benchmark_id']==BENCHMARK and contract['task']['board']==[MODE.SIZE,MODE.SIZE] and contract['task']['players']==MODE.PLAYERS
 assert contract['task']['maps']==MODE.MAPS and contract['task']['round_limit']==limit and contract['task']['map_seeds']==[MODE.MAP_SEED_LOW,MODE.MAP_SEED_HIGH]
 assert (HERE/contract['task']['mode_definition']).is_file()
 brief=(HERE/'task/brief.md').read_text()
 assert 'same outcome rule' in brief and 'bootstraps' not in brief and 'no reward' not in brief
 for text in (f'{MODE.SIZE}×{MODE.SIZE} board',f'reaches {limit} rounds',f'{limit}-round cap',f'`size` {MODE.SIZE}',f'`maps` {MODE.MAPS}',f'`max_turns` {limit} '):assert text in brief,text
 rules=[HERE/'task/game_rules/rules/mode-and-maps.md',HERE/'wiki_maintenance/source_markdown/rules/mode-and-maps.md',HERE/'wiki_maintenance/raw_build/rules/mode-and-maps.md']
 for page in rules:assert f'`{BENCHMARK}`' in page.read_text() and f'round limit of{limit} rounds' in page.read_text() and f'an{MODE.SIZE}×{MODE.SIZE} board' in page.read_text(),page
 assert f'a round limit of {limit} rounds' in (HERE/'CONTRACT.md').read_text() and f'a round limit of {limit} rounds' in (HERE/'README.md').read_text()
 from harness.games.polytopia import SPEC
 assert SPEC['mode']==f'Bardur vs Bardur 1v1, {MODE.SIZE}x{MODE.SIZE}, fog of war, {MODE.MAPS} maps, {limit}-round cap'
 # no written page of the task still describes the earlier mode
 for page in [HERE/'task/brief.md',*sorted((HERE/'task/game_rules').rglob('*.md'))]:
  text=page.read_text()
  for old in ('14×14','14x14','proposed-v1','200-round','200 rounds','y*14','196','131076'):assert old not in text or (old in ('14x14',) and page.name=='scope-and-limitations.md'),(page.name,old)

def test_a_release_needs_the_map_table_and_practice_opponents_of_this_mode(tmp_path,monkeypatch):
 """prepare() checks both local inputs before writing anything: a missing one, or one of another mode, stops it."""
 import hashlib,numpy as np
 from simulators.polytopia.autoresearch import prepare as constructor
 cache=tmp_path/'cache';cache.mkdir();monkeypatch.setattr(build_context,'CACHE',cache);monkeypatch.setattr(build_context,'PANEL_DIR',cache/'practice')
 out=tmp_path/'release'
 with pytest.raises(FileNotFoundError,match='(?s)Missing the training map table.*--env game-autoresearch-project experiments/modal_map_table.py::build.*--mix drylands --size 11'):constructor.prepare(out)
 n,maps=MODE.SIZE**2,MODE.MAP_SEED_HIGH-MODE.MAP_SEED_LOW
 def table(lo=MODE.MAP_SEED_LOW,hi=MODE.MAP_SEED_HIGH,types=(MODE.MAPS,),tiles=n,count=maps):
  np.savez_compressed(cache/MODE.MAP_TABLE,lo=np.int64(lo),hi=np.int64(hi),types=np.array(types),kind=np.zeros(count,np.int8),
   **{k:np.zeros((count,tiles),np.int16) for k in ('terrain','structure','resource')},capitals=np.zeros((count,2),np.int16))
 for wrong in (dict(tiles=196),dict(types=('classic','drylands')),dict(lo=0),dict(count=10)):  # e.g. the 14x14 table renamed
  table(**wrong)
  with pytest.raises(ValueError,match='not the map table of this game mode'):constructor.prepare(out)
 table();assert build_context.map_table()==cache/MODE.MAP_TABLE
 with pytest.raises(FileNotFoundError,match='Missing practice opponent starter-200m'):constructor.prepare(out)
 def opponent(label,size=MODE.SIZE,sha=None):
  folder=cache/'practice'/label;folder.mkdir(parents=True,exist_ok=True);(folder/'final.npz').write_bytes(b'weights of '+label.encode())
  write(folder/'final.json',{'config':{'size':size,'players':2},'sha256':sha or hashlib.sha256((folder/'final.npz').read_bytes()).hexdigest()})
  (folder/'network.py').write_text('# network');(folder/'ar_arch.json').write_text('{}')
 for label in build_context.PANEL[:-1]:opponent(label)
 with pytest.raises(FileNotFoundError,match='Missing practice opponent research-top'):constructor.prepare(out)
 opponent('research-top',size=14)   # the earlier mode's checkpoint
 with pytest.raises(ValueError,match='research-top was trained for a 14x14 board'):constructor.prepare(out)
 opponent('research-top',sha='0'*64)
 with pytest.raises(ValueError,match='does not match the digest'):constructor.prepare(out)
 opponent('research-top');assert list(build_context.panel())==list(build_context.PANEL)
 assert not out.exists()   # nothing was written by any refused attempt
 with pytest.raises(ValueError,match='panel is true'):constructor.prepare(out,panel='no')

def test_the_earlier_modes_local_inputs_are_not_picked_up():
 # The 14x14 table and practice checkpoints keep their own names in the git-ignored cache; this mode reads others.
 assert build_context.MAP_NAME=='table-drylands-11-2p-200000-300000.npz'!='table-proposed-v1-14-2p-200000-300000.npz'
 assert build_context.PANEL_DIR==HERE/'.cache/practice-drylands-11'

def test_full_practice_and_grading_refuse_a_release_without_opponents(tmp_path):
 write(tmp_path/'private/schedule.json',{'seeds':[1000000],'opponent_seed':0})
 with pytest.raises(ValueError,match='prepared without practice opponents'):grade_submission(None,'unused',tmp_path/'release',tmp_path/'private',False)

def test_new_map_tables_are_built_only_in_the_project_environment():
 text=(HERE.parent/'experiments/modal_map_table.py').read_text()
 assert "ENVIRONMENT = 'game-autoresearch-project'" in text and "environment_name=ENVIRONMENT" in text
 assert "os.environ.get('MODAL_ENVIRONMENT') != ENVIRONMENT" in text and "'staging') != 'staging'" not in text
 assert "from_name('polytopia-structured-rl')" not in text and f"VOLUME = '{build_context.MAP_TABLE_VOLUME}'" in text

def test_contract_paths_and_noncalibrated_score():
 d=json.loads((HERE/'benchmark.json').read_text());assert d['budget']['research_usd']==DEFAULT['budget_usd']==23.0
 assert d['session_contract']==DEFAULT['version'] and d['benchmark_id']==DEFAULT['benchmark_id']
 assert (d['resources']['cpu_cores'],d['resources']['memory_gib']*1024)==(TRAIN_SHAPE['cpu_cores'],TRAIN_SHAPE['memory_mib'])
 assert (d['budget']['min_run_seconds'],d['budget']['min_lease_seconds'])==(DEFAULT['min_run_seconds'],DEFAULT['min_lease_seconds'])
 assert (d['starter']['policy_parameter_cap'],d['starter']['critic_parameter_cap'])==(DEFAULT['policy_parameter_cap'],DEFAULT['critic_parameter_cap'])
 assert d['evaluation']['normalized_reference_score'] is None
 for rel in [*d['starter']['editable_files'],d['starter']['recipe'],*d['agent'].values()]:assert (HERE/rel).is_file()
 assert all(command.startswith('.venv-polytopia-ar/bin/python -m harness ') for command in d['commands'].values())
 from harness.games.polytopia import SPEC
 assert SPEC['research_usd']==d['budget']['research_usd'] and SPEC['smoke_research_usd']==d['budget']['smoke_research_usd']
 assert SPEC['wall_seconds']==d['budget']['research_wall_seconds'] and SPEC['final_train_seconds']==d['budget']['fresh_final_train_seconds']
 assert SPEC['leases'] is True and SPEC['agents']==('terminus-2','claude-code')
 assert SPEC['budget_mode']==d['budget']['default_mode']=='seconds' and SECONDS_DEFAULT['budget_seconds']==SPEC['research_seconds']
 assert (SPEC['research_seconds'],SPEC['smoke_research_seconds'])==(d['budget']['seconds_mode']['research_seconds'],d['budget']['seconds_mode']['smoke_research_seconds'])==(14400,600)
 assert (ROOT/SPEC['contract'])==HERE/'benchmark.json'

def test_checkpoint_optimizer_reload(tmp_path,runtime):
 # Checkpoint persistence for inference and optimizer tensors; not exact environment resume.
 import numpy as np,jax,jax.numpy as jnp
 from polytopia_bot.rl.runner import Config,save_checkpoint,load_checkpoint
 params={'weights':jnp.arange(12,dtype=jnp.float32).reshape(3,4)};opt={'moment':jnp.ones((3,4))}
 p=tmp_path/'x.npz';save_checkpoint(p,params,Config(),stage='test',optimizer=opt)
 loaded=load_checkpoint(p,params);np.testing.assert_array_equal(loaded['weights'],params['weights'])
 with np.load(p,allow_pickle=False) as z:np.testing.assert_array_equal(z['o0'],opt['moment'])
 p.write_bytes(b'tampered')
 with pytest.raises(ValueError):load_checkpoint(p,params)

def test_milestone_missing_prior_submission_is_explicit(tmp_path):
 l=session(tmp_path);l.queue('a',*run(300));l.settle('a',receipt(1,251),'complete')
 points={p['fraction']:p for p in l.public()['points']}
 assert points[str(1/3)]['submission'] is None and points[str(1/3)]['charged']==pytest.approx(250*R) and str(2/3) not in points

def test_teardown_crossing_milestone_does_not_adopt_late_submission(tmp_path):
 l=session(tmp_path);(tmp_path/'bundle').mkdir()
 l.queue('old',*run(60));l.settle('old',receipt(1,2,'sb-0'),'complete');l.attach('old',tmp_path/'bundle','sha-old');l.submit('old')
 l.queue('a',*run(300));l.settle('a',receipt(100,301),'complete');l.attach('a',tmp_path/'bundle','sha-late');l.submit('a')
 points={p['fraction']:p['submission'] for p in l.public()['points']}
 assert l.submissions()[points[str(1/3)]-1]['job']=='old'   # the policy already submitted when the third was crossed
 l.finalize();final={p['fraction']:p['submission'] for p in l.public()['points']}[str(1.0)]
 assert l.submissions()[final-1]['job']=='a'

def test_duplicate_settlement_and_unresolved_close(tmp_path):
 l=session(tmp_path);l.queue('x',*run(100))
 with pytest.raises(ValueError):l.finalize()
 with pytest.raises(ValueError):l.settle('x',{'provider_id':'sb-1','started_at':70,'finished_at':1},'failed')
 with pytest.raises(ValueError):l.settle('x',{'provider_id':'sb-1','started_at':1},'failed')   # no end: not a settlement
 l.start('x','sb-1',1.)
 with pytest.raises(ValueError,match='identity differs'):l.settle('x',receipt(1,70,'sb-other'),'failed')
 with pytest.raises(ValueError,match='start differs'):l.settle('x',receipt(5,70),'failed')
 l.settle('x',receipt(1,70),'failed')
 with pytest.raises(ValueError):l.settle('x',receipt(1,2),'complete')
 assert l.used()==pytest.approx(69*R)
 l.annotate('x','Stopped by request',stopped=True);assert (l.job('x')['status'],l.job('x')['charged'])==('stopped',pytest.approx(69*R))

def test_balanced_schedule_and_cap_labels(runtime):
 sys.path.insert(0,str(HERE/'v1'))
 try:from grade import summarize
 finally:sys.path.remove(str(HERE/'v1'))
 rows=[{'seed':41,'seat':1,'reward':1.,'capped':True},{'seed':41,'seat':2,'reward':-1.,'capped':False}]
 result=summarize(rows,[41])
 assert result['raw_score_0_100']==50 and result['natural_games']==1 and result['capped_games']==1
 assert result['normalized_reference_score'] is None
 with pytest.raises(ValueError):summarize(rows[:1],[41])
 with pytest.raises(ValueError):summarize(rows+rows,[41])

def test_panel_inventory_and_public_boundary():
 text=(BACKEND/'provider.py').read_text();assert ".add_local_dir(self.release/'host-private-panel'" not in text
 assert "user=uid,group=uid,extra_groups=[]" in (HERE/'v1/grade.py').read_text()
 assert "RustGames(SIZE,1,map_batch=True,max_turns=ROUND_LIMIT,players=PLAYERS,no_resign=True)" in (HERE/'v1/grade.py').read_text()
 assert "MapSource(MAPS,SIZE)" in (HERE/'v1/grade.py').read_text()

def test_provider_reserves_before_create_with_fixed_resources_in_the_project_environment():
 text=(BACKEND/'provider.py').read_text();assert text.index('self.account.reserve(role')<text.index('sb=modal.Sandbox.create')
 assert text.index('ledger.reserve(name,grant)')<text.index("sb=self._create('training-'+name,grant,True")
 # a lease reserves before its worker is created too, is network-blocked like every worker, and mounts only /scratch
 lease=text[text.index(' def hold(self'):text.index(' def lease_live(self')]
 assert lease.index('ledger.reserve(name,grant)')<lease.index("sb=self._create('lease-'+name,grant,shape=shape,volumes={SCRATCH:")
 assert text.count('modal.Sandbox.create(')==1 and 'block_network=True' in text and 'secrets=[]' in text
 assert "UNPRIVILEGED='setpriv --reuid=65534 --regid=65534 --clear-groups --no-new-privs'" in text
 assert "(TRAIN_SHAPE['cpu_cores'],TRAIN_SHAPE['memory_mib']//1024) if gpu" in text and "timeout=max(1,int(seconds-15))" in text and "gpu=hardware.GPU if gpu else None" in text
 # Every worker is created in the project environment. A Modal environment named staging is never named; the
 # word alone is allowed, since the trainer handoff has a staging directory.
 import re
 assert set(re.findall(r"environment_name=([\w.'\"]+)",text))=={'ENVIRONMENT'}
 assert not re.search(r"env\w*\s*[=:]\s*['\"]staging|--env[ =]staging",text,re.I)
 assert 'staging' not in (BACKEND/'transport.py').read_text()
 from harness.policy import ENVIRONMENT
 assert ENVIRONMENT=='game-autoresearch-project'

def test_a_session_lease_cap_is_validated():
 from simulators.polytopia.autoresearch.contract import validate_session
 assert validate_session(dict(budget_mode='seconds',budget_seconds=600,max_lease_seconds=900))['max_lease_seconds']==900
 for bad in (30,1.5,True,'900'):
  with pytest.raises(ValueError,match='max_lease_seconds'):validate_session(dict(budget_mode='seconds',budget_seconds=600,max_lease_seconds=bad))

def test_cpu_only_workers_reserve_one_core_under_the_shape_cap():
 text=(BACKEND/'provider.py').read_text()
 assert 'CPU_REQUEST,MEMORY_REQUEST_MIB=1.0,2048' in text and 'lean=not gpu and shape is None' in text
 assert 'cpu=(min(CPU_REQUEST,cores),cores) if lean else (cores,cores)' in text

def test_worker_entrypoints_named_by_the_provider_exist():
 text=(BACKEND/'provider.py').read_text()
 for name in ('train.py','grade.py','preflight.py'):assert "'/opt/v1/"+name+"'" in text and (HERE/'v1'/name).is_file()
 assert "'/opt/v1/policy.py'" in (HERE/'v1/grade.py').read_text() and (HERE/'v1/policy.py').is_file()
 assert ".add_local_dir(self.release/'v1','/opt/v1'" in text

def test_parameter_cap_rejects_before_gpu_reservation(tmp_path,monkeypatch):
 monkeypatch.setitem(sys.modules,'modal',types.SimpleNamespace())
 from harness.games.polytopia.provider import Provider
 release=tmp_path/'release';release.mkdir();(release/'source.txt').write_text('frozen');seal(release)
 workspace=tmp_path/'workspace';workspace.mkdir();(workspace/'recipe.json').write_text('{}')
 provider=object.__new__(Provider);provider.release=release;provider.parameter_counts=lambda ws:{'policy':700000,'critic':1,'errors':['policy parameter cap exceeded']}
 class NoGpu:
  def reserve(self,*args):raise AssertionError('GPU must not be reserved after static cap failure')
 with pytest.raises(ValueError,match='parameter cap'):provider.train(NoGpu(),'bad',1200,workspace,tmp_path/'out')

def test_stop_before_allocation_reserves_nothing(tmp_path,monkeypatch):
 monkeypatch.setitem(sys.modules,'modal',types.SimpleNamespace())
 from harness.games.polytopia.provider import Provider
 release=tmp_path/'release';release.mkdir();(release/'source.txt').write_text('frozen');seal(release)
 workspace=tmp_path/'workspace';workspace.mkdir();(workspace/'recipe.json').write_text('{}')
 provider=object.__new__(Provider);provider.release=release;provider.parameter_counts=lambda ws:{'policy':1,'critic':1,'errors':[]}
 class NoGpu:
  def reserve(self,*args):raise AssertionError('A stopped run must not reserve')
 with pytest.raises(RuntimeError,match='before any GPU allocation'):provider.train(NoGpu(),'late',1200,workspace,tmp_path/'out',stop=lambda:True)

def test_own_training_log_is_readable_after_failure_but_cannot_address_private_files(tmp_path):
 ctl=Controller(tmp_path,tmp_path/'release',None,dict(DEFAULT))
 ctl.ledger.queue('trial-1',*run(300));ctl.ledger.settle('trial-1',receipt(1,200),'failed','Training failed; see retained stderr')
 logs=tmp_path/'jobs/trial-1';logs.mkdir(parents=True)
 (logs/'stdout.log').write_text('progress\n'*1000)
 (logs/'stderr.log').write_text('ValueError: shape mismatch in own learner')
 private=tmp_path/'private';private.mkdir();(private/'stderr.log').write_text('private marker')
 result=ctl.log('trial-1',tail=80)
 assert result['stderr']=='ValueError: shape mismatch in own learner'
 assert result['stdout'].count('\n')==79 and result['status']=='failed' and 'retained stderr' in result['failure']
 assert result['charged_usd']==pytest.approx(199*R) and result['worker_seconds']==199
 for name in ('../private','private','../../private/stderr.log',None,7):
  with pytest.raises(ValueError,match='Unknown run'):ctl.log(name)
 assert 'private marker' not in json.dumps(result)

def test_full_panel_schedule_and_equal_opponent_weight(tmp_path):
 release=tmp_path/'release';private=tmp_path/'private';seeds=list(range(1000000,1000128))
 write(release/'host-private-panel/inventory.json',{'panel':[{'name':str(i),'directory':str(i)} for i in range(4)]})
 write(private/'schedule.json',{'seeds':seeds,'opponent_seed':0})
 class FakeProvider:
  def evaluate(self,bundle,seeds,opponent_seed,path,opponent_bundle):
   rows=[{'seed':s,'seat':seat,'reward':1 if Path(opponent_bundle).name=='0' else -1,'capped':False} for s in seeds for seat in (1,2)]
   write(path,{'details':rows,'cpu_seconds':1});return {}
 result=grade_submission(FakeProvider(),'unused',release,private,False)
 assert result['games']==1024 and result['raw_score_0_100']==25
 assert len(result['by_opponent'])==4 and result['by_seat']=={'1':25.,'2':25.}

def _panel(tmp_path):
 release=tmp_path/'release';private=tmp_path/'private';seeds=list(range(1000000,1000128))
 write(release/'host-private-panel/inventory.json',{'panel':[{'name':str(i),'directory':str(i)} for i in range(4)]})
 write(private/'schedule.json',{'seeds':seeds,'opponent_seed':0})
 return release,private

def test_a_lost_grade_worker_is_replayed_and_its_records_kept(tmp_path):
 release,private=_panel(tmp_path)
 class NotFoundError(Exception):pass
 calls=[]
 class FlakyProvider:
  def evaluate(self,bundle,seeds,opponent_seed,path,opponent_bundle):
   key=(Path(opponent_bundle).name,seeds[0]);calls.append(key)
   write(Path(path).with_suffix('.worker.json'),{'sandbox_id':'sb-'+str(len(calls))})
   if key==('2',1000008) and calls.count(key)==1:raise NotFoundError('The Sandbox is unavailable')
   rows=[{'seed':s,'seat':seat,'reward':1,'capped':False} for s in seeds for seat in (1,2)]
   write(path,{'details':rows,'cpu_seconds':1});return {}
 result=grade_submission(FlakyProvider(),'unused',release,private,False)
 assert result['games']==1024 and calls.count(('2',1000008))==2 and len(calls)==129
 shard=private/'grades/2'
 assert json.loads((shard/'8.lost-1.worker.json').read_text())['sandbox_id']!=json.loads((shard/'8.worker.json').read_text())['sandbox_id']
 assert json.loads((shard/'8.lost-1.error.json').read_text())['type']=='NotFoundError'

def test_a_failed_evaluation_is_not_replayed(tmp_path):
 release,private=_panel(tmp_path)
 calls=[]
 class BrokenProvider:
  def evaluate(self,bundle,seeds,opponent_seed,path,opponent_bundle):
   calls.append(seeds[0]);raise RuntimeError('CPU evaluation failed')
 with pytest.raises(RuntimeError,match='CPU evaluation failed'):grade_submission(BrokenProvider(),'unused',release,private,False)
 assert len(calls)==len(set(calls)) and not list(private.rglob('*.lost-*'))

def test_a_worker_lost_on_every_attempt_fails_the_grade(tmp_path):
 release,private=_panel(tmp_path)
 class Dead:
  def evaluate(self,bundle,seeds,opponent_seed,path,opponent_bundle):raise ConnectionResetError('GOAWAY')
 with pytest.raises(ConnectionResetError):grade_submission(Dead(),'unused',release,private,False)
