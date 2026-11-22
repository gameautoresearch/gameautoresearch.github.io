"""Immutable offline preparation: minimal public context and private grading data."""
from pathlib import Path
import json
import secrets
import shutil
import subprocess
import time
import numpy as np
from .contract import HERE,ROOT,GAME,RUNTIME,DEFAULT,TRAIN_SHAPE,BUNDLED,PANEL_TENSOR_CAP,rate,digest,sha,frozen_manifest,public_seeds,write_json
from .bundle import export_existing

CONTROL=('__init__.py','contract.py','bundle.py','rollout.py','wire.py','policy_worker.py','arena.py','grading.py','train_entry.py','training_limits.py','evaluate_entry.py','export_entry.py','flops.py','learned.py')

# What the researcher starts from. `ppo` is the working PPO starter. `empty` is boilerplate only: self-play plumbing
# with no learning, an empty learner and a linear placeholder network, its own brief and configs, and nothing to
# copy: neither the box nor an unprivileged process on a worker can read the project's training code, the
# upstream engine's reward helpers, scripted players and example trainer, or the wrapper's scripted players.
# `slowsim` is the empty starter on an editable simulator: the box's `simulator/` is a plain-Python implementation of
# the rules (`slow_sim/generals_sim`, checked tick for tick against the engine by `slow_sim/parity.py`) that the
# researcher may read and edit like the training files, and that the boilerplate trains on. Neither the box nor an
# unprivileged worker process can read or import the pinned JAX engine or the wrapper; practice, grading and
# tournaments still play on them, as root.
STARTERS={'ppo':dict(agent='starter',configs={'starter.json':'starter.json','smoke.json':'smoke.json'},brief=None),
          'empty':dict(agent='starter_empty',configs={'starter.json':'empty.json','smoke.json':'empty-smoke.json'},brief='brief-empty.md'),
          'slowsim':dict(agent='starter_slowsim',configs={'starter.json':'slowsim.json','smoke.json':'slowsim-smoke.json'},
                         brief='brief-slowsim.md',simulator='slow_sim/generals_sim',rules='game_rules_slowsim')}
# What the box shows of an editable-simulator release: the plain simulator, editable, and nothing of the engine.
SLOW_SIM=('__init__.py','game.py','view.py')
SLOW_README='''# Simulator

`generals_sim/` is the Generals simulator your training runs on: the rules of one mid-size 1v1 game
(`game.py`), what a player sees of it and the action codec (`view.py`). It is yours: read, run and change it like
the files in `training/`. Its files travel with `training/` and `configs/` to a worker (`/work/simulator`, on the
Python path there), and Python in this box can import it too (`cd simulator && python -c "import generals_sim"`).
The game itself is defined in `../game_rules/`; practice, grading and tournaments play it with the host's own
simulator of those rules, so what you change here changes only your own training games.
'''
# The constants an unprivileged process of an editable-simulator release imports in place of the wrapper.
SLOW_ENVIRONMENT='''"""Board and action sizes of the mid-size 1v1 mode (sides 12..15, padded to 16)."""
PAD_TO = 16
BOARD_SHAPE = (PAD_TO, PAD_TO)
ACTION_COUNT = 1+PAD_TO*PAD_TO*8
'''
# The engine's rule modules, byte for byte as pinned. Its package files are rewritten below because the pinned
# ones import the reward helpers, the scripted players and the renderer.
PUBLIC_ENGINE=('core/action.py','core/env.py','core/game.py','core/grid.py','core/observation.py',
               'modifiers/__init__.py','modifiers/build_castles.py','modifiers/deathtouch.py')
ENGINE_INIT='''# JAX-based Generals.io implementation: the rule modules a training process imports.
from generals.core.env import GeneralsEnv
from generals.core.game import GameState, GameInfo, create_initial_state, step, get_observation
from generals.core.observation import Observation
from generals.core.action import compute_valid_move_mask, create_action
from generals.core.grid import generate_grid
'''
PUBLIC_CONTROL=('__init__.py','contract.py','bundle.py','wire.py','flops.py','learned.py','policy_worker.py')
# Wrapper functions that are players or training aids, not rules: left out of what a researcher can read or import.
WRAPPER_WITHHELD=('potential','scripted_logits')
# The same selection for the box's read-only `simulator/` (harness.workspace.stage): paths under it.
BOX_WITHHELD=(r'^vendor/engines/generals-bots/generals/(agents|gui|remote|assets)/'
              r'|^vendor/engines/generals-bots/generals/core/(rewards|rendering|config)\.py$'
              r'|^simulators/generals/autoresearch/rollout\.py$')

def without(source,names):
 """`source` (Python text) without the top-level functions `names`."""
 import ast
 lines=source.splitlines(keepends=True);drop=set();found=set()
 for node in ast.parse(source).body:
  if isinstance(node,ast.FunctionDef) and node.name in names:
   first=min([node.lineno]+[d.lineno for d in node.decorator_list])
   drop.update(range(first-1,node.end_lineno));found.add(node.name)
 if found!=set(names):raise ValueError('Wrapper function to withhold is missing: '+', '.join(sorted(set(names)-found)))
 text=''.join(line for i,line in enumerate(lines) if i not in drop)
 while '\n\n\n\n' in text:text=text.replace('\n\n\n\n','\n\n\n')
 compile(text,'environment.py','exec')
 return text

def unprivileged_tree(starter='empty'):
 """{relative path: bytes}: everything an unprivileged process can import in an empty-starter release.

 An editable-simulator release (`slowsim`) gets no engine and no wrapper: its training imports the simulator the
 researcher sends, and the control modules only need the board and action sizes."""
 if starter=='slowsim':
  files={'integrations/__init__.py':b'','integrations/generals/__init__.py':(GAME/'__init__.py').read_bytes(),
         'integrations/generals/environment.py':SLOW_ENVIRONMENT.encode()}
  for n in PUBLIC_CONTROL:files['integrations/generals_autoresearch/'+n]=(HERE/n).read_bytes()
  return files
 engine=ROOT/'candidates/generals-bots/generals'
 files={'generals/'+n:(engine/n).read_bytes() for n in PUBLIC_ENGINE}
 files['generals/__init__.py']=ENGINE_INIT.encode();files['generals/core/__init__.py']=b''
 files['integrations/__init__.py']=b''
 for n in RUNTIME:files['integrations/generals/'+n]=(GAME/n).read_bytes()
 files['integrations/generals/environment.py']=without((GAME/'environment.py').read_text(),WRAPPER_WITHHELD).encode()
 for n in PUBLIC_CONTROL:files['integrations/generals_autoresearch/'+n]=(HERE/n).read_bytes()
 return files

def box_views():
 """What the box shows in place of the three files the unprivileged tree rewrites, by path under `simulator/`."""
 tree=unprivileged_tree()
 return {'simulators/generals/environment.py':tree['integrations/generals/environment.py'].decode(),
         'vendor/engines/generals-bots/generals/__init__.py':ENGINE_INIT,
         'vendor/engines/generals-bots/generals/core/__init__.py':''}

def save_pool(pool,path):
 np.savez_compressed(path,**{k:np.asarray(v) for k,v in pool._asdict().items()})

def prepare(out,include_existing=True,private_maps=128,opponent_release=None,panel=None,starter='ppo'):
 """`starter`: 'ppo' (the working PPO starter) or 'empty' (STARTERS). An empty release seals its own brief as
 public/brief.md, shows the box only what an unprivileged worker process can import, and stages that as
 frozen/unprivileged for the worker image."""
 from integrations.generals.environment import build_pool,RULESET
 import jax
 if starter not in STARTERS:raise ValueError('starter is one of: '+', '.join(STARTERS))
 chosen=STARTERS[starter]
 out=Path(out).resolve()
 if out.exists():raise FileExistsError('Choose a new release; historical manifests and seeds are immutable')
 if not 2<=private_maps<=512:raise ValueError('Private map count must be 2..512')
 out.mkdir(parents=True);frozen=out/'frozen';public=out/'public';private=out/'private'
 frozen.mkdir();public.mkdir();private.mkdir(mode=0o700)
 binding=frozen_manifest();game=frozen/'integrations/generals';control=frozen/'integrations/generals_autoresearch'
 (frozen/'integrations').mkdir();(frozen/'integrations/__init__.py').write_text('')
 game.mkdir();control.mkdir()
 for n in RUNTIME:shutil.copy2(GAME/n,game/n)
 for n in CONTROL:shutil.copy2(HERE/n,control/n)
 (control/'starter').mkdir();shutil.copy2(HERE/'starter/policy.py',control/'starter/policy.py')
 source=ROOT/'candidates/generals-bots';dest=frozen/'candidates/generals-bots'
 for n in binding['upstream']:
  target=dest/n;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source/n,target)
 shutil.copytree(source/'.git',dest/'.git',ignore=shutil.ignore_patterns('hooks','logs','config','*.lock'))
 (dest/'.git/config').write_text('[core]\n\trepositoryformatversion = 0\n\tbare = false\n\tfilemode = false\n')
 if subprocess.check_output(['git','rev-parse','HEAD'],cwd=dest,text=True).strip()!=binding['commit']:raise ValueError('Staged engine pin differs')
 shutil.copytree(HERE/chosen['agent'],public/'agent',ignore=shutil.ignore_patterns('__pycache__'))
 (public/'recipes').mkdir()
 for name,source in chosen['configs'].items():shutil.copy2(HERE/'recipes'/source,public/'recipes'/name)
 shutil.copy2(HERE/'benchmark.json',public/'benchmark.json')
 shutil.copy2(HERE/'CONTRACT.md',public/'CONTRACT.md')
 if chosen['brief']:shutil.copy2(HERE/'task'/chosen['brief'],public/'brief.md')
 # The researcher sees only the canonical workspace built from this public tree.
 from harness.workspace import stage as stage_workspace
 if starter in ('empty','slowsim'):
  for name,data in unprivileged_tree(starter).items():
   target=frozen/'unprivileged'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
 if starter=='slowsim':
  sim=public/'simulator'/'generals_sim';sim.mkdir(parents=True)
  for n in SLOW_SIM:shutil.copy2(HERE/chosen['simulator']/n,sim/n)
  stage_workspace('generals',public,withhold_paths=r'.',simulator_readme=SLOW_README,
                  rules=HERE/'task'/chosen['rules'],
                  extra_editable={'simulator/generals_sim/'+n:'simulator/generals_sim/'+n for n in SLOW_SIM})
 elif starter=='empty':
  stage_workspace('generals',public,withhold_paths=BOX_WITHHELD,views=box_views())
 else:stage_workspace('generals',public)
 seeds=public_seeds();known=set(sum((seeds[k] for k in ('train','practice','heldout')),[]))
 if len(seeds['practice'])<DEFAULT['practice_maps']:raise ValueError('Public practice bank is smaller than the contract schedule')
 save_pool(build_pool(seeds['train']),public/'train-maps.npz');save_pool(build_pool(seeds['practice']),public/'practice-maps.npz')
 provenance={'kind':'canonical pinned generator'}
 # Two independent private schedules, disjoint from every public historical map.
 for split in ('grade','calibration'):
  chosen=[]
  while len(chosen)<private_maps:
   seed=secrets.randbits(32)
   if seed in known:continue
   known.add(seed);chosen.append(seed)
  write_json(private/(split+'-seeds.json'),{'split':'heldout' if split=='grade' else split,'seeds':chosen},exclusive=True)
  save_pool(build_pool(chosen),private/(split+'-maps.npz'))
 write_json(private/'anchors.json',{'qualified':False,'status':'full_budget_multi_seed_calibration_not_run'},exclusive=True)
 bots=[];rated={}
 if panel and opponent_release:raise ValueError('Give a sealed panel or an earlier release, not both')
 if panel:
  # A sealed, rated panel (ladder.py `seal`): learned opponents with their own source and their Elo on one
  # ladder with the scripted opponents. verify() fails closed unless every package is bound to exactly
  # this release's frozen engine and rules.
  from .bundle import verify
  from .ladder import read_panel
  old=Path(panel).resolve();catalog=read_panel(old)
  for p in catalog['bots']:
   if p['kind']!='policy':raise ValueError('A sealed panel holds policy opponents')
   verify(old/p['bundle'],p['bundle_sha256'],cap=PANEL_TENSOR_CAP)
   dest=private/'bots'/p['name'];shutil.copytree(old/p['bundle'],dest)
   verify(dest,p['bundle_sha256'],cap=PANEL_TENSOR_CAP)
   bots.append({**{k:p[k] for k in ('name','kind','bundle_sha256','elo','practice','qualification')},'bundle':'bots/'+p['name']})
  rated={n:v['elo'] for n,v in catalog.get('scripted',{}).items()}
  shutil.copy2(old/'bots.json',private/'panel-source.json')
  if catalog.get('ladder'):shutil.copy2(old/catalog['ladder'],private/'ladder.json')
 elif opponent_release:
  # Reuse previously sealed, strictly imported policies. Do not deserialize a
  # historical training checkpoint under a different full implementation hash.
  # verify() enforces this release's exact frozen engine/rules and file hashes.
  from .bundle import verify
  old=Path(opponent_release).resolve();audit_release(old)
  for p in json.loads((old/'private/panel.json').read_text())['official']:
   if p['kind'] not in BUNDLED:
    if p.get('elo') is not None:rated[p['name']]=p['elo']
    continue
   source=old/'private'/p['bundle'];verify(source,p['bundle_sha256'],cap=PANEL_TENSOR_CAP)
   dest=private/'bots'/p['name'];shutil.copytree(source,dest)
   verify(dest,p['bundle_sha256'],cap=PANEL_TENSOR_CAP);bots.append(p)
 elif include_existing:
  # Retained development checkpoints of an older simulator source. Where the files still exist,
  # export_existing fails closed on their source binding; where they do not, the panel is scripted only.
  base=ROOT/'evidence/generals-implementation/fullsize-20261108'
  for name,path in [('starter-early',base/'cpu-cli-release/train/latest.msgpack'),
                    ('trained-fullsize',base/'profile/h100-default-strict-4/seed-42/batch-2048/starting.msgpack')]:
   if not path.exists():continue
   identity=export_existing(path,private/'bots'/name)
   bots.append({'name':name,'kind':'checkpoint','bundle':'bots/'+name,'bundle_sha256':identity,
    'qualification':'source-bound development policy; competitive strength unqualified'})
 scripted=[dict({'name':n,'kind':n},**({'elo':rated[n]} if n in rated else {})) for n in ('random','expansion','hunter')]
 # Practice opponents: the two weak scripted ones, and the learned opponents the panel marks for practice
 # (historical checkpoint panels have no mark: their first bot).
 marked=[b for b in bots if b.get('practice')] if any('practice' in b for b in bots) else bots[:1]
 practice=scripted[:2]+marked
 official=scripted+bots
 write_json(private/'panel.json',{'practice':practice,'development':official,'official':official,
  'score_status':('fixed-panel points and Elo on the panel ladder ready' if any('elo' in p for p in official) else 'fixed-panel points ready')+
   '; optional reference normalization requires calibration'},exclusive=True)
 write_json(public/'practice-opponents.json',[{'name':p['name'],'kind':p['kind']} for p in practice])
 write_json(public/'session.json',DEFAULT);write_json(public/'engine-manifest.json',binding)
 write_json(public/'map-bank.json',{'rules_sha256':digest(RULESET),'train_seed_commitment':digest(seeds['train']),
  'train_maps':len(seeds['train']),'practice_maps':len(seeds['practice']),
  'train_sha256':sha(public/'train-maps.npz'),'practice_sha256':sha(public/'practice-maps.npz'),'provenance':provenance})
 shutil.copytree(HERE/'hidden',private/'reference',ignore=shutil.ignore_patterns('__pycache__'))
 plan=[{'name':f'{family}-{seed}','family':family,'seed':seed,'allocated_seconds':int(DEFAULT['budget_usd']/rate(TRAIN_SHAPE)),  # the whole budget on one training worker
 
  'recipe':json.loads((HERE/('recipes/starter.json' if family=='starter' else 'hidden/reference.json')).read_text())}
  for family in ('starter','reference') for seed in (101,211,307)]
 write_json(private/'calibration-plan.json',{'status':'not_run','jobs':plan,
  'gate':'same full allocation for every anchor; three paired training seeds; reference gap >=.05 and paired map/training-seed bootstrap lower bound >0'},exclusive=True)
 manifest=dict(version=DEFAULT['version'],created_unix=time.time(),engine=binding,
  public_files={str(p.relative_to(public)):sha(p) for p in public.rglob('*') if p.is_file()},
  frozen_files={str(p.relative_to(frozen)):sha(p) for p in frozen.rglob('*') if p.is_file()},
  private_commitments={p.name:sha(p) for p in private.glob('*') if p.is_file()},
  imported_development_bots=len(bots),gpu_jobs_started=0,model_calls=0,starter=starter,
  grading_ready=private_maps>=DEFAULT['grade_maps'],normalized_anchor_ready=False)
 write_json(out/'release.json',manifest,exclusive=True);audit_release(out)
 return manifest

def audit_release(out):
 out=Path(out);m=json.loads((out/'release.json').read_text())
 for label in ('public','frozen'):
  root=out/label;actual={str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}
  if actual!=set(m[label+'_files']):raise ValueError('Staged file inventory drift')
  for n,value in m[label+'_files'].items():
   if sha(root/n)!=value or (root/n).is_symlink():raise ValueError('Staged source drift: '+n)
  for p in root.rglob('*'):
   if p.is_symlink() or any(v in ('private','hidden') for v in p.relative_to(root).parts) or p.name in ('grade-seeds.json','calibration-seeds.json','weights.npz') or p.suffix in ('.key',):raise ValueError('Private asset staged: '+str(p))
 for n,value in m['private_commitments'].items():
  if sha(out/'private'/n)!=value:raise ValueError('Private grading data drift: '+n)
 if m.get('starter')=='slowsim':
  # No engine and no wrapper reach an unprivileged process: exactly the tree unprivileged_tree('slowsim') builds.
  root=out/'frozen/unprivileged';actual={str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}
  if actual!={'integrations/__init__.py','integrations/generals/__init__.py','integrations/generals/environment.py',
              *('integrations/generals_autoresearch/'+n for n in PUBLIC_CONTROL)}:raise ValueError('Unprivileged tree of an editable-simulator release differs')
  if (root/'integrations/generals/environment.py').read_text()!=SLOW_ENVIRONMENT:raise ValueError('Unprivileged sizes module differs')
 elif (out/'frozen/unprivileged').is_dir():
  # What an unprivileged process plays on is the pinned engine's own rule modules and the frozen wrapper's rules.
  for n in PUBLIC_ENGINE:
   if sha(out/'frozen/unprivileged/generals'/n)!=sha(out/'frozen/candidates/generals-bots/generals'/n):raise ValueError('Unprivileged engine differs from the pin: '+n)
  wrapper=without((out/'frozen/integrations/generals/environment.py').read_text(),WRAPPER_WITHHELD)
  if (out/'frozen/unprivileged/integrations/generals/environment.py').read_text()!=wrapper:raise ValueError('Unprivileged wrapper differs from the frozen one')
  for n in RUNTIME:
   if n!='environment.py' and sha(out/'frozen/unprivileged/integrations/generals'/n)!=sha(out/'frozen/integrations/generals'/n):raise ValueError('Unprivileged runtime drift: '+n)
 if 'task_files' in m:
  root=out/'task';actual={str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}
  if actual!=set(m['task_files']):raise ValueError('Task inventory drift')
  for n,value in m['task_files'].items():
   if (root/n).is_symlink() or sha(root/n)!=value:raise ValueError('Task source drift: '+n)
 from .bundle import verify
 catalog=json.loads((out/'private/panel.json').read_text())
 for p in catalog['official']:
  if p['kind'] in BUNDLED:verify(out/'private'/p['bundle'],p['bundle_sha256'],cap=PANEL_TENSOR_CAP)
 return {'public_files':len(m['public_files']),'frozen_files':len(m['frozen_files']),'hidden_files_staged':0,
  'grading_ready':m['grading_ready'],'normalized_anchor_ready':False}

def audit_map_banks(out):
 """Inspect every prepared board against RULESET; return aggregates without exposing private seeds."""
 import hashlib
 from integrations.generals.environment import map_dimensions,RULESET,BOARD_SHAPE
 from integrations.generals.audit import distances
 out=Path(out);audit_release(out)
 public=public_seeds()
 historical=set(sum((public[k] for k in ('train','practice','heldout')),[]))
 (low,high),(army_low,army_high)=RULESET['num_castles_range'],RULESET['castle_val_range']
 seen_private=set();seen_boards=set();result={}
 for split in ('train','practice','grade','calibration'):
  private=split in ('grade','calibration');base=out/('private' if private else 'public')
  seeds=json.loads((base/(split+'-seeds.json')).read_text())['seeds'] if private else public[split]
  if private:
   if len(set(seeds))!=len(seeds) or set(seeds)&(historical|seen_private):raise ValueError('Private seed overlap')
   seen_private.update(seeds)
  dims=map_dimensions(seeds);ds=[];castles=[]
  with np.load(base/(split+'-maps.npz'),allow_pickle=False) as bank:
   if bank['armies'].shape!=(len(seeds),*BOARD_SHAPE):raise ValueError('Bank shape differs')
   for i,(h,w) in enumerate(dims):
    p=bank['passable'][i];g=bank['general_positions'][i];own=bank['ownership'][i]
    if p[h:,:].any() or p[:,w:].any():raise ValueError('Map dimensions/padding differ')
    d=int(distances(p,tuple(g[0]))[tuple(g[1])]);ds.append(d)
    c=bank['castles'][i];castles.append(int(c.sum()));a=bank['armies'][i][c]
    if not (RULESET['min_generals_distance']<=d<9999 and low<=c.sum()<=high and ((army_low<=a)&(a<army_high)).all()):raise ValueError('Native map contract differs')
    if own.sum(axis=(1,2)).tolist()!=[1,1] or bank['time'][i]!=0 or bank['eliminated'][i].any():raise ValueError('Invalid initial state')
    identity=hashlib.sha256(b''.join(bank[k][i].tobytes() for k in ('armies','ownership','generals','castles','mountains','passable'))).hexdigest()
    if identity in seen_boards:raise ValueError('Duplicate board across partitions')
    seen_boards.add(identity)
  result[split]={'maps':len(seeds),'shapes':len(set(map(tuple,dims.tolist()))),'min_spawn_distance':min(ds),
   'max_spawn_distance':max(ds),'castle_range':[min(castles),max(castles)],'sha256':sha(base/(split+'-maps.npz'))}
 return {'partitions':result,'boards':len(seen_boards),'private_seed_overlap':0,'duplicate_boards':0,'invalid_boards':0}
