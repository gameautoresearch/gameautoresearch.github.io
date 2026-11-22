"""The empty starter (prepare(starter='empty')): self-play plumbing with no learning, a brief that says what castles
are worth, and nothing to copy: neither the box nor an unprivileged worker process can read the project's training
code, the engine's reward helpers and scripted players, or the wrapper's scripted players."""
import json
import re
import subprocess
import sys
from pathlib import Path
import numpy as np
import pytest
from integrations.generals.environment import RULESET
from integrations.generals_autoresearch import prepare as P
from integrations.generals_autoresearch.contract import HERE,ROOT,GAME,sha

# Code that learns, or that hands over a reward or a player to copy.
LEARNING=re.compile(r'optax|value_and_grad|jax\.grad\(|gae_lambda|clip_eps|entropy_coef|learning_rate|advantage|castle_reward|composite_reward|scripted_logits|HunterAgent|ExpanderAgent|def potential')

def pinned():
 try:
  from integrations.generals_autoresearch.contract import frozen_manifest;frozen_manifest()
 except Exception as exc:pytest.skip(f'Pinned Generals engine metadata unavailable: {exc}')

def test_empty_starter_files_hold_no_learning_and_the_brief_states_the_castle_facts():
 for name in ('policy.py','learner.py','train.py'):
  assert not LEARNING.search((HERE/'starter_empty'/name).read_text()),name
 for name in ('empty.json','empty-smoke.json'):
  assert set(json.loads((HERE/'recipes'/name).read_text()))=={'num_envs','block_steps','save_seconds','progress_seconds'}
 brief=(HERE/'task/brief-empty.md').read_text();flat=' '.join(brief.split())
 assert RULESET['id'] in brief and 'boilerplate only' in brief and 'Castles decide games' in brief
 lo,hi=RULESET['castle_val_range']
 assert f'{lo}–{hi-1} army' in flat and '61–64 %' in flat and '82–85 %' in flat and 'tick 100' in flat
 assert not re.search(r'PPO|GAE|rollout\.py|make_collect|residual',brief)
 # The limits and worker sections are the PPO brief's: an edit there must reach both.
 other=(HERE/'task/brief.md').read_text()
 assert brief[brief.index('## Workers'):].replace("the training command on the files","the starter's training command on the files").replace('The modules shown under `simulator/` import','The engine and the two helper modules import')==other[other.index('## Workers'):]

def test_unprivileged_tree_is_the_pinned_rules_without_players_or_reward_helpers():
 pinned()
 tree=P.unprivileged_tree();engine=ROOT/'candidates/generals-bots/generals'
 for name in P.PUBLIC_ENGINE:assert tree['generals/'+name]==(engine/name).read_bytes()
 assert not any(re.search(r'rewards|agents|rendering|rollout|arena|starter|gui|remote',name) for name in tree)
 for name,data in tree.items():
  assert not LEARNING.search(data.decode()),name
 wrapper=tree['integrations/generals/environment.py'].decode();full=(GAME/'environment.py').read_text()
 assert 'def scripted_logits' in full and 'def potential' in full
 kept=[line for line in full.splitlines() if line.strip()];shown=[line for line in wrapper.splitlines() if line.strip()]
 assert set(shown)<=set(kept) and 'def transition(states, actions):' in shown and 'def build_pool(seeds, dimensions=None):' in shown
 assert all(re.search(P.BOX_WITHHELD,name) for name in ('vendor/engines/generals-bots/generals/agents/hunter_agent.py',
  'vendor/engines/generals-bots/generals/core/rewards.py','simulators/generals/autoresearch/rollout.py'))
 assert not any(re.search(P.BOX_WITHHELD,name) for name in ('vendor/engines/generals-bots/generals/core/game.py',
  'simulators/generals/environment.py','simulators/generals/autoresearch/bundle.py'))

def test_empty_release_shows_and_ships_only_that_tree_and_its_boilerplate_runs_on_it(tmp_path):
 pinned()
 release=tmp_path/'release'
 manifest=P.prepare(release,include_existing=False,private_maps=2,starter='empty')
 assert manifest['starter']=='empty' and P.audit_release(release)['public_files']
 box=release/'public/agent_workspace';shown={str(p.relative_to(box)) for p in box.rglob('*') if p.is_file()}
 assert not any(re.search(r'agents/|rewards\.py|rollout\.py|rendering\.py|gui/|remote/',name) for name in shown)
 for name in shown:
  if name.endswith(('.py','.json')):assert not LEARNING.search((box/name).read_text()),name
 assert (box/'training/learner.py').read_text()==(HERE/'starter_empty/learner.py').read_text()
 assert json.loads((box/'configs/train.json').read_text())==json.loads((HERE/'recipes/empty.json').read_text())
 assert (release/'public/brief.md').read_text()==(HERE/'task/brief-empty.md').read_text()
 tree=release/'frozen/unprivileged'
 assert {str(p.relative_to(tree)) for p in tree.rglob('*') if p.is_file()}==set(P.unprivileged_tree())
 assert (box/'simulator/simulators/generals/environment.py').read_text()==(tree/'integrations/generals/environment.py').read_text()
 from harness.games.generals.provider import unprivileged_pythonpath,PYTHONPATH
 assert unprivileged_pythonpath(release)=='/opt/gr/unprivileged'!=PYTHONPATH
 # The boilerplate, importing from that tree alone: it plays, saves a network and learns nothing.
 run=tmp_path/'run';run.mkdir()
 for name in ('policy.py','learner.py','train.py'):
  (run/name).write_text((box/'training'/name).read_text().replace("'/data/train-maps.npz'",repr(str(release/'public/train-maps.npz'))))
 cfg=dict(json.loads((box/'configs/smoke.json').read_text()),num_envs=32,block_steps=8);(run/'config.json').write_text(json.dumps(cfg))
 env={'PATH':'/usr/bin:/bin','HOME':str(tmp_path),'JAX_PLATFORMS':'cpu','PYTHONPATH':str(tree)}
 done=subprocess.run([sys.executable,'train.py','--config','config.json','--out',str(run/'out'),'--seed','5','--seconds','25'],cwd=run,env=env,capture_output=True,text=True,timeout=600)
 assert done.returncode==0,done.stderr[-3000:]
 result=json.loads((run/'out/result.json').read_text())
 assert result['block']>=1 and result['joint_ticks']==result['block']*32*8 and json.loads((run/'out/model.json').read_text())==cfg
 code=('import jax,jax.numpy as jnp,json,policy;from integrations.generals_autoresearch.bundle import load_weights;'
  'from integrations.generals.environment import BOARD_SHAPE,ACTION_COUNT;m=policy.create({});'
  "fresh=m.init(jax.random.split(jax.random.PRNGKey(5),3)[1],jnp.zeros((1,*BOARD_SHAPE,14)),jnp.ones((1,ACTION_COUNT),bool))['params'];"
  "saved=load_weights('out/weights.npz');print(all(bool((a==b).all()) for a,b in zip(jax.tree.leaves(fresh),jax.tree.leaves(saved))))")
 same=subprocess.run([sys.executable,'-c',code],cwd=run,env=env,capture_output=True,text=True,timeout=300)
 assert same.stdout.strip()=='True',same.stderr[-2000:]

def test_ppo_release_is_unchanged_by_the_option(tmp_path):
 pinned()
 release=tmp_path/'release';manifest=P.prepare(release,include_existing=False,private_maps=2)
 assert manifest['starter']=='ppo' and not (release/'frozen/unprivileged').exists() and not (release/'public/brief.md').exists()
 box=release/'public/agent_workspace'
 assert (box/'training/learner.py').read_text()==(HERE/'starter/learner.py').read_text()
 assert (box/'simulator/simulators/generals/autoresearch/rollout.py').is_file()
 from harness.games.generals.provider import unprivileged_pythonpath,PYTHONPATH
 assert unprivileged_pythonpath(release)==PYTHONPATH
