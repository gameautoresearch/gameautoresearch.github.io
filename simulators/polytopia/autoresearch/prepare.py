"""Offline release assembly: pinned worker runtime, map table, worker entrypoints, host-only panel and workspace.

`prepare(out, panel=False)` builds a bootstrap release without the practice opponents. It exists because the opponents
are themselves starter runs of this game mode: such a release can train (and run a smoke session, whose practice and
grading play a random network), which is how the opponents are made; full practice and grading refuse it.
"""
import json,shutil
from pathlib import Path
from . import build_context
from .v1.core import HERE,write,seal,digest

PANEL=build_context.PANEL

# Reference learning code in the read-only simulator sources the box shows: optimizers, gradients, advantage and return
# estimation, PPO. An empty-starter release withholds every such file from the box (workers never had the reference
# trainer: the staged runtime carries the empty starter's files).
LEARNING_CODE=(r'adam_step|adam_init|value_and_grad|jax\.grad\(|segment_gae|gae_lambda\s*\*|advantage\s*=|\bppo\.loss'
               r'|clipped surrogate|policy gradient|optax|critic_w|family_embedding|parameter_embedding')
# The browser player (engine-wasm: a CPU port of the PPO starter's network and a turn search over it) is research
# tooling, not part of the training task: an empty-starter box does not show it.
RESEARCH_PATHS=r'^simulators/polytopia/engine-wasm/'
# What the researcher starts from. `ppo` is the working clipped-PPO starter; `empty` is boilerplate only (the GPU
# rollout plumbing with no learning, an empty learner and a linear placeholder network), with its own brief and recipes.
STARTERS={'ppo':dict(recipes={'starter':'starter','smoke':'smoke','starter_arch':'starter_arch'},brief='brief.md'),
          'empty':dict(recipes={'starter':'empty','smoke':'empty-smoke','starter_arch':'empty_arch'},brief='brief-empty.md',withhold=LEARNING_CODE,withhold_paths=RESEARCH_PATHS)}

def prepare(out,panel=True,starter='ppo'):
 """`panel`: true (the current panel, build_context.PANEL), 'ladder' (the earlier ladder: random, starter-20m/65m/200m)
 or false (a bootstrap release). `starter`: 'ppo' or 'empty' (STARTERS); the release seals its brief as public/brief.md."""
 if panel not in (True,False,'ladder','strong'):raise ValueError('panel is true (the default), ladder or false')
 if starter not in STARTERS:raise ValueError('starter is one of: '+', '.join(STARTERS))
 chosen=STARTERS[starter];labels=build_context.PANELS['ladder' if panel=='ladder' else 'strong']
 out=Path(out).resolve()
 if out.exists():raise FileExistsError('Choose a new release directory; releases are never rebuilt in place')
 # Local inputs are checked before anything is written: the map table and the practice opponents of this game mode.
 # A missing input, or one made for another mode, stops here with what to build. Nothing here contacts a provider.
 table=build_context.map_table();folders=build_context.panel(labels) if panel else {}
 out.mkdir(parents=True)
 build_context.stage(out/'runtime',starter)
 (out/'maps').mkdir();shutil.copy2(table,out/'maps'/table.name)
 # /opt/v1 on every worker: train.py, grade.py, policy.py and preflight.py are named by the provider.
 shutil.copytree(HERE/'v1',out/'v1',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
 (out/'public/recipes').mkdir(parents=True)
 for name,native in chosen['recipes'].items():shutil.copy2(HERE/'recipes'/(native+'.json'),out/'public/recipes'/(name+'.json'))
 shutil.copy2(HERE/'task'/chosen['brief'],out/'public/brief.md')
 shutil.copy2(HERE/'benchmark.json',out/'benchmark.json')
 shutil.copy2(HERE/'base.Dockerfile',out/'base.Dockerfile')
 (out/'base-context/repo').mkdir(parents=True)
 for engine in ('engine-rs','engine-cu'):shutil.copytree(out/'runtime'/engine,out/'base-context/repo'/engine)
 # Host-only inventory: Provider.image deliberately does not mount this directory.
 inventory=[]
 for label in folders:
  source=folders[label]
  target=out/'host-private-panel'/label;target.mkdir(parents=True)
  for f in build_context.PANEL_FILES:shutil.copy2(source/f,target/f)
  meta=json.loads((target/'final.json').read_text())
  if meta['sha256']!=digest(target/'final.npz'):raise ValueError('Panel checkpoint integrity failure')
  inventory.append({'name':label,'directory':'host-private-panel/'+label,'provenance':build_context.PROVENANCE[label]+'; not a calibrated reference','checkpoint_sha256':digest(target/'final.npz'),'network_sha256':digest(target/'network.py'),'checkpoint_metadata':meta})
 if panel:write(out/'host-private-panel/inventory.json',{'panel':inventory,'map_count':128,'seats_per_map':2,'weighting':'equal opponent then paired-seat map mean','normalized_reference_score':None})
 # No existing opponents or private seeds/weights are ever included in the public image.
 # The researcher sees only the canonical workspace built from the public recipes and the staged starter.
 from harness.workspace import stage as stage_workspace
 stage_workspace('polytopia',out/'public',training_source=out/'runtime',withhold=chosen.get('withhold'),withhold_paths=chosen.get('withhold_paths'))
 write(out/'starter.json',{'starter':starter,'panel':list(labels) if panel else []})
 return seal(out)
