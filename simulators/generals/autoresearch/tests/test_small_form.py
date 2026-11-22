"""The benchmark's form (full-size again since 2026-11-14; the file keeps the name it got with the 8x8 form):
one ruleset behind every declaration, the practice schedule, and a training run that keeps its last network
when it is ended before its own end."""
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import pytest
from integrations.generals.environment import RULESET,BOARD_SHAPE,ACTION_COUNT
from integrations.generals_autoresearch.contract import HERE,GAME,ROOT,DEFAULT,public_seeds
from integrations.generals_autoresearch.ledger import Ledger


def test_every_declaration_states_the_one_ruleset():
 assert RULESET['id']=='generals-research-1v1-medium-v2'==DEFAULT['environment_id']
 assert (RULESET['min_grid_size'],RULESET['max_grid_size'],RULESET['pad_to'],RULESET['truncation'])==(12,15,16,1000)
 assert (RULESET['mountain_density_range'],RULESET['num_castles_range'],RULESET['castle_val_range'],RULESET['min_generals_distance'])==([.20,.20],[4,6],[10,16],10)
 assert BOARD_SHAPE==(16,16) and ACTION_COUNT==2049
 from integrations.generals_autoresearch.contract import sha
 contract=json.loads((HERE/'benchmark.json').read_text());task=contract['task']
 assert {k:task[k] for k in RULESET}==RULESET
 assert task['observation_shape']==[*BOARD_SHAPE,14] and task['action_count']==ACTION_COUNT
 assert task['policy_tensor_cap']==DEFAULT['policy_tensor_cap'] and task['policy_flops_cap']==DEFAULT['policy_flops_cap']
 assert contract['budget']['max_run_seconds']==DEFAULT['max_run_seconds']
 assert contract['budget']['research_usd']==DEFAULT['budget_usd']
 assert contract['evaluation']['practice_maps']==DEFAULT['practice_maps']==16
 assert contract['evaluation']['private_grade_maps']==DEFAULT['grade_maps']
 from harness.games.generals.provider import ModalProvider as worker
 from harness.games.generals import SPEC,SMOKE
 reserve=contract['budget']['export_reserve_seconds']
 assert (reserve['fraction_of_run'],reserve['minimum'],reserve['maximum'])==(worker.EXPORT_RESERVE_FRACTION,worker.MIN_EXPORT_RESERVE,worker.MAX_EXPORT_RESERVE)==(.05,40,60)
 assert contract['budget']['min_run_seconds']==DEFAULT['min_run_seconds']==120 and 'min_run_seconds' not in SMOKE
 assert SPEC['smoke_final_train_seconds']>=DEFAULT['min_run_seconds'] and SMOKE['max_run_seconds']>=DEFAULT['min_run_seconds']
 # The shortest run still trains: measured live, about 16 s of worker setup and 18 s to the first update,
 # then the trainer's own 10 s closing margin and the reserve.
 assert DEFAULT['min_run_seconds']-worker.export_reserve(DEFAULT['min_run_seconds'])-10-16-18>=30
 for text in ((HERE/'CONTRACT.md').read_text(),(HERE/'task/brief.md').read_text()):
  flat=' '.join(text.split())
  assert 'at least 40' in flat and 'at most 60' in flat and '120' in flat
 for text in ((HERE/'CONTRACT.md').read_text(),(HERE/'task/brief.md').read_text(),(HERE/'task/game_rules/mode.md').read_text()):
  assert RULESET['id'] in text
 brief=(HERE/'task/brief.md').read_text()
 assert f'[batch, {ACTION_COUNT}]' in brief and '[batch, 16, 16, 14]' in brief and '16 public maps' in brief
 assert 'stopped_early' in brief and 'cut_short' in brief


def test_neutral_castles_are_cheap_and_pay_like_a_general_once_captured():
 """medium-v2 (2026-11-19): castles cost 10..15, not 40..50. At the old price the graded networks took a castle
 in 1 game of 96 and buying one lost games; this pins the price and the income a capture buys."""
 import jax,jax.numpy as jnp
 from generals.core import game
 from integrations.generals.environment import build_pool
 pool=build_pool(list(range(7000,7032)))
 castles=np.asarray(pool.castles);armies=np.asarray(pool.armies);counts=castles.sum((1,2))
 assert ((counts>=4)&(counts<=6)).all() and armies[castles].min()>=10 and armies[castles].max()<=15
 s=jax.tree.map(lambda x:x[0],pool);mountains=np.asarray(s.mountains);generals=np.asarray(s.generals)
 steps=((-1,0),(1,0),(0,-1),(0,1))
 r,c,d,nr,nc=next((r,c,d,r-dr,c-dc) for r,c in np.argwhere(castles[0]) for d,(dr,dc) in enumerate(steps)
  if 0<=r-dr<16 and 0<=c-dc<16 and not mountains[r-dr,c-dc] and not castles[0][r-dr,c-dc] and not generals[r-dr,c-dc])
 cost=int(s.armies[r,c]);passes=jnp.array([[1,0,0,0,0]]*2,jnp.int32)
 def with_stack(army):return s._replace(armies=s.armies.at[nr,nc].set(army),ownership=s.ownership.at[0,nr,nc].set(True),
  ownership_neutral=s.ownership_neutral.at[nr,nc].set(False))
 def step(state,move=None):return game.step(state,passes if move is None else passes.at[0].set(jnp.array(move,jnp.int32)),general_trade=True)[0]
 move=[0,int(nr),int(nc),d,0]
 tie=step(with_stack(cost+1),move)
 assert not tie.ownership[0,r,c] and int(tie.armies[r,c])==0
 won=step(with_stack(cost+2),move)
 assert won.ownership[0,r,c] and won.castles[r,c] and int(won.armies[r,c])==1
 later=won
 for _ in range(2*cost):later=step(later)
 # One army every second tick (and the 50-tick land army when it falls due): the price is back in about 2*cost ticks.
 assert cost<=int(later.armies[r,c])-1<=cost+1
 idle=s
 for _ in range(20):idle=step(idle)
 assert int(idle.armies[r,c])==cost


def test_practice_schedule_is_the_unchanged_public_seed_manifest():
 from simulators.generals.autoresearch.contract import seed_manifest
 original=seed_manifest();seeds=public_seeds()
 assert [len(original[k]) for k in ('train','practice')]==[256,16]
 assert len(original['heldout']) in (0,32)
 assert seeds['train']==original['train'] and seeds['heldout']==original['heldout']
 assert len(seeds['practice'])==DEFAULT['practice_maps'] and seeds['practice'][:16]==original['practice']
 everything=seeds['train']+seeds['practice']+seeds['heldout']
 assert len(set(everything))==len(everything)
 assert public_seeds()==seeds


def test_only_a_complete_packaged_run_is_marked_as_ended_early(tmp_path):
 l=Ledger(tmp_path,dict(DEFAULT,budget_usd=1.0,min_run_seconds=10));start=100
 for name,status,sha,early in (('own-end','complete','h1',None),('stopped','complete','h2','stopped_early'),
   ('cut','complete','h3','cut_short'),('lost','stopped',None,None)):
  l.queue(name,30,{'kind':'train'});l.start(name,'provider-'+name,start)
  l.finish(name,start+20,status,'/owned/'+name if sha else None,sha,early=early);start+=100
 jobs={j['name']:j for j in l.public()['jobs']}
 assert jobs['stopped']['status']=='complete' and jobs['stopped']['stopped_early'] is True and 'cut_short' not in jobs['stopped']
 assert jobs['cut']['status']=='complete' and jobs['cut']['cut_short'] is True and 'stopped_early' not in jobs['cut']
 assert not {'stopped_early','cut_short'}&set(jobs['own-end']) and not {'stopped_early','cut_short'}&set(jobs['lost'])
 assert l.early()=={'stopped':'stopped_early','cut':'cut_short'}
 from integrations.generals_autoresearch.contract import TRAIN_SHAPE,rate
 assert l.used()==pytest.approx(80*rate(TRAIN_SHAPE)) and l.submit('stopped')==1  # charged like any run; submittable
 for status,sha,early in (('stopped',None,'stopped_early'),('failed',None,'cut_short'),('complete',None,'cut_short'),('complete','h','paused')):
  l.queue('bad',30,{});l.start('bad','provider-bad',900)
  with pytest.raises(ValueError,match='ended early'):l.finish('bad',910,status,'/owned/bad' if sha else None,sha,early=early)
  l.finish('bad',910,'failed');l.db.execute("DELETE FROM jobs WHERE name='bad'")
 l.queue('again',30,{});l.start('again','provider-again',1000);l.uncertain('again','controller interruption')
 l.reconcile('again',dict(provider_id='provider-again',started_at=1000,finished_at=1010),'complete','/owned/again','h4',early='stopped_early')
 assert next(j for j in l.public()['jobs'] if j['name']=='again')['stopped_early'] is True


def start_starter(folder,seconds,**settings):
 """The real starter trainer on CPU, one thread, against a small generated bank; returns the process and its config."""
 from integrations.generals.environment import build_pool
 from integrations.generals_autoresearch.prepare import save_pool
 folder.mkdir();bank=folder/'train-maps.npz';save_pool(build_pool(list(range(900,916))),bank)
 cfg=dict(json.loads((HERE/'recipes/smoke.json').read_text()),num_envs=8,rollout_steps=8,minibatches=2,compute_dtype='float32',**settings)
 (folder/'config.json').write_text(json.dumps(cfg));out=folder/'out'
 code=('import sys,runpy\n'
  'import integrations.generals_autoresearch.rollout as rollout\n'
  f'load=rollout.load_pool;rollout.load_pool=lambda path:load({str(bank)!r})\n'  # the worker path is /data/train-maps.npz
  f'sys.path.insert(0,{str(HERE/"starter")!r})\n'
  f'sys.argv=["train.py","--config",{str(folder/"config.json")!r},"--out",{str(out)!r},"--seed","3","--seconds",{str(seconds)!r}]\n'
  f'runpy.run_path({str(HERE/"starter/train.py")!r},run_name="__main__")\n')
 env={'PATH':os.environ.get('PATH',''),'JAX_PLATFORMS':'cpu','PYTHONPATH':f'{ROOT}{os.pathsep}{ROOT/"candidates/generals-bots"}',
  'OMP_NUM_THREADS':'1','XLA_FLAGS':'--xla_cpu_multi_thread_eigen=false'}
 log=(folder/'trainer.log').open('w')
 return subprocess.Popen([sys.executable,'-c',code],cwd=folder,env=env,stdout=log,stderr=subprocess.STDOUT),cfg

def progress_rows(out):
 return [json.loads(line) for line in (out/'progress.jsonl').read_text().splitlines() if line.endswith('}')]

COUNTS=('learner_samples','captures_won','captures_lost','truncations','invalid_actions')

def test_starter_trainer_killed_mid_run_leaves_a_network_that_seals_and_plays(tmp_path):
 """The real starter on CPU: saved early and again while training, whole files only, usable after SIGKILL."""
 import jax.numpy as jnp
 from integrations.generals_autoresearch.bundle import seal,verify,load_weights
 try:
  from integrations.generals_autoresearch.contract import frozen_manifest;frozen_manifest()
 except Exception as exc:pytest.skip(f'Pinned Generals engine metadata unavailable: {exc}')
 folder=tmp_path/'run';proc,cfg=start_starter(folder,900,save_seconds=1,progress_seconds=.25);out=folder/'out'
 try:
  seen=set();deadline=time.monotonic()+600
  while len(seen)<3:  # the first save and at least two later ones, while the trainer is still running
   assert proc.poll() is None,(folder/'trainer.log').read_text()[-3000:]
   assert time.monotonic()<deadline,'the starter did not keep saving'
   if (out/'weights.npz').exists():
    try:
     with np.load(out/'weights.npz') as a:seen.add(tuple(float(np.abs(a[k]).sum()) for k in sorted(a.files)))
    except (OSError,ValueError,EOFError) as exc:pytest.fail(f'A partly written weights file was visible: {exc}')
   time.sleep(.2)
  assert json.loads((out/'model.json').read_text())==cfg
 finally:
  proc.send_signal(signal.SIGKILL);proc.wait()
 assert not (out/'result.json').exists()  # it never reached its own end
 rows=progress_rows(out)
 assert len(rows)>=3 and rows[-1]['invalid_actions']==0 and sum(r['updates_in_row'] for r in rows)==rows[-1]['update']
 source=tmp_path/'export'  # what export_entry does on a worker after the trainer is ended
 source.mkdir()
 for name in ('policy.py','learner.py','train.py'):shutil.copy2(HERE/'starter'/name,source/name)
 for name in ('model.json','weights.npz'):shutil.copy2(out/name,source/name)
 identity=seal(source,tmp_path/'bundle',{'kind':'allocated_worker_output'})
 assert verify(tmp_path/'bundle',identity)['tensors']>0
 from integrations.generals_autoresearch.starter.policy import create
 logits,value=create(cfg).apply({'params':load_weights(tmp_path/'bundle/weights.npz')},jnp.zeros((2,*BOARD_SHAPE,14)),jnp.ones((2,ACTION_COUNT),bool))
 assert logits.shape==(2,ACTION_COUNT) and value.shape==(2,) and bool(jnp.isfinite(logits).all())


def test_starter_progress_is_one_row_every_few_seconds_with_exact_running_totals(tmp_path):
 """Two real runs of the same seed: a row per update as the reference, and the paced rows a run now writes.

 At about 46 updates a second on an L40S, a row per update made progress.jsonl megabytes long within minutes;
 the host's reads of it stalled and a 900 s final lost its network.
 """
 every,_=start_starter(tmp_path/'every',60,progress_seconds=0)
 # A CPU update on the 24x24 board takes about half a second, so the paced run writes a row every 5 seconds.
 paced,_=start_starter(tmp_path/'paced',60,progress_seconds=5)
 for name,proc in (('every',every),('paced',paced)):
  assert proc.wait(timeout=600)==0,(tmp_path/name/'trainer.log').read_text()[-3000:]
 reference=progress_rows(tmp_path/'every/out');rows=progress_rows(tmp_path/'paced/out')
 assert [r['update'] for r in reference]==list(range(1,len(reference)+1)) and all(r['updates_in_row']==1 for r in reference)
 updates=[r['update'] for r in rows]
 assert updates[0]==1 and updates==sorted(set(updates)) and sum(r['updates_in_row'] for r in rows)==updates[-1]
 assert updates[-1]>5*len(rows)  # many updates to a row
 gaps=np.diff([r['elapsed_seconds'] for r in rows])
 assert (gaps[:-1]>=4.75).all() and len(rows)<=rows[-1]['elapsed_seconds']/5+3  # paced; the closing row may come sooner
 by_update={r['update']:r for r in reference};compared=0;previous=0
 for row in rows:
  if row['update'] in by_update:
   assert {k:row[k] for k in COUNTS}=={k:by_update[row['update']][k] for k in COUNTS},row  # totals so far, exactly
   window=[by_update[u] for u in range(previous+1,row['update']+1)]
   assert len(window)==row['updates_in_row']
   for key in ('loss','entropy','kl'):assert row[key]==pytest.approx(np.mean([w[key] for w in window]),rel=1e-4,abs=1e-6)
   compared+=1
  previous=row['update']
 assert compared>=3
 result=json.loads((tmp_path/'paced/out/result.json').read_text())  # the closing row: the run's final totals
 assert {k:result[k] for k in rows[-1]}==rows[-1] and result['update']==updates[-1]
 assert (tmp_path/'paced/out/progress.jsonl').stat().st_size<400*(len(rows)+1)
