"""Editable training loop. Each run starts fresh; host allocation deadlines are authoritative."""
import time
START=time.monotonic()
import argparse
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
import policy
import learner
from integrations.generals_autoresearch.rollout import load_pool,initial_carry,make_collect
from integrations.generals_autoresearch.bundle import save_weights

def main():
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--out',required=True)
 p.add_argument('--seed',type=int,required=True);p.add_argument('--seconds',type=float,required=True);a=p.parse_args()
 cfg=json.loads(Path(a.config).read_text());out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 pool=load_pool('/data/train-maps.npz');model=policy.create(cfg)
 key,ik,ck=jax.random.split(jax.random.PRNGKey(a.seed),3)
 state=learner.create_state(model,ik,cfg);initial=jax.tree.map(lambda x:np.asarray(x).copy(),state.params)
 # The graded limit is compute per move (FLOPs of one forward pass for one board); fail early, not at grading.
 from integrations.generals_autoresearch.flops import check;check(model,state.params)
 carry=initial_carry(pool,ck,cfg['num_envs']);collect=make_collect(model,cfg);update=learner.make_update(model,cfg)
 @jax.jit
 def iteration(state,carry):
  carry,data=collect(state.params,carry,pool);key,uk=jax.random.split(carry.key);carry=carry._replace(key=key)
  state,metrics=update(state,data,uk)
  counts=jnp.array([data['wins'].sum(),data['losses'].sum(),data['truncated'].sum(),data['invalid'].sum()])
  return state,carry,metrics,counts
 # A run that is stopped or cut off at its time limit is packaged from the files below, so they are
 # written early and often, each replaced whole: model.json now, weights.npz after the first update
 # and then at least every `save_seconds` of the config (30 by default).
 every=float(cfg.get('save_seconds',30))
 tmp=out/'model.json.tmp';tmp.write_text(json.dumps(cfg,indent=2)+'\n');tmp.replace(out/'model.json')
 def save():save_weights(state.params,out/'weights.npz')
 # progress.jsonl gets one row for the first update and then one about every `progress_seconds` (2 by
 # default), not one per update: at tens of updates a second a row each made the file megabytes long.
 # A row covers the updates since the previous row (`updates_in_row`): loss, entropy, kl and
 # iteration_seconds are their means; update, learner_samples and the four counts are totals so far.
 pace=float(cfg.get('progress_seconds',2))
 u=0;totals=np.zeros(4,np.int64);path=out/'progress.jsonl';saved=logged=time.monotonic();window=[];spent=0.
 def record():
  m=np.mean(window,0)
  row=dict(update=u,updates_in_row=len(window),learner_samples=u*cfg['num_envs']*cfg['rollout_steps'],
    iteration_seconds=spent/len(window),loss=float(m[0]),entropy=float(m[3]),kl=float(m[4]),
    captures_won=int(totals[0]),captures_lost=int(totals[1]),truncations=int(totals[2]),
    invalid_actions=int(totals[3]),elapsed_seconds=time.monotonic()-START)
  with path.open('a') as f:f.write(json.dumps(row)+'\n')
  window.clear()
  return row
 while time.monotonic()-START < a.seconds-8:
  t=time.monotonic();state,carry,m,counts=iteration(state,carry);jax.block_until_ready((m,counts));u+=1
  m=np.asarray(m);counts=np.asarray(counts);totals+=counts
  if not np.isfinite(m).all() or counts[3]:raise RuntimeError('Nonfinite learner or illegal action')
  window.append(m);spent+=time.monotonic()-t
  if u==1 or time.monotonic()-logged>=pace:row=record();logged=time.monotonic();spent=0.
  if u==1 or time.monotonic()-saved>=every:save();saved=time.monotonic()
 if u==0:raise RuntimeError('No PPO update completed in the allocation')
 if window:row=record()  # the last updates always get their row: result.json is the run's final totals
 save()
 change=sum(float(np.sum((np.asarray(x)-y)**2)) for x,y in zip(jax.tree.leaves(state.params),jax.tree.leaves(initial)))**.5
 (out/'result.json').write_text(json.dumps(dict(**row,parameter_l2_change=change,
  actual_devices=[d.device_kind for d in jax.devices()],fresh_initialization_seed=a.seed),indent=2)+'\n')

if __name__=='__main__':main()
