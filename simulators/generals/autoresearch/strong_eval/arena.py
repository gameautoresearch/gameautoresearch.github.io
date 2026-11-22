"""Exact-task paired cross-play. All neural agents see only native public fields."""
from pathlib import Path
import contextlib,json,time
import numpy as np
from ..contract import write_json
from .client import Client
from .registry import verify_entry


def evaluate(candidate,panel,pool_path,map_indices,*,output=None,development=False,batch_size=16,record=False):
 import jax,jax.numpy as jnp
 from generals.agents import HunterAgent
 from generals.agents.harvester_agent import HarvesterAgent
 from generals.core.action import compute_valid_move_mask
 from simulators.generals.environment import observations,transition,decode_action,encode_action,scripted_logits
 from ..rollout import load_pool
 from .codec import raw_observation,entrant_mask
 from ..grading import by_map
 if not panel or len({v['name'] for v in panel})!=len(panel):raise ValueError('Distinct nonempty opponents required')
 candidate_meta=verify_entry(candidate);pool=load_pool(pool_path);indices=list(map_indices)
 if not indices or len(set(indices))!=len(indices) or min(indices)<0 or max(indices)>=len(pool.time):raise ValueError('Invalid map indices')
 view=jax.jit(lambda states:(observations(states),jax.vmap(raw_observation)(observations(states))))
 directions=jax.jit(jax.vmap(compute_valid_move_mask));advance=jax.jit(transition);hunter=HunterAgent()
 hunt=jax.jit(lambda obs:encode_action(jax.vmap(lambda o:hunter.act(o,jax.random.PRNGKey(0)))(obs)))
 harvester=HarvesterAgent();harvest=jax.jit(lambda obs:encode_action(jax.vmap(lambda o:harvester.act(o,jax.random.PRNGKey(0)))(obs)))
 script=jax.jit(lambda obs,keys,tick,kind:jax.vmap(jax.random.categorical)(jax.vmap(lambda k:jax.random.fold_in(k,tick))(keys),scripted_logits(obs,kind)).astype(jnp.int32))
 rows=[];isolation=[];started=time.monotonic();clients={};opponent_clients={};parts=[(op,seat,indices[c:c+batch_size]) for op in panel for seat in (0,1) for c in range(0,len(indices),batch_size)]
 with contextlib.ExitStack() as lifetime:
  for part_number,(op,seat,chosen_maps) in enumerate(parts):
   for old_key in list(opponent_clients):
    if old_key[0]!=op['name']:
     opponent_clients.pop(old_key).close()
   real=len(chosen_maps);padded_maps=chosen_maps+[chosen_maps[-1]]*(batch_size-real);count=batch_size;rr=jnp.arange(count);states=jax.tree.map(lambda v:v[jnp.asarray(padded_maps)],pool);active=np.arange(count)<real
   keys=jax.vmap(lambda i:jax.random.fold_in(jax.random.PRNGKey(71091),i))(jnp.asarray(padded_maps))
   passes=np.zeros((count,2),np.int32);cities=np.zeros((count,2),np.int32);actions_record=[]
   with contextlib.ExitStack() as stack:
    key=(seat,count)
    if key not in clients:clients[key]=lifetime.enter_context(Client(candidate,65534-2*seat,development))
    own=clients[key];other=None
    if op['kind']=='neural':
     peer=(op['name'],seat,count)
     if peer not in opponent_clients:opponent_clients[peer]=lifetime.enter_context(Client(op['entry'],65533-2*seat,development))
     other=opponent_clients[peer]
    for tick in range(2048):
     obs,raw=view(states);mine=jax.tree.map(lambda v:v[:,seat],obs);enemy=jax.tree.map(lambda v:v[:,1-seat],obs)
     md=directions(mine.armies,mine.owned_cells,mine.mountains);od=directions(enemy.armies,enemy.owned_cells,enemy.mountains)
     own_ids=own.choose(np.asarray(raw[:,seat]),np.asarray(md))
     if own.calls==1:isolation.append(dict(role='candidate',seat=seat,identity=own.meta['identity'],**getattr(own,'observed_isolation',{})))
     if other:
      opp_ids=other.choose(np.asarray(raw[:,1-seat]),np.asarray(od))
      if other.calls==1:isolation.append(dict(role='opponent',seat=1-seat,identity=other.meta['identity'],**getattr(other,'observed_isolation',{})))
     elif op['kind']=='hunter':opp_ids=np.asarray(hunt(enemy))
     elif op['kind']=='harvester':opp_ids=np.asarray(harvest(enemy))
     elif op['kind'] in ('random','expansion'):opp_ids=np.asarray(script(enemy,keys,jnp.int32(tick),int(op['kind']=='expansion')))
     else:raise ValueError('Unknown opponent kind')
     if not np.asarray(entrant_mask(od))[np.arange(count),opp_ids].all():raise RuntimeError('Opponent illegal action; no imputation')
     ids=jnp.zeros((count,2),jnp.int32).at[:,seat].set(own_ids).at[:,1-seat].set(opp_ids)
     passes[:,seat]+=active&(own_ids==0);passes[:,1-seat]+=active&(opp_ids==0)
     if record:actions_record.append(np.asarray(ids))
     result=advance(states,decode_action(ids));done=np.asarray(result.terminated|result.truncated);reward=np.asarray(result.reward[:,seat]);natural=np.asarray(result.terminated);cap=np.asarray(result.truncated)
     # Scoring diagnostics never enter the player process.
     current_city=np.asarray((result.final_state.ownership&result.final_state.castles[:,None]).sum((-1,-2)))
     cities=np.maximum(cities,current_city)
     for i in np.flatnonzero(active&done):
      rows.append(dict(opponent=op['name'],map_index=chosen_maps[i],seat=seat,score=float((reward[i]+1)/2),natural=bool(natural[i]),truncated=bool(cap[i]),capture_win=bool(reward[i]>0),capture_loss=bool(reward[i]<0),length=tick+1,passes=int(passes[i,seat]),opponent_passes=int(passes[i,1-seat]),max_owned_cities=int(cities[i,seat]),opponent_max_owned_cities=int(cities[i,1-seat])))
     states=jax.tree.map(lambda n,o:jnp.where(jnp.asarray(active).reshape((count,)+(1,)*(n.ndim-1)),n,o),result.final_state,states);active&=~done
     if output and tick%128==0:write_json(Path(output)/'progress.json',dict(part=part_number+1,parts=len(parts),tick=tick,opponent=op['name'],seat=seat,games_complete=len(rows)))
     if not active.any():break
    if active.any():raise RuntimeError('Unfinished cases; no partial score')
   if record and output:np.savez_compressed(Path(output)/f'actions-part-{part_number:03d}.npz',actions=np.asarray(actions_record),map_indices=np.asarray(chosen_maps),seat=np.asarray(seat))
   if output:write_json(Path(output)/'rows-progress.json',dict(rows=rows))
 if len(rows)!=2*len(indices)*len(panel):raise RuntimeError('Incomplete full-task schedule')
 by_map(rows)
 return dict(rows=rows,games=len(rows),seconds=time.monotonic()-started,illegal_actions=0,simulator_faults=0,inference_platform='cuda',candidate_identity=candidate_meta['identity'],candidate_class=candidate_meta['inference_class'],general_trade=True,episode_cap=2048,process_isolation=isolation,information='Per-player isolated process sees raw14player-visibleplanesandlegalmaskonly;AverageJoememoryderivedonlyfromownobservationhistory',unprivileged_policy_observed=not development,development_only=development)
