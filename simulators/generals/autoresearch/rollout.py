"""Frozen fast self-play plumbing. Models receive only visible encoded observations."""
from typing import NamedTuple
import jax
import jax.numpy as jnp
import numpy as np
from generals.core.game import GameState
from integrations.generals.environment import observations,encode,action_mask,decode_action,transition,reset_done,RULESET,BOARD_SHAPE,ACTION_COUNT
from integrations.generals.packed import pack_observation

class Carry(NamedTuple):
 states:object
 key:object
 seats:object
 episodes:object

def logprob(logits,ids):return jnp.take_along_axis(jax.nn.log_softmax(logits),ids[...,None],-1)[...,0]

def load_pool(path):
 with np.load(path,allow_pickle=False) as a:
  if set(a.files)!=set(GameState._fields):raise ValueError('Unexpected map-bank fields')
  pool=GameState(**{k:jnp.asarray(a[k]) for k in GameState._fields})
 if pool.armies.shape[1:]!=BOARD_SHAPE or pool.ownership.shape[1:]!=(2,*BOARD_SHAPE):raise ValueError('Map bank does not match '+RULESET['id'])
 return pool

def initial_carry(pool,key,batch):
 key,k=jax.random.split(key)
 states=jax.tree.map(lambda a:a[jax.random.randint(k,(batch,),0,len(pool.time))],pool)
 return Carry(states,key,jnp.arange(batch)%2,jnp.zeros(batch,jnp.int32))

def make_collect(model,cfg):
 b,t=cfg['num_envs'],cfg['rollout_steps'];rows=jnp.arange(b)
 if not 1<=t<=RULESET['truncation']:raise ValueError('Rollout horizon must fit the episode cap')
 @jax.jit
 def collect(params,carry,pool):
  def tick(loop,_):
   c,caps,cap_seats=loop
   key,ka,kr=jax.random.split(c.key,3)
   obs=observations(c.states);x=encode(obs);mask=action_mask(obs)
   logits,value=model.apply({'params':params},x.reshape((2*b,*BOARD_SHAPE,14)),mask.reshape((2*b,ACTION_COUNT)))
   logits=logits.reshape((b,2,ACTION_COUNT));value=value.reshape((b,2))
   ids=jax.random.categorical(ka,logits).astype(jnp.int32)
   r=transition(c.states,decode_action(ids));done=r.terminated|r.truncated
   caps=jax.tree.map(lambda n,o:jnp.where(r.truncated.reshape((b,)+(1,)*(n.ndim-1)),n,o),r.final_state,caps)
   cap_seats=jnp.where(r.truncated,c.seats,cap_seats)
   own=jax.tree.map(lambda a:a[rows,c.seats],obs)
   out=pack_observation(own,mask[rows,c.seats])
   out.update(action=ids[rows,c.seats],old_logp=logprob(logits[rows,c.seats],ids[rows,c.seats]),
    value=value[rows,c.seats],reward=r.reward[rows,c.seats],terminated=r.terminated,truncated=r.truncated,
    wins=r.reward[rows,c.seats]>0,losses=r.reward[rows,c.seats]<0,
    invalid=~jnp.take_along_axis(mask,ids[...,None],-1)[...,0],
    land=own.owned_land_count,next_land=r.final_state.ownership.sum((2,3))[rows,c.seats])
   states=reset_done(r.final_state,done,pool,jax.random.randint(kr,(b,),0,len(pool.time)))
   return (Carry(states,key,jnp.where(done,1-c.seats,c.seats),c.episodes+done),caps,cap_seats),out
  (final,caps,seats),data=jax.lax.scan(tick,(carry,carry.states,carry.seats),None,length=t)
  def critic(states,seats):
   obs=observations(states)
   return model.apply({'params':params},encode(obs)[rows,seats],action_mask(obs)[rows,seats])[1]
  tail=critic(final.states,final.seats)
  cap_value=jax.lax.cond(data['truncated'].any(),lambda _:critic(caps,seats),lambda _:jnp.zeros(b),None)
  data['next_value']=jnp.where(data['truncated'],cap_value[None],jnp.concatenate([data['value'][1:],tail[None]]))
  data['next_value']=jnp.where(data['terminated'],0.,data['next_value'])
  return final,data
 return collect
