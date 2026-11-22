"""Isolated inference: native public observations only, bounded explicit memory."""
import argparse,json,sys
from pathlib import Path


def main():
 protocol=sys.stdout.buffer;sys.stdout=sys.stderr
 from ..wire import read_frame,reply
 from .codec import decode,entrant_mask,entrant_observation,to_entrant
 import jax,jax.numpy as jnp,numpy as np
 p=argparse.ArgumentParser();p.add_argument('--entry',required=True);a=p.parse_args();root=Path(a.entry);meta=json.loads((root/'registry.json').read_text());kind=meta['inference_class'];memory=None
 if kind=='entrant-stateless-600k-v1':
  import importlib.util
  from ..bundle import load_weights
  b=root/'bundle';spec=importlib.util.spec_from_file_location('registered_policy',b/'policy.py');module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
  model=module.create(json.loads((b/'model.json').read_text()));params=load_weights(b/'weights.npz')
  @jax.jit
  def act(raw,directions):
   x=entrant_observation(raw);mask=entrant_mask(directions)
   def one(xx,mm):return model.apply({'params':params},xx[None],mm[None])[0][0]
   logits=jax.vmap(one)(x,mask);ids=jnp.argmax(jnp.where(mask,logits,-jnp.inf),-1)
   return ids,jnp.all(jnp.isfinite(jnp.where(mask,logits,0.)))
 else:
  import equinox as eqx
  from simulators.generals.averagejoe.bootstrap import activate
  activate()
  from simulators.generals.averagejoe.framework import recipe
  from networks import build_network
  from networks.common import init_obs_state,augment_obs,reset_done_envs,decode_action
  from networks.transformer import greedy_action_transformer
  cfg=recipe()
  if cfg.to_dict()!=json.loads((root/'config.json').read_text()):raise ValueError('Published model configuration mismatch')
  model=eqx.tree_deserialise_leaves(root/'model.eqx',build_network(cfg,jax.random.PRNGKey(0)))
  leaves=[v for v in jax.tree.leaves(model) if eqx.is_array(v)]
  if sum(v.size for v in leaves)!=meta['parameters'] or not all(np.isfinite(np.asarray(v)).all() for v in leaves):raise ValueError('Invalid trusted reference tensors')
  one=init_obs_state(24,24)
  if sum(v.nbytes for v in jax.tree.leaves(one))>65536:raise ValueError('Reference persistent-memory cap exceeded')
  @eqx.filter_jit
  def act_ref(raw,directions,memory):
   memory=reset_done_envs(memory,raw[:,13,0,0]==0)
   x,m=jax.vmap(augment_obs)(raw,memory);temporal=jnp.stack([m.opponent_army_history,m.opponent_land_history],1)
   logits=jax.vmap(lambda xx,mm,tt:model._forward(xx,mm,tt)[0])(x,directions,temporal)
   actions=jax.vmap(lambda i:decode_action(i,24))(jnp.argmax(logits,-1))
   return to_entrant(actions),m,jnp.all(jnp.isfinite(logits))
 while (value:=read_frame(sys.stdin.buffer)) is not None:
  raw,directions=decode(value);mask=np.asarray(entrant_mask(directions))
  if kind=='entrant-stateless-600k-v1':
   ids,finite=act(raw,directions)
   if not bool(finite):raise ValueError('Nonfinite entrant logits')
  else:
   if memory is None:memory=jax.tree.map(lambda v:jnp.broadcast_to(v,(len(raw),)+v.shape),one)
   if len(jax.tree.leaves(memory)[0])!=len(raw):raise ValueError('Stable per-game reference slots required')
   ids,memory,finite=act_ref(raw,directions,memory)
   if not bool(finite):raise ValueError('Nonfinite reference logits')
  ids=np.asarray(ids)
  if ids.shape!=(len(raw),) or not mask[np.arange(len(raw)),ids].all():raise ValueError('Illegal neural response')
  protocol.write(reply(ids));protocol.flush()

if __name__=='__main__':main()
