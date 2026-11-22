"""Editable plain PPO: outcome reward, visible critic, clipped loss and per-decision GAE."""
import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState
from integrations.generals.packed import PACKED_KEYS,unpack_observation
from integrations.generals.environment import BOARD_SHAPE,ACTION_COUNT
from integrations.generals_autoresearch.rollout import logprob

def create_state(model,key,cfg):
 params=model.init(key,jnp.zeros((1,*BOARD_SHAPE,14)),jnp.ones((1,ACTION_COUNT),bool))['params']
 tx=optax.chain(optax.clip_by_global_norm(cfg['max_grad_norm']),optax.adam(cfg['learning_rate'],eps=1e-5))
 return TrainState.create(apply_fn=model.apply,params=params,tx=tx).replace(step=jnp.int32(0))

def advantages(data,gamma,lam):
 delta=data['reward']+gamma*(~data['terminated'])*data['next_value']-data['value']
 def tick(acc,pair):
  d,end=pair;acc=d+gamma*lam*(~end)*acc
  return acc,acc
 _,adv=jax.lax.scan(tick,jnp.zeros_like(delta[0]),(delta,data['terminated']|data['truncated']),reverse=True)
 return adv,adv+data['value']

def make_update(model,cfg):
 n=cfg['num_envs']*cfg['rollout_steps'];mb=cfg['minibatches']
 if n%mb:raise ValueError('Minibatches must divide rollout samples')
 @jax.jit
 def update(state,data,key):
  adv,ret=advantages(data,cfg['gamma'],cfg['gae_lambda'])
  adv=(adv-adv.mean())/(adv.std()+1e-8)
  batch={k:data[k].reshape((n,)+data[k].shape[2:]) for k in (*PACKED_KEYS,'action','old_logp','value')}
  batch.update(adv=adv.reshape(n),ret=ret.reshape(n))
  def loss(params,part):
   x,mask=unpack_observation(part);logits,value=model.apply({'params':params},x,mask)
   lp=logprob(logits,part['action']);ratio=jnp.exp(lp-part['old_logp'])
   pg=-jnp.mean(jnp.minimum(ratio*part['adv'],jnp.clip(ratio,1-cfg['clip_eps'],1+cfg['clip_eps'])*part['adv']))
   vl=.5*jnp.mean((value-part['ret'])**2)
   entropy=-jnp.mean(jnp.sum(jax.nn.softmax(logits)*jax.nn.log_softmax(logits),-1))
   kl=jnp.mean((ratio-1)-(lp-part['old_logp']))
   total=pg+cfg['value_coef']*vl-cfg['entropy_coef']*entropy
   return total,jnp.array([total,pg,vl,entropy,kl])
  def epoch(c,_):
   st,k=c;k,pk=jax.random.split(k);indices=jax.random.permutation(pk,n).reshape((mb,n//mb))
   def step(st,ix):
    (_,metrics),grads=jax.value_and_grad(loss,has_aux=True)(st.params,jax.tree.map(lambda a:a[ix],batch))
    return st.apply_gradients(grads=grads),jnp.concatenate([metrics,optax.global_norm(grads)[None]])
   st,metrics=jax.lax.scan(step,st,indices)
   return (st,k),metrics.mean(0)
  (state,_),metrics=jax.lax.scan(epoch,(state,key),None,length=cfg['epochs'])
  return state,metrics.mean(0)
 return update
