"""Editable small spatial policy and visible-state value function."""
import jax.numpy as jnp
from flax import linen as nn

class Policy(nn.Module):
 width:int=16
 compute_dtype:str='bfloat16'
 @nn.compact
 def __call__(self,x,mask):
  dtype=jnp.bfloat16 if self.compute_dtype=='bfloat16' else jnp.float32
  for _ in range(3):x=nn.relu(nn.Conv(self.width,(3,3),padding='SAME',dtype=dtype)(x))
  moves=nn.Conv(8,(1,1),kernel_init=nn.initializers.orthogonal(.01),dtype=dtype)(x).reshape((x.shape[0],-1))
  pooled=jnp.mean(x,axis=(1,2))
  passed=nn.Dense(1,kernel_init=nn.initializers.orthogonal(.01),dtype=dtype)(pooled)
  logits=jnp.concatenate([passed,moves],-1).astype(jnp.float32)
  value=nn.Dense(1,dtype=dtype)(nn.tanh(nn.Dense(self.width,dtype=dtype)(pooled)))[...,0].astype(jnp.float32)
  return jnp.where(mask,logits,-1e9),value

def create(config):return Policy(width=int(config.get('width',16)),compute_dtype=config.get('compute_dtype','bfloat16'))
