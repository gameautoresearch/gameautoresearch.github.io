"""Only native public Observation fields enter either neural interface."""
import io,struct
import numpy as np
MAX_BYTES=4*1024*1024
SPATIAL=('armies','generals','cities','mountains','neutral_cells','owned_cells','opponent_cells','fog_cells','structures_in_fog')
SCALARS=('owned_land_count','owned_army_count','opponent_land_count','opponent_army_count','timestep')


def raw_observation(obs):
 import jax.numpy as jnp
 shape=obs.armies.shape
 return jnp.stack([getattr(obs,k) for k in SPATIAL]+[jnp.broadcast_to(getattr(obs,k)[...,None,None],shape) for k in SCALARS],axis=-3).astype(jnp.float32)


def entrant_observation(raw):
 import jax.numpy as jnp
 raw=jnp.asarray(raw);x=jnp.moveaxis(raw,-3,-1)
 x=x.at[...,0].set(jnp.log1p(x[...,0])/6.)
 for i in (9,11):x=x.at[...,i].divide(24*24)
 for i in (10,12):x=x.at[...,i].set(jnp.log1p(x[...,i])/8.)
 return x.at[...,13].divide(2048.)


def entrant_mask(direction_mask):
 import jax.numpy as jnp
 mask=jnp.asarray(direction_mask);joint=jnp.repeat(mask[...,None],2,axis=-1).reshape(mask.shape[:-3]+(4608,))
 return jnp.concatenate([jnp.ones(mask.shape[:-3]+(1,),bool),joint],axis=-1)


def to_entrant(actions):
 import jax.numpy as jnp
 a=jnp.asarray(actions)
 return jnp.where(a[...,0]>0,0,1+((a[...,1]*24+a[...,2])*4+a[...,3])*2+a[...,4]).astype(jnp.int32)


def encode(raw,directions):
 f=io.BytesIO();np.savez(f,raw=np.asarray(raw,np.float32),directions=np.packbits(directions.reshape(len(raw),-1),axis=-1,bitorder='little'))
 value=f.getvalue()
 if len(value)>MAX_BYTES:raise ValueError('Public frame too large')
 return struct.pack('>I',len(value))+value


def decode(value):
 if len(value)>MAX_BYTES:raise ValueError('Public frame too large')
 with np.load(io.BytesIO(value),allow_pickle=False) as f:
  if set(f.files)!={'raw','directions'}:raise ValueError('Only public observation and direction mask allowed')
  raw=f['raw'];packed=f['directions']
 if raw.dtype!=np.float32 or raw.ndim!=4 or raw.shape[1:]!=(14,24,24) or not 1<=len(raw)<=32 or not np.isfinite(raw).all():raise ValueError('Bad raw public observations')
 if packed.dtype!=np.uint8 or packed.shape!=(len(raw),288):raise ValueError('Bad public mask')
 return raw,np.unpackbits(packed,axis=-1,count=2304,bitorder='little').reshape(len(raw),24,24,4).astype(bool)
