"""Lossless compact storage of visible boards for thousands of simultaneous GPU games.

Nine spatial and five scalar float planes otherwise repeat booleans/scalars at
every board tile. Store raw visible armies, eight binary planes in one byte, five
public scalars, and a bitpacked joint mask, and rebuild the float planes only where
they are needed. This accepts Observation, never privileged GameState.
"""
import jax.numpy as jnp
from .environment import SPATIAL, SCALARS, RULESET

PACKED_KEYS = ('packed_armies', 'packed_bits', 'packed_scalars', 'packed_mask')


def pack_observation(obs, mask):
    bits = jnp.zeros_like(obs.armies, dtype=jnp.uint8)
    for i, name in enumerate(SPATIAL[1:]):
        bits = bits | (getattr(obs, name).astype(jnp.uint8) << i)
    return dict(packed_armies=obs.armies,
                packed_bits=bits,
                packed_scalars=jnp.stack([getattr(obs, name) for name in SCALARS], -1),
                packed_mask=jnp.packbits(mask, axis=-1, bitorder='little'))


def unpack_observation(batch):
    armies, bits, scalars = (batch[k] for k in PACKED_KEYS[:3])
    planes = [jnp.log1p(armies.astype(jnp.float32))/6.]
    planes += [((bits >> i) & 1).astype(jnp.float32) for i in range(8)]
    for i in range(5):
        x = scalars[..., i].astype(jnp.float32)
        x = x/(armies.shape[-2]*armies.shape[-1]) if i in (0, 2) else (x/RULESET['truncation'] if i == 4 else jnp.log1p(x)/8)
        planes.append(jnp.broadcast_to(x[..., None, None], armies.shape))
    mask = jnp.unpackbits(batch['packed_mask'], axis=-1, count=1+armies.shape[-2]*armies.shape[-1]*8, bitorder='little').astype(bool)
    return jnp.stack(planes, -1), mask


def rollout_inputs(obs, x, masks, rows, seats, packed):
    if not packed:
        return dict(x=x[rows, seats], mask=masks[rows, seats])
    # Selecting visible fields before packing avoids retaining both seats.
    import jax
    selected = jax.tree.map(lambda a: a[rows, seats], obs)
    return pack_observation(selected, masks[rows, seats])
