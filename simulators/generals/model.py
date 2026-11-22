"""Small spatial PPO policy; actor and critic consume identical visible inputs."""
import jax
import jax.numpy as jnp
from flax import linen as nn


def historical_policy(model, cfg):
    convolution = cfg.get('historical_convolution', model.convolution)
    if convolution == 'implicit':
        from .implicit import HistoricalPolicy
        return HistoricalPolicy(model.width, model.compute_dtype)
    return Policy(model.width, convolution=convolution, compute_dtype=model.compute_dtype)


def patch_convolution(lhs, rhs, window_strides, padding, lhs_dilation=None,
                      rhs_dilation=None, dimension_numbers=None,
                      feature_group_count=1, batch_group_count=1,
                      precision=None, preferred_element_type=None):
    """The same NHWC convolution expressed as patches and a matrix multiply.

    XLA CPU's grouped convolution lowering is costly for the tiny, per-episode
    historical networks. Patches also support the vmap's independent weights.
    Parameters, padding and gradients retain the original Conv contract. Float
    accumulation order can differ; this is an explicit execution option.
    """
    if feature_group_count != 1 or batch_group_count != 1:
        raise ValueError("Patch convolution expects ungrouped input before vmap")
    dims = jax.lax.conv_dimension_numbers(lhs.shape, rhs.shape, dimension_numbers)
    if dims != jax.lax.ConvDimensionNumbers((0, 3, 1, 2), (3, 2, 0, 1), (0, 3, 1, 2)):
        raise ValueError("Patch convolution requires NHWC/HWIO/NHWC")
    patches = jax.lax.conv_general_dilated_patches(
        lhs, rhs.shape[:2], window_strides, padding,
        lhs_dilation=lhs_dilation, rhs_dilation=rhs_dilation,
        dimension_numbers=dimension_numbers, precision=precision)
    # conv_general_dilated_patches orders each patch as input-channel, row, col.
    weights = rhs.transpose(2, 0, 1, 3).reshape((-1, rhs.shape[-1]))
    return jnp.matmul(patches, weights, precision=precision,
                      preferred_element_type=preferred_element_type)


def slice_convolution(lhs, rhs, window_strides, padding, lhs_dilation=None,
                      rhs_dilation=None, dimension_numbers=None,
                      feature_group_count=1, batch_group_count=1,
                      precision=None, preferred_element_type=None):
    """Static slices avoid the identity-convolution used by JAX's patch helper."""
    if (window_strides != (1, 1) or padding != 'SAME'
            or lhs_dilation not in (None, (1, 1)) or rhs_dilation not in (None, (1, 1))
            or feature_group_count != 1 or batch_group_count != 1):
        raise ValueError('Slice convolution supports the frozen policy shapes only')
    h, w = lhs.shape[1:3]
    kh, kw = rhs.shape[:2]
    padded = jnp.pad(lhs, ((0, 0), (kh//2, kh//2), (kw//2, kw//2), (0, 0)))
    patches = jnp.concatenate([padded[:, i:i+h, j:j+w, :]
                               for i in range(kh) for j in range(kw)], axis=-1)
    return jnp.matmul(patches, rhs.reshape((-1, rhs.shape[-1])), precision=precision,
                      preferred_element_type=preferred_element_type)


class Policy(nn.Module):
    width: int = 32
    convolution: str = "direct"
    compute_dtype: str = "float32"

    @nn.compact
    def __call__(self, x, mask):
        if self.convolution not in ("direct", "patch", "slice"):
            raise ValueError("convolution must be direct, patch or slice")
        if self.compute_dtype not in ('float32', 'bfloat16'):
            raise ValueError('compute_dtype must be float32 or bfloat16')
        dtype = jnp.float32 if self.compute_dtype == 'float32' else jnp.bfloat16
        conv = {'patch': patch_convolution, 'slice': slice_convolution}.get(self.convolution)
        for _ in range(3):
            x = nn.relu(nn.Conv(self.width, (3, 3), padding="SAME", conv_general_dilated=conv, dtype=dtype)(x))
        # C-order output is row, col, direction, split, matching the joint codec.
        moves = nn.Conv(8, (1, 1), kernel_init=nn.initializers.orthogonal(.01), conv_general_dilated=conv, dtype=dtype)(x).reshape((x.shape[0], -1))
        pooled = jnp.mean(x, axis=(1, 2))
        passed = nn.Dense(1, kernel_init=nn.initializers.orthogonal(.01), dtype=dtype)(pooled)
        logits = jnp.concatenate([passed, moves], -1).astype(jnp.float32)
        value = nn.Dense(1, dtype=dtype)(nn.tanh(nn.Dense(self.width, dtype=dtype)(pooled)))[..., 0].astype(jnp.float32)
        return jnp.where(mask, logits, -1e9), value
