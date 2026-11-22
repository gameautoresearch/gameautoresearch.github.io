"""GPU historical-policy inference without materializing spatial patches.

One Pallas/Triton program computes64 pixels of a board per convolution.
Neighborhood loads, BF16 dot, rounding, bias and activation share a kernel.
Float32 parameters and the Flax policy's named parameter tree are preserved.
This inference-only option deliberately does not replace the autodiff learner.
"""
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plt


def convolution(x, w, bias, *, relu=True, interpret=False):
    b, h, width, ci = x.shape
    kh, kw, wi, co = w.shape
    if wi != ci or (kh, kw) not in ((3, 3), (1, 1)):
        raise ValueError('Implicit historical convolution requires NHWC/HWIO policy shapes')
    if x.dtype != jnp.bfloat16:
        raise ValueError('Implicit historical convolution requires BF16 compute')
    ksize = kh*kw*ci
    bk = 1 << (ksize-1).bit_length()
    bn = max(16, 1 << (co-1).bit_length())

    def kernel(xr, wr, br, yr):
        bi = pl.program_id(0)
        m = pl.program_id(1)*64+jnp.arange(64)
        k = jnp.arange(bk)
        n = jnp.arange(bn)
        yy = m[:, None]//width+k[None, :]//(kw*ci)-kh//2
        xx = m[:, None]%width+(k[None, :]//ci)%kw-kw//2
        a = plt.load(xr.at[bi, yy, xx, k[None, :]%ci],
                     mask=(m[:,None]<h*width)&(yy>=0)&(yy<h)&(xx>=0)&(xx<width)&(k[None, :]<ksize), other=0)
        weights = plt.load(wr.at[k[:, None], n[None, :]],
                           mask=(k[:, None]<ksize)&(n[None, :]<co), other=0)
        # Flax Conv first rounds the BF16 convolution output, then adds a
        # BF16 bias. Preserve both roundings rather than fusing an FP32 bias.
        z = plt.dot(a, weights).astype(x.dtype)
        bb = plt.load(br.at[n], mask=n<co, other=0).astype(x.dtype)
        z = (z+bb[None, :]).astype(x.dtype)
        if relu:
            z = jnp.maximum(z, 0)
        plt.store(yr.at[bi, m[:, None]//width, m[:, None]%width, n[None, :]], z,
                  mask=(m[:,None]<h*width)&(n[None, :]<co))

    return pl.pallas_call(kernel, out_shape=jax.ShapeDtypeStruct((b, h, width, co), x.dtype),
        grid=(b, (h*width+63)//64), interpret=interpret, name='generals_implicit_conv',
        compiler_params=plt.CompilerParams(num_warps=4))(
            x, w.reshape((ksize, co)).astype(x.dtype), bias)


def apply(params, x, mask, *, interpret=False):
    x = x.astype(jnp.bfloat16)
    for i in range(3):
        p = params[f'Conv_{i}']
        x = convolution(x, p['kernel'], p['bias'], interpret=interpret)
    p = params['Conv_3']
    moves = convolution(x, p['kernel'], p['bias'], relu=False,
                        interpret=interpret).reshape((x.shape[0], -1))
    pooled = jnp.mean(x, (1, 2))

    def dense(name, value):
        p = params[name]
        return jnp.matmul(value, p['kernel'].astype(jnp.bfloat16))+p['bias'].astype(jnp.bfloat16)

    passed = dense('Dense_0', pooled)
    # Flax names the outer Dense constructor before evaluating its inner arg.
    value = dense('Dense_1', jnp.tanh(dense('Dense_2', pooled)))[..., 0].astype(jnp.float32)
    logits = jnp.concatenate([passed, moves], -1).astype(jnp.float32)
    return jnp.where(mask, logits, -1e9), value


class HistoricalPolicy:
    def __init__(self, width, compute_dtype):
        if width != 16 or compute_dtype != 'bfloat16':
            raise ValueError('Implicit historical inference supports width16 BF16 only')
        if jax.default_backend() != 'gpu':
            raise ValueError('Implicit historical inference requires a CUDA GPU')

    def apply(self, variables, x, mask):
        return apply(variables['params'], x, mask)
