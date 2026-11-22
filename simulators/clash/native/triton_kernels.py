"""Optional fused device arithmetic; no simulator/rules transitions live here."""
import torch
import triton
import triton.language as tl

@triton.jit
def _gae(reward,value,done,trunc,last,adv,returns,N:tl.constexpr,T:tl.constexpr,GAMMA:tl.constexpr,LAM:tl.constexpr,BLOCK:tl.constexpr):
    seat=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);valid=seat<N
    carry=tl.full((BLOCK,),0,tl.float32);next_value=tl.load(last+seat,valid,0)
    for reverse in range(T):
        tick=T-1-reverse;at=tick*N+seat
        r=tl.load(reward+at,valid,0);v=tl.load(value+at,valid,0);d=tl.load(done+at,valid,0);tv=tl.load(trunc+at,valid,0)
        nxt=tl.where(d,tv,next_value)
        carry=(r+GAMMA*nxt-v)+((GAMMA*LAM)*(1-d.to(tl.float32)))*carry
        tl.store(adv+at,carry,valid);tl.store(returns+at,carry+v,valid);next_value=v

def gae(reward,value,done,trunc,last,gamma,lam):
    if reward.device.type!='cuda':raise ValueError('Triton GAE requires CUDA')
    advantage=torch.empty_like(reward);returns=torch.empty_like(value)
    _gae[(triton.cdiv(reward.shape[1],256),)](reward,value,done,trunc,last,advantage,returns,reward.shape[1],reward.shape[0],gamma,lam,256,enable_fp_fusion=False)
    return advantage,returns
