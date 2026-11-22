"""Opt-in fixed-capacity PPO action terms using the SAME Policy parameters.

Ported from the qualified evidence-only native-packed-head-20261108 reference.
The complete encoder and critic retain all N samples. Only policy_logits runs on
one stable, unique active-first prefix of a HOST-STATIC capacity. No ragged
nonzero/boolean compaction, CPU device count read, sampler, or model replacement.
An undersized direct branch fails closed; PackedHeadLearnerGraph selects a safe
capacity on device and always retains full-N/full-200 fallbacks.
"""
from __future__ import annotations
from collections.abc import Mapping
import torch
from torch.distributions import Categorical

INPUT_KEYS=('obs','mask','action','logp','advantage','return','weight')
METRIC_NAMES=('policy_loss','value_loss','entropy','approx_kl','gradient_norm')
HEAD_PARAMETER_NAMES=('tiles','slot.weight','slot.bias','hand_query.0.weight','hand_query.0.bias')

def active_rows(mask):
    if mask.ndim!=2 or mask.shape[1]<2:raise ValueError('mask must retain the complete two-dimensional action layout')
    mask=mask.bool()
    return mask[:,1:].any(1)|~mask[:,0]

def head_bounds(n):
    if type(n) is not int or n<1:raise ValueError('positive static minibatch size required')
    return tuple(sorted({min(value,n) for value in (32,64,128,n)}))

def capacity_for_cpu(mask,requested=None):
    """Reference/qualification selector ONLY. CUDA uses PackedHeadBranchGraph."""
    if mask.device.type!='cpu':raise ValueError('CPU selector cannot synchronize device counts')
    bounds=head_bounds(mask.shape[0]);count=int(active_rows(mask).sum())
    if requested is None:capacity=next(bound for bound in bounds if count<=bound)
    else:
        if type(requested) is not int or not 1<=requested<=mask.shape[0]:raise ValueError('requested capacity outside full N')
        capacity=requested if count<=requested else mask.shape[0]
    return capacity,{'active_count':count,'capacity':capacity,'full_N_fallback':requested is not None and count>requested}

def packed_action_terms(policy,batch,actions,capacity=None,*,return_selection=False):
    """Fixed N/cap terms; extra inactive prefix rows contribute exact zeros.

    capacity=None is the safe full-N branch. A smaller static branch asserts its
    precondition ON DEVICE instead of silently dropping active rows. Values and
    full-N scatter outputs preserve every sample. Nonempty dummy head operations
    keep head gradients present exact zero on wholly forced minibatches, allowing
    existing Adam momentum/steps to evolve normally.
    """
    if not torch.are_deterministic_algorithms_enabled() or torch.is_deterministic_algorithms_warn_only_enabled():raise ValueError('packed head prototype requires strict deterministic algorithms')
    mask=batch['mask'].bool();n=mask.shape[0]
    if torch.is_autocast_enabled(mask.device.type):raise ValueError('packed head prototype requires strict FP32 without autocast')
    capacity=n if capacity is None else capacity
    if type(capacity) is not int or not 1<=capacity<=n:raise ValueError('static capacity must lie within full N')
    if actions.shape!=(n,) or actions.dtype!=torch.int64 or actions.device!=mask.device:raise ValueError('actions must retain one int64 value per row on the same device')
    active=active_rows(mask)
    torch._assert_async(active.to(torch.int32).sum()<=capacity,'packed policy head would drop active rows; select the full-N fallback')
    context,h=policy.encode(batch)
    if context.dtype!=torch.float32:raise ValueError('packed head prototype requires FP32 model parameters/context')
    order=torch.argsort((~active).to(torch.int32),stable=True)
    selected=order[:capacity]
    d=Categorical(logits=policy.policy_logits(context.index_select(0,selected),h.index_select(0,selected),mask.index_select(0,selected)),validate_args=False)
    selected_active=active.index_select(0,selected)
    # Explicit forced terms also cover dummy fillers; keep both branches in the
    # autograd graph so policy-exclusive gradients are tensors rather than None.
    new=torch.where(selected_active,d.log_prob(actions.index_select(0,selected)),torch.zeros(capacity,device=context.device,dtype=context.dtype))
    ent=torch.where(selected_active,d.entropy(),torch.zeros_like(new))
    logp=context.new_zeros(n).scatter(0,selected,new)
    entropy=context.new_zeros(n).scatter(0,selected,ent)
    result=(logp,entropy,policy.value(context).squeeze(-1))
    return (result,selected) if return_selection else result

def ppo_loss(terms,fields,c):
    """Original PPO domains/denominators; no selected-cap mean or lost rows."""
    new,entropy,value=terms;old=fields['logp'];ratio=(new-old).exp();w=fields['weight'];a=fields['advantage']
    pg=-(torch.minimum(ratio*a,ratio.clamp(1-c['clip'],1+c['clip'])*a)*w).sum()/w.sum().clamp_min(1)
    vl=(value-fields['return']).square().mean()
    ent=(entropy*w).sum()/w.sum().clamp_min(1)
    loss=pg+c['value_coef']*vl-c['entropy_coef']*ent
    with torch.no_grad():kl=((ratio-1)-(new-old)).mean()
    return loss,torch.stack((pg.detach(),vl.detach(),ent.detach(),kl))

def packed_minibatch(trainer,fields,capacity=None):
    """Complete eager update control: same finite/zero/backward/clip/Adam order."""
    if trainer.c.get('compile_loss',False):raise ValueError('packed-head prototype does not silently combine an unqualified compiled loss')
    batch=trainer.encoder(fields['obs'],fields['mask'])
    loss,metrics=ppo_loss(packed_action_terms(trainer.model,batch,fields['action'],capacity),fields,trainer.c)
    torch._assert_async(torch.isfinite(loss),'nonfinite packed PPO loss')
    trainer.optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad=torch.nn.utils.clip_grad_norm_(trainer.model.parameters(),trainer.c['max_grad_norm'])
    trainer.optimizer.step()
    return torch.cat((metrics,grad.detach().reshape(1)))

def workload(policy,n,capacity):
    """Analytical traffic/work only; sort, scalar work and all encoder work extra."""
    if not 1<=capacity<=n:raise ValueError('invalid capacity')
    slots=policy.s.hand_slots;tiles=policy.s.cols*policy.s.rows
    flops=2*policy.slot.in_features*policy.slot.out_features+2*slots*policy.hand_query[0].in_features*policy.hand_query[0].out_features+2*slots*policy.width*tiles
    return {'N':n,'head_capacity':capacity,'all_N_encoder_and_critic':True,'head_linear_flops_full':n*flops,'head_linear_flops_packed':capacity*flops,'head_logits_float32_bytes_full':n*policy.s.actions*4,'head_logits_float32_bytes_packed':capacity*policy.s.actions*4,'mask_selector_scan_bytes':n*policy.s.actions,'stable_sort_key_and_index_elements':n,'head_mask_gather_bytes':capacity*policy.s.actions,'two_full_N_scalar_outputs_bytes':2*n*4,'scope':'analytical only; no timing or total-system speed claim; backward/embedding/sort/gather/scatter overhead excluded'}
