"""Policy-only active-row collection and batched post-rollout value inference.

Weights stay fixed throughout a rollout. Forced wait rows need no policy network, but
still receive a learned value for GAE. No decisions or simulation ticks are skipped.
"""
from __future__ import annotations
import numpy as np
import torch
from torch.distributions import Categorical


def sample_actions(model,obs,masks,c,timer,greedy=False):
    masks=np.asarray(masks).reshape(-1,model.s.actions)
    active=masks[:,1:].any(1)|~masks[:,0].astype(bool)
    ids=np.flatnonzero(active)
    packed_ids=ids
    bucket=c.get('actor_batch_bucket',0)
    if len(ids) and bucket:
        size=((len(ids)+bucket-1)//bucket)*bucket
        packed_ids=np.pad(ids,(0,size-len(ids)),constant_values=ids[-1])
    with torch.no_grad():
        with timer.section('pack_and_transfer'):
            batch=model.pack(obs[packed_ids],masks[packed_ids],c['device'],c['trim_entities'],bucket_override=c.get('inference_entity_bucket')) if len(ids) else None
        with timer.section('policy_inference'):
            logits=torch.full(masks.shape,-1e9,device=c['device']);logits[:,0]=0
            if batch is not None:
                # Use distribution() to retain the existing optional CUDA Graph cache.
                # Only active probabilities matter; the all-row sample below keeps the
                # sampler's original shape and advances RNG even for forced rows.
                d,_=model.distribution(batch)
                logits[torch.as_tensor(ids,device=c['device'])]=d.logits[:len(ids)]
            d=Categorical(logits=logits,validate_args=False)
            action=d.logits.argmax(-1) if greedy else d.sample()
            logp=d.log_prob(action)
        with timer.section('actions_to_cpu'):
            result=torch.stack((action.float(),logp),1).cpu().numpy()
    return result[:,0].astype(np.int32),result[:,1]


def predict_values(model,obs,c,timer):
    size=c.get('value_batch_size',1024)
    output=[]
    with torch.no_grad():
        for start in range(0,len(obs),size):
            with timer.section('value_pack_and_transfer'):
                batch=model.pack(obs[start:start+size],device=c['device'],trim=c['trim_entities'])
            with timer.section('batched_values'):
                output.append(model.values(batch).cpu().numpy())
    return np.concatenate(output)


def evaluation_probabilities(model,obs,masks,c,timer):
    """Skip neural work for a point mass; caller still advances every seat RNG."""
    masks=np.asarray(masks).reshape(-1,model.s.actions)
    ids=np.flatnonzero(masks[:,1:].any(1)|~masks[:,0].astype(bool))
    probabilities=np.zeros(masks.shape,dtype=np.float64);probabilities[:,0]=1
    if len(ids):
        packed_ids=ids
        bucket=c.get('actor_batch_bucket',0)
        if bucket:
            size=((len(ids)+bucket-1)//bucket)*bucket
            packed_ids=np.pad(ids,(0,size-len(ids)),constant_values=ids[-1])
        with timer.section('pack_and_transfer'):
            batch=model.pack(obs[packed_ids],masks[packed_ids],c['device'],c['trim_entities'],bucket_override=c.get('inference_entity_bucket'))
        with torch.no_grad(),timer.section('policy_inference'):
            d,_=model.distribution(batch)
            probabilities[ids]=d.probs[:len(ids)].cpu().numpy()
    return probabilities
