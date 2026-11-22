"""Episodic PPO with advantage normalization and KL-guarded epochs."""
import numpy as np
import torch
from contextlib import nullcontext
from ..collection import gae


def update(model, optimizer, episodes, c, device, anchor=None, timer=None):
    span = timer.measure if timer is not None else lambda name: nullcontext()
    with span('ppo_assembly_transfer'):
        inputs = _prepare(model, episodes, c, device, anchor)
    with span('ppo_optimization'):
        return _steps(model, optimizer, *inputs, c=c, device=device)


def _prepare(model, episodes, c, device, anchor):
    rows = []
    adv = []
    ret = []
    for ep in episodes:
        a, r = gae(ep, c['gamma'], c['gae_lambda'])
        rows.extend(ep)
        adv.extend(a)
        ret.extend(r)
    if len(rows) < 2:
        raise ValueError('No complete natural trajectories')
    if hasattr(episodes, 'pack'):
        f, cards, masks = episodes.pack(rows, device)
    else:
        f = torch.tensor(np.stack([r['features'] for r in rows]), device=device)
        cards = torch.tensor(np.stack([r['cards'] for r in rows]), device=device, dtype=torch.long)
        masks = torch.tensor(np.stack([r['mask'] for r in rows]), device=device)
    actions = torch.tensor([r['action'] for r in rows], device=device)
    old = torch.tensor([r['logp'] for r in rows], device=device)
    adv = torch.tensor(np.array(adv, dtype=np.float32), device=device)
    ret = torch.tensor(np.array(ret, dtype=np.float32), device=device)
    adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
    if c.get('preprocess_numeric', False):
        # A whole-cohort sign/abs/log1p expression can allocate several complete
        # feature buffers at once. Bound its scratch independently of cohort size.
        encoded = torch.empty_like(f)
        for begin in range(0, len(f), 8192):
            encoded[begin:begin+8192].copy_(model.encode_numeric(f[begin:begin+8192]))
        f = encoded
    anchor_coef = float(c.get('anchor_kl_coef', 0))
    if not np.isfinite(anchor_coef) or anchor_coef < 0:
        raise ValueError('Finite nonnegative incumbent KL coefficient required')
    anchor_probs = anchor_logp = None
    if anchor_coef:
        if anchor is None:
            raise ValueError('Frozen visible-only incumbent required for KL anchoring')
        with torch.no_grad():
            anchor_dist, _ = anchor.distribution(f, cards, masks, numeric_encoded=c.get('preprocess_numeric', False))
            anchor_probs, anchor_logp = anchor_dist.probs, anchor_dist.logits

    return rows, f, cards, masks, actions, old, adv, ret, anchor_probs, anchor_logp, anchor_coef


def _steps(model, optimizer, rows, f, cards, masks, actions, old, adv, ret,
           anchor_probs, anchor_logp, anchor_coef, c, device):
    target_kl = float(c.get('target_kl', 0.03))
    before = torch.cat([p.detach().flatten() for p in model.parameters()]).clone()
    losses = []
    kls = []
    anchor_kls = []
    stop = False
    for _ in range(int(c['ppo_epochs'])):
        if stop:
            break
        for ids in torch.randperm(len(rows), device=device).split(int(c['minibatch_size'])):
            if c.get('ppo_backward_graph',False) and str(device).startswith('cuda') and len(ids)==int(c['minibatch_size']) and not anchor_coef:
                from .ppo_graph import backward
                packed=backward(model,(f[ids],cards[ids],masks[ids],actions[ids],old[ids],adv[ids],ret[ids]),c).cpu().numpy()
                if not packed[2]:raise ValueError('Nonfinite PPO loss')
                optimizer.step()
                losses.append(float(packed[0]));kls.append(float(packed[1]))
                if kls[-1]>1.5*target_kl and target_kl>0:stop=True;break
                continue
            dist, v = model.distribution(f[ids], cards[ids], masks[ids], numeric_encoded=c.get('preprocess_numeric', False))
            logp = dist.log_prob(actions[ids])
            ratio = (logp - old[ids]).exp()
            loss = (-torch.minimum(ratio * adv[ids],
                                   ratio.clamp(1 - c['clip'], 1 + c['clip']) * adv[ids]).mean()
                    + c['value_coef'] * ((v - ret[ids]) ** 2).mean()
                    - c['entropy_coef'] * dist.entropy().mean())
            if anchor_coef:
                anchor_kl = (anchor_probs[ids] * (anchor_logp[ids] - dist.logits)).sum(-1).mean()
                loss = loss + anchor_coef * anchor_kl
                anchor_kls.append(float(anchor_kl.detach()))
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite PPO loss')
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), c['max_grad_norm'])
            optimizer.step()
            losses.append(float(loss.detach()))
            with torch.no_grad():
                kls.append(float((old[ids] - logp).mean()))
            if kls[-1] > 1.5 * target_kl and target_kl > 0:
                stop = True
                break
    delta = float((torch.cat([p.detach().flatten() for p in model.parameters()]) - before).norm())
    return dict(loss=float(np.mean(losses)), samples=len(rows),
                approx_kl=float(np.mean(kls)), parameter_delta_l2=delta,
                incumbent_kl=float(np.mean(anchor_kls)) if anchor_kls else 0.,
                incumbent_kl_coef=anchor_coef)
