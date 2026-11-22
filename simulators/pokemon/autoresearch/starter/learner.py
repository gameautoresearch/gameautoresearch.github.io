"""Small ordinary clipped PPO; no population, curriculum or shaping options."""
import numpy as np
import torch
from integrations.pokemon.network import batch

def update(model, optimizer, samples, config, device):
    if not samples:
        return {'samples':0, 'updates':0}
    x = batch([s['obs'] for s in samples], device)
    action = torch.tensor([s['action'] for s in samples], device=device)
    old = torch.tensor([s['logp'] for s in samples], device=device)
    target = torch.tensor([s['target'] for s in samples], device=device)
    advantage = torch.tensor([s['advantage'] for s in samples], device=device)
    advantage = (advantage-advantage.mean())/(advantage.std(unbiased=False)+1e-8)
    losses = []
    for _ in range(config['ppo_epochs']):
        for ix in torch.randperm(len(samples), device=device).split(config['minibatch']):
            dist, value = model.distribution({k:v[ix] for k,v in x.items()})
            ratio = (dist.log_prob(action[ix])-old[ix]).exp()
            surrogate = torch.minimum(ratio*advantage[ix], ratio.clamp(1-config['clip'],1+config['clip'])*advantage[ix])
            loss = -surrogate.mean()+config['value_coef']*.5*(value-target[ix]).square().mean()-config['entropy_coef']*dist.entropy().mean()
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite PPO loss')
            optimizer.zero_grad(set_to_none=True); loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])
            if not torch.isfinite(norm): raise FloatingPointError('Nonfinite gradient')
            optimizer.step(); losses.append(float(loss.detach().cpu()))
    return {'samples':len(samples), 'updates':len(losses), 'loss':float(np.mean(losses))}
