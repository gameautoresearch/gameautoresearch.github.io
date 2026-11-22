"""Plain episodic PPO; no reference recipes, shaping or privileged critic."""
import numpy as np
import torch
from rollout import gae

def update(model,optimizer,episodes,c,device):
 rows=[];adv=[];ret=[]
 for ep in episodes:
  a,r=gae(ep,c['gamma'],c['gae_lambda']);rows.extend(ep);adv.extend(a);ret.extend(r)
 if len(rows)<2:raise ValueError('No complete natural trajectories')
 f=torch.tensor(np.stack([r['features'] for r in rows]),device=device)
 cards=torch.tensor(np.stack([r['cards'] for r in rows]),device=device,dtype=torch.long)
 masks=torch.tensor(np.stack([r['mask'] for r in rows]),device=device)
 actions=torch.tensor([r['action'] for r in rows],device=device)
 old=torch.tensor([r['logp'] for r in rows],device=device)
 adv=torch.tensor(np.array(adv),device=device);ret=torch.tensor(np.array(ret),device=device)
 adv=(adv-adv.mean())/(adv.std(unbiased=False)+1e-8)
 before=torch.cat([p.detach().flatten() for p in model.parameters()]).clone();losses=[]
 for _ in range(c['ppo_epochs']):
  for ids in torch.randperm(len(rows),device=device).split(c['minibatch_size']):
   dist,v=model.distribution(f[ids],cards[ids],masks[ids]);ratio=(dist.log_prob(actions[ids])-old[ids]).exp()
   loss=-torch.minimum(ratio*adv[ids],ratio.clamp(1-c['clip'],1+c['clip'])*adv[ids]).mean()+c['value_coef']*((v-ret[ids])**2).mean()-c['entropy_coef']*dist.entropy().mean()
   if not torch.isfinite(loss):raise ValueError('Nonfinite PPO loss')
   optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),c['max_grad_norm']);optimizer.step();losses.append(float(loss.detach()))
 delta=float((torch.cat([p.detach().flatten() for p in model.parameters()])-before).norm())
 return dict(loss=float(np.mean(losses)),samples=len(rows),parameter_delta_l2=delta)
