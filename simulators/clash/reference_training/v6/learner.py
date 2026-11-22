"""Masked PPO; teacher loss is explicitly separated in metrics."""
import torch

def gae(r,v,d,b,last,gamma,lam):
 carry=torch.zeros_like(last);adv=torch.empty_like(r)
 for t in reversed(range(len(r))):
  nxt=last if t==len(r)-1 else v[t+1];nxt=torch.where(d[t],b[t],nxt)
  carry=r[t]+gamma*nxt-v[t]+gamma*lam*(~d[t])*carry;adv[t]=carry
 return adv,adv+v

def update(model,opt,s,last,c,teacher_coef):
 adv,ret=gae(s['reward'],s['value'],s['done'],s['bootstrap'],last,c['gamma'],c['gae_lambda'])
 flat={k:v.flatten(0,1) for k,v in s.items()};a=adv.flatten();active=flat['mask'].sum(-1)>1
 norm=a[active] if active.any() else a;a=(a-norm.mean())/(norm.std(unbiased=False)+1e-6)
 rows=[]
 for _ in range(c['epochs']):
  for ix in torch.randperm(len(a),device=a.device).split(c['minibatch']):
   logits,v=model(flat['obs'][ix],flat['mask'][ix]);dist=torch.distributions.Categorical(logits=logits)
   ratio=(dist.log_prob(flat['action'][ix])-flat['logp'][ix]).exp();w=active[ix].float()
   pol=(-torch.minimum(ratio*a[ix],ratio.clamp(1-c['clip'],1+c['clip'])*a[ix])*w).sum()/w.sum().clamp_min(1)
   if teacher_coef:
    demo_logits,_=model(flat['demo_obs'][ix],flat['demo_mask'][ix]);demo_dist=torch.distributions.Categorical(logits=demo_logits)
    demo_active=flat['demo_mask'][ix].sum(-1)>1;deploy=(flat['demo_teacher'][ix]>0)&demo_active;wait=(~deploy)&demo_active
    # Calibrate when to deploy to the real demonstration prior. Learn card and
    # placement conditionally on deployment, without inventing a 50% play prior.
    tile=demo_logits[:,1:2305].reshape(-1,4,576);marginal=tile.logsumexp(-1)
    gate=marginal.logsumexp(-1)-demo_logits[:,0]
    weight=demo_active.float();dw=deploy.float();denom=dw.sum().clamp_min(1)
    gate_loss=(torch.nn.functional.binary_cross_entropy_with_logits(gate,deploy.float(),reduction='none')*weight).sum()/weight.sum().clamp_min(1)
    target=flat['demo_teacher'][ix];slot=((target-1).clamp_min(0)//576).long();position=((target-1).clamp_min(0)%576).long()
    card_loss=(torch.nn.functional.cross_entropy(marginal,slot,reduction='none')*dw).sum()/denom
    selected=tile[torch.arange(len(tile),device=tile.device),slot]
    tile_loss=(torch.nn.functional.cross_entropy(selected,position,reduction='none')*dw).sum()/denom
    bc=gate_loss+.5*(card_loss+tile_loss)
   else:bc=logits.new_zeros(())
   loss=pol+c['value_coef']*(v-ret.flatten()[ix]).square().mean()-c['entropy_coef']*dist.entropy().mean()+teacher_coef*bc
   if not torch.isfinite(loss):raise ValueError('nonfinite PPO loss')
   opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),c['max_grad_norm']);opt.step()
   rows.append(torch.stack((loss.detach(),pol.detach(),bc.detach())))
 avg=torch.stack(rows).mean(0).cpu().tolist()
 return dict(loss=avg[0],policy_loss=avg[1],teacher_nll=avg[2] if teacher_coef else None,teacher_coef=teacher_coef,optimizer_steps=len(rows))
