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
   bc=(-dist.log_prob(flat['teacher'][ix])*w).sum()/w.sum().clamp_min(1)
   loss=pol+c['value_coef']*(v-ret.flatten()[ix]).square().mean()-c['entropy_coef']*dist.entropy().mean()+teacher_coef*bc
   if not torch.isfinite(loss):raise ValueError('nonfinite PPO loss')
   opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),c['max_grad_norm']);opt.step()
   rows.append(torch.stack((loss.detach(),pol.detach(),bc.detach())))
 avg=torch.stack(rows).mean(0).cpu().tolist()
 return dict(loss=avg[0],policy_loss=avg[1],teacher_nll=avg[2],teacher_coef=teacher_coef,optimizer_steps=len(rows))
