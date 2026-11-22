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
    # Equal class mass prevents abundant wait labels hiding deployment failures.
    bw=deploy.float()/deploy.sum().clamp_min(1)+wait.float()/wait.sum().clamp_min(1)
    bc=(-demo_dist.log_prob(flat['demo_teacher'][ix])*bw).sum()/bw.sum().clamp_min(1)
   else:bc=logits.new_zeros(())
   # Exploration regularizes the five meaningful macro decisions. Counting
   # hundreds of placement tiles in H(action) otherwise rewards deployment over
   # waiting simply because a card has many legal locations.
   tile_logits=logits[:,1:2305].reshape(-1,4,576)
   macro_logits=torch.cat((logits[:,:1],tile_logits.logsumexp(-1)),1)
   macro_entropy=torch.distributions.Categorical(logits=macro_logits).entropy().mean()
   value_error=(v-ret.flatten()[ix]).square().mean()
   loss=pol+c['value_coef']*value_error-c['entropy_coef']*macro_entropy+teacher_coef*bc
   if not torch.isfinite(loss):raise ValueError('nonfinite PPO loss')
   opt.zero_grad();loss.backward();gradient_norm=torch.nn.utils.clip_grad_norm_(model.parameters(),c['max_grad_norm']);opt.step()
   log_ratio=dist.log_prob(flat['action'][ix])-flat['logp'][ix]
   approximate_kl=((ratio-1)-log_ratio).mean();clip_fraction=((ratio-1).abs()>c['clip']).float().mean()
   rows.append(torch.stack((loss.detach(),pol.detach(),bc.detach(),value_error.detach(),macro_entropy.detach(),dist.entropy().mean().detach(),approximate_kl.detach(),clip_fraction.detach(),gradient_norm.detach())))
 avg=torch.stack(rows).mean(0).cpu().tolist()
 return dict(loss=avg[0],policy_loss=avg[1],teacher_nll=avg[2] if teacher_coef else None,value_mse=avg[3],macro_entropy=avg[4],joint_action_entropy_diagnostic=avg[5],approximate_kl=avg[6],clip_fraction=avg[7],gradient_norm_before_clipping=avg[8],teacher_coef=teacher_coef,optimizer_steps=len(rows))
