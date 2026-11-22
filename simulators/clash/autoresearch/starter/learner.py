"""Plain clipped PPO and GAE, with terminal/truncation boundaries kept separate."""
import torch
def gae(reward,value,done,bootstrap,last,gamma,lam):
 carry=torch.zeros_like(last);adv=torch.empty_like(reward)
 for t in reversed(range(len(reward))):
  nxt=last if t==len(reward)-1 else value[t+1]
  nxt=torch.where(done[t],bootstrap[t],nxt)
  carry=reward[t]+gamma*nxt-value[t]+gamma*lam*(~done[t])*carry;adv[t]=carry
 return adv,adv+value
def eager_reference_update(model,optimizer,store,last,c):
 advantage,target=gae(store['reward'],store['value'],store['done'],store['bootstrap'],last,c['gamma'],c['gae_lambda'])
 a=advantage.flatten();a=(a-a.mean())/(a.std(unbiased=False)+1e-8)
 flat={k:v.flatten(0,1) for k,v in store.items()};metrics=[]
 for epoch in range(c['epochs']):
  for idx in torch.randperm(len(a),device=a.device).split(c['minibatch']):
   logits,value=model(flat['obs'][idx],flat['mask'][idx]);dist=torch.distributions.Categorical(logits=logits)
   ratio=(dist.log_prob(flat['action'][idx])-flat['logp'][idx]).exp()
   policy=-torch.minimum(ratio*a[idx],ratio.clamp(1-c['clip'],1+c['clip'])*a[idx]).mean()
   loss=policy+c['value_coef']*(value-target.flatten()[idx]).square().mean()-c['entropy_coef']*dist.entropy().mean()
   if not torch.isfinite(loss):raise ValueError('nonfinite PPO loss')
   optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),c['max_grad_norm']);optimizer.step()
   metrics.append(float(loss.detach()))
 return {'loss':sum(metrics)/len(metrics),'optimizer_steps':len(metrics)}


def update(trainer,data):
 """Default uses qualified graph/bucket PPO; replace here for learner research."""
 from simulators.clash.native.train import Trainer
 return Trainer.learn(trainer,data)
