"""Device collector/PPO using the existing policy and player decision boundaries."""
import copy
import hashlib
import json

import torch
from torch.distributions import Categorical

from integrations.pokemon.efficient import unpack_tensor


def decision_gae(rewards, values, ready, boundaries, terminal, truncated_bootstrap, final_value, gamma=.99, lam=.95):
    """Waiting phases accumulate reward into the preceding actual player action."""
    advantage=torch.empty_like(values)
    last_value=final_value.clone();last_adv=torch.zeros_like(final_value);accumulated=torch.zeros_like(final_value)
    for t in range(len(values)-1,-1,-1):
        boundary=boundaries[t,:,None]
        last_value=torch.where(boundary,torch.where(terminal[t,:,None],0,truncated_bootstrap[t]),last_value)
        last_adv=torch.where(boundary,0,last_adv)
        accumulated=torch.where(boundary,0,accumulated)+rewards[t]
        current=accumulated+gamma*last_value-values[t]+gamma*lam*last_adv
        advantage[t]=torch.where(ready[t],current,0)
        last_value=torch.where(ready[t],values[t],last_value)
        last_adv=torch.where(ready[t],current,last_adv)
        accumulated=torch.where(ready[t],0,accumulated)
    return advantage


class Trainer:
    def __init__(self, env, model, optimizer, config):
        self.env,self.model,self.optimizer=env,model,optimizer
        self.steps=config['rollout_steps'];self.minibatch=config['minibatch'];self.epochs=config['ppo_epochs']
        self.faint_reward=0;self.value_coef=config['value_coef'];self.config=config
        steps=self.steps
        self.device=env.obs.device
        if next(model.parameters()).device!=self.device:raise ValueError('Model and native environment must share a device')
        env.allocate_rollout(steps)
        self.observations=torch.empty((steps,env.n,2,700),device=self.device)
        self.actions=torch.empty((steps,env.n,2),dtype=torch.long,device=self.device)
        self.values=torch.empty((steps,env.n,2),device=self.device)
        self.logp=torch.empty_like(self.values)
        self.ready=torch.empty((steps,env.n,2),dtype=torch.bool,device=self.device)
        self.updates=0

    def collect_update(self):
        env=self.env
        with torch.no_grad():
            for t in range(self.steps):
                self.observations[t].copy_(env.obs)
                flat=env.obs.reshape(-1,700)
                logits,value=self.model(unpack_tensor(flat))
                ready=flat[:,686:].bool().any(-1)
                distribution=Categorical(logits=logits,validate_args=False)
                actions=distribution.sample()
                self.actions[t].copy_(actions.reshape(env.n,2))
                self.values[t].copy_(value.reshape(env.n,2))
                self.logp[t].copy_(distribution.log_prob(actions).reshape(env.n,2))
                self.ready[t].copy_(ready.reshape(env.n,2))
                env.step(torch.where(ready,actions,-1).reshape(env.n,2),t,faint_reward=self.faint_reward)
            # One batch-level fault check, including every pre-reset transition.
            if env.flags[...,2].any().item():
                codes,counts=torch.unique(env.flags[...,2],return_counts=True)
                typed={int(k):int(v)for k,v in zip(codes.cpu(),counts.cpu())if int(k)}
                raise RuntimeError('Native transition fault codes before PPO: '+json.dumps(typed))
            final_value=self.model(unpack_tensor(env.obs.reshape(-1,700)))[1].reshape(env.n,2)
            views,selected=env.truncated_views()
            boundary_values=torch.zeros_like(self.values)
            boundary_values.reshape(-1)[selected]=self.model(unpack_tensor(views))[1]
            terminal=env.flags[...,0].bool();boundary=terminal|env.flags[...,1].bool()
            advantage=decision_gae(env.rewards,self.values,self.ready,boundary,terminal,boundary_values,final_value,self.config['gamma'],self.config['gae_lambda'])
            valid=self.ready.reshape(-1)
            observations=self.observations.reshape(-1,700)[valid]
            actions=self.actions.reshape(-1)[valid]
            old_logp=self.logp.reshape(-1)[valid]
            targets=(advantage+self.values).reshape(-1)[valid]
            advantage=advantage.reshape(-1)[valid]
            count=len(advantage)
            if not count:raise RuntimeError('No ready player decisions in rollout')
            advantage=(advantage-advantage.mean())/(advantage.std(unbiased=False)+1e-8)
        optimizer_updates=0
        for _ in range(self.epochs):
            for index in torch.randperm(count,device=self.device).split(self.minibatch):
                logits,value=self.model(unpack_tensor(observations[index]))
                distribution=Categorical(logits=logits,validate_args=False)
                ratio=(distribution.log_prob(actions[index])-old_logp[index]).exp()
                policy=-torch.minimum(ratio*advantage[index],ratio.clamp(1-self.config['clip'],1+self.config['clip'])*advantage[index]).mean()
                loss=policy+.5*self.value_coef*(value-targets[index]).square().mean()-self.config['entropy_coef']*distribution.entropy().mean()
                self.optimizer.zero_grad(set_to_none=True);loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),self.config['max_grad_norm']);self.optimizer.step();optimizer_updates+=1
        self.updates+=1
        return dict(samples=count,optimizer_updates=optimizer_updates,natural=int(terminal.sum()),
            official_draws=int((terminal & (env.rewards==0).all(-1)).sum()),
            truncations=int(env.flags[...,1].sum()),faults=0)

