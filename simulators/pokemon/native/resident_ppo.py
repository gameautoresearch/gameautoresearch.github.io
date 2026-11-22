"""Device collector/PPO using the existing policy and player decision boundaries."""
import copy
import hashlib
import json

import torch
from torch.distributions import Categorical

from ..efficient import unpack_tensor


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


class ResidentPPO:
    def __init__(self, env, model, optimizer, *, steps=32, minibatch=4096, epochs=4,faint_reward=.1,value_coef=1.):
        self.env,self.model,self.optimizer=env,model,optimizer
        self.steps,self.minibatch,self.epochs=steps,minibatch,epochs;self.faint_reward=faint_reward;self.value_coef=value_coef
        self.device=env.obs.device
        if next(model.parameters()).device!=self.device:raise ValueError('Model and native environment must share a device')
        env.allocate_rollout(steps)
        self.observations=torch.empty((steps,env.n,2,700),device=self.device)
        self.actions=torch.empty((steps,env.n,2),dtype=torch.long,device=self.device)
        self.values=torch.empty((steps,env.n,2),device=self.device)
        self.logp=torch.empty_like(self.values)
        self.ready=torch.empty((steps,env.n,2),dtype=torch.bool,device=self.device)
        self.updates=0
        self.training_signature=hashlib.sha256(json.dumps(dict(steps=steps,minibatch=minibatch,epochs=epochs,faint_reward=faint_reward,value_coef=value_coef,
            optimizer=optimizer.defaults,hidden=model.hidden,embedding=model.embedding_size),sort_keys=True,default=str).encode()).hexdigest()

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
            if env.flags[...,2].any().item():raise RuntimeError('Native simulator fault before PPO update')
            final_value=self.model(unpack_tensor(env.obs.reshape(-1,700)))[1].reshape(env.n,2)
            views,selected=env.truncated_views()
            boundary_values=torch.zeros_like(self.values)
            boundary_values.reshape(-1)[selected]=self.model(unpack_tensor(views))[1]
            terminal=env.flags[...,0].bool();boundary=terminal|env.flags[...,1].bool()
            advantage=decision_gae(env.rewards,self.values,self.ready,boundary,terminal,boundary_values,final_value)
            valid=self.ready.reshape(-1)
            observations=self.observations.reshape(-1,700)[valid]
            actions=self.actions.reshape(-1)[valid]
            old_logp=self.logp.reshape(-1)[valid]
            targets=(advantage+self.values).reshape(-1)[valid]
            advantage=advantage.reshape(-1)[valid]
            count=len(advantage)
            if not count:raise RuntimeError('No ready player decisions in rollout')
            advantage=(advantage-advantage.mean())/(advantage.std(unbiased=False)+1e-8)
        for _ in range(self.epochs):
            for index in torch.randperm(count,device=self.device).split(self.minibatch):
                logits,value=self.model(unpack_tensor(observations[index]))
                distribution=Categorical(logits=logits,validate_args=False)
                ratio=(distribution.log_prob(actions[index])-old_logp[index]).exp()
                policy=-torch.minimum(ratio*advantage[index],ratio.clamp(.8,1.2)*advantage[index]).mean()
                loss=policy+.5*self.value_coef*(value-targets[index]).square().mean()-.01*distribution.entropy().mean()
                self.optimizer.zero_grad(set_to_none=True);loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),.5);self.optimizer.step()
        self.updates+=1
        return dict(samples=count,natural=int(terminal.sum()),truncations=int(env.flags[...,1].sum()),faults=0)

    def checkpoint(self):
        return dict(model=copy.deepcopy(self.model.state_dict()),optimizer=copy.deepcopy(self.optimizer.state_dict()),
                    environment=self.env.snapshot(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(self.device) if self.device.type=='cuda' else None,
                    updates=self.updates,source_signature=self.env.source_signature,binary_sha256=self.env.binary_sha256,
                    rules_sha256=self.env.rules_sha256,backend=self.env.backend,initial_sha256=self.env.initial_sha256,
                    integration_signature=self.env.integration_signature,training_signature=self.training_signature)

    def restore(self, checkpoint):
        for name in ('source_signature','binary_sha256','rules_sha256','backend','initial_sha256','integration_signature'):
            if checkpoint[name]!=getattr(self.env,name):raise ValueError(f'Native training checkpoint {name} changed')
        if checkpoint['training_signature']!=self.training_signature:raise ValueError('Native training checkpoint configuration changed')
        self.model.load_state_dict(checkpoint['model']);self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.env.restore(checkpoint['environment']);self.updates=checkpoint['updates']
        torch.set_rng_state(checkpoint['torch_rng'].cpu())
        if self.device.type=='cuda':torch.cuda.set_rng_state(checkpoint['cuda_rng'].cpu(),self.device)
        elif checkpoint['cuda_rng'] is not None:raise ValueError('CUDA RNG state in a CPU-only checkpoint')
