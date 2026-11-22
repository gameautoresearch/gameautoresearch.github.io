"""Editable baseline: three-role Deep Monte Carlo regression with RMSprop."""
import torch
from integrations.doudizhu.model import update
from integrations.doudizhu.environment import ROLES

def make_optimizer(policy,config):
    return {r:torch.optim.RMSprop(policy.roles[r].parameters(),lr=config['learning_rate'],alpha=.99,eps=1e-5) for r in ROLES}

def learn(policy,optimizer,samples,rng,config):
    return update(policy,optimizer,samples,config['batch_size'],1,rng,shuffle_backend='torch')
