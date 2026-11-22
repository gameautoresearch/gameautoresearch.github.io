"""Shared categorical embeddings; per-mon and per-action encoders; masked PPO."""
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

KEYS = ("tokens", "mons", "global", "global_tokens", "action_tokens", "actions", "mask")

def batch(observations, device):
    return {k: torch.as_tensor(np.asarray([o[k] for o in observations]),
                              dtype=torch.long if k in ("tokens", "global_tokens", "action_tokens") else
                              torch.bool if k == "mask" else torch.float32, device=device) for k in KEYS}

class Policy(nn.Module):
    def __init__(self, hidden=128, embedding=16):
        super().__init__()
        self.hidden, self.embedding_size = hidden, embedding
        self.embedding = nn.Embedding(16384, embedding, padding_idx=0)
        self.mon = nn.Sequential(nn.Linear(12*embedding+24, hidden), nn.Tanh())
        self.trunk = nn.Sequential(nn.Linear(hidden*12+40+4*embedding, hidden), nn.Tanh(),
                                   nn.Linear(hidden, hidden), nn.Tanh())
        self.action = nn.Sequential(nn.Linear(3*embedding+12, hidden), nn.Tanh())
        self.logits = nn.Linear(hidden, 14)
        self.value = nn.Linear(hidden, 1)
        self.action_bias = nn.Linear(hidden, 1)
        nn.init.orthogonal_(self.logits.weight, gain=.01)
        nn.init.zeros_(self.logits.bias)
    def forward(self, x):
        m = self.mon(torch.cat((self.embedding(x["tokens"]).flatten(2), x["mons"]), dim=-1)).flatten(1)
        z = self.trunk(torch.cat((m, x["global"], self.embedding(x["global_tokens"]).flatten(1)), dim=-1))
        a = self.action(torch.cat((self.embedding(x["action_tokens"]).flatten(2), x["actions"]), dim=-1))
        logits = self.logits(z) + (a*z[:, None, :]).sum(-1) / self.hidden**.5 + self.action_bias(a).squeeze(-1)
        return logits.masked_fill(~x["mask"], -1e9), self.value(z).squeeze(-1)
    def distribution(self, x):
        logits, value = self(x)
        if not x["mask"].any(-1).all():
            raise ValueError("Policy loss/sampling requested on a waiting or terminal seat")
        return Categorical(logits=logits), value

def create(config):
    return Policy(**config)
