"""Position-aware masked MLP with card embeddings; visible inputs only."""
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
import taverngym as tg


def _ortho(layer, gain):
    if isinstance(layer, nn.Linear):
        nn.init.orthogonal_(layer.weight, gain)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)


class Policy(nn.Module):
    def __init__(self, hidden=128, embedding=8):
        super().__init__()
        self.arch = dict(hidden=hidden, embedding=embedding)
        self.cards = nn.Embedding(tg.N_CARDS, embedding, padding_idx=0)
        nn.init.normal_(self.cards.weight, std=0.05)
        with torch.no_grad():
            self.cards.weight[0].zero_()
        in_dim = tg.OBS_FLOATS + tg.OBS_CARDS * embedding
        self.body = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.actor = nn.Linear(hidden, tg.N_ACTIONS)
        self.critic = nn.Linear(hidden, 1)
        self.body.apply(lambda m: _ortho(m, np.sqrt(2.0)))
        _ortho(self.actor, 0.01)
        _ortho(self.critic, 1.0)

    @staticmethod
    def encode_numeric(features):
        return torch.sign(features) * torch.log1p(features.abs())

    def forward(self, features, cards, numeric_encoded=False):
        f = features if numeric_encoded else self.encode_numeric(features)
        emb = self.cards(cards.long()).flatten(1)
        h = self.body(torch.cat((f, emb), 1))
        return self.actor(h), self.critic(h).squeeze(-1)

    def distribution(self, features, cards, mask, numeric_encoded=False):
        logits, value = self(features, cards, numeric_encoded=numeric_encoded)
        return Categorical(logits=logits.masked_fill(~mask.bool(), -1e9), validate_args=not getattr(self,"fast_inference",False)), value


@torch.no_grad()
def act(policy, obs, indices, device, deterministic=False):
    if hasattr(policy, 'remote_act'):
        return policy.remote_act(obs, indices)
    f = torch.tensor(obs['features'][indices], device=device)
    c = torch.tensor(obs['cards'][indices], device=device, dtype=torch.long)
    m = torch.tensor(obs['action_mask'][indices], device=device)
    dist, value = policy.distribution(f, c, m)
    a = dist.logits.argmax(-1) if deterministic else dist.sample()
    return a.cpu().numpy(), dist.log_prob(a).cpu().numpy(), value.cpu().numpy()
