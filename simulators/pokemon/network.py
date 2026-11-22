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

def scripted(obs, rng, kind="power"):
    legal = np.flatnonzero(obs["mask"])
    if not len(legal):
        return None
    if kind == "random":
        return int(rng.choice(legal))
    # Public base power, type chart, accuracy and STAB; ignores hidden ability/items.
    moves = [a for a in legal if a < 8]
    if moves:
        scores = []
        for a in moves:
            f = obs["actions"][a]
            score = f[0] * f[1] * f[8] * (1+.5*f[9]) + .002*f[2] - .001*f[10]
            scores.append(score)
        best = max(scores)
        return int(rng.choice([a for a, s in zip(moves, scores) if abs(s-best) < 1e-9]))
    return int(max(legal, key=lambda a: obs["actions"][a][0]))


def scripted_batch(observations,rngs,kinds):
    """Vectorize public-power scores; keep each battle/seat's exact RNG stream."""
    features=np.asarray([o['actions'] for o in observations],dtype=np.float64)
    masks=np.asarray([o['mask'] for o in observations],dtype=bool)
    f=features[:,:8]
    scores=f[:,:,0]*f[:,:,1]*f[:,:,8]*(1+.5*f[:,:,9])+.002*f[:,:,2]-.001*f[:,:,10]
    move_legal=masks[:,:8]
    best=np.where(move_legal,scores,-np.inf).max(-1)
    tied=move_legal & (np.abs(scores-best[:,None])<1e-9)
    result=[]
    for i,(rng,kind) in enumerate(zip(rngs,kinds,strict=True)):
        legal=np.flatnonzero(masks[i])
        if not len(legal):result.append(None)
        elif kind=='random':result.append(int(rng.choice(legal)))
        elif move_legal[i].any():result.append(int(rng.choice(np.flatnonzero(tied[i]))))
        else:result.append(int(legal[np.argmax(features[i,legal,0])]))
    return result
