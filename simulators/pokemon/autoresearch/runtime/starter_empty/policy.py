"""Placeholder policy: one linear layer over the 40 global features. Replace it with your own architecture.

`forward(x)` takes the seven-key observation dictionary and returns `(logits, value)`:
logits `[batch, 14]` (masked with `x["mask"]`) and value `[batch]`. The keys and shapes are
`tokens [12,12]`, `mons [12,24]`, `global [40]`, `global_tokens [4]`, `action_tokens [14,3]`,
`actions [14,12]`, `mask [14]` (see `simulator/simulators/pokemon/efficient.py` and `view.cjs`).
"""
from torch import nn


class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = nn.Linear(40, 14)
        self.value = nn.Linear(40, 1)

    def forward(self, x):
        features = x["global"]
        logits = self.logits(features)
        return logits.masked_fill(~x["mask"], -1e9), self.value(features).squeeze(-1)


def create(config):
    return Policy(**config)
