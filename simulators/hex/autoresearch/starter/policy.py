"""Editable policy. `create(model_config)` and `act(policy, board, to_play, legal)` are the evaluation API.

The network sees the board from the side to move (`canonical`): plane 0 holds its
stones, plane 1 the opponent's, and for white the board is transposed so the side
to move always joins top to bottom. It answers in that frame; `from_canonical`
maps the chosen cell back to the real board.
"""
import torch
from torch import nn

from simulators.hex.engine import CELLS, canonical, canonical_legal, from_canonical


class Net(nn.Module):
    def __init__(self, hidden=256, depth=2):
        super().__init__()
        layers, width = [], 2 * CELLS
        for _ in range(depth):
            layers += [nn.Linear(width, hidden), nn.ReLU()]
            width = hidden
        self.body = nn.Sequential(*layers)
        self.policy = nn.Linear(width, CELLS)
        self.value = nn.Linear(width, 1)

    def forward(self, planes):
        """(n, 2, 13, 13) canonical planes -> (move logits (n, 169), value in [-1, 1] (n,)) for the side to move."""
        h = self.body(planes.flatten(1))
        return self.policy(h), torch.tanh(self.value(h)).squeeze(1)


def create(config):
    return Net(config.get('hidden', 256), config.get('depth', 2))


def masked_logits(policy, board, to_play, legal):
    """Logits in the canonical frame with illegal cells at -inf, plus the canonical planes and values."""
    planes = canonical(board, to_play)
    logits, value = policy(planes)
    mask = canonical_legal(legal, to_play)
    return logits.masked_fill(~mask, float('-inf')), value


def act(policy, board, to_play, legal):
    """One legal cell (0..168, row-major on the real board) per position. Greedy: the most likely move."""
    board, to_play, legal = torch.as_tensor(board), torch.as_tensor(to_play), torch.as_tensor(legal)
    logits, _ = masked_logits(policy, board, to_play, legal)
    return from_canonical(logits.argmax(1), to_play).tolist()
