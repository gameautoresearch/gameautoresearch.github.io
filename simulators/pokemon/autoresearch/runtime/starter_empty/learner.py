"""Boilerplate learner: plays rollouts with the current policy and learns nothing.

There is no learning algorithm here. `collect_update` shows how to drive the environment and returns the
counts `train.py` records; the update that changes the model's weights is yours to write.

The environment (`env`, built by `train.py`) holds `env.n` battles, each with two seats played by this policy:
- `env.obs`: `[env.n, 2, 700]` float observations of the current step. The last 14 values are the legal-action
  mask; a seat whose mask is all zero is waiting and must be sent action -1.
- `env.allocate_rollout(steps)`: once, before stepping; `steps` is at most 256.
- `env.step(actions, t, faint_reward=0)`: `actions` is `[env.n, 2]` (action 0-13, or -1 for a waiting seat), `t`
  the index within the allocated rollout. `faint_reward` must stay 0. It fills, for index `t`:
  `env.rewards[t]` `[env.n, 2]` (+1/-1/0 at a natural end, otherwise 0) and `env.flags[t]` `[env.n, 3]`:
  column 0 natural end, column 1 cut off at the battle length limit, column 2 a nonzero fault code. A battle that
  ends is replaced by a new one, and `env.obs` already shows the next battle's first step.
- `env.truncated_views()`: after a rollout, `(views, selected)`: the `[count, 700]` observations at the cut-off
  steps (for bootstrapping a value there) and which flattened `[steps, env.n, 2]` entries they belong to.
`unpack_tensor(flat)` turns `[batch, 700]` into the seven-key dictionary the policy takes.
"""
import torch
from torch.distributions import Categorical

from integrations.pokemon.efficient import unpack_tensor

MASK = slice(686, 700)  # the last 14 observation values: the legal-action mask


class Trainer:
    def __init__(self, env, model, optimizer, config):
        self.env, self.model, self.optimizer, self.config = env, model, optimizer, config
        self.steps = config['rollout_steps']
        env.allocate_rollout(self.steps)

    def collect_update(self):
        env = self.env
        decisions = 0
        with torch.no_grad():
            for t in range(self.steps):
                flat = env.obs.reshape(-1, 700)
                ready = flat[:, MASK].bool().any(-1)
                logits, _ = self.model(unpack_tensor(flat))
                actions = Categorical(logits=logits, validate_args=False).sample()
                env.step(torch.where(ready, actions, -1).reshape(env.n, 2), t, faint_reward=0)
                decisions += int(ready.sum())
            if env.flags[..., 2].any().item():
                raise RuntimeError('Native transition fault flags in this rollout')
            terminal = env.flags[..., 0].bool()
        # TODO: learn from the rollout. Nothing here changes the model's weights.
        return dict(samples=decisions, optimizer_updates=0, natural=int(terminal.sum()),
                    official_draws=int((terminal & (env.rewards == 0).all(-1)).sum()),
                    truncations=int(env.flags[..., 1].sum()), faults=0)
