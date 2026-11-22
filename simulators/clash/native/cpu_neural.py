"""Opt-in CPU-only use of the existing Policy's live-row/forced-head paths.

No model, learned tensor, optimizer, sampling rule, public observation or mask
is replaced. Policy.encode owns the ragged scatter/reduction (including first
max-index ties); Policy.action_terms owns point-mass forced-row terms and empty
head gradients. All PPO values, ratios and KL retain every minibatch row.

This module is not wired into Trainer. Dense FP32 remains its independent
control. Shape changes may alter floating-point rounding: qualify gradients,
Adam and exact resume IN the selected mode before production use. A local CPU
component benchmark is not an allocated-H100/full-PPO throughput comparison.
"""
from __future__ import annotations

import torch


def normalize_advantages(advantage, weight):
    """Original native Trainer normalization, including all-forced fallback."""
    norm = torch.where(weight.sum() > 0, weight, torch.ones_like(weight))
    count = norm.sum()
    mean = (advantage * norm).sum() / count
    std = torch.sqrt(((advantage - mean).square() * norm).sum() / count)
    return (advantage - mean) / (std + 1e-8)


def ppo_objective(logp, entropy, value, actions, old_logp, advantage, returns, weight,
                  *, clip=.2, value_coef=.5, entropy_coef=.001):
    """Identical full-N PPO arithmetic; policy weight does NOT filter KL/value."""
    n = actions.numel()
    if n == 0 or any(tensor.ndim != 1 or tensor.numel() != n for tensor in
                     (logp, entropy, value, actions, old_logp, advantage, returns, weight)):
        raise ValueError('PPO terms must be matching nonempty full-minibatch vectors')
    ratio = (logp - old_logp).exp()
    denominator = weight.sum().clamp_min(1)
    policy = -(torch.minimum(ratio * advantage, ratio.clamp(1 - clip, 1 + clip) * advantage) * weight).sum() / denominator
    value_loss = (value - returns).square().mean()
    ent = (entropy * weight).sum() / denominator
    loss = policy + value_coef * value_loss - entropy_coef * ent
    kl = ((ratio - 1) - (logp - old_logp)).mean()
    return loss, {'policy_loss': policy, 'value_loss': value_loss, 'entropy': ent, 'kl': kl,
                  'ratio': ratio, 'logp': logp, 'values': value, 'policy_denominator': denominator,
                  'value_rows': n, 'kl_rows': n}


class CpuNeuralAdapter:
    """Borrow one CPU FP32 Policy without changing parameter/state-dict identity.

    CPU global-max entity trimming precedes Policy's int64 conversion; it keeps
    at least one row and all 200 whenever needed. The original raw tensor/store
    is untouched. All actor calls retain full-N categorical sampling/RNG; only
    learner action_terms avoids materializing policy heads for forced waits.
    """

    def __init__(self, policy, *, trim_entities=True, guard_finite=True):
        if type(trim_entities) is not bool or type(guard_finite) is not bool:
            raise ValueError('CPU adapter flags must be boolean')
        if not callable(getattr(policy, 'action_terms', None)):
            raise ValueError('CPU adapter requires the original Policy action_terms API')
        self.policy = policy
        self.trim_entities, self.guard_finite = trim_entities, guard_finite
        self.parameter_ids = tuple(id(parameter) for parameter in policy.parameters())
        self.state_keys = tuple(policy.state_dict())
        self._closed = False
        self._old_flags = {key: (key in policy.__dict__, getattr(policy, key, None))
                           for key in ('ragged_entities', 'sparse_policy_head')}
        self.validate_parameters()
        policy.ragged_entities = True
        policy.sparse_policy_head = True

    def validate_parameters(self):
        parameters = tuple(self.policy.parameters())
        if tuple(map(id, parameters)) != self.parameter_ids or tuple(self.policy.state_dict()) != self.state_keys:
            raise RuntimeError('CPU adapter parameter/state-dict identity changed')
        if any(parameter.device.type != 'cpu' or parameter.dtype != torch.float32 for parameter in parameters):
            raise ValueError('CPU adapter requires existing CPU FP32 parameters')
        if self.guard_finite:
            # Check every call: skipping a dead row/head must not hide corrupt
            # weights, including changes through .data that bypass versioning.
            for name, value in self.policy.named_parameters():
                if not torch.isfinite(value).all():
                    raise ValueError('nonfinite CPU policy parameter: ' + name)

    def prepare(self, batch):
        if self._closed:
            raise RuntimeError('CPU neural adapter is closed')
        header, entities = batch['header'], batch['entities']
        if header.device.type != 'cpu' or entities.device.type != 'cpu':
            raise ValueError('CPU neural adapter cannot move or accept CUDA inputs')
        if header.ndim != 2 or entities.ndim != 3 or header.shape[0] != entities.shape[0] or not len(header):
            raise ValueError('CPU neural batch has invalid header/entity shapes')
        counts = header[:, self.policy.s.header['entity_count'][0]]
        if not ((counts >= 0) & (counts <= self.policy.s.max_entities)).all():
            raise ValueError('CPU entity count is outside the complete public schema')
        needed = max(1, int(counts.max()))
        if entities.shape[1] < needed or entities.shape[1] > self.policy.s.max_entities:
            raise ValueError('CPU batch drops live entities or exceeds the public schema')
        if 'mask' in batch:
            mask = batch['mask']
            if mask.device.type != 'cpu' or mask.dtype != torch.bool or mask.shape != (len(header), self.policy.s.actions):
                raise ValueError('CPU mask must be full-size boolean public input')
            if not mask.any(-1).all():
                raise ValueError('CPU neural batch contains an empty legal mask')
        self.validate_parameters()
        return dict(batch, entities=entities[:, :needed]) if self.trim_entities else batch

    def distribution(self, batch):
        return self.policy.distribution(self.prepare(batch))

    def values(self, batch):
        return self.policy.values(self.prepare(batch))

    def sample_batch(self, batch, greedy=False):
        # Original full-N Categorical.sample is preserved, including forced rows.
        return self.policy.sample_batch(self.prepare(batch), greedy=greedy)

    def action_terms(self, batch, actions):
        if actions.device.type != 'cpu' or actions.ndim != 1 or actions.numel() != len(batch['header']):
            raise ValueError('CPU actions must retain the complete minibatch')
        if actions.dtype not in (torch.int32, torch.int64) or not ((actions >= 0) & (actions < self.policy.s.actions)).all():
            raise ValueError('CPU action index is outside the public schema')
        if not batch['mask'].gather(1, actions.long()[:, None]).all():
            raise ValueError('CPU selected action is closed by its public legal mask')
        return self.policy.action_terms(self.prepare(batch), actions)

    def ppo_loss(self, batch, actions, old_logp, advantage, returns, weight, **config):
        logp, entropy, value = self.action_terms(batch, actions)
        return ppo_objective(logp, entropy, value, actions, old_logp, advantage, returns, weight, **config)

    def metadata(self):
        return {'mode': 'cpu-existing-ragged-selected-action', 'trim_entities': self.trim_entities,
                'guard_finite': self.guard_finite, 'max_entities': self.policy.s.max_entities,
                'all_parameters_and_state_dict_preserved': True, 'actor_sampling': 'original full-N Policy.sample_batch',
                'learner_terms': 'original Policy.action_terms; full-N values/ratios/KL',
                'production_trainer_integrated': False}

    def close(self):
        if not self._closed:
            for key, (present, value) in self._old_flags.items():
                if present:
                    setattr(self.policy, key, value)
                else:
                    delattr(self.policy, key)
            self._closed = True

    def __enter__(self): return self
    def __exit__(self, *args): self.close()
