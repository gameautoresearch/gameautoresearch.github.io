"""Editable learner: REINFORCE with a learned value baseline and an entropy bonus, Adam."""
import torch
import torch.nn.functional as F


def make_optimizer(policy, config):
    return torch.optim.Adam(policy.parameters(), lr=config['learning_rate'])


def learn(policy, optimizer, batch, config):
    """One pass over a self-play batch in shuffled minibatches.

    batch: planes (N, 2, 13, 13), mask (N, 169) legal cells, action (N,) canonical cells,
    outcome (N,) +1 if the side to move went on to win, else -1.
    """
    n = len(batch['action'])
    order = torch.randperm(n, device=batch['action'].device)
    totals = {'policy_loss': 0., 'value_loss': 0., 'entropy': 0.}
    steps = 0
    for _ in range(config.get('epochs', 1)):
        for start in range(0, n, config['minibatch']):
            index = order[start:start + config['minibatch']]
            logits, value = policy(batch['planes'][index])
            logits = logits.masked_fill(~batch['mask'][index], -1e9)
            log_probs = F.log_softmax(logits, 1)
            chosen = log_probs.gather(1, batch['action'][index, None]).squeeze(1)
            outcome = batch['outcome'][index]
            advantage = (outcome - value).detach()
            policy_loss = -(advantage * chosen).mean()
            value_loss = F.mse_loss(value, outcome)
            entropy = -(log_probs.exp() * log_probs.clamp(min=-1e4)).sum(1).mean()
            loss = policy_loss + config['value_coef'] * value_loss - config['entropy_coef'] * entropy
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), config.get('max_grad_norm', 1.0))
            optimizer.step()
            totals['policy_loss'] += policy_loss.item()
            totals['value_loss'] += value_loss.item()
            totals['entropy'] += entropy.item()
            steps += 1
    return {k: v / max(steps, 1) for k, v in totals.items()} | {'updates': steps}
