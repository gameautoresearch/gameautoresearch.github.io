"""DouZero-style three-role DMC action-value networks with factored history inference."""
from __future__ import annotations

from contextlib import nullcontext
from functools import wraps
import numpy as np
import torch
from torch import nn

from .environment import ROLES


def precision_context(policy):
    if policy.device.type=='cuda' and getattr(policy,'compute_dtype','float32')=='bfloat16':
        return torch.autocast('cuda',dtype=torch.bfloat16)
    return nullcontext()


def inference_precision(fn):
    @wraps(fn)
    def wrapped(self,*args,**kwargs):
        with precision_context(self):return fn(self,*args,**kwargs)
    return wrapped


class RoleNet(nn.Module):
    def __init__(self, role, hidden=512, depth=5, history=128):
        super().__init__()
        self.depth = depth
        self.lstm = nn.LSTM(162, history, batch_first=True)
        dims = [373 if role == 'landlord' else 484] + [hidden] * depth + [1]
        dims[0] += history
        for i in range(len(dims) - 1):
            setattr(self, f'dense{i + 1}', nn.Linear(dims[i], dims[i + 1]))

    def encode_history(self, z):
        return self.lstm(z)[0][:, -1, :]

    def head(self, h, x):
        x = torch.cat((h, x), dim=-1)
        for i in range(1, self.depth + 2):
            if i==self.depth+1 and x.device.type=='cuda' and torch.is_autocast_enabled('cuda'):
                # Keep the scalar Q head in FP32; avoid quantizing argmax tie breaks
                # to BF16 even when the expensive hidden matrices use tensor cores.
                with torch.autocast('cuda',enabled=False):
                    x=getattr(self,f'dense{i}')(x.float())
            else:
                x = getattr(self, f'dense{i}')(x)
            if i <= self.depth:
                x = torch.relu(x)
        return x.squeeze(-1)

    def forward(self, z, x):
        return self.head(self.encode_history(z), x)


class Policy(nn.Module):
    def __init__(self, architecture=None):
        super().__init__()
        self.architecture = architecture or {'hidden': 512, 'depth': 5, 'history': 128}
        self.roles = nn.ModuleDict({r: RoleNet(r, **self.architecture) for r in ROLES})
        self.compute_dtype='float32'

    def _apply(self, fn, recurse=True):
        # CUDA graphs bind parameter storage. Device/dtype migrations invalidate it.
        self.__dict__.pop('_graph_scorers',None)
        return super()._apply(fn,recurse=recurse)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        self.__dict__.pop('_graph_scorers',None)
        return super().load_state_dict(state_dict,strict=strict,assign=assign)

    @property
    def device(self):
        return next(self.parameters()).device

    @torch.inference_mode()
    @inference_precision
    def values_device(self,role,static,history,actions,owners,max_candidates=16384):
        """Resident public tensors only; ragged heads return device Q values."""
        net=self.roles[role]
        h=net.encode_history(history.float())
        static=static.float()
        values=[]
        for offset in range(0,len(owners),max_candidates):
            owner=owners[offset:offset+max_candidates]
            x=torch.cat((static[owner],actions[offset:offset+max_candidates].float()),dim=1)
            values.append(net.head(h[owner],x))
        return torch.cat(values)

    @torch.inference_mode()
    @inference_precision
    def values_arrays(self,role,static,history,actions,counts,max_candidates=4096):
        """Public-only bulk scoring; no environment pointer or hidden hand access."""
        z=torch.as_tensor(history,dtype=torch.float32,device=self.device)
        net=self.roles[role]
        h=net.encode_history(z)
        owners=np.repeat(np.arange(len(counts)),counts)
        values=[]
        for offset in range(0,len(actions),max_candidates):
            owner=owners[offset:offset+max_candidates]
            x=np.concatenate((static[owner],actions[offset:offset+max_candidates]),axis=1)
            xx=torch.as_tensor(x,dtype=torch.float32,device=self.device)
            oo=torch.as_tensor(owner,device=self.device)
            values.append(net.head(h[oo],xx).cpu().numpy())
        q=np.concatenate(values)
        if not np.isfinite(q).all():
            raise FloatingPointError('Non-finite policy prediction')
        return q

    @torch.inference_mode()
    @inference_precision
    def values_many(self, observations, max_candidates=4096, factored=True, cache_encoding=True,
                    encoding_backend='direct', inference_backend='eager'):
        """Return ragged Q arrays. No padding and no invalid-action logits."""
        result = [None] * len(observations)
        if inference_backend=='graph' and self.compute_dtype!='float32':
            raise ValueError('Autocast inference uses the eager/resident path; FP32 graphs bind separate weight storage')
        for role in ROLES:
            ids = [i for i, obs in enumerate(observations) if obs.role == role]
            if not ids:
                continue
            encoded = [observations[i].encode(cache_encoding, encoding_backend) for i in ids]
            counts = [len(e[2]) for e in encoded]
            static = np.stack([e[0] for e in encoded])
            z = torch.as_tensor(np.stack([e[1] for e in encoded]), dtype=torch.float32, device=self.device)
            actions = np.concatenate([e[2] for e in encoded])
            owners = np.repeat(np.arange(len(ids)), counts)
            net = self.roles[role]
            if inference_backend == 'graph':
                if not factored or self.device.type != 'cuda':
                    raise ValueError('Graph inference requires factored history on CUDA')
                from .graph import GraphScorer
                if not hasattr(self, '_graph_scorers'):
                    self._graph_scorers = {}
                if role not in self._graph_scorers:
                    self._graph_scorers[role] = GraphScorer(net)
                flat = self._graph_scorers[role].values(static, z, actions, owners, max_candidates)
                split = np.split(flat, np.cumsum(counts)[:-1])
                for i, values in zip(ids, split):
                    if not np.isfinite(values).all():
                        raise FloatingPointError('Non-finite policy prediction')
                    result[i] = values
                continue
            if inference_backend != 'eager':
                raise ValueError('Unknown inference backend')
            if factored:
                h = net.encode_history(z)
            values = []
            for offset in range(0, len(actions), max_candidates):
                end = offset + max_candidates
                owner = owners[offset:end]
                owner_tensor = torch.as_tensor(owner, device=self.device)
                x = np.concatenate((static[owner], actions[offset:end]), axis=1)
                x = torch.as_tensor(x, dtype=torch.float32, device=self.device)
                if factored:
                    v = net.head(h[owner_tensor], x)
                else:
                    v = net(z[owner_tensor], x)
                values.append(v.cpu().numpy())
            split = np.split(np.concatenate(values), np.cumsum(counts)[:-1])
            for i, values in zip(ids, split):
                if not np.isfinite(values).all():
                    raise FloatingPointError('Non-finite policy prediction')
                result[i] = values
        return result


def update(policy, optimizer, samples, batch_size, epochs, rng, max_grad_norm=40, shuffle_backend='numpy'):
    """Undiscounted selected-action terminal-return regression (Deep Monte Carlo)."""
    # cuDNN RNN backward requires train-mode reserve-space from its forward pass.
    # Do not rely on every caller remembering to undo inference/evaluation mode.
    was_training = policy.training
    policy.train()
    losses = []
    updates = {r: 0 for r in ROLES}
    try:
        for role in ROLES:
            data = samples[role]
            if not len(data):
                continue
            if hasattr(data,'x'):
                xx,zz,yy=data.x,data.z,data.y
            else:
                xx,zz,yy=np.stack([d[0] for d in data]),np.stack([d[1] for d in data]),[d[2] for d in data]
            x = torch.as_tensor(xx, dtype=torch.float32, device=policy.device)
            z = torch.as_tensor(zz, dtype=torch.float32, device=policy.device)
            y = torch.as_tensor(yy, dtype=torch.float32, device=policy.device)
            for _ in range(epochs):
                chunks=max(1,(len(data)+batch_size-1)//batch_size)
                if shuffle_backend=='torch':
                    permutation=torch.randperm(len(data),device=policy.device)
                    batches=torch.tensor_split(permutation,chunks)
                elif shuffle_backend=='numpy':
                    batches=np.array_split(rng.permutation(len(data)),chunks)
                else:
                    raise ValueError('Unknown learner shuffle protocol')
                for indices in batches:
                    indices = torch.as_tensor(indices, device=policy.device)
                    with precision_context(policy):
                        pred = policy.roles[role](z[indices], x[indices])
                        loss = (pred.float() - y[indices]).square().mean()
                    if not torch.isfinite(loss):
                        raise FloatingPointError('Non-finite learner loss')
                    optimizer[role].zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(policy.roles[role].parameters(), max_grad_norm, error_if_nonfinite=True)
                    optimizer[role].step()
                    losses.append(float(loss.detach()))
                    updates[role] += 1
        return {'mse': float(np.mean(losses)) if losses else None, 'updates': updates,
                'samples': {r: len(samples[r]) for r in ROLES}}
    finally:
        policy.train(was_training)
