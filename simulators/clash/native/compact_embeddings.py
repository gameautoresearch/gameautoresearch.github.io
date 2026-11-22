"""Deterministic small-support embedding gradient reduction for the frozen task.

Forward lookup and the full parameter/checkpoint vocabulary stay unchanged.
Only backward reduction is replaced: a fixed-order GEMM over the task's reachable
IDs avoids sorting large padded entity batches. Reduction roundoff can differ
from torch embedding backward, so parity tolerance is explicitly qualified.
"""
import torch
from torch import nn

class _Embedding(torch.autograd.Function):
    @staticmethod
    def forward(ctx,weight,indices,compact,allowed,padding):
        ctx.save_for_backward(compact,allowed);ctx.shape=weight.shape;ctx.padding=padding
        return torch.nn.functional.embedding(indices,weight)
    @staticmethod
    def backward(ctx,gradient):
        compact,allowed=ctx.saved_tensors
        hot=torch.nn.functional.one_hot(compact.reshape(-1),num_classes=allowed.numel()).to(gradient.dtype)
        if ctx.padding>=0:hot[:,ctx.padding]=0
        reduced=hot.T@gradient.reshape(-1,gradient.shape[-1]);result=gradient.new_zeros(ctx.shape)
        result.index_copy_(0,allowed,reduced)
        return result,None,None,None,None

class CompactEmbedding(nn.Module):
    def __init__(self,original,allowed):
        super().__init__();self.num_embeddings=original.num_embeddings;self.embedding_dim=original.embedding_dim;self.weight=original.weight
        allowed=sorted(set(allowed));mapping=torch.full((self.num_embeddings,),-1,device=self.weight.device,dtype=torch.int64);ids=torch.tensor(allowed,device=self.weight.device,dtype=torch.int64);mapping[ids]=torch.arange(len(allowed),device=self.weight.device)
        self.register_buffer('support_mapping',mapping,persistent=False);self.register_buffer('support_ids',ids,persistent=False)
        self.compact_padding=allowed.index(original.padding_idx) if original.padding_idx is not None else -1
    def forward(self,indices):
        compact=self.support_mapping[indices]
        torch._assert_async((compact>=0).all(),'embedding ID outside frozen fixed8 support')
        if not torch.is_grad_enabled():return torch.nn.functional.embedding(indices,self.weight)
        return _Embedding.apply(self.weight,indices,compact,self.support_ids,self.compact_padding)

def install(model,data):
    reachable=set(data['cards'])|set(data['units'])|set(data['projectiles'])|set(data['areas'])
    # Dense observation padding is all-zero, so Policy.encode performs an ID1
    # lookup before applying the entity alive mask. ID0 is the explicit -1
    # missing-value lookup. Retain both even though unused rows contribute zero.
    cards=[0,1]+[i+1 for i,name in enumerate(data['vocabulary']) if name in reachable]
    # This task has stun as its only buff; preserve the exact exported vocabulary index.
    # Derive the actual buff spelling from the exported scaled configs as well.
    from data_binding import validate_data
    validate_data()
    import re
    from pathlib import Path
    header=(Path(__file__).parent/'generated-v2/config.hpp').read_text()
    match=re.search(r's\.stun\.id\s*=\s*(-?\d+)',header)
    if match is None:raise ValueError('missing frozen stun ID')
    buffs=[0,1,int(match.group(1))+1]
    supports={'cards':cards,'buffs':buffs,'kind':range(model.kind.num_embeddings),'side':range(model.side.num_embeddings),'state':range(model.state.num_embeddings)}
    for name,allowed in supports.items():setattr(model,name,CompactEmbedding(getattr(model,name),allowed))
    return {name:list(getattr(model,name).support_ids.cpu().numpy()) for name in supports}
