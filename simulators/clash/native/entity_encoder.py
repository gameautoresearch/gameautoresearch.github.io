"""Experimental fixed8 live entity encoder; never installed by default.

Triton computes live-row features/MLP and recomputes its local VJP. Parameter
reductions use deterministic FP32 cuBLAS GEMMs (no floating scatter atomics).
Dense alive mean/max remains Policy's original reduction, including first ties.
This staging deliberately retains dense backward scratch; it is a candidate to
measure, not a claim that fusion or recomputation saves end-to-end time.
"""
from __future__ import annotations
import os, types
import torch
import torch.nn.functional as F

PARAMETERS=('cards.weight','buffs.weight','kind.weight','side.weight','state.weight',
            'entity_net.0.weight','entity_net.0.bias','entity_net.2.weight','entity_net.2.bias')


def fixed8_support(model,data):
    reachable=set(data['cards'])|set(data['units'])|set(data['projectiles'])|set(data['areas'])
    return {'cards':sorted({0,1}|{i+1 for i,name in enumerate(data['vocabulary']) if name in reachable}),
            'buffs':sorted({0,1,data['buff_names'].index('ZapFreeze')+1}),
            'kind':list(range(model.kind.num_embeddings)), 'side':list(range(model.side.num_embeddings)),
            'state':list(range(model.state.num_embeddings))}


def _indices(e,spec):
    # Widen BEFORE adding one: int32 max must clamp, never wrap to missing ID0.
    cols=spec.columns
    return {'cards':(e[...,cols['row']].long()+1).clamp(0,spec.shapes[0][0]-1),
            'buffs':(e[...,spec.buff_columns].long()+1).clamp(0,spec.shapes[1][0]-1),
            'kind':e[...,cols['kind']].clamp(0,7),'side':e[...,cols['side']].clamp(0,3),
            'state':(e[...,cols['state']].long()+1).clamp(0,spec.shapes[4][0]-1)}


def _features(raw,parameters,spec):
    ids=_indices(raw,spec);c,b,k,s,t=parameters[:5]
    return torch.cat((F.embedding(ids['cards'],c),F.embedding(ids['kind'],k),
        F.embedding(ids['side'],s),F.embedding(ids['state'],t),
        F.embedding(ids['buffs'],b).flatten(-2),
        (raw[...,spec.numeric_columns].float()/spec.scales).clamp(-10,10),
        ((raw[...,spec.columns['flags'],None].long()&spec.flag_bits)!=0).float()),-1)


class _Entity(torch.autograd.Function):
    @staticmethod
    def forward(ctx,raw,counts,spec,*parameters):
        spec.validate(raw,counts,parameters)
        ctx.spec=spec;ctx.save_for_backward(raw,counts,*parameters)
        if spec.backend=='torch-reference':
            x=_features(raw,parameters,spec)
            a1=F.linear(x,parameters[5],parameters[6]).tanh()
            z=F.linear(a1,parameters[7],parameters[8]).tanh()
            alive=torch.arange(raw.shape[1],device=raw.device)[None,:]<counts[:,None]
            return z.masked_fill(~alive[...,None],0.)
        from entity_encoder_kernels import forward
        return forward(raw,counts,parameters,spec)

    @staticmethod
    def backward(ctx,gradient):
        raw,counts,*p=ctx.saved_tensors;spec=ctx.spec
        if torch.is_grad_enabled():raise RuntimeError('experimental entity encoder supports first-order training gradients only')
        spec.validate_runtime(raw)
        gradient=gradient.contiguous()
        # The public operation is a pooling-only Z producer. Original pooling
        # gives exactly zero inactive gradients; fail closed on another consumer.
        alive=torch.arange(raw.shape[1],device=raw.device)[None,:]<counts[:,None]
        torch._assert_async(torch.isfinite(gradient).all(),'nonfinite entity upstream gradient')
        torch._assert_async((torch.where(alive[...,None],0.,gradient)==0).all(),
                            'entity encoder Z must be consumed by alive pooling')
        if spec.backend=='torch-reference':
            x=_features(raw,p,spec);a1=F.linear(x,p[5],p[6]).tanh()
            a2=F.linear(a1,p[7],p[8]).tanh();dz2=gradient*(1-a2*a2)
            dz1=(dz2@p[7])*(1-a1*a1);de=dz1@p[5][:,:36]
        else:
            from entity_encoder_kernels import vjp
            x,a1,dz1,dz2,de=vjp(raw,counts,gradient,p,spec)
        m=raw.shape[0]*raw.shape[1]
        x=x.reshape(m,72);a1=a1.reshape(m,64);dz1=dz1.reshape(m,64);dz2=dz2.reshape(m,64);de=de.reshape(m,36)
        # No per-world giant parameter partial arrays. cuBLAS GEMMs have fixed
        # shapes/order within a source/runtime-bound recipe; TF32 is forbidden.
        w1=dz1.T@x; b1=dz1.sum(0);w2=dz2.T@a1;b2=dz2.sum(0)
        if spec.backend=='triton':
            from entity_encoder_kernels import embedding_gradients
            grads=embedding_gradients(raw,de,spec)
        else:
            ids=_indices(raw,spec);grads={};offsets={'cards':(0,12),'kind':(12,16),'side':(16,19),'state':(19,24)}
            for name in ('cards','buffs','kind','side','state'):
                allowed=spec.support_ids[name];mapping=spec.maps[name]
                idx=ids[name].reshape(m,-1)
                result=torch.zeros(spec.shapes[('cards','buffs','kind','side','state').index(name)],device=raw.device,dtype=torch.float32)
                reduced=torch.zeros((len(allowed),result.shape[1]),device=raw.device,dtype=torch.float32)
                for slot in range(idx.shape[1]):
                    compact=mapping[idx[:,slot]]
                    hot=F.one_hot(compact,num_classes=len(allowed)).float()
                    if name in ('cards','buffs'):hot[:,spec.padding[name]]=0.
                    lo,hi=(24+slot*4,28+slot*4) if name=='buffs' else offsets[name]
                    reduced=reduced+hot.T@de[:,lo:hi]
                result.index_copy_(0,allowed,reduced);grads[name]=result
        return (None,None,None,grads['cards'],grads['buffs'],grads['kind'],grads['side'],grads['state'],w1,b1,w2,b2)


class LiveEntityEncoder:
    """External adapter: owns no Parameters and adds no state_dict names.

    Only Policy64/e12, 3 buffs, 24 numeric and 12 flag features are accepted.
    All selected rows are validated, including inactive padding. Nonfinite or
    potentially overflowing parameters are rejected, not hidden by live skips.
    The torch-reference backend is correctness-only, never a CPU speed baseline.
    """
    def __init__(self,model,data,*,backend='triton',math_mode='fp32_ieee'):
        if backend not in ('triton','torch-reference'):raise ValueError('unknown entity backend')
        if math_mode!='fp32_ieee':raise ValueError('only separately qualified strict fp32_ieee is implemented')
        if (model.width,model.embedding,model.s.buff_slots,len(model.entity_num_idx),len(model.s.flag_names),model.entity_net[0].in_features)!=(64,12,3,24,12,72):raise ValueError('requires frozen Policy64/e12 layout')
        self.model=model;self.backend=backend;self.math_mode=math_mode;self.columns=dict(model.s.entity)
        self.numeric_columns=tuple(model.entity_num_idx);self.buff_columns=list(model.buff_idx)
        self.scales=model.entity_scale;self.flag_bits=model.flag_bits
        self.shapes=[tuple(getattr(model,n).weight.shape) for n in ('cards','buffs','kind','side','state')]
        support=fixed8_support(model,data);device=model.cards.weight.device
        self.support_ids={n:torch.tensor(v,device=device,dtype=torch.long) for n,v in support.items()}
        self.maps={};self.padding={n:support[n].index(0) for n in ('cards','buffs')}
        for name,shape in zip(('cards','buffs','kind','side','state'),self.shapes):
            table=torch.full((shape[0],),-1,device=device,dtype=torch.long)
            table[self.support_ids[name]]=torch.arange(len(support[name]),device=device);self.maps[name]=table
        self.metadata={'experimental':True,'backend':backend,'math_mode':math_mode,
            'backward':'Triton recompute VJP + deterministic FP32 linear GEMMs; ordered Triton support embedding reductions',
            'pool':'unmodified dense alive mean and first-index max',
            'nonfinite':'reject all nonfinite and conservative overflow-unsafe encoder parameters',
            'scratch_floats_per_selected_row':72+64+64+64+36,'support':support,
            'gradient_order':1,
            'qualification_required':'actual GPU output/gradient/Adam/graph/resume; no throughput claim'}

    def validate(self,raw,counts,p):
        if raw.dtype!=torch.int32 or counts.dtype not in (torch.int32,torch.int64):raise TypeError('raw entity rows/counts must be integer public tensors')
        if raw.ndim!=3 or raw.shape[2]!=34 or not 1<=raw.shape[1]<=200 or counts.shape!=(raw.shape[0],):raise ValueError('invalid public row/count shape')
        if raw.stride(-1)!=1 or raw.stride(-2)!=34:raise ValueError('requires contiguous 34-int entity rows')
        if any(q.dtype!=torch.float32 or q.device!=raw.device or not q.is_contiguous() for q in p):raise ValueError('contiguous FP32 parameters on input device required')
        if counts.device!=raw.device:raise ValueError('count device differs')
        if self.backend=='triton' and not raw.is_cuda:raise ValueError('Triton entity encoder requires CUDA')
        self.validate_runtime(raw)
        torch._assert_async(((counts>=0)&(counts<=raw.shape[1])).all(),'entity count outside selected all-row capacity')
        torch._assert_async(torch.stack([torch.isfinite(q).all() for q in p]).all(),'nonfinite encoder parameter')
        # Numeric/flags bounded by 10/1. Embedding magnitude bounds both GEMMs.
        emb=torch.stack([q.abs().max().double() for q in p[:5]]).max().clamp_min(10.)
        bound1=72*emb*p[5].abs().max().double()+p[6].abs().max().double()
        bound2=64*p[7].abs().max().double()+p[8].abs().max().double()
        torch._assert_async((torch.maximum(bound1,bound2)<torch.finfo(torch.float32).max/2),'overflow-unsafe encoder parameters')
        for name,indices in _indices(raw,self).items():
            torch._assert_async((self.maps[name][indices]>=0).all(),'embedding ID outside frozen fixed8 support')

    def validate_runtime(self,raw):
        if raw.is_cuda:
            if torch.is_autocast_enabled('cuda') or torch.backends.cuda.matmul.allow_tf32 or torch.get_float32_matmul_precision()!='highest':raise ValueError('strict entity candidate requires autocast off and TF32 disabled')
            if not torch.are_deterministic_algorithms_enabled() or os.environ.get('CUBLAS_WORKSPACE_CONFIG') not in (':4096:8',':16:8'):raise ValueError('deterministic algorithms and cuBLAS workspace config required before CUDA initialization')

    def uninstall(self):
        self.model.encode=self.original_encode;delattr(self.model,'_live_entity_adapter')

    def __call__(self,raw,counts):
        named=dict(self.model.named_parameters());return _Entity.apply(raw,counts,self,*[named[n] for n in PARAMETERS])


def _encoded(this,batch):
    adapter=this._live_entity_adapter
    h=batch['header'].long();e=batch['entities']
    counts=h[:,this.s.header['entity_count'][0]]
    alive=torch.arange(e.shape[1],device=e.device)[None,:]<counts[:,None]
    ci=this.card_indices if getattr(this,'device_indices',False) else this.card_idx
    gi=this.global_indices if getattr(this,'device_indices',False) else this.global_idx
    card=this.cards((h[:,ci]+1).clamp(0,this.cards.num_embeddings-1)).flatten(1)
    g=(h[:,gi].float()/this.global_scale).clamp(-10,10)
    z=adapter(e,counts)
    mean=(z*alive[:,:,None]).sum(1)/alive.sum(1).clamp_min(1)[:,None]
    maximum=z.masked_fill(~alive[:,:,None],-1e4).max(1).values
    maximum=torch.where(alive.any(1)[:,None],maximum,torch.zeros_like(maximum))
    return this.global_net(torch.cat((card,g,mean,maximum),-1)),h


def install(model,data,*,backend='triton',math_mode='fp32_ieee'):
    """Opt in before creating graphs/compiled losses; adds no Parameters."""
    if hasattr(model,'_live_entity_adapter'):raise ValueError('entity encoder already installed')
    if getattr(model,'ragged_entities',False):raise ValueError('disable original ragged encoder before installing candidate')
    for name in ('_inference_graphs','_action_graphs','_compiled_loss_cache'):
        if getattr(model,name,None):raise ValueError('install encoder before creating captured/compiled model paths')
    adapter=LiveEntityEncoder(model,data,backend=backend,math_mode=math_mode)
    adapter.original_encode=model.encode
    model._live_entity_adapter=adapter;model.encode=types.MethodType(_encoded,model)
    return adapter
