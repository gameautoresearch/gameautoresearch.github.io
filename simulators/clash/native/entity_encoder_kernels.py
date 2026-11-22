"""Lazy-imported strict-FP32 Triton stages for entity_encoder.py.

No atomic operations. Linear-gradient reductions deliberately stay in cuBLAS;
this bounds prototype size and avoids one full parameter gradient per world.
"""
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _features(Raw,Cards,Buffs,Kind,Side,State,Scale,world,row,k,
              S0:tl.constexpr,S1:tl.constexpr,R:tl.constexpr,
              NC:tl.constexpr,NB:tl.constexpr,NT:tl.constexpr,
              ROW:tl.constexpr,KIND:tl.constexpr,SIDE:tl.constexpr,STATE:tl.constexpr,
              FLAGS:tl.constexpr,BUFFCOLS:tl.constexpr,NUMCOLS:tl.constexpr,alive):
    base=Raw+world*S0+row*S1
    card=tl.minimum(tl.maximum(tl.load(base+ROW,alive,other=-1).to(tl.int64)+1,0),NC-1)
    kind=tl.minimum(tl.maximum(tl.load(base+KIND,alive,other=0),0),7)
    side=tl.minimum(tl.maximum(tl.load(base+SIDE,alive,other=0),0),3)
    state=tl.minimum(tl.maximum(tl.load(base+STATE,alive,other=-1).to(tl.int64)+1,0),NT-1)
    x=tl.load(Cards+card[:,None]*12+k[None,:],alive[:,None]&(k[None,:]<12),other=0)
    x+=tl.load(Kind+kind[:,None]*4+(k[None,:]-12),alive[:,None]&(k[None,:]>=12)&(k[None,:]<16),other=0)
    x+=tl.load(Side+side[:,None]*3+(k[None,:]-16),alive[:,None]&(k[None,:]>=16)&(k[None,:]<19),other=0)
    x+=tl.load(State+state[:,None]*5+(k[None,:]-19),alive[:,None]&(k[None,:]>=19)&(k[None,:]<24),other=0)
    for slot in tl.static_range(3):
        buff=tl.minimum(tl.maximum(tl.load(base+BUFFCOLS[slot],alive,other=-1).to(tl.int64)+1,0),NB-1)
        x+=tl.load(Buffs+buff[:,None]*4+(k[None,:]-24-slot*4),alive[:,None]&(k[None,:]>=24+slot*4)&(k[None,:]<28+slot*4),other=0)
    col=tl.full((128,),0,tl.int32)
    for j in tl.static_range(24):col=tl.where(k==36+j,NUMCOLS[j],col)
    numeric=tl.load(base[:,None]+col[None,:],alive[:,None]&(k[None,:]>=36)&(k[None,:]<60),other=0).to(tl.float32)
    scale=tl.load(Scale+(k-36),(k>=36)&(k<60),other=1.)
    x+=tl.minimum(tl.maximum(tl.div_rn(numeric,scale[None,:]),-10.),10.)
    flags=tl.load(base+FLAGS,alive,other=0)
    shift=tl.minimum(tl.maximum(k-60,0),11)
    x+=tl.where((k[None,:]>=60)&(k[None,:]<72),((flags[:,None]&(1<<shift[None,:]))!=0).to(tl.float32),0.)
    return x


@triton.jit
def _stage(Raw,Count,Cards,Buffs,Kind,Side,State,W1,B1,W2,B2,Scale,
           Z,Gradient,X,A,D1,D2,DE,
           S0:tl.constexpr,S1:tl.constexpr,CS:tl.constexpr,R:tl.constexpr,
           NC:tl.constexpr,NB:tl.constexpr,NT:tl.constexpr,
           ROW:tl.constexpr,KIND:tl.constexpr,SIDE:tl.constexpr,STATE:tl.constexpr,
           FLAGS:tl.constexpr,BUFFCOLS:tl.constexpr,NUMCOLS:tl.constexpr,BACK:tl.constexpr):
    world=tl.program_id(0);tile=tl.program_id(1)
    rows=tile*16+tl.arange(0,16);k=tl.arange(0,128);h=tl.arange(0,64)
    count=tl.load(Count+world*CS);live=rows<count;exists=rows<R
    # Uniform control skips expensive tensor products for fully dead tiles.
    if tile*16<count:
        x=_features(Raw,Cards,Buffs,Kind,Side,State,Scale,world,rows,k,S0,S1,R,
            NC,NB,NT,ROW,KIND,SIDE,STATE,FLAGS,BUFFCOLS,NUMCOLS,live)
        w1=tl.load(W1+h[None,:]*72+k[:,None],k[:,None]<72,other=0.)
        a=libdevice.tanh(tl.dot(x,w1,input_precision="ieee")+tl.load(B1+h)[None,:])
        w2=tl.load(W2+h[None,:]*64+h[:,None])
        z=libdevice.tanh(tl.dot(a,w2,input_precision="ieee")+tl.load(B2+h)[None,:])
        z=tl.where(live[:,None],z,0.)
        if BACK:
            grad=tl.load(Gradient+(world*R+rows[:,None])*64+h[None,:],live[:,None],other=0.)
            d2=grad*(1.-z*z)
            w2normal=tl.load(W2+h[:,None]*64+h[None,:])
            d1=tl.dot(d2,w2normal,input_precision="ieee")*(1.-a*a)
            d1=tl.where(live[:,None],d1,0.)
            w1embed=tl.load(W1+h[:,None]*72+h[None,:],h[None,:]<36,other=0.)
            de=tl.dot(d1,w1embed,input_precision="ieee")
            x=tl.where(live[:,None],x,0.);a=tl.where(live[:,None],a,0.)
    else:
        z=tl.full((16,64),0.,tl.float32)
        if BACK:
            x=tl.full((16,128),0.,tl.float32);a=tl.full((16,64),0.,tl.float32)
            d1=tl.full((16,64),0.,tl.float32);d2=tl.full((16,64),0.,tl.float32);de=tl.full((16,64),0.,tl.float32)
    if BACK:
        tl.store(X+(world*R+rows[:,None])*72+k[None,:],x,exists[:,None]&(k[None,:]<72))
        tl.store(A+(world*R+rows[:,None])*64+h[None,:],a,exists[:,None])
        tl.store(D1+(world*R+rows[:,None])*64+h[None,:],d1,exists[:,None])
        tl.store(D2+(world*R+rows[:,None])*64+h[None,:],d2,exists[:,None])
        tl.store(DE+(world*R+rows[:,None])*36+h[None,:],de,exists[:,None]&(h[None,:]<36))
    else:tl.store(Z+(world*R+rows[:,None])*64+h[None,:],z,exists[:,None])


def _launch(raw,counts,p,spec,z,gradient,scratch,back):
    c=spec.columns
    _stage[(raw.shape[0],triton.cdiv(raw.shape[1],16))](raw,counts,*p,spec.scales,
        z,gradient,*scratch,raw.stride(0),raw.stride(1),counts.stride(0),raw.shape[1],
        spec.shapes[0][0],spec.shapes[1][0],spec.shapes[4][0],c['row'],c['kind'],c['side'],c['state'],c['flags'],
        tuple(spec.buff_columns),spec.numeric_columns,back,num_warps=4,enable_fp_fusion=False)


def forward(raw,counts,p,spec):
    z=torch.empty((*raw.shape[:2],64),device=raw.device,dtype=torch.float32)
    _launch(raw,counts,p,spec,z,z,(z,z,z,z,z),False)
    return z


def vjp(raw,counts,gradient,p,spec):
    scratch=tuple(torch.empty((*raw.shape[:2],n),device=raw.device,dtype=torch.float32) for n in (72,64,64,64,36))
    _launch(raw,counts,p,spec,gradient,gradient,scratch,True)
    return scratch


@triton.jit
def _embedding_partials(Raw,DE,Allowed,Partial,
                        S0:tl.constexpr,S1:tl.constexpr,R:tl.constexpr,M:tl.constexpr,
                        F:tl.constexpr,D:tl.constexpr,N:tl.constexpr,NAME:tl.constexpr,
                        COL:tl.constexpr,BUFFCOLS:tl.constexpr,OFFSET:tl.constexpr):
    rows=tl.program_id(0)*256+tl.arange(0,256)
    feature=tl.program_id(1)*32+tl.arange(0,32)
    valid=rows<M;livefeature=feature<F
    target=tl.load(Allowed+feature//D,livefeature,other=0)
    component=feature%D
    world=rows//R;local=rows%R
    base=Raw+world*S0+local*S1
    value=tl.full((256,32),0.,tl.float32)
    if NAME==1: # buffs: fixed slot order, then fixed row tree
        for slot in tl.static_range(3):
            encoded=tl.minimum(tl.maximum(tl.load(base+BUFFCOLS[slot],valid,other=-1).to(tl.int64)+1,0),N-1)
            derivative=tl.load(DE+rows[:,None]*36+24+slot*4+component[None,:],valid[:,None]&livefeature[None,:],other=0.)
            value+=tl.where((encoded[:,None]==target[None,:])&(target[None,:]!=0),derivative,0.)
    else:
        encoded=tl.load(base+COL,valid,other=0).to(tl.int64)
        if NAME==0 or NAME==4:encoded+=1
        encoded=tl.minimum(tl.maximum(encoded,0),N-1)
        derivative=tl.load(DE+rows[:,None]*36+OFFSET+component[None,:],valid[:,None]&livefeature[None,:],other=0.)
        same=encoded[:,None]==target[None,:]
        if NAME==0:same=same&(target[None,:]!=0)
        value=tl.where(same,derivative,0.)
    reduced=tl.sum(value,axis=0)
    tl.store(Partial+tl.program_id(0)*F+feature,reduced,livefeature)


@triton.jit
def _embedding_finish(Partial,Allowed,Result,F:tl.constexpr,D:tl.constexpr,G:tl.constexpr):
    feature=tl.program_id(0)*32+tl.arange(0,32);group=tl.arange(0,128)
    total=tl.full((32,),0.,tl.float32)
    # Ascending chunks with fixed tree within each: scheduling cannot change sum.
    for start in range(tl.cdiv(G,128)):
        row=start*128+group
        v=tl.load(Partial+row[:,None]*F+feature[None,:],(row[:,None]<G)&(feature[None,:]<F),other=0.)
        total+=tl.sum(v,axis=0)
    target=tl.load(Allowed+feature//D,feature<F,other=0)
    tl.store(Result+target*D+feature%D,total,feature<F)


def embedding_gradients(raw,de,spec):
    out={};m=raw.shape[0]*raw.shape[1];groups=triton.cdiv(m,256)
    columns=(spec.columns['row'],0,spec.columns['kind'],spec.columns['side'],spec.columns['state'])
    for index,(name,offset,shape) in enumerate(zip(('cards','buffs','kind','side','state'),(0,24,12,16,19),spec.shapes)):
        ids=spec.support_ids[name];n,d=shape;f=ids.numel()*d
        partial=torch.empty((groups,f),device=raw.device,dtype=torch.float32)
        result=torch.zeros(shape,device=raw.device,dtype=torch.float32)
        _embedding_partials[(groups,triton.cdiv(f,32))](raw,de,ids,partial,raw.stride(0),raw.stride(1),raw.shape[1],m,
            f,d,n,index,columns[index],tuple(spec.buff_columns),offset,num_warps=4,enable_fp_fusion=False)
        _embedding_finish[(triton.cdiv(f,32),)](partial,ids,result,f,d,groups,num_warps=4,enable_fp_fusion=False)
        out[name]=result
    return out
