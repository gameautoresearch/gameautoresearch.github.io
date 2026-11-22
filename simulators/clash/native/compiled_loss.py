"""Optional fullgraph Inductor PPO loss on the SAME Policy/encoder parameters.

This module only computes the loss and detached metric components. The caller
keeps finite-loss checking, zero_grad, backward, gradient clipping and the existing
Adam step, in that order. No actor, sampler, simulator, optimizer or model is
replaced; model.state_dict() names and parameter objects remain unchanged.

Integration after separate CUDA/resume qualification::

    loss, (pg, vl, entropy, kl) = compiled_ppo_loss(trainer, fields)
    torch._assert_async(torch.isfinite(loss), 'nonfinite native PPO loss')
    trainer.optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad = torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), max_norm)
    trainer.optimizer.step()

Warm the FULL forward/backward outside any outer CUDA capture. Inductor internal
CUDA graphs and timing-based autotuning are explicitly disabled. Compilation,
CUDA capture compatibility, GPU math and exact resume within this optional mode
still require their own source-bound device qualification.
"""
from __future__ import annotations
from collections.abc import Mapping
from types import MappingProxyType
import argparse,json,sys,time
from pathlib import Path

INPUT_KEYS=('obs','mask','action','logp','advantage','return','weight')
METRIC_NAMES=('policy_loss','value_loss','entropy','approx_kl')
_OPTIONS=MappingProxyType({
    'triton.cudagraphs':False,
    'max_autotune':False,
    'max_autotune_gemm':False,
    'max_autotune_pointwise':False,
    'coordinate_descent_tuning':False,
    'triton.autotune_pointwise':False,
    'triton.autotune_cublasLt':False,
})
LOSS_KEYS=('clip','value_coef','entropy_coef')

def compile_recipe():
    """Serializable source-bound compiler recipe; no environment-dependent mode."""
    return {'backend':'inductor','mode':'default','fullgraph':True,'dynamic':False,'options':dict(_OPTIONS),'internal_cuda_graphs':False,'timing_based_autotuning':False}


def _body(model,encoder,clip,value_coef,entropy_coef,obs,mask,action,old,adv,target,weight):
    """Exactly the existing learner's Policy/Categorical/PPO forward expression."""
    import torch
    batch=encoder(obs,mask)
    distribution,value=model.distribution(batch)
    new=distribution.log_prob(action)
    ratio=(new-old).exp();w=weight;a=adv
    pg=-(torch.minimum(ratio*a,ratio.clamp(1-clip,1+clip)*a)*w).sum()/w.sum().clamp_min(1)
    vl=(value-target).square().mean()
    entropy=(distribution.entropy()*w).sum()/w.sum().clamp_min(1)
    loss=pg+value_coef*vl-entropy_coef*entropy
    with torch.no_grad():kl=((ratio-1)-(new-old)).mean()
    return loss,(pg.detach(),vl.detach(),entropy.detach(),kl)


def _validate(trainer,fields):
    import torch
    from gpu_features import DeviceBatchEncoder
    if not isinstance(fields,Mapping) or set(fields)!=set(INPUT_KEYS):raise ValueError('compiled loss needs exactly '+','.join(INPUT_KEYS))
    if not isinstance(trainer.encoder,DeviceBatchEncoder):raise TypeError('compiled loss requires the existing static DeviceBatchEncoder')
    if getattr(trainer.model,'ragged_entities',False) or getattr(trainer.model,'sparse_policy_head',False):raise ValueError('compiled fullgraph loss requires the existing static dense policy')
    if not torch.are_deterministic_algorithms_enabled() or torch.is_deterministic_algorithms_warn_only_enabled():raise ValueError('compiled PPO prototype requires strict deterministic algorithms')
    if not torch.is_grad_enabled():raise ValueError('compiled PPO loss must be evaluated with gradients enabled')
    parameters=tuple(trainer.model.parameters())
    if not parameters:raise ValueError('policy has no parameters')
    device=parameters[0].device
    if any(p.device!=device for p in parameters):raise ValueError('policy parameters must share one device')
    action=fields['action']
    if not isinstance(action,torch.Tensor) or action.ndim!=1 or not action.numel():raise ValueError('nonempty one-dimensional action tensor required')
    size=action.shape[0]
    for name in INPUT_KEYS:
        value=fields[name]
        if not isinstance(value,torch.Tensor) or value.device!=device:raise ValueError(name+' must be on the model device')
        if name in ('obs','mask'):
            if value.ndim!=2 or value.shape[0]!=size:raise ValueError(name+' minibatch shape differs')
        elif value.shape!=(size,):raise ValueError(name+' must have one entry per sample')
    if fields['obs'].dtype!=torch.int32 or fields['mask'].dtype not in (torch.uint8,torch.bool) or action.dtype!=torch.int64:raise ValueError('raw observation/mask/action dtype differs from the existing learner')
    for name in ('logp','advantage','return','weight'):
        if fields[name].dtype!=torch.float32:raise ValueError(name+' must retain float32 dtype')
    layout=trainer.encoder.layout
    if fields['obs'].shape[1]!=layout[0] or fields['mask'].shape[1]!=layout[4]:raise ValueError('complete public observation/mask widths must be retained')
    return parameters,device


def _key(trainer,fields,parameters,device):
    import torch
    encoder=trainer.encoder
    autocast=torch.is_autocast_enabled(device.type)
    return (id(trainer.model),tuple((id(p),tuple(p.shape),tuple(p.stride()),p.dtype,p.device) for p in parameters),
            (type(encoder),tuple(encoder.layout),encoder.count_index,encoder.copy),
            tuple((name,tuple(fields[name].shape),tuple(fields[name].stride()),fields[name].dtype,fields[name].device) for name in INPUT_KEYS),
            tuple((name,trainer.c[name]) for name in LOSS_KEYS),
            tuple((name,getattr(trainer.model,name,False)) for name in ('device_indices','ragged_entities','sparse_policy_head')),
            bool(trainer.model.training),autocast,torch.get_autocast_dtype(device.type) if autocast else None)


class _Entry:
    def __init__(self,trainer):
        import torch
        # Function compilation closes over existing objects, rather than wrapping
        # or assigning model=torch.compile(model), so checkpoint names do not move.
        self.model=trainer.model;self.encoder=trainer.encoder
        self.constants=tuple(trainer.c[name] for name in LOSS_KEYS)
        def forward(obs,mask,action,old,adv,target,weight):
            return _body(self.model,self.encoder,*self.constants,obs,mask,action,old,adv,target,weight)
        self.function=torch.compile(forward,backend='inductor',fullgraph=True,dynamic=False,options=dict(_OPTIONS))
        self.prepared=False
    def __call__(self,fields):
        result=self.function(*(fields[name] for name in INPUT_KEYS))
        self.prepared=True
        return result


def compiled_ppo_loss(trainer,fields):
    """Return (loss,(pg,vl,entropy,kl)); use trainer-owned static-shape cache.

    No input is copied, mutated, read on the CPU, normalized or resampled here.
    Metrics are detached scalar outputs. The loss keeps the ordinary autograd
    connection to the SAME parameters used by the caller's existing optimizer.
    """
    import torch
    parameters,device=_validate(trainer,fields)
    key=_key(trainer,fields,parameters,device)
    if not hasattr(trainer,'_compiled_loss_cache'):trainer._compiled_loss_cache={}
    cache=trainer._compiled_loss_cache
    capturing=device.type=='cuda' and torch.cuda.is_current_stream_capturing()
    if key not in cache:
        if capturing:raise RuntimeError('warm compiled PPO forward/backward outside CUDA capture before using this static bucket/shape')
        cache[key]=_Entry(trainer)
    entry=cache[key]
    if capturing and not entry.prepared:raise RuntimeError('compiled PPO loss shape has not completed an uncaptured forward warmup')
    return entry(fields)


def cache_summary(trainer):
    """Host metadata only; useful for receipts without touching sample contents."""
    rows=[]
    for key,entry in getattr(trainer,'_compiled_loss_cache',{}).items():
        rows.append({'entity_rows':entry.encoder.entity_rows,'input_layouts':[{'name':name,'shape':list(shape),'stride':list(stride),'dtype':str(dtype),'device':str(device)} for name,shape,stride,dtype,device in key[3]],'prepared':entry.prepared})
    return {'recipe':compile_recipe(),'entries':rows}


def clear_cache(trainer):
    """Release Python wrapper references; does not mutate any model/Adam state."""
    if hasattr(trainer,'_compiled_loss_cache'):trainer._compiled_loss_cache.clear()


def cpu_qualification(out,*,fixture=None,compact=False,integrated=False,atol=2e-6,rtol=2e-5):
    """Actual Inductor CPU semantic/gradient/Adam control on public fixtures.

    All observations/masks remain genuine native/Java-qualified public frames.
    Synthetic advantage/return and old-policy offsets test clipping, but do not
    alter simulator transitions. This is not a throughput or GPU qualification.
    """
    import copy,hashlib,random,numpy as np,torch
    from types import SimpleNamespace
    here=Path(__file__).resolve().parent;sys.path.insert(0,str(here));sys.path.insert(0,str(here.parent))
    from train import schema
    from model import Policy
    from gpu_features import DeviceBatchEncoder
    from learner_graph import eager_minibatch
    out=Path(out);out.mkdir(parents=True,exist_ok=False);started=time.perf_counter()
    if fixture is None:fixture=here.parents[2]/'evidence/clash-implementation/native-training-20261108/compact-embedding-cpu-3/real-observations.npz'
    fixture=Path(fixture);records=np.load(fixture);obs=records['observations'].reshape(-1,7024);mask=records['masks'].reshape(-1,2307)
    data=json.loads((here/'generated-v2/task-data.json').read_text());s=schema(data)
    active=np.flatnonzero(mask.sum(1)>1);forced=np.flatnonzero(mask.sum(1)==1);small_active=active[obs[active,48]<=32];small_forced=forced[obs[forced,48]<=32]
    if len(active)<1 or len(forced)<4 or len(small_active)<1:raise ValueError('fixture lacks required real forced/active public frames')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.manual_seed(81108)
    reference=Policy(s,len(data['vocabulary']),len(data['buff_names']),width=16,embedding=4)
    torch.nn.init.normal_(reference.slot.weight,std=.01)
    if compact:
        from compact_embeddings import install
        install(reference,data)
    reference.device_indices=True;compiled=copy.deepcopy(reference)
    c={'clip':.2,'value_coef':.5,'entropy_coef':.001,'max_grad_norm':.5}
    def trainer(model):return SimpleNamespace(model=model,encoder=DeviceBatchEncoder(s,copy=False),c=dict(c),optimizer=torch.optim.Adam(model.parameters(),lr=.0003,eps=1e-5,fused=False))
    a,b=trainer(reference),trainer(compiled);b.c['compile_loss']=integrated;names=tuple(compiled.state_dict());ids=tuple(id(p) for p in compiled.parameters());cases=[]
    def choose(values,n):return values[np.arange(n)%len(values)]
    for rows,size,kind in ((200,8,'mixed'),(200,8,'forced'),(200,4,'mixed-tail'),(64,8,'mixed'),(32,8,'mixed'),(96,8,'mixed')):
        aa,ff=(small_active,small_forced) if rows==32 else (active,forced)
        if kind=='forced':indices=choose(ff,size)
        else:indices=np.concatenate((choose(aa,size//2),choose(ff,size-size//2)))
        a.encoder=DeviceBatchEncoder(s,entity_rows=rows,copy=False);b.encoder=DeviceBatchEncoder(s,entity_rows=rows,copy=False)
        raw=torch.from_numpy(obs[indices].copy());legal=torch.from_numpy(mask[indices].copy())
        action=torch.tensor([int(np.flatnonzero(row)[-1]) for row in mask[indices]],dtype=torch.int64)
        weight=(legal.sum(-1)>1).float()
        with torch.no_grad():old=a.model.distribution(a.encoder(raw,legal))[0].log_prob(action)
        old=old+torch.linspace(-.45,.45,size)*weight
        fields={'obs':raw,'mask':legal,'action':action,'logp':old,'advantage':torch.linspace(-1.2,1.2,size),'return':torch.linspace(.2,.8,size),'weight':weight}
        before={name:value.clone() for name,value in fields.items()}
        expected=eager_minibatch(a,fields)
        rng_cpu=torch.get_rng_state().clone();rng_python=random.getstate();rng_numpy=np.random.get_state()
        loss,parts=compiled_ppo_loss(b,fields)
        expected_loss=expected[0]+c['value_coef']*expected[1]-c['entropy_coef']*expected[2]
        torch.testing.assert_close(loss.detach(),expected_loss,atol=atol,rtol=rtol)
        torch._assert_async(torch.isfinite(loss),'nonfinite compiled PPO loss')
        if integrated:actual=eager_minibatch(b,fields)
        else:
            b.optimizer.zero_grad(set_to_none=True);loss.backward();grad=torch.nn.utils.clip_grad_norm_(b.model.parameters(),c['max_grad_norm']);b.optimizer.step()
            actual=torch.stack((*parts,grad.detach()))
        assert torch.equal(torch.get_rng_state(),rng_cpu) and random.getstate()==rng_python,'compiled loss consumed research RNG'
        after_numpy=np.random.get_state();assert after_numpy[0]==rng_numpy[0] and np.array_equal(after_numpy[1],rng_numpy[1]) and after_numpy[2:]==rng_numpy[2:],'compiled loss consumed NumPy RNG'
        torch.testing.assert_close(actual,expected,atol=atol,rtol=rtol)
        max_gradient=max(float((x.grad-y.grad).abs().max()) for x,y in zip(a.model.parameters(),b.model.parameters()))
        max_parameter=max(float((x-y).detach().abs().max()) for x,y in zip(a.model.parameters(),b.model.parameters()))
        for x,y in zip(a.model.parameters(),b.model.parameters()):
            torch.testing.assert_close(x,y,atol=atol,rtol=rtol);torch.testing.assert_close(x.grad,y.grad,atol=atol,rtol=rtol)
            for name,value in a.optimizer.state[x].items():torch.testing.assert_close(b.optimizer.state[y][name],value,atol=atol,rtol=rtol)
        for name,value in fields.items():assert torch.equal(value,before[name]),'compiled loss mutated input '+name
        assert tuple(compiled.state_dict())==names and tuple(id(p) for p in compiled.parameters())==ids
        cases.append({'entity_rows':rows,'minibatch':size,'kind':kind,'real_source_indices':indices.tolist(),'maximum_entity_count':int(raw[:,48].max()),'active_rows':int(weight.sum()),'loss_difference':float((loss.detach()-expected_loss).abs()),'maximum_metric_difference':float((actual-expected).abs().max()),'maximum_clipped_gradient_difference':max_gradient,'maximum_post_adam_parameter_difference':max_parameter})
    summary=cache_summary(b);assert len(summary['entries'])==5,'forced/active contents should share one static layout entry'
    result={'status':'passed','device':'cpu','actual_inductor_executed':True,'fullgraph':True,'dynamic':False,'compact_embedding_grad':compact,'optional_minibatch_integration_exercised':integrated,'atol':atol,'rtol':rtol,'cases':cases,'cache':summary,'same_parameter_objects_and_state_dict_names':True,'model_gradients_adam_and_input_immutability_checked':True,'torch_python_numpy_rng_unchanged':True,'fixture':str(fixture),'fixture_sha256':hashlib.sha256(fixture.read_bytes()).hexdigest(),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'torch_version':torch.__version__,'setup_compile_qualification_seconds':time.perf_counter()-started,'timing_comparison':False,'cuda_executed':False,'cuda_capture_and_exact_resume_qualification':'pending actual source-bound GPU allocation'}
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True);return result

if __name__=='__main__':
    here=Path(__file__).resolve().parent;sys.path.insert(0,str(here));sys.path.insert(0,str(here.parent))
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--cpu-check',action='store_true');parser.add_argument('--compact-embedding-grad',action='store_true');parser.add_argument('--integrated',action='store_true');parser.add_argument('--fixture',type=Path);parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    if not args.cpu_check:parser.error('use --cpu-check; this module never launches GPU/cloud work')
    cpu_qualification(args.out,fixture=args.fixture,compact=args.compact_embedding_grad,integrated=args.integrated)
