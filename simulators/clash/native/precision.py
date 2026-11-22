"""Explicit OPTIONAL neural precision modes; frozen integer simulation is untouched.

Configure before actor/learner graph capture. FP32 remains the default. TF32 is
an explicit CUDA matmul policy; BF16 autocasts ONLY Policy.forward/values, with
FP32 master parameters and autocast weight caching DISABLED. Loss, backward,
clipping, Adam, GAE, sampling and all simulator operations stay outside wrappers.

Cross-mode numeric/throughput results are not equal-strength claims. Checkpoints
must bind requested/actual precision metadata and qualify exact resume IN mode.
"""
from __future__ import annotations
import copy,random,sys,types

MODES=('fp32','tf32','bf16')
# Numerical qualification bounds, not guarantees about policy quality. Reporting
# includes absolute/L2 errors; the caller can request tighter bounds explicitly.
TOLERANCES={
    'fp32':dict(output_atol=0.,output_rtol=0.,gradient_atol=0.,gradient_rtol=0.,gradient_relative_l2=0.),
    'tf32':dict(output_atol=.003,output_rtol=.003,gradient_atol=.0001,gradient_rtol=.03,gradient_relative_l2=.02),
    'bf16':dict(output_atol=.1,output_rtol=.02,gradient_atol=.001,gradient_rtol=.15,gradient_relative_l2=.1),
}


def _autocast_forward(self,*args,**kwargs):
    import torch
    with torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False):
        logits,value=self._precision_original_forward(*args,**kwargs)
    return logits.float(),value.float()


def _autocast_values(self,*args,**kwargs):
    import torch
    with torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False):
        value=self._precision_original_values(*args,**kwargs)
    return value.float()


def matmul_runtime_flags():
    import torch
    result={}
    for name in ('allow_tf32','allow_fp16_reduced_precision_reduction','allow_bf16_reduced_precision_reduction','allow_fp16_accumulation'):
        try:result[name]=bool(getattr(torch.backends.cuda.matmul,name))
        except AttributeError:result[name]=None
    return result


def configure_precision(model,mode='fp32',device=None):
    """Configure one fixed8 Policy without replacing ANY parameter or buffer.

    Repeating the same mode is idempotent. Changing mode requires fresh captured
    graphs; known Policy graph caches cause an explicit rejection of hot changes.
    External graph owners must likewise be rebuilt by the caller.
    """
    import torch
    if mode not in MODES:raise ValueError('precision mode must be fp32, tf32 or bf16')
    parameters=tuple(model.parameters())
    if not parameters:raise ValueError('precision configuration requires model parameters')
    actual_device=parameters[0].device
    requested=torch.device(device) if device is not None else actual_device
    if requested.type!=actual_device.type or requested.index is not None and requested.index!=actual_device.index:
        raise ValueError('requested precision device differs from model parameter device')
    if any(p.device!=actual_device or p.dtype!=torch.float32 for p in parameters):raise ValueError('precision modes preserve a single-device FP32 master parameter set')
    if not callable(getattr(model,'values',None)):raise ValueError('precision model must expose forward and values')
    if actual_device.type!='cuda' and mode!='fp32':raise ValueError('non-fp32 precision requires an actual CUDA model; CPU modes stay fp32')
    if actual_device.type not in ('cpu','cuda'):raise ValueError('precision qualification supports CPU or CUDA devices')
    old=getattr(model,'_precision_mode','fp32')
    if mode!=old and any(bool(getattr(model,name,{})) for name in ('_action_graphs','_inference_graphs')):
        raise ValueError('precision mode changed after graph capture; recreate model graph owners')
    supported=None
    if actual_device.type=='cuda':
        if not torch.cuda.is_available():raise ValueError('actual CUDA device required')
        with torch.cuda.device(actual_device):supported=bool(torch.cuda.is_bf16_supported(including_emulation=False))
        if mode=='bf16' and not supported:raise ValueError('bf16 mode requires native GPU bfloat16 support')
    # Validation is complete BEFORE mutating global policy or module attributes.
    torch.backends.cuda.matmul.allow_tf32=(mode=='tf32')
    actual_tf32=bool(torch.backends.cuda.matmul.allow_tf32)
    if actual_tf32!=(mode=='tf32'):raise ValueError('actual CUDA matmul TF32 policy differs from requested precision')
    if not hasattr(model,'_precision_original_forward'):
        model._precision_original_forward=model.forward
        model._precision_original_values=model.values
    if mode=='bf16':
        if old!='bf16' or getattr(model.forward,'__func__',None) is not _autocast_forward:
            model.forward=types.MethodType(_autocast_forward,model)
            model.values=types.MethodType(_autocast_values,model)
    else:
        model.forward=model._precision_original_forward
        model.values=model._precision_original_values
    model._precision_mode=mode
    metadata={'matmul_runtime_flags':matmul_runtime_flags(),'mode':mode,'device':str(actual_device),'cuda_matmul_allow_tf32':actual_tf32,
              'native_bf16_supported':supported,'parameter_dtype':'float32','output_dtype':'float32',
              'autocast_dtype':'bfloat16' if mode=='bf16' else None,
              'autocast_cache_enabled':False if mode=='bf16' else None,
              'scope':'forward_and_values_only' if mode=='bf16' else 'explicit_cuda_matmul_policy',
              'configure_before_graph_capture':True}
    model._precision_metadata=metadata.copy()
    return metadata


def _rng():
    import numpy as np,torch
    return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state().clone(),
            'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def _restore_rng(state):
    import numpy as np,torch
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch'])
    if state['cuda'] is not None:torch.cuda.set_rng_state_all(state['cuda'])


def _exact(actual,expected,label):
    import numpy as np,torch
    if isinstance(actual,torch.Tensor):assert isinstance(expected,torch.Tensor) and actual.dtype==expected.dtype and actual.shape==expected.shape and torch.equal(actual,expected),label
    elif isinstance(actual,np.ndarray):np.testing.assert_array_equal(actual,expected,err_msg=label)
    elif isinstance(actual,dict):
        assert actual.keys()==expected.keys(),label+' keys'
        for key in actual:_exact(actual[key],expected[key],label+'.'+str(key))
    elif isinstance(actual,(tuple,list)):
        assert type(actual)==type(expected) and len(actual)==len(expected),label
        for index,(a,b) in enumerate(zip(actual,expected)):_exact(a,b,label+'.'+str(index))
    else:assert actual==expected,(label,actual,expected)


def _objective(model,batch,actions=None):
    import torch
    from torch.distributions import Categorical
    logits,value=model(batch)
    assert logits.dtype==value.dtype==torch.float32
    assert not torch.is_autocast_enabled('cuda'),'autocast leaked beyond neural forward'
    distribution=Categorical(logits=logits,validate_args=False)
    if actions is None:actions=batch['mask'].long().argmax(-1)
    loss=(value.square()+.3*value-.2*distribution.log_prob(actions)).mean()
    assert loss.dtype==torch.float32
    return loss,(logits,value)


def _capture_forward(model,batch):
    import torch
    if torch.cuda.is_current_stream_capturing():raise ValueError('construct precision qualification graph outside capture')
    rng=_rng();stream=torch.cuda.Stream(device=next(model.parameters()).device);current=torch.cuda.current_stream()
    stream.wait_stream(current)
    with torch.no_grad(),torch.cuda.stream(stream):
        for _ in range(2):model(batch);model.values(batch)
    current.wait_stream(stream);torch.cuda.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.no_grad(),torch.cuda.graph(graph,stream=stream):
        outputs=model(batch);values=model.values(batch)
    current.wait_stream(stream);torch.cuda.synchronize()
    _exact(_rng(),rng,'RNG-free forward graph construction')
    return graph,outputs,values


def qualify_precision(model_factory,packed_batch,mode='bf16',*,tolerance=None):
    """Actual-device callable; never allocates a cloud worker or simulator.

    ``model_factory()`` returns a fresh SAME Policy/FP32 parameter recipe on CUDA,
    including any already-qualified optional compact embeddings. ``packed_batch``
    is the existing encoder dictionary from real public observations/legal masks.
    Input tensors are cloned; every caller RNG stream/global TF32 flag is restored.
    """
    import torch
    from torch.distributions import Categorical
    if mode not in MODES:raise ValueError('unknown precision mode')
    if not torch.cuda.is_available():raise ValueError('actual allocated CUDA hardware is required')
    if torch.cuda.device_count()!=1:raise ValueError('precision qualifier requires exactly one visible allocated GPU')
    if torch.is_autocast_enabled('cuda'):raise ValueError('qualify precision outside any enclosing autocast scope')
    limits=dict(TOLERANCES[mode] if tolerance is None else tolerance)
    if set(limits)!=set(TOLERANCES[mode]) or any(float(v)<0 for v in limits.values()):raise ValueError('explicit complete nonnegative tolerance dictionary required')
    saved_rng=_rng();original_tf32=bool(torch.backends.cuda.matmul.allow_tf32);graphs=[]
    try:
        reference=model_factory();candidate=model_factory();candidate.load_state_dict(reference.state_dict())
        if any(not p.is_cuda for p in reference.parameters()):raise ValueError('model factory must return actual CUDA parameters')
        if not getattr(reference,'device_indices',False) or not getattr(candidate,'device_indices',False):raise ValueError('model factory must enable Policy.device_indices before graph qualification')
        batch={name:value.detach().clone() for name,value in packed_batch.items()}
        if not {'header','entities','mask'}.issubset(batch):raise ValueError('provide full existing packed public batch including mask')
        names=list(candidate.state_dict());aliases=[(name,id(p),p.data_ptr(),p.dtype) for name,p in candidate.named_parameters()]
        configure_precision(reference,'fp32');reference.zero_grad(set_to_none=True)
        ref_loss,ref_output=_objective(reference,batch);ref_loss.backward()
        ref_grad=[p.grad.detach().clone() for p in reference.parameters()]
        metadata=configure_precision(candidate,mode);candidate.zero_grad(set_to_none=True)
        candidate_loss,output=_objective(candidate,batch);candidate_loss.backward()
        assert names==list(candidate.state_dict())
        assert aliases==[(name,id(p),p.data_ptr(),p.dtype) for name,p in candidate.named_parameters()]
        mask=batch['mask'].bool();maxima={}
        # Mask sentinels may round in BF16, but remain effectively impossible.
        # Compare every legal logit and normalized log-probability, all values.
        for label,actual,want in [('legal_logits',output[0][mask],ref_output[0][mask]),('values',output[1],ref_output[1]),('legal_log_probabilities',output[0].log_softmax(-1)[mask],ref_output[0].log_softmax(-1)[mask])]:
            torch.testing.assert_close(actual,want,atol=limits['output_atol'],rtol=limits['output_rtol'],msg=label)
            maxima[label]=float((actual-want).detach().abs().max()) if actual.numel() else 0.
        assert (output[0][~mask]<-1e8).all() and (ref_output[0][~mask]<-1e8).all()
        norm=error=0.;gradient_max=0.
        for p,want in zip(candidate.parameters(),ref_grad):
            assert p.dtype==p.grad.dtype==torch.float32
            torch.testing.assert_close(p.grad,want,atol=limits['gradient_atol'],rtol=limits['gradient_rtol'])
            gradient_max=max(gradient_max,float((p.grad-want).abs().max()));error+=float((p.grad-want).double().square().sum());norm+=float(want.double().square().sum())
        relative_l2=(error/max(norm,1e-30))**.5
        assert relative_l2<=limits['gradient_relative_l2'],('gradient relative L2',relative_l2,limits)
        graph,graph_output,graph_values=_capture_forward(candidate,batch);graphs.append(graph)
        graph.replay();before=[v.clone() for v in graph_output];torch.cuda.synchronize()
        optimizer=torch.optim.Adam(candidate.parameters(),lr=.003,eps=1e-5,fused=True)
        optimizer.zero_grad(set_to_none=True);loss,_=_objective(candidate,batch);loss.backward()
        assert not torch.is_autocast_enabled('cuda')
        torch.nn.utils.clip_grad_norm_(candidate.parameters(),.5);optimizer.step()
        assert all(p.dtype==torch.float32 for p in candidate.parameters())
        with torch.no_grad():eager=candidate(batch);eager_values=candidate.values(batch)
        graph.replay();torch.cuda.synchronize()
        for actual,want in zip(graph_output,eager):torch.testing.assert_close(actual,want,atol=0.,rtol=0.)
        torch.testing.assert_close(graph_values,eager_values,atol=0.,rtol=0.)
        assert any(not torch.equal(a,b) for a,b in zip(before,graph_output)),'captured graph did not observe updated FP32 parameters'
        # A new model, Adam and graph recreate the same in-mode continuation.
        weights=copy.deepcopy(candidate.state_dict());adam=copy.deepcopy(optimizer.state_dict());continuation_rng=_rng()
        recreated=model_factory();recreated.load_state_dict(weights);recreated_metadata=configure_precision(recreated,mode)
        resumed=torch.optim.Adam(recreated.parameters(),lr=.003,eps=1e-5,fused=True);resumed.load_state_dict(adam)
        _restore_rng(continuation_rng)
        second_graph,second_output,second_values=_capture_forward(recreated,batch);graphs.append(second_graph)
        graph.replay();second_graph.replay();torch.cuda.synchronize()
        for actual,want in zip(second_output,graph_output):torch.testing.assert_close(actual,want,atol=0.,rtol=0.)
        torch.testing.assert_close(second_values,graph_values,atol=0.,rtol=0.)
        _exact(metadata,recreated_metadata,'recreated precision metadata')
        def continuation(model,opt):
            terms=[]
            for _ in range(2):
                opt.zero_grad(set_to_none=True);logits,value=model(batch);distribution=Categorical(logits=logits,validate_args=False);action=distribution.sample()
                assert not torch.is_autocast_enabled('cuda')
                loss=(value.square()-.2*distribution.log_prob(action)).mean();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),.5);opt.step()
                terms.append((action.clone(),loss.detach().clone()))
            torch.cuda.synchronize()
            return terms,copy.deepcopy(model.state_dict()),copy.deepcopy(opt.state_dict()),_rng()
        _restore_rng(continuation_rng);expected=continuation(candidate,optimizer)
        _restore_rng(continuation_rng);actual=continuation(recreated,resumed)
        _exact(actual,expected,'exact in-mode recreated model/Adam/sampling/RNG continuation')
        return {'status':'passed','mode':mode,'metadata':metadata,'tolerances':limits,'cross_mode_maximum_absolute_error':maxima,'gradient_maximum_absolute_error':gradient_max,'gradient_relative_l2':relative_l2,'master_parameters_and_state_dict_aliases_preserved':True,'outputs_float32':True,'loss_backward_clip_and_adam_outside_autocast':True,'captured_forward_reads_updated_fp32_weights':True,'in_mode_graph_recreation_exact':True,'in_mode_model_adam_sampling_all_rng_continuation_exact':True,'complete_trainer_resume_still_required':True,'equal_strength_claimed':False,'gpu':torch.cuda.get_device_name(),'timing_comparison':False}
    finally:
        torch.cuda.synchronize()
        for graph in graphs:graph.reset()
        _restore_rng(saved_rng);torch.backends.cuda.matmul.allow_tf32=original_tf32
