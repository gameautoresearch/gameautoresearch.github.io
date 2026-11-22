"""Optional complete single-minibatch CUDA graph for the existing PPO learner.

The model, optimizer, encoding, PPO expression, clipping and Adam implementation
remain those owned by Trainer. Advantage normalization, minibatch ordering, KL
stopping, update counts and checkpoints remain outside this module.

Integration::

    fields = dict(obs=..., mask=..., action=..., logp=...,
                  advantage=normalized_advantage, **{'return': ...}, weight=...)
    graph = LearnerGraph(trainer, fields)  # optimizer groups: capturable=True
    metrics = graph(fields)              # borrowed CUDA tensor, five entries
    history.append(metrics.clone())      # retain before the next replay

A distinct graph may be cached for a final smaller minibatch. All graphs refer to
exactly the same model/Adam state, and restore warmup mutations in place. Creating
or loading a different model/optimizer after capture requires constructing new
graphs; validate_bindings() rejects stale tensor references.
"""
from __future__ import annotations
from collections.abc import Mapping
import copy
import random

INPUT_KEYS=('obs','mask','action','logp','advantage','return','weight')
METRIC_NAMES=('policy_loss','value_loss','entropy','approx_kl','gradient_norm')


def eager_minibatch(trainer, fields):
    """The exact Trainer.learn minibatch expression and update order.

    This also provides the independent eager control for qualification. No action
    sampling, advantage normalization, optimizer replacement or alternate loss.
    """
    import torch
    c=trainer.c
    compiled=bool(c.get('compile_loss',False))
    if compiled:
        from compiled_loss import compiled_ppo_loss
        loss,(pg,vl,entropy,kl)=compiled_ppo_loss(trainer,fields)
    else:
        batch=trainer.encoder(fields['obs'],fields['mask'])
        distribution,value=trainer.model.distribution(batch)
        new=distribution.log_prob(fields['action']);old=fields['logp']
        ratio=(new-old).exp();w=fields['weight'];a=fields['advantage']
        pg=-(torch.minimum(ratio*a,ratio.clamp(1-c['clip'],1+c['clip'])*a)*w).sum()/w.sum().clamp_min(1)
        vl=(value-fields['return']).square().mean()
        entropy=(distribution.entropy()*w).sum()/w.sum().clamp_min(1)
        loss=pg+c['value_coef']*vl-c['entropy_coef']*entropy
    torch._assert_async(torch.isfinite(loss),'nonfinite native PPO loss')
    trainer.optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad=torch.nn.utils.clip_grad_norm_(trainer.model.parameters(),c['max_grad_norm'])
    trainer.optimizer.step()
    with torch.no_grad():
        if not compiled:kl=((ratio-1)-(new-old)).mean()
        return torch.stack((pg.detach(),vl.detach(),entropy.detach(),kl,grad.detach()))


class _TrainingSnapshot:
    """Value snapshots only; restore NEVER replaces graph-referenced tensors.

    Adam lazily initializes state during warmup. Previously absent Adam tensor
    state is retained at the same address and zeroed (including step=0), which is
    exactly its initialization value. Existing state is copied back in place.
    """
    def __init__(self,trainer):
        import numpy as np
        import torch
        self.trainer=trainer
        self.model_values=[(v,v.detach().clone()) for v in (*trainer.model.parameters(),*trainer.model.buffers())]
        self.grads=[(p,p.grad,None if p.grad is None else p.grad.detach().clone()) for p in trainer.model.parameters()]
        self.optimizer_values={p:{k:(v.detach().clone() if torch.is_tensor(v) else copy.deepcopy(v)) for k,v in state.items()} for p,state in trainer.optimizer.state.items()}
        self.cpu_rng=torch.get_rng_state().clone()
        self.cuda_rng=torch.cuda.get_rng_state_all() if any(p.is_cuda for p in trainer.model.parameters()) else None
        self.python_rng=random.getstate();self.numpy_rng=np.random.get_state()

    def restore(self):
        import numpy as np
        import torch
        with torch.no_grad():
            for target,saved in self.model_values:target.copy_(saved)
            for parameter,state in self.trainer.optimizer.state.items():
                original=self.optimizer_values.get(parameter,{})
                for name,target in state.items():
                    if torch.is_tensor(target):
                        if name in original:
                            saved=original[name]
                            if not torch.is_tensor(saved) or saved.shape!=target.shape or saved.dtype!=target.dtype:raise RuntimeError('Adam warmup changed state schema')
                            target.copy_(saved)
                        else:
                            # Adam's lazily allocated moments, AMSGrad maximum
                            # and step all have zero initial values.
                            if name not in ('step','exp_avg','exp_avg_sq','max_exp_avg_sq'):raise RuntimeError('unexpected newly initialized Adam state '+name)
                            target.zero_()
                    elif name in original:state[name]=copy.deepcopy(original[name])
                    else:raise RuntimeError('unexpected new non-tensor Adam state '+name)
            for parameter,original,saved in self.grads:
                parameter.grad=original
                if original is not None:original.copy_(saved)
        torch.set_rng_state(self.cpu_rng)
        if self.cuda_rng is not None:torch.cuda.set_rng_state_all(self.cuda_rng)
        random.setstate(self.python_rng);np.random.set_state(self.numpy_rng)


def _optimizer_signature(optimizer):
    """Python options are captured constants; tensor options retain addresses."""
    import torch
    result=[]
    for group in optimizer.param_groups:
        options=[]
        for name,value in sorted(group.items()):
            if name=='params':continue
            if torch.is_tensor(value):options.append((name,'tensor',id(value),value.data_ptr(),tuple(value.shape),str(value.dtype),str(value.device)))
            else:options.append((name,copy.deepcopy(value)))
        result.append((tuple(id(p) for p in group['params']),tuple(options)))
    return tuple(result)


class LearnerGraph:
    """Capture forward, PPO loss, backward, clip-grad and existing Adam step.

    Construction has no real update: warmup and capture mutations are restored
    before returning. Calling this object performs exactly ONE optimizer step.
    Returned metrics are graph-owned CUDA storage, overwritten on the next call.
    """
    def __init__(self,trainer,example:Mapping,*,warmups=2):
        import torch
        if warmups<1:raise ValueError('at least one optimizer warmup is required')
        if not isinstance(trainer.optimizer,torch.optim.Adam):raise TypeError('learner graph requires the existing torch.optim.Adam')
        self.trainer=trainer;self.parameters=tuple(trainer.model.parameters())
        if not self.parameters or any(not p.is_cuda for p in self.parameters):raise ValueError('learner graph requires CUDA model parameters')
        self.device=self.parameters[0].device
        if any(p.device!=self.device for p in self.parameters):raise ValueError('all learner parameters must share one CUDA device')
        for group in trainer.optimizer.param_groups:
            if not group.get('capturable',False):raise ValueError('set capturable=True on the existing Adam before learner graph creation')
            if group.get('differentiable',False):raise ValueError('differentiable Adam is outside the eager learner recipe')
        self.inputs={};self.layout={}
        self._validate_fields(example,initialize=True)
        self.options=_optimizer_signature(trainer.optimizer)
        self.loss_recipe=tuple((name,trainer.c[name]) for name in ('clip','value_coef','entropy_coef','max_grad_norm'))
        self.buffer_bindings=tuple((id(v),v.data_ptr(),tuple(v.shape),v.dtype,v.device) for v in trainer.model.buffers())
        self.graph=torch.cuda.CUDAGraph();self.stream=torch.cuda.Stream(device=self.device)
        snapshot=_TrainingSnapshot(trainer)
        current=torch.cuda.current_stream(self.device)
        self.stream.wait_stream(current)
        try:
            with torch.enable_grad(),torch.cuda.stream(self.stream):
                for _ in range(warmups):eager_minibatch(trainer,self.inputs)
            current.wait_stream(self.stream);torch.cuda.synchronize(self.device)
            snapshot.restore();torch.cuda.synchronize(self.device)
            # zero_grad(set_to_none=True) inside eager_minibatch leaves the graph
            # owning its freshly allocated static gradient tensors. Replay writes
            # those outputs again; it does not accumulate previous minibatches.
            with torch.enable_grad(),torch.cuda.graph(self.graph,stream=self.stream):
                self.metrics=eager_minibatch(trainer,self.inputs)
            self.gradients=tuple(p.grad for p in self.parameters)
            self.state_bindings=self._state_bindings()
        finally:
            current.wait_stream(self.stream)
            torch.cuda.synchronize(self.device)
            snapshot.restore()
            torch.cuda.synchronize(self.device)
        self.validate_bindings()

    def _validate_fields(self,fields,*,initialize=False):
        import torch
        if initialize:self.compiled_loss_enabled=bool(self.trainer.c.get('compile_loss',False))
        elif self.compiled_loss_enabled!=bool(self.trainer.c.get('compile_loss',False)):raise RuntimeError('compile_loss changed after learner graph capture')
        if set(fields)!=set(INPUT_KEYS):raise ValueError('minibatch keys must be exactly '+','.join(INPUT_KEYS))
        size=fields['action'].shape[0]
        if size<1:raise ValueError('empty learner minibatch')
        for name in INPUT_KEYS:
            value=fields[name]
            if not isinstance(value,torch.Tensor) or value.device!=self.device:raise ValueError(name+' must be a tensor on the model CUDA device')
            if value.shape[0]!=size or (name not in ('obs','mask') and value.shape!=(size,)):raise ValueError(name+' minibatch shape differs')
            if name in ('obs','mask') and value.ndim!=2:raise ValueError(name+' must have two dimensions')
            layout=(tuple(value.shape),value.dtype,value.device)
            if initialize:
                self.layout[name]=layout;self.inputs[name]=value.detach().clone()
            elif layout!=self.layout[name]:raise ValueError(name+' differs from the captured shape/dtype/device')
        if fields['action'].dtype!=torch.int64:raise ValueError('actions must retain int64 dtype')
        if fields['obs'].dtype!=torch.int32 or fields['mask'].dtype not in (torch.uint8,torch.bool):raise ValueError('raw observation/mask dtype differs from native learner')
        for name in ('logp','advantage','return','weight'):
            if fields[name].dtype!=torch.float32:raise ValueError(name+' must retain float32 dtype')

    def _state_bindings(self):
        import torch
        return tuple((id(p),name,id(value),value.data_ptr(),tuple(value.shape),value.dtype,value.device) for p in self.parameters for name,value in sorted(self.trainer.optimizer.state.get(p,{}).items()) if torch.is_tensor(value))

    def validate_bindings(self):
        """Fail if load_state_dict or model replacement invalidated a graph."""
        if tuple(id(p) for p in self.trainer.model.parameters())!=tuple(id(p) for p in self.parameters):raise RuntimeError('learner model parameters replaced after capture')
        if tuple((id(v),v.data_ptr(),tuple(v.shape),v.dtype,v.device) for v in self.trainer.model.buffers())!=self.buffer_bindings:raise RuntimeError('learner model buffers replaced after capture')
        if tuple((name,self.trainer.c[name]) for name,_ in self.loss_recipe)!=self.loss_recipe:raise RuntimeError('PPO loss recipe changed after graph capture')
        if _optimizer_signature(self.trainer.optimizer)!=self.options:raise RuntimeError('Adam options or parameters changed after graph capture')
        if self._state_bindings()!=self.state_bindings:raise RuntimeError('Adam state tensors replaced after graph capture; recreate learner graphs')

    def __call__(self,fields:Mapping):
        self._validate_fields(fields);self.validate_bindings()
        for name in INPUT_KEYS:self.inputs[name].copy_(fields[name])
        # A second graph for a smaller tail minibatch can have distinct gradient
        # outputs, but both refer to the SAME model and moment/step tensors.
        for parameter,gradient in zip(self.parameters,self.gradients):parameter.grad=gradient
        self.graph.replay()
        return self.metrics


def qualify_learner_graph(trainer,minibatches,*,atol=0.,rtol=0.):
    """Compare real graph updates with eager, then restore the caller in place.

    Checks every metric, parameter, model buffer and Adam state after every step,
    including construction's step/RNG preservation. Mixed minibatch sizes create
    separate graphs and exercise shared optimizer state. No timing is reported.
    """
    import torch
    minibatches=list(minibatches)
    if not minibatches:raise ValueError('qualification requires actual valid minibatches')
    saved=_TrainingSnapshot(trainer);references=[];graphs={}
    def values():
        return ([v.detach().clone() for v in (*trainer.model.parameters(),*trainer.model.buffers())],{p:{k:v.detach().clone() for k,v in state.items() if torch.is_tensor(v)} for p,state in trainer.optimizer.state.items()})
    def check_values(expected):
        model,state=expected
        for value,want in zip((*trainer.model.parameters(),*trainer.model.buffers()),model):torch.testing.assert_close(value,want,atol=atol,rtol=rtol)
        for p,fields in state.items():
            for name,want in fields.items():torch.testing.assert_close(trainer.optimizer.state[p][name],want,atol=atol,rtol=rtol)
    try:
        for fields in minibatches:
            metrics=eager_minibatch(trainer,fields).clone();references.append((metrics,values()))
        saved.restore();torch.cuda.synchronize()
        for step,(fields,(want_metrics,want_values)) in enumerate(zip(minibatches,references)):
            key=tuple((name,tuple(fields[name].shape),fields[name].dtype) for name in INPUT_KEYS)
            if key not in graphs:
                before=values();cpu_rng=torch.get_rng_state().clone();cuda_rng=torch.cuda.get_rng_state_all()
                graphs[key]=LearnerGraph(trainer,fields)
                check_values(before)
                assert torch.equal(torch.get_rng_state(),cpu_rng)
                assert all(torch.equal(a,b) for a,b in zip(torch.cuda.get_rng_state_all(),cuda_rng))
            actual=graphs[key](fields).clone()
            torch.testing.assert_close(actual,want_metrics,atol=atol,rtol=rtol)
            check_values(want_values)
        torch.cuda.synchronize()
        return {'status':'passed','steps':len(minibatches),'graph_shapes':len(graphs),'metric_names':METRIC_NAMES,'atol':atol,'rtol':rtol,'model_and_adam_after_every_step':True,'warmup_model_adam_rng_restored':True,'timing_claims':False}
    finally:
        saved.restore();torch.cuda.synchronize()
