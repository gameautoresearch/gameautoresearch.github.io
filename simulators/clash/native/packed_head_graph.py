"""Opt-in complete PPO graphs with one device-only entity x packed-head SWITCH.

No production trainer integration. Model/Adam/RNG ownership follows the separately
qualified LearnerGraph lifecycle. Children are RNG-free, unconditional Torch
keep_graph graphs. Sampling stays outside. Default4 entity x<=4 head cases avoid
replacing the existing entity-bucket path with an unconditional200-row encoder.
"""
from collections.abc import Mapping
from contextlib import contextmanager
import torch
from learner_graph import LearnerGraph,_TrainingSnapshot,_optimizer_signature
from gpu_features import DeviceBatchEncoder
from packed_head import INPUT_KEYS,head_bounds,packed_minibatch

@contextmanager
def _encoder(trainer,encoder):
    original=trainer.encoder;trainer.encoder=encoder
    try:yield
    finally:trainer.encoder=original

class PackedHeadLearnerGraph(LearnerGraph):
    """Borrowed metrics; flattened children entity-major, head-capacity-minor.

    Each branch encodes/values ALL N, keeps its existing entity bound, executes a
    nonempty static-cap head, scatters full-N scalar terms and updates the SAME
    Adam. Branch-local gradient operands stay retained; shared gradient outputs
    expose the actual selected branch to the caller. No CPU data-count read.
    """
    def __init__(self,trainer,example:Mapping,*,entity_bounds=(32,64,96,200),warmups=2):
        from packed_head_branch import PackedHeadBranchGraph
        if type(warmups) is not int or warmups<1:raise ValueError('at least one warmup required')
        if trainer.c.get('compile_loss',False) or trainer.c.get('rollout_graph',False):raise ValueError('prototype requires compile_loss=False and rollout_graph=False')
        if not isinstance(trainer.optimizer,torch.optim.Adam):raise TypeError('existing torch.optim.Adam required')
        if not torch.are_deterministic_algorithms_enabled() or torch.is_deterministic_algorithms_warn_only_enabled():raise ValueError('strict deterministic FP32 required')
        self.trainer=trainer;self.parameters=tuple(trainer.model.parameters())
        if not self.parameters or any(not p.is_cuda or p.dtype!=torch.float32 for p in self.parameters):raise ValueError('actual CUDA FP32 parameters required')
        self.device=self.parameters[0].device
        if torch.cuda.is_current_stream_capturing():raise RuntimeError('construct outside CUDA capture')
        if torch.is_autocast_enabled('cuda') or torch.backends.cuda.matmul.allow_tf32:raise ValueError('packed head has only strict FP32 qualification')
        if any(p.device!=self.device for p in self.parameters):raise ValueError('one model device required')
        if not getattr(trainer.model,'device_indices',False) or getattr(trainer.model,'ragged_entities',False) or getattr(trainer.model,'sparse_policy_head',False):raise ValueError('existing static device-indexed encoder/head required')
        for group in trainer.optimizer.param_groups:
            if not group.get('capturable',False) or group.get('differentiable',False):raise ValueError('existing Adam needs capturable=True, differentiable=False')
        self.entity_bounds=tuple(entity_bounds)
        if not self.entity_bounds or len(self.entity_bounds)>4 or tuple(sorted(set(self.entity_bounds)))!=self.entity_bounds or any(type(n) is not int or n<1 for n in self.entity_bounds) or self.entity_bounds[-1]!=200:raise ValueError('<=4 increasing entity bounds ending at full200 required')
        self.inputs={};self.layout={};self._validate_fields(example,initialize=True);self.inputs={k:v.contiguous() for k,v in self.inputs.items()}
        self.head_caps=head_bounds(len(self.inputs['action']));self.original_encoder=trainer.encoder
        self.model_identity=id(trainer.model);self.parameter_bindings=self._parameter_bindings();self.input_bindings=tuple((name,id(v),v.data_ptr()) for name,v in self.inputs.items());self.capture_bounds=(self.entity_bounds,self.head_caps)
        self.encoder_binding=(tuple(trainer.encoder.layout),trainer.encoder.count_index,trainer.encoder.copy)
        self.options=_optimizer_signature(trainer.optimizer);self.loss_recipe=tuple((name,trainer.c[name]) for name in ('clip','value_coef','entropy_coef','max_grad_norm'))
        self.buffer_bindings=tuple((id(v),v.data_ptr(),tuple(v.shape),v.dtype,v.device) for v in trainer.model.buffers());self.training_mode=trainer.model.training
        self.encoders=tuple(DeviceBatchEncoder(trainer.schema,entity_rows=bound,copy=False) for bound in self.entity_bounds)
        self.children=[];self.child_metrics=[];self.child_gradients=[];self.shared_gradients=tuple(torch.zeros_like(p) for p in self.parameters);self.metrics=torch.empty(5,device=self.device,dtype=torch.float32)
        self.closed=False;self.branch=None;self.count_index=trainer.schema.header['entity_count'][0]
        actual={k:v.clone() for k,v in self.inputs.items()};counts=self.inputs['obs'][:,self.count_index]
        torch._assert_async(((counts>=0)&(counts<=200)).all(),'entity count outside complete public schema')
        snapshot=_TrainingSnapshot(trainer);current=torch.cuda.current_stream(self.device);stream=torch.cuda.Stream(device=self.device)
        try:
            for bound,encoder in zip(self.entity_bounds,self.encoders):
                for cap in self.head_caps:
                    for name,value in self.inputs.items():value.copy_(actual[name])
                    # Synthetic inputs are PRIVATE constructor-only warmup values.
                    # Real masks/counts/fields are restored before the first replay.
                    self.inputs['obs'][:,self.count_index].clamp_(max=bound)
                    self.inputs['mask'].zero_();self.inputs['mask'][:,0]=1;self.inputs['action'].zero_()
                    stream.wait_stream(current)
                    with _encoder(trainer,encoder):
                        with torch.enable_grad(),torch.cuda.stream(stream):
                            for _ in range(warmups):packed_minibatch(trainer,self.inputs,cap)
                        current.wait_stream(stream);torch.cuda.synchronize(self.device);snapshot.restore();torch.cuda.synchronize(self.device)
                        child=torch.cuda.CUDAGraph(keep_graph=True)
                        with torch.enable_grad(),torch.cuda.graph(child,stream=stream):
                            metrics=packed_minibatch(trainer,self.inputs,cap)
                            for parameter,target in zip(self.parameters,self.shared_gradients):
                                if parameter.grad is None:raise RuntimeError('packed PPO unexpectedly left a parameter unused')
                                target.copy_(parameter.grad)
                            self.metrics.copy_(metrics)
                        self.children.append(child);self.child_metrics.append(metrics);self.child_gradients.append(tuple(p.grad for p in self.parameters))
                    current.wait_stream(stream);torch.cuda.synchronize(self.device);snapshot.restore();torch.cuda.synchronize(self.device)
            self.state_bindings=self._state_bindings()
            for name,value in self.inputs.items():value.copy_(actual[name])
            torch.cuda.synchronize(self.device)
            self.branch=PackedHeadBranchGraph(self.inputs['obs'],self.inputs['mask'],self.count_index,self.entity_bounds,self.head_caps,self.children)
            self.selection=self.branch.selection;self.selected_bucket=self.branch.selected_bucket;self.flat_bucket=self.branch.flat_bucket;self.build_binding=self.branch.build_binding
        finally:
            current.wait_stream(stream);torch.cuda.synchronize(self.device);snapshot.restore()
            for name,value in self.inputs.items():value.copy_(actual[name])
            trainer.encoder=self.original_encoder;torch.cuda.synchronize(self.device)
        self.validate_bindings()
    def _validate_fields(self,fields,*,initialize=False):
        super()._validate_fields(fields,initialize=initialize)
        if fields['obs'].shape[1]!=self.trainer.schema.seat_ints or fields['mask'].shape[1]!=self.trainer.schema.actions:raise ValueError('complete player-visible raw input widths required')
    def _parameter_bindings(self):
        return tuple((id(p),p.data_ptr(),tuple(p.shape),tuple(p.stride()),p.dtype,p.device) for p in self.trainer.model.parameters())
    def validate_bindings(self):
        super().validate_bindings()
        if id(self.trainer.model)!=self.model_identity or self._parameter_bindings()!=self.parameter_bindings:raise RuntimeError('model or parameter storage changed after capture')
        if tuple((name,id(v),v.data_ptr()) for name,v in self.inputs.items())!=self.input_bindings:raise RuntimeError('owned graph input storage changed after capture')
        if (self.entity_bounds,self.head_caps)!=self.capture_bounds:raise RuntimeError('captured entity/head bounds changed')
        if (tuple(self.trainer.encoder.layout),self.trainer.encoder.count_index,self.trainer.encoder.copy)!=self.encoder_binding:raise RuntimeError('encoder layout changed after capture')
        if self.trainer.encoder is not self.original_encoder or self.trainer.model.training!=self.training_mode:raise RuntimeError('encoder or model mode changed after capture')
        if torch.is_autocast_enabled('cuda') or torch.backends.cuda.matmul.allow_tf32 or not torch.are_deterministic_algorithms_enabled() or torch.is_deterministic_algorithms_warn_only_enabled():raise RuntimeError('numeric mode changed after strict FP32 capture')
    def __call__(self,fields):
        if self.closed:raise RuntimeError('packed-head graph is closed')
        if torch.cuda.is_current_stream_capturing():raise RuntimeError('conditional graph cannot be an outer captured child')
        self._validate_fields(fields);self.validate_bindings()
        for name in INPUT_KEYS:self.inputs[name].copy_(fields[name])
        for parameter,gradient in zip(self.parameters,self.shared_gradients):parameter.grad=gradient
        self.branch.replay();return self.metrics
    def close(self):
        if self.closed:return
        torch.cuda.synchronize(self.device)
        if self.branch is not None:self.branch.close()
        for child in self.children:child.reset()
        self.closed=True
