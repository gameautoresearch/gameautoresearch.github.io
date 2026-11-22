"""Optional GPU-selected entity buckets for the existing complete PPO update.

No trainer defaults or game observations are changed. Every branch receives the
same complete [N,7024] public observations and [N,2307] masks; only the learned
encoder's padded entity dimension changes. A CUDA 12.8 conditional SWITCH picks
32/64/96/200 from the device maximum count, with 200 as the complete fallback.

Integration after separate qualification::
    graph = BucketLearnerGraph(trainer, fields)  # capturable existing Adam
    metrics = graph(fields).clone()             # one complete Adam update

Construction restores all warmup/capture model, Adam and RNG mutations. Replay
contains no RNG, sampling, advantage normalization, simulator or CPU count read.
Cross-bucket numerical parity is tolerance-based; exact resume must be qualified
WITHIN this optional mode. Changing/loading model or Adam requires new graphs.
"""
from __future__ import annotations
from collections.abc import Mapping
from contextlib import contextmanager
import json
from pathlib import Path

from learner_graph import (INPUT_KEYS,METRIC_NAMES,LearnerGraph,_TrainingSnapshot,
                           _optimizer_signature,eager_minibatch)
from gpu_features import DeviceBatchEncoder

DEFAULT_BUCKETS=(32,64,96,200)


def validate_buckets(schema,buckets=DEFAULT_BUCKETS):
    buckets=tuple(buckets)
    if not buckets or any(type(x) is not int or x<1 for x in buckets) or tuple(sorted(set(buckets)))!=buckets or buckets[-1]!=schema.max_entities:
        raise ValueError('entity buckets must be strictly increasing positive integers ending at the complete schema capacity')
    if schema.seat_ints!=7024 or schema.actions!=2307 or schema.max_entities!=200:
        raise ValueError('bucket learner requires the frozen fixed8 public schema')
    return buckets


@contextmanager
def _encoder(trainer,encoder):
    original=trainer.encoder;trainer.encoder=encoder
    try:yield
    finally:trainer.encoder=original


def eager_bucket_minibatch(trainer,fields,entity_rows):
    """Qualification control: exact existing update with a known-safe row bound."""
    with _encoder(trainer,DeviceBatchEncoder(trainer.schema,entity_rows=entity_rows,copy=False)):
        return eager_minibatch(trainer,fields)


class BucketLearnerGraph(LearnerGraph):
    """Four ordinary Torch child graphs selected entirely by device counts.

    All branch graph handles and branch-local gradient operands are retained.
    Every selected branch updates the SAME model parameters and Adam state, then
    copies its gradients and metrics to shared observable output addresses.
    """
    def __init__(self,trainer,example:Mapping,*,buckets=DEFAULT_BUCKETS,warmups=2):
        import torch
        from device_branch import DeviceBranchGraph
        if warmups<1:raise ValueError('at least one warmup is required')
        if trainer.c.get('rollout_graph',False):raise ValueError('bucket learner prototype requires rollout_graph=False; conditional graphs cannot be nested')
        if not isinstance(trainer.optimizer,torch.optim.Adam):raise TypeError('bucket learner requires the existing torch.optim.Adam')
        if not getattr(trainer.model,'device_indices',False):raise ValueError('enable Policy.device_indices before bucket capture')
        if getattr(trainer.model,'ragged_entities',False) or getattr(trainer.model,'sparse_policy_head',False):raise ValueError('bucket learner requires the existing static dense encoder/head')
        self.trainer=trainer;self.parameters=tuple(trainer.model.parameters());self.buckets=validate_buckets(trainer.schema,buckets)
        if not self.parameters or any(not p.is_cuda for p in self.parameters):raise ValueError('bucket learner requires CUDA model parameters')
        self.device=self.parameters[0].device
        if any(p.device!=self.device for p in self.parameters):raise ValueError('bucket learner parameters must share one CUDA device')
        for group in trainer.optimizer.param_groups:
            if not group.get('capturable',False):raise ValueError('existing Adam must have capturable=True before bucket capture')
            if group.get('differentiable',False):raise ValueError('differentiable Adam is outside the existing learner recipe')
        self.original_encoder=trainer.encoder
        self.inputs={};self.layout={};self._validate_fields(example,initialize=True)
        self.inputs={k:v.contiguous() for k,v in self.inputs.items()}
        self.options=_optimizer_signature(trainer.optimizer)
        self.loss_recipe=tuple((name,trainer.c[name]) for name in ('clip','value_coef','entropy_coef','max_grad_norm'))
        self.buffer_bindings=tuple((id(v),v.data_ptr(),tuple(v.shape),v.dtype,v.device) for v in trainer.model.buffers())
        self.encoders=tuple(DeviceBatchEncoder(trainer.schema,entity_rows=n,copy=False) for n in self.buckets)
        self.children=[];self.child_metrics=[];self.child_gradients=[];self.shared_gradients=tuple(torch.zeros_like(p) for p in self.parameters)
        self.metrics=torch.empty(5,device=self.device,dtype=torch.float32);self.closed=False;self.branch=None
        self.count_index=trainer.schema.header['entity_count'][0]
        actual_inputs={k:v.clone() for k,v in self.inputs.items()}
        counts=self.inputs['obs'][:,self.count_index]
        torch._assert_async(((counts>=0)&(counts<=trainer.schema.max_entities)).all(),'bucket learner entity count outside complete public schema')
        snapshot=_TrainingSnapshot(trainer);current=torch.cuda.current_stream(self.device);stream=torch.cuda.Stream(device=self.device)
        try:
            for bound,encoder in zip(self.buckets,self.encoders):
                for name,value in self.inputs.items():value.copy_(actual_inputs[name])
                # These are private synthetic WARMUP inputs only. The real complete
                # minibatch is restored before the first GPU-selected real update.
                self.inputs['obs'][:,self.count_index].clamp_(max=bound)
                stream.wait_stream(current)
                with _encoder(trainer,encoder):
                    with torch.enable_grad(),torch.cuda.stream(stream):
                        for _ in range(warmups):eager_minibatch(trainer,self.inputs)
                    current.wait_stream(stream);torch.cuda.synchronize(self.device)
                    snapshot.restore();torch.cuda.synchronize(self.device)
                    child=torch.cuda.CUDAGraph(keep_graph=True)
                    with torch.enable_grad(),torch.cuda.graph(child,stream=stream):
                        metrics=eager_minibatch(trainer,self.inputs)
                        for parameter,target in zip(self.parameters,self.shared_gradients):
                            if parameter.grad is None:raise RuntimeError('frozen dense PPO unexpectedly left a parameter unused')
                            target.copy_(parameter.grad)
                        self.metrics.copy_(metrics)
                    self.children.append(child);self.child_metrics.append(metrics);self.child_gradients.append(tuple(p.grad for p in self.parameters))
                current.wait_stream(stream);torch.cuda.synchronize(self.device);snapshot.restore();torch.cuda.synchronize(self.device)
            self.state_bindings=self._state_bindings()
            for name,value in self.inputs.items():value.copy_(actual_inputs[name])
            torch.cuda.synchronize(self.device)
            self.branch=DeviceBranchGraph(self.inputs['obs'],self.count_index,self.buckets,self.children)
            self.build_binding=self.branch.build_binding
            self.selection=self.branch.selection
        finally:
            current.wait_stream(stream);torch.cuda.synchronize(self.device)
            snapshot.restore()
            for name,value in self.inputs.items():value.copy_(actual_inputs[name])
            trainer.encoder=self.original_encoder
            torch.cuda.synchronize(self.device)
        self.validate_bindings()

    def _validate_fields(self,fields,*,initialize=False):
        super()._validate_fields(fields,initialize=initialize)
        if fields['obs'].shape[1]!=self.trainer.schema.seat_ints or fields['mask'].shape[1]!=self.trainer.schema.actions:
            raise ValueError('bucket inputs must retain complete public observations and legal masks')

    def validate_bindings(self):
        super().validate_bindings()
        if self.trainer.encoder is not self.original_encoder:raise RuntimeError('trainer encoder replaced after bucket capture')

    def __call__(self,fields:Mapping):
        import torch
        if self.closed:raise RuntimeError('bucket learner is closed')
        if torch.cuda.is_current_stream_capturing():raise RuntimeError('conditional bucket learner cannot replay inside a captured outer graph')
        self._validate_fields(fields);self.validate_bindings()
        for name in INPUT_KEYS:self.inputs[name].copy_(fields[name])
        for parameter,gradient in zip(self.parameters,self.shared_gradients):parameter.grad=gradient
        self.branch.replay()
        return self.metrics

    def close(self):
        import torch
        if self.closed:return
        torch.cuda.synchronize(self.device)
        if self.branch is not None:self.branch.close()
        for child in self.children:child.reset()
        self.closed=True


def _values(trainer,*,gradients=False):
    import torch
    result={'model':[v.detach().clone() for v in (*trainer.model.parameters(),*trainer.model.buffers())],
            'adam':[{k:v.detach().clone() for k,v in trainer.optimizer.state.get(p,{}).items() if torch.is_tensor(v)} for p in trainer.model.parameters()]}
    if gradients:result['gradients']=[None if p.grad is None else p.grad.detach().clone() for p in trainer.model.parameters()]
    return result


def _check_values(trainer,expected,*,atol,rtol):
    import torch
    maximum=0.
    for value,want in zip((*trainer.model.parameters(),*trainer.model.buffers()),expected['model']):
        torch.testing.assert_close(value,want,atol=atol,rtol=rtol)
        if value.is_floating_point():maximum=max(maximum,float((value-want).detach().abs().max()))
    for p,fields in zip(trainer.model.parameters(),expected['adam']):
        for name,want in fields.items():torch.testing.assert_close(trainer.optimizer.state[p][name],want,atol=atol,rtol=rtol)
    if 'gradients' in expected:
        for p,want in zip(trainer.model.parameters(),expected['gradients']):
            if want is None:assert p.grad is None
            else:torch.testing.assert_close(p.grad,want,atol=atol,rtol=rtol)
    return maximum


def qualify_bucket_learner(trainer,minibatches,*,atol=2e-6,rtol=2e-5):
    """Untimed GPU dense-vs-selected-bucket updates; restores caller state.

    Host count reads BELOW belong only to qualification reporting. Normal replay
    selection has no .item(), CPU scalar, D2H count transfer or Python bucket test.
    This does not replace enabled-mode full Trainer checkpoint/resume qualification.
    """
    import torch
    minibatches=list(minibatches)
    if not minibatches:raise ValueError('qualification needs actual valid minibatches')
    saved=_TrainingSnapshot(trainer);references=[];graphs={};maximum=0.;selected=[]
    try:
        for fields in minibatches:
            metrics=eager_minibatch(trainer,fields).clone();references.append((metrics,_values(trainer,gradients=True)))
        saved.restore();torch.cuda.synchronize()
        for fields,(want_metrics,want_values) in zip(minibatches,references):
            key=tuple((name,tuple(fields[name].shape),fields[name].dtype) for name in INPUT_KEYS)
            if key not in graphs:
                before=_values(trainer);cpu_rng=torch.get_rng_state().clone();cuda_rng=torch.cuda.get_rng_state_all()
                graphs[key]=BucketLearnerGraph(trainer,fields)
                _check_values(trainer,before,atol=0.,rtol=0.)
                assert torch.equal(torch.get_rng_state(),cpu_rng)
                assert all(torch.equal(a,b) for a,b in zip(torch.cuda.get_rng_state_all(),cuda_rng))
            actual=graphs[key](fields).clone();torch.testing.assert_close(actual,want_metrics,atol=atol,rtol=rtol)
            maximum=max(maximum,_check_values(trainer,want_values,atol=atol,rtol=rtol))
            count=int(fields['obs'][:,trainer.schema.header['entity_count'][0]].max().cpu())
            index=next(i for i,bound in enumerate(DEFAULT_BUCKETS) if count<=bound)
            observed=graphs[key].selection.cpu().tolist()
            assert observed==[index,count,0],(observed,index,count)
            selected.append(DEFAULT_BUCKETS[index])
        torch.cuda.synchronize()
        return {'status':'passed','steps':len(minibatches),'graph_shapes':len(graphs),'buckets':DEFAULT_BUCKETS,'expected_selected_buckets':selected,'atol':atol,'rtol':rtol,'maximum_model_absolute_difference':maximum,'metrics_model_adam_and_clipped_gradients_compared_every_step':True,'warmup_model_adam_rng_restored_exact':True,'exact_resume_qualification_required':True,'timing_comparison':False}
    finally:
        for graph in graphs.values():graph.close()
        saved.restore();torch.cuda.synchronize()


def cpu_semantic_qualification(out,*,compact=False):
    """Portable qualification of bucket slicing using retained REAL public frames.

    Includes synthetic boundary populations formed by repeating real exposed
    entity rows. Those are neural-input fixtures, never simulator transitions.
    No CUDA helper is imported or launched by this control.
    """
    import argparse,copy,sys,numpy as np,torch
    from types import SimpleNamespace
    from model import Policy
    from train import schema
    here=Path(__file__).resolve().parent;root=here.parents[2];data=json.loads((here/'generated-v2/task-data.json').read_text());s=schema(data)
    source=root/'evidence/clash-implementation/native-training-20261108/compact-embedding-cpu-3/real-observations.npz'
    records=np.load(source);real_obs=records['observations'].reshape(-1,s.seat_ints);real_masks=records['masks'].reshape(-1,s.actions)
    assert len(real_obs)>0;torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.manual_seed(81108)
    reference=Policy(s,len(data['vocabulary']),len(data['buff_names']),64,12)
    torch.nn.init.normal_(reference.slot.weight,std=.01)
    if compact:
        from compact_embeddings import install
        install(reference,data)
    live_counts=real_obs[:,s.header['entity_count'][0]]
    entities=real_obs[:,s.header_ints:].reshape(-1,s.max_entities,s.entity_ints)
    stunned=np.flatnonzero((entities[...,s.entity['buff_0']]==244).any(1))
    indices=np.unique(np.concatenate((np.linspace(0,len(real_obs)-1,min(12,len(real_obs)),dtype=int),[live_counts.argmax()],stunned[:1])))
    real_obs=real_obs[indices];real_masks=real_masks[indices]
    populations=(None,0,1,31,32,33,63,64,65,95,96,97,199,200)
    result=[]
    for count in populations:
        # Fixed minibatch shape, with the first row at the exact boundary.
        obs=real_obs.copy();mask=real_masks.copy()
        if count is not None:
            for lane in range(len(obs)):
                rows=obs[lane,s.header_ints:].reshape(s.max_entities,s.entity_ints);n=int(obs[lane,s.header['entity_count'][0]])
                template=rows[:max(1,n)].copy();rows[:]=0
                if count:rows[:count]=template[np.arange(count)%len(template)]
                obs[lane,s.header['entity_count'][0]]=count
        fields={'obs':torch.from_numpy(obs),'mask':torch.from_numpy(mask),'action':torch.zeros(len(obs),dtype=torch.int64),'logp':torch.zeros(len(obs)),'advantage':torch.linspace(-1,1,len(obs)),'return':torch.linspace(.2,.8,len(obs)),'weight':torch.ones(len(obs))}
        dense=copy.deepcopy(reference);bucket=copy.deepcopy(reference);maximum_count=int(obs[:,s.header['entity_count'][0]].max());bound=next(n for n in DEFAULT_BUCKETS if maximum_count<=n)
        c={'clip':.2,'value_coef':.5,'entropy_coef':.001,'max_grad_norm':.5}
        a=SimpleNamespace(model=dense,encoder=DeviceBatchEncoder(s,copy=False),schema=s,c=c,optimizer=torch.optim.Adam(dense.parameters(),lr=.0003,eps=1e-5))
        b=SimpleNamespace(model=bucket,encoder=DeviceBatchEncoder(s,copy=False),schema=s,c=c,optimizer=torch.optim.Adam(bucket.parameters(),lr=.0003,eps=1e-5))
        da=dense(a.encoder(fields['obs'],fields['mask']))
        db=bucket(DeviceBatchEncoder(s,entity_rows=bound,copy=False)(fields['obs'],fields['mask']))
        forward=max(float((x-y).detach().abs().max()) for x,y in zip(da,db))
        for x,y in zip(da,db):torch.testing.assert_close(x,y,atol=2e-6,rtol=2e-5)
        # Real PPO log probabilities prevent manufactured exp overflow.
        with torch.no_grad():fields['logp']=dense.distribution(a.encoder(fields['obs'],fields['mask']))[0].log_prob(fields['action'])
        ma=eager_minibatch(a,fields);mb=eager_bucket_minibatch(b,fields,bound)
        torch.testing.assert_close(ma,mb,atol=2e-6,rtol=2e-5)
        maximum=_check_values(b,_values(a,gradients=True),atol=2e-6,rtol=2e-5)
        gradient=max(float((pa.grad-pb.grad).abs().max()) for pa,pb in zip(dense.parameters(),bucket.parameters()))
        result.append({'population':'unmodified_real_public_observations' if count is None else count,'maximum_count':maximum_count,'bucket':bound,'forward_max_absolute_difference':forward,'clipped_gradient_max_absolute_difference':gradient,'post_adam_model_max_absolute_difference':maximum})
    out.mkdir(parents=True,exist_ok=False)
    summary={'status':'passed','device':'cpu','compact_embedding_grad':compact,'source_real_public_observations':str(source),'boundary_cases':result,'atol':2e-6,'rtol':2e-5,'scope':'dense-vs-bucket neural/PPO semantics, synthetic boundaries populated only from actual public entity rows; conditional CUDA routing and exact bucket-mode resume NOT yet qualified','GPU_launched':False,'timing_comparison':False}
    (out/'result.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary))


if __name__=='__main__':
    import argparse,sys
    here=Path(__file__).resolve().parent;sys.path.insert(0,str(here.parent))
    parser=argparse.ArgumentParser();parser.add_argument('--cpu-check',action='store_true');parser.add_argument('--compact-embedding-grad',action='store_true');parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    if not args.cpu_check:parser.error('use --cpu-check locally; GPU qualification is called only on an explicitly allocated worker')
    cpu_semantic_qualification(args.out,compact=args.compact_embedding_grad)
