"""Actual allocated-GPU precision qualification from native PUBLIC rollouts.

Run explicitly on an already allocated worker. This script never launches Modal
or another worker, changes training defaults, or claims equal policy strength.
"""
from pathlib import Path
import argparse,copy,hashlib,json,os,sys,time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE));sys.path.insert(0,str(HERE.parent))


def _forward_check(actual,expected,mask,limits):
    import torch
    mask=mask.bool();errors={}
    for name,a,b in [('legal_logits',actual[0][mask],expected[0][mask]),('values',actual[1],expected[1]),('legal_log_probabilities',actual[0].log_softmax(-1)[mask],expected[0].log_softmax(-1)[mask])]:
        torch.testing.assert_close(a,b,atol=limits['output_atol'],rtol=limits['output_rtol'],msg=name)
        errors[name]=float((a-b).abs().max()) if a.numel() else 0.
    assert (actual[0][~mask]<-1e8).all() and (expected[0][~mask]<-1e8).all()
    assert actual[0].dtype==actual[1].dtype==torch.float32
    return errors


def _boundary_observations(raw,s,population):
    """Neural fixture only; repeats exposed rows, never a simulator transition."""
    import torch
    value=raw.clone();rows=value[:,s.header_ints:].reshape(-1,s.max_entities,s.entity_ints)
    template=rows[:,:1].clone();rows.zero_()
    if population:rows[:,:population].copy_(template.expand(-1,population,-1))
    value[:,s.header['entity_count'][0]]=population
    return value


def qualify_bucket_actor_precision(factory,raw,masks,s,mode):
    import torch
    from bucket_actor import BucketActor
    from gpu_features import DeviceBatchEncoder
    from precision import configure_precision,TOLERANCES,_objective,_rng,_restore_rng,_exact
    before_rng=_rng();before_tf32=torch.backends.cuda.matmul.allow_tf32;actor=None
    model=factory();metadata=configure_precision(model,mode);encoder=DeviceBatchEncoder(s,copy=False)
    limits=TOLERANCES[mode].copy()
    # Shape-only rounding tolerance already exists in strict-FP32 bucket mode.
    limits['output_atol']=max(limits['output_atol'],2e-6);limits['output_rtol']=max(limits['output_rtol'],2e-5)
    try:
        rng=_rng();actor=BucketActor(model,len(raw),graph_sampler=True);_exact(_rng(),rng,'bucket actor construction RNG')
        reports=[]
        for label,state in [('real-public',raw)]+[(str(n),_boundary_observations(raw,s,n)) for n in (0,32,33,64,65,96,97,200)]:
            with torch.no_grad():
                expected=model(encoder(state,masks));actual=actor.forward(state,masks)
                errors=_forward_check(actual,expected,masks,limits)
            selected=actor._branch.selection.cpu().tolist();maximum=int(state[:,s.header['entity_count'][0]].max().cpu())
            expected_index=next(i for i,bound in enumerate((32,64,96,200)) if maximum<=bound)
            assert selected==[expected_index,maximum,0]
            reports.append({'fixture':label,'selection':selected,'maximum_absolute_errors':errors})
        with torch.no_grad():old_values=actor.forward(raw,masks)[1].clone()
        optimizer=torch.optim.Adam(model.parameters(),lr=.003,eps=1e-5,fused=True)
        optimizer.zero_grad(set_to_none=True);loss,_=_objective(model,encoder(raw,masks));loss.backward()
        assert not torch.is_autocast_enabled('cuda');torch.nn.utils.clip_grad_norm_(model.parameters(),.5);optimizer.step()
        with torch.no_grad():
            expected=model(encoder(raw,masks));actual=actor.forward(raw,masks)
            updated_errors=_forward_check(actual,expected,masks,limits)
        assert not torch.equal(old_values,actual[1]),'conditional actor did not observe updated FP32 weights'
        rng=_rng()
        with torch.no_grad():expected_sample=tuple(x.clone() for x in actor.sample(raw,masks))
        expected_rng=_rng();_restore_rng(rng)
        with torch.no_grad():actual_sample=tuple(x.clone() for x in actor.sample(raw,masks))
        _exact(actual_sample,expected_sample,'in-mode bucket sampler replay');_exact(_rng(),expected_rng,'in-mode bucket sampler RNG')
        return {'status':'passed','mode':mode,'metadata':metadata,'boundary_fixtures':reports,'post_adam_maximum_absolute_errors':updated_errors,'captured_actor_observes_updated_fp32_weights':True,'in_mode_sampler_rng_replay_exact':True,'conditional_build':actor._branch.build_binding,'tolerances':limits,'full_trainer_exact_resume_still_required':True}
    finally:
        if actor is not None:actor.close()
        _restore_rng(before_rng);torch.backends.cuda.matmul.allow_tf32=before_tf32


def qualify_bucket_learner_precision(factory,fields,s,mode,c):
    """Exact SAME-BUCKET eager/conditional updates, not loose BF16-vs-FP32 tests."""
    import torch
    from types import SimpleNamespace
    from gpu_features import DeviceBatchEncoder
    from bucket_learner import BucketLearnerGraph,eager_bucket_minibatch,_values,_check_values
    from learner_graph import _TrainingSnapshot
    from precision import configure_precision,_rng,_restore_rng,_exact
    before_rng=_rng();before_tf32=torch.backends.cuda.matmul.allow_tf32;graph=None
    model=factory();metadata=configure_precision(model,mode)
    optimizer=torch.optim.Adam(model.parameters(),lr=.003,eps=1e-5,fused=True,capturable=True)
    trainer=SimpleNamespace(model=model,optimizer=optimizer,schema=s,encoder=DeviceBatchEncoder(s,copy=False),c=dict(c,rollout_graph=False))
    examples=[('real-public',fields)]
    for population in (0,32,33,64,65,96,97,200):
        example={name:value.clone() for name,value in fields.items()};example['obs']=_boundary_observations(example['obs'],s,population)
        examples.append((str(population),example))
    snapshot=_TrainingSnapshot(trainer)
    try:
        references=[]
        for label,example in examples:
            maximum=int(example['obs'][:,s.header['entity_count'][0]].max().cpu());bound=next(n for n in (32,64,96,200) if maximum<=n)
            metrics=eager_bucket_minibatch(trainer,example,bound).clone();references.append((metrics,_values(trainer,gradients=True),maximum,bound))
        snapshot.restore();torch.cuda.synchronize();rng=_rng()
        graph=BucketLearnerGraph(trainer,fields);_exact(_rng(),rng,'bucket learner construction RNG')
        reports=[]
        for (label,example),(expected_metrics,expected_values,maximum,bound) in zip(examples,references):
            actual=graph(example).clone();torch.testing.assert_close(actual,expected_metrics,atol=0.,rtol=0.)
            _check_values(trainer,expected_values,atol=0.,rtol=0.)
            selected=graph.selection.cpu().tolist();assert selected==[(32,64,96,200).index(bound),maximum,0]
            reports.append({'fixture':label,'selection':selected,'metrics_model_adam_clipped_gradients_exact':True})
        return {'status':'passed','mode':mode,'metadata':metadata,'same_bucket_eager_vs_graph':reports,'conditional_build':graph.build_binding,'comparison_atol':0.,'comparison_rtol':0.,'full_trainer_exact_resume_still_required':True}
    finally:
        if graph is not None:graph.close()
        snapshot.restore();_restore_rng(before_rng);torch.backends.cuda.matmul.allow_tf32=before_tf32


def main(args):
    started=time.perf_counter();args.out.mkdir(parents=True,exist_ok=False)
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise RuntimeError('exactly one already allocated CUDA device required; no CPU fallback/cloud launch')
    from train import Trainer
    from model import Policy
    from precision import qualify_precision
    config=json.loads(args.config.read_text())
    config.update(backend='native-cuda',device='cuda',precision='fp32',envs=args.envs,rollout_steps=args.rollout_steps,epochs=1,max_ticks=0,opponent_mode='selfplay',warmstart_steps=0,reward_shaping=0.,target_kl=0.,graph_sampling=False,rollout_graph=False,learner_graph=False,bucket_actor=False,bucket_learner=False,compact_embedding_grad=args.compact_embedding_grad,optimizer_fused=True,torch_threads=1)
    if args.width is not None:config['width']=args.width
    if 'hardware_allocation' in config:config['hardware_allocation'].update(gpu_model=torch.cuda.get_device_name(),gpu_count=1,cpu_threads=int(os.getenv('CLASH_CPU_CORES','4')),ram_gib=int(os.getenv('CLASH_RAM_GIB','16')))
    (args.out/'collection-config.json').write_text(json.dumps(config,indent=2)+'\n')
    trainer=Trainer(config,args.data)
    try:
        data=trainer.collect();active=torch.nonzero(data['mask'].sum(-1)>1).flatten();forced=torch.nonzero(data['mask'].sum(-1)==1).flatten()
        if not active.numel() or not forced.numel():raise AssertionError('real rollout must expose active and forced-wait public states')
        n=args.batch_size
        if n<4:raise ValueError('batch-size must be at least four')
        selected=torch.cat((active[torch.arange(n//2,device=active.device)%active.numel()],forced[torch.arange(n-n//2,device=forced.device)%forced.numel()]))
        # Include the busiest public state and one actual stun state when present.
        count_index=trainer.schema.header['entity_count'][0];selected[1]=data['obs'][:,count_index].argmax()
        entity=data['obs'][:,trainer.schema.header_ints:].reshape(-1,trainer.schema.max_entities,trainer.schema.entity_ints)
        buffs=entity[..., [trainer.schema.entity[f'buff_{i}'] for i in range(trainer.schema.buff_slots)]]
        stun=torch.nonzero((buffs==244).any(-1).any(-1)).flatten()
        if stun.numel():selected[-1]=stun[0]
        raw=data['obs'][selected].clone();masks=data['mask'][selected].clone();packed=trainer.encoder(raw,masks)
        selected_active=int((masks.sum(-1)>1).sum().cpu());selected_forced=int((masks.sum(-1)==1).sum().cpu())
        assert selected_active and selected_forced,'selected public minibatch must retain active and forced rows'
        weights={name:value.detach().clone() for name,value in trainer.model.state_dict().items()}
        def factory():
            model=Policy(trainer.schema,len(trainer.data['vocabulary']),len(trainer.data['buff_names']),config['width'],config['embedding']).to(trainer.device)
            if args.compact_embedding_grad:
                from compact_embeddings import install
                install(model,trainer.data)
            model.load_state_dict(weights);model.device_indices=True;model.graph_sampling=False
            return model
        mode_report=qualify_precision(factory,packed,args.mode)
        (args.out/'precision.json').write_text(json.dumps(mode_report,indent=2)+'\n')
        results={'precision':mode_report}
        if args.bucket_actor:results['bucket_actor']=qualify_bucket_actor_precision(factory,raw,masks,trainer.schema,args.mode)
        if args.bucket_learner:
            fields={name:data[key][selected].clone() for name,key in (('obs','obs'),('mask','mask'),('action','action'),('logp','logp'),('return','return'))}
            advantage=data['advantage'];weight=(data['mask'].sum(-1)>1).float() if config.get('active_policy_loss',True) else torch.ones_like(advantage)
            norm=torch.where(weight.sum()>0,weight,torch.ones_like(weight));mean=(advantage*norm).sum()/norm.sum();std=torch.sqrt(((advantage-mean).square()*norm).sum()/norm.sum())
            fields.update(advantage=((advantage-mean)/(std+1e-8))[selected],weight=weight[selected])
            results['bucket_learner']=qualify_bucket_learner_precision(factory,fields,trainer.schema,args.mode,config)
        result={'status':'passed','mode':args.mode,'gpu':torch.cuda.get_device_name(),'collection_precision':'fp32','compact_embedding_grad':args.compact_embedding_grad,'real_rollout_samples':data['action'].numel(),'active_rollout_samples':active.numel(),'forced_rollout_samples':forced.numel(),'stun_public_rows':stun.numel(),'selected_public_rows':selected.cpu().tolist(),'selected_active_rows':selected_active,'selected_forced_rows':selected_forced,'selected_max_entities':int(raw[:,count_index].max().cpu()),'results':results,'source_binding':trainer.binding,'engine_build':trainer.backend_build,'execution_runtime':trainer.execution_runtime,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'process_setup_compile_collection_qualification_seconds':time.perf_counter()-started,'full_trainer_exact_resume_still_required':True,'equal_strength_claimed':False,'timing_comparison':False}
        (args.out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps({key:value for key,value in result.items() if key not in ('source_binding','engine_build','results')}),flush=True)
    finally:trainer.env.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True);parser.add_argument('--mode',choices=['fp32','tf32','bf16'],required=True);parser.add_argument('--config',type=Path,default=HERE.parent/'configs/cuda-learning.json');parser.add_argument('--data',type=Path,default=HERE/'generated-v2/task-data.json');parser.add_argument('--envs',type=int,default=4);parser.add_argument('--rollout-steps',type=int,default=96);parser.add_argument('--batch-size',type=int,default=32);parser.add_argument('--width',type=int);parser.add_argument('--compact-embedding-grad',action='store_true');parser.add_argument('--bucket-actor',action='store_true');parser.add_argument('--bucket-learner',action='store_true');main(parser.parse_args())
