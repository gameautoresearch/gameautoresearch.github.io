"""Optional SAME-precision eager/Inductor + exact combined-mode resume qualifier.

Run only on an already allocated GPU; this helper launches no cloud workers and
changes no existing integration/default. Strict FP32 stays the production default.
BF16/TF32 compiled losses are compared to SAME-mode eager at the existing strict
2e-6/2e-5 compiled-loss tolerance. Failures retain generated dtype/cast evidence;
this helper never silently relaxes BF16 bounds or claims equal policy strength.
"""
from pathlib import Path
import argparse,copy,hashlib,json,os,signal,subprocess,sys,time,traceback
from contextlib import contextmanager
from unittest.mock import patch
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE));sys.path.insert(0,str(HERE.parent))


def dump(path,value):Path(path).write_text(json.dumps(value,indent=2)+'\n')


def code_inventory(codes):
    """Actual generated code lines, not a claim inferred from requested mode."""
    result={'bfloat16_cast_or_allocation':[],'float32_cast_or_allocation':[],'matmul_precision':[]}
    for index,code in enumerate(codes):
        for line in code.splitlines():
            if line.lstrip().startswith('#'):continue
            if ('bfloat16' in line or 'bf16' in line) and any(word in line for word in ('empty_strided','alloc','to(','to_dtype','extern_kernels','dot(')):result['bfloat16_cast_or_allocation'].append({'source':index,'line':line.strip()})
            if ('float32' in line or 'fp32' in line) and any(word in line for word in ('empty_strided','alloc','to(','to_dtype','extern_kernels','dot(')):result['float32_cast_or_allocation'].append({'source':index,'line':line.strip()})
            if any(word in line for word in ('allow_tf32','ALLOW_TF32','input_precision','extern_kernels.mm','extern_kernels.addmm')):result['matmul_precision'].append({'source':index,'line':line.strip()})
    return result


@contextmanager
def capture_inductor_sources(codes):
    """Observe cold generation AND hot cached module loads without cache resets."""
    from torch._inductor.graph import GraphLowering
    from torch._inductor.codecache import PyCodeCache
    original=PyCodeCache.load_by_key_path;seen=set()
    def retain(code):
        if isinstance(code,str):
            digest=hashlib.sha256(code.encode()).hexdigest()
            if digest not in seen:codes.append(code);seen.add(digest)
    def load(*args,**kwargs):
        module=original(*args,**kwargs);name=getattr(module,'__file__',None)
        if name and Path(name).is_file():retain(Path(name).read_text())
        return module
    with patch.object(GraphLowering,'save_output_code',retain),patch.object(PyCodeCache,'load_by_key_path',side_effect=load):yield


def numerical_worker(args):
    import torch
    import torch._functorch.config as functorch_config
    from types import SimpleNamespace
    from train import Trainer,binding
    from learner_graph import eager_minibatch,_TrainingSnapshot,qualify_learner_graph
    from bucket_learner import _values,_check_values,_encoder
    from gpu_features import DeviceBatchEncoder
    from compiled_loss import compiled_ppo_loss,cache_summary,compile_recipe
    from qualify_compiled_loss import make_minibatches,assert_static_entries
    from precision import _rng,_exact,matmul_runtime_flags
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise RuntimeError('one already allocated GPU required; no cloud launch/CPU fallback')
    args.out.mkdir(parents=True,exist_ok=False);source=binding(args.data);codes=[];records=[];trainer=None
    config=json.loads(args.config.read_text());config.update(backend=args.backend,device='cuda',precision=args.precision,compile_loss=True,envs=4,threads=args.cpu_threads,torch_threads=1,width=64,embedding=12,minibatch=args.full_minibatch,rollout_steps=96,epochs=1,target_kl=0.,max_ticks=0,graph_sampling=False,rollout_graph=False,learner_graph=True,bucket_actor=False,bucket_learner=False,optimizer_fused=True,opponent_mode='selfplay',warmstart_steps=0,reward_shaping=0.,compact_embedding_grad=args.compact_embedding_grad)
    config['hardware_allocation']={'gpu_model':torch.cuda.get_device_name(),'gpu_count':1,'cpu_threads':args.cpu_threads,'ram_gib':int(os.getenv('CLASH_RAM_GIB','64'))}
    if args.backend=='native-cuda':config.update(cuda_launch_threads=1,cuda_render_mode=4)
    dump(args.out/'config.json',config)
    try:
        trainer=Trainer(config,args.data);data=trainer.collect();active=torch.nonzero(data['mask'].sum(-1)>1).flatten();forced=torch.nonzero(data['mask'].sum(-1)==1).flatten()
        if not active.numel() or not forced.numel():raise AssertionError('real public rollout needs both active and forced rows')
        weight=(data['mask'].sum(-1)>1).float();norm=torch.where(weight.sum()>0,weight,torch.ones_like(weight));mean=(data['advantage']*norm).sum()/norm.sum();std=torch.sqrt(((data['advantage']-mean).square()*norm).sum()/norm.sum());advantage=(data['advantage']-mean)/(std+1e-8)
        minibatches,sources=make_minibatches(data,advantage,weight,active,forced,full_minibatch=args.full_minibatch,tail_minibatch=args.tail_minibatch)
        # Separate descriptor: no mutation of Trainer's frozen production recipe.
        core=SimpleNamespace(model=trainer.model,optimizer=trainer.optimizer,encoder=trainer.encoder,schema=trainer.schema,c=dict(trainer.c))
        cases=[(f'real-{index}',200,fields) for index,fields in enumerate(minibatches)]
        for rows in (32,64,96,200):
            for size,template in ((args.full_minibatch,minibatches[1]),(args.tail_minibatch,minibatches[2])):
                fields={name:value.clone() for name,value in template.items()};entities=fields['obs'][:,trainer.schema.header_ints:].reshape(-1,200,trainer.schema.entity_ints);exposed=entities[:,:1].clone();entities.zero_();entities[:,:rows].copy_(exposed.expand(-1,rows,-1));fields['obs'][:,trainer.schema.header['entity_count'][0]]=rows
                cases.append((f'neural-boundary-{rows}x{size}',rows,fields))
        dump(args.out/'fixtures.json',{'real_minibatches':sources,'cases':[{'label':label,'entity_rows':rows,'minibatch':len(fields['action'])} for label,rows,fields in cases],'synthetic_scope':'boundary fixtures repeat real exposed rows; no simulator transition/rule is modified'})
        saved=_TrainingSnapshot(core);references=[];parameter_names=[name for name,_ in core.model.named_parameters()]
        core.c['compile_loss']=False
        for label,rows,fields in cases:
            with _encoder(core,DeviceBatchEncoder(core.schema,entity_rows=rows,copy=False)):metrics=eager_minibatch(core,fields).clone()
            references.append((metrics,_values(core,gradients=True)))
        saved.restore();torch.cuda.synchronize();core.c['compile_loss']=True
        assert not torch.is_autocast_enabled('cuda'),'outer loss/autograd context must stay outside autocast'
        with capture_inductor_sources(codes):
            for (label,rows,fields),(expected,want) in zip(cases,references):
                before_rng=_rng()
                with _encoder(core,DeviceBatchEncoder(core.schema,entity_rows=rows,copy=False)):
                    loss,parts=compiled_ppo_loss(core,fields)
                    assert loss.dtype==torch.float32 and all(value.dtype==torch.float32 for value in parts)
                    assert not torch.is_autocast_enabled('cuda'),'compiled loss leaked the nested forward autocast scope'
                    torch._assert_async(torch.isfinite(loss),'nonfinite combined precision/compiled loss');core.optimizer.zero_grad(set_to_none=True);loss.backward()
                    assert all(p.dtype==p.grad.dtype==torch.float32 for p in core.model.parameters())
                    grad=torch.nn.utils.clip_grad_norm_(core.model.parameters(),core.c['max_grad_norm']);core.optimizer.step();actual=torch.stack((*parts,grad.detach()))
                _exact(_rng(),before_rng,'compiled update must consume no research RNG')
                grad_errors=[{'parameter':name,'maximum_absolute_error':float((p.grad-g).abs().max())} for name,p,g in zip(parameter_names,core.model.parameters(),want['gradients'])]
                record={'case':label,'entity_rows':rows,'minibatch':len(fields['action']),'metric_maximum_absolute_error':float((actual-expected).abs().max()),'clipped_gradient_errors':grad_errors,'loss_dtype':str(loss.dtype),'outer_autocast_enabled':torch.is_autocast_enabled('cuda')}
                records.append(record);dump(args.out/'numerical-progress.json',records)
                torch.testing.assert_close(actual,expected,atol=2e-6,rtol=2e-5)
                _check_values(core,want,atol=2e-6,rtol=2e-5)
        saved.restore();torch.cuda.synchronize()
        assert_static_entries(cache_summary(core),args.full_minibatch,args.tail_minibatch)
        # Same COMBINED-mode compiled control, exact graph replay, including
        # graph constructor restoration and real optimizer updates after capture.
        graph=qualify_learner_graph(core,minibatches,atol=0.,rtol=0.)
        inspection=retain_sources(args.out,codes,args.precision)
        assert codes,'actual Inductor generated source was not observed; cast inspection remains unresolved'
        if args.precision=='bf16':assert inspection['inventory']['bfloat16_cast_or_allocation'],'requested BF16 was not observed in actual generated cast/allocation code'
        assert binding(args.data)==source,'source changed during combined-mode qualification'
        result={'status':'passed','precision':args.precision,'backend':args.backend,'width':64,'embedding':12,'compact_embedding_grad':args.compact_embedding_grad,'numerical_reference':'SAME precision eager, same model/Adam/data; not FP32 baseline','atol':2e-6,'rtol':2e-5,'tolerance_widened_for_bf16':False,'cases':records,'compiled_mode_graph_exact':graph,'inspection':inspection,'compile_recipe':compile_recipe(),'cache':cache_summary(core),'precision_runtime':trainer.execution_runtime['precision'],'matmul_flags':matmul_runtime_flags(),'source_binding':trainer.binding,'backend_build':trainer.backend_build,'auxiliary_builds':trainer.auxiliary_builds,'execution_runtime':trainer.execution_runtime,'full_combined_trainer_resume_required':True,'timing_comparison':False,'equal_strength_claimed':False}
        dump(args.out/'result.json',result)
    except Exception as exc:
        inspection=retain_sources(args.out,codes,args.precision)
        dump(args.out/'failure.json',{'status':'failed','precision':args.precision,'exception_type':type(exc).__name__,'error':str(exc),'traceback':traceback.format_exc(),'completed_cases':records,'inspection':inspection,'backward_pass_autocast':str(functorch_config.backward_pass_autocast),'tolerance_widened':False});raise
    finally:
        if trainer is not None:trainer.close()


def retain_sources(out,codes,precision):
    import torch._functorch.config as config
    folder=out/'inductor-generated';folder.mkdir(exist_ok=True);files=[]
    for index,code in enumerate(codes):
        digest=hashlib.sha256(code.encode()).hexdigest();path=folder/f'{index:02d}-{digest[:12]}.py'
        if not path.exists():path.write_text(code)
        files.append({'file':path.name,'sha256':digest})
    report={'precision':precision,'actual_generated_sources':files,'inventory':code_inventory(codes),'backward_pass_autocast':str(config.backward_pass_autocast),'outer_forward_context':'autocast off; BF16 forward context is nested inside Policy.forward and exits before PPO/Categorical loss','cache_reset_or_compile_options_overridden':False,'interpretation':'Actual dtype/cast source retained for diagnosis; requested precision alone does not prove dtype lowering. No BF16 tolerance loosening.'}
    dump(out/'cast-inspection.json',report);return report


def resume_worker(args):
    import torch
    import test_training
    # The existing qualifier has no CPU-thread CLI. Extend only its config
    # factory, preserving ALL comparisons/source guards/tamper checks unchanged.
    original=test_training.qualification_config
    def configured(parsed,gpu):
        config=original(parsed,gpu);config['threads']=args.cpu_threads
        config['hardware_allocation']={'gpu_model':torch.cuda.get_device_name(),'gpu_count':1,'cpu_threads':args.cpu_threads,'ram_gib':int(os.getenv('CLASH_RAM_GIB','64'))}
        return config
    command=['--out',str(args.out),'--data',str(args.data),'--backend',args.backend,'--envs',str(args.resume_envs),'--width','64','--embedding','12','--minibatch',str(args.full_minibatch),'--precision',args.precision,'--compile-loss','--learner-graph','--gae-kernel','triton']
    if args.compact_embedding_grad:command+=['--compact-embedding-grad']
    if args.bucket_actor:command+=['--bucket-actor']
    if args.bucket_learner:command+=['--bucket-learner']
    with patch.object(test_training,'qualification_config',configured):test_training.main(test_training.parse_args(command))


def run_child(command,log,remaining):
    with log.open('w') as output:
        process=subprocess.Popen(command,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
        try:code=process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            try:os.killpg(process.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            process.wait();raise TimeoutError('bounded combined-mode qualifier/process group exceeded remaining budget')
    if code:raise RuntimeError('combined-mode phase failed; see '+str(log))


def main(args):
    from train import binding
    from snapshots import available_cpu_workers
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise RuntimeError('one already allocated GPU required; this helper launches no cloud/GPU workers')
    if not 1<=args.cpu_threads<=available_cpu_workers():raise ValueError('CPU threads must fit attached allocation')
    if args.full_minibatch<=args.tail_minibatch or args.tail_minibatch<2 or args.resume_envs<1 or args.timeout_seconds<=0:raise ValueError('positive bounded production shape/window required')
    args.out.mkdir(parents=True,exist_ok=False);started=time.monotonic();source=binding(args.data);failure=None;completed=[]
    base=[sys.executable,str(Path(__file__).resolve()),'--precision',args.precision,'--backend',args.backend,'--config',str(args.config),'--data',str(args.data),'--cpu-threads',str(args.cpu_threads),'--full-minibatch',str(args.full_minibatch),'--tail-minibatch',str(args.tail_minibatch),'--resume-envs',str(args.resume_envs)]
    for flag,enabled in [('--compact-embedding-grad',args.compact_embedding_grad),('--bucket-actor',args.bucket_actor),('--bucket-learner',args.bucket_learner)]:
        if enabled:base.append(flag)
    try:
        phases=[('numerical','--numerical-worker')]+([] if args.skip_resume else [('resume','--resume-worker')])
        for name,flag in phases:
            remaining=args.timeout_seconds-(time.monotonic()-started)
            if remaining<1:raise TimeoutError('whole combined-mode qualifier budget exhausted')
            run_child(base+['--out',str(args.out/name),flag],args.out/(name+'.log'),remaining)
            assert binding(args.data)==source,'source changed between combined-mode qualification phases'
            completed.append(name)
    except BaseException as exc:failure={'type':type(exc).__name__,'message':str(exc)}
    result={'status':'failed' if failure else 'passed','error':failure,'precision':args.precision,'backend':args.backend,'completed_phases':completed,'combined_mode_qualified':not failure and not args.skip_resume,'exact_full_trainer_resume_completed':'resume' in completed,'requested_recipe':{'width':64,'embedding':12,'compile_loss':True,'precision':args.precision,'compact_embedding_grad':args.compact_embedding_grad,'bucket_actor':args.bucket_actor,'bucket_learner':args.bucket_learner,'cpu_threads':args.cpu_threads,'full_minibatch':args.full_minibatch,'tail_minibatch':args.tail_minibatch,'resume_envs':args.resume_envs},'source_binding':source,'timing_comparison':False,'equal_strength_claimed':False,'GPU_worker_launched':False}
    dump(args.out/'result.json',result);print(json.dumps({key:value for key,value in result.items() if key!='source_binding'}),flush=True)
    if failure:raise RuntimeError(failure['message'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);p.add_argument('--precision',choices=['fp32','tf32','bf16'],required=True);p.add_argument('--backend',choices=['native-cuda','native-cpu-cuda'],default='native-cuda');p.add_argument('--config',type=Path,default=HERE.parent/'configs/cuda-learning.json');p.add_argument('--data',type=Path,default=HERE/'generated-v2/task-data.json');p.add_argument('--cpu-threads',type=int,default=16);p.add_argument('--full-minibatch',type=int,default=1024);p.add_argument('--tail-minibatch',type=int,default=48);p.add_argument('--resume-envs',type=int,default=32);p.add_argument('--timeout-seconds',type=float,default=1500.);p.add_argument('--compact-embedding-grad',action='store_true');p.add_argument('--bucket-actor',action='store_true');p.add_argument('--bucket-learner',action='store_true');p.add_argument('--skip-resume',action='store_true');p.add_argument('--numerical-worker',action='store_true',help=argparse.SUPPRESS);p.add_argument('--resume-worker',action='store_true',help=argparse.SUPPRESS);args=p.parse_args()
    if args.numerical_worker:numerical_worker(args)
    elif args.resume_worker:resume_worker(args)
    else:main(args)
