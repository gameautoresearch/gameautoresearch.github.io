"""Bounded CPU/CUDA qualification for optional complete compiled PPO loss.

No cloud launcher or timing comparison. Dense/compact modes run in separate
processes. CUDA controls cover eager values/gradients/Adam, full-update capture,
all four bucket contexts, default full48/tail16 or explicit production shapes,
and reuse after real existing Adam updates.
Independent-process full Trainer resume remains a separate test_training check.
"""
from __future__ import annotations
from pathlib import Path
import argparse,json,os,subprocess,sys,time,traceback
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE));sys.path.insert(0,str(HERE.parent))

def dump(path,value):Path(path).write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')

def compiler_diagnostics():
    import torch._dynamo.config as config
    from torch._dynamo.utils import counters
    return {'limits':{name:getattr(config,name,None) for name in ('recompile_limit','accumulated_recompile_limit')},'counters':{name:dict(values) for name,values in counters.items()}}

def minibatch_cases(full_minibatch=48,tail_minibatch=16):
    if type(full_minibatch) is not int or type(tail_minibatch) is not int or not 2<=tail_minibatch<full_minibatch:raise ValueError('two distinct minibatch sizes with full > tail >= 2 required')
    return (('forced',full_minibatch),('mixed',full_minibatch),('mixed',tail_minibatch),('mixed',full_minibatch),('forced',tail_minibatch),('mixed',tail_minibatch))


def make_minibatches(data,advantage,weight,active,forced,*,full_minibatch=48,tail_minibatch=16):
    """Repeat genuine public rows when requested batches exceed unique rows."""
    import torch
    if active.numel()<1 or forced.numel()<1:raise AssertionError('real rollout lacks active/forced samples')
    def choose(values,n,offset):return values[(torch.arange(n,device=values.device)+offset)%values.numel()]
    minibatches=[];sources=[]
    for j,(kind,n) in enumerate(minibatch_cases(full_minibatch,tail_minibatch)):
        if kind=='forced':ix=choose(forced,n,j*7)
        else:
            na=min(active.numel(),n//2);ix=torch.cat((choose(active,na,j*3),choose(forced,n-na,j*7)))
        fields={name:data[name][ix] for name in ('obs','mask','action','logp','return')};fields.update(advantage=advantage[ix],weight=weight[ix])
        minibatches.append(fields);sources.append({'kind':kind,'size':n,'source_indices':ix.cpu().tolist(),'active_samples':int(fields['weight'].sum().cpu())})
    return minibatches,sources


def assert_static_entries(summary,full_minibatch=48,tail_minibatch=16):
    minibatch_cases(full_minibatch,tail_minibatch)
    expected={(rows,size) for rows in (32,64,96,200) for size in (full_minibatch,tail_minibatch)}
    actual=[(entry['entity_rows'],next(layout['shape'][0] for layout in entry['input_layouts'] if layout['name']=='action')) for entry in summary['entries']]
    assert len(actual)==len(expected)==8 and set(actual)==expected,('four buckets times two requested minibatch shapes must require exactly eight entries',actual,sorted(expected))


def qualify_cuda(out,config_path,data_path,*,compact=False,width=32,full_minibatch=48,tail_minibatch=16):
    import torch
    from train import Trainer
    from learner_graph import eager_minibatch,_TrainingSnapshot,qualify_learner_graph
    from bucket_learner import _values,_check_values,qualify_bucket_learner
    from compiled_loss import cache_summary,compile_recipe
    if not torch.cuda.is_available():raise RuntimeError('actual CUDA hardware required; no cloud launch is performed')
    minibatch_cases(full_minibatch,tail_minibatch)
    if type(width) is not int or width<1:raise ValueError('positive CUDA policy width required')
    out=Path(out);out.mkdir(parents=True,exist_ok=False);started=time.perf_counter()
    config=json.loads(Path(config_path).read_text())
    config.update(backend='native-cuda',device='cuda',envs=4,width=width,minibatch=full_minibatch,
        rollout_steps=64,epochs=1,target_kl=0.,max_ticks=0,graph_sampling=False,
        rollout_graph=False,learner_graph=True,bucket_actor=False,bucket_learner=False,
        optimizer_fused=True,compile_loss=True,compact_embedding_grad=compact,
        opponent_mode='selfplay',warmstart_steps=0,active_policy_loss=True)
    if 'hardware_allocation' in config:config['hardware_allocation'].update(gpu_model=torch.cuda.get_device_name(),gpu_count=1,cpu_threads=int(os.getenv('CLASH_CPU_CORES','4')),ram_gib=int(os.getenv('CLASH_RAM_GIB','16')))
    dump(out/'config.json',config);trainer=None;phase='construction'
    try:
        trainer=Trainer(config,data_path)
        parameter_ids=tuple(id(p) for p in trainer.model.parameters());optimizer_id=id(trainer.optimizer);state_names=tuple(trainer.model.state_dict())
        log=(out/'progress.jsonl').open('w')
        def run(name,fn):
            nonlocal phase
            phase=name;log.write(json.dumps({'event':'start','phase':name})+'\n');log.flush()
            result=fn();torch.cuda.synchronize()
            record={'phase':name,'result':result,'compiled_cache':cache_summary(trainer)}
            dump(out/(name+'.json'),record);log.write(json.dumps({'event':'passed','phase':name,'cache_entries':len(record['compiled_cache']['entries'])})+'\n');log.flush()
            print(json.dumps({'event':'passed','phase':name,'cache_entries':len(record['compiled_cache']['entries'])}),flush=True)
            return record
        data=trainer.collect();active=torch.nonzero(data['mask'].sum(-1)>1,as_tuple=False).flatten();forced=torch.nonzero(data['mask'].sum(-1)==1,as_tuple=False).flatten()
        if active.numel()<1 or forced.numel()<1:raise AssertionError('real rollout lacks active/forced samples')
        weight=(data['mask'].sum(-1)>1).float();norm=torch.where(weight.sum()>0,weight,torch.ones_like(weight));count=norm.sum()
        mean=(data['advantage']*norm).sum()/count;std=torch.sqrt(((data['advantage']-mean).square()*norm).sum()/count);advantage=(data['advantage']-mean)/(std+1e-8)
        minibatches,sources=make_minibatches(data,advantage,weight,active,forced,full_minibatch=full_minibatch,tail_minibatch=tail_minibatch)
        populations=[];bucket_batches=list(minibatches)
        for population in (0,32,33,64,65,96,97,200):
            fields={name:value.clone() for name,value in minibatches[1].items()}
            rows=fields['obs'][:,trainer.schema.header_ints:].reshape(-1,200,trainer.schema.entity_ints)
            template=rows[:,:1].clone();rows.zero_()
            if population:rows[:,:population].copy_(template.expand(-1,population,-1))
            fields['obs'][:,trainer.schema.header['entity_count'][0]]=population
            bucket_batches.append(fields);populations.append(population)
        dump(out/'minibatches.json',{'real':sources,'synthetic_neural_boundary_populations':populations,'source':'actual native public rollout; boundary rows repeat exposed rows; no simulator transition is changed'})
        def eager_control():
            saved=_TrainingSnapshot(trainer);original=trainer.c.get('compile_loss',False);references=[];maximum_model=maximum_metric=maximum_gradient=0.
            try:
                trainer.c['compile_loss']=False
                for fields in minibatches:references.append((eager_minibatch(trainer,fields).clone(),_values(trainer,gradients=True)))
                saved.restore();torch.cuda.synchronize();trainer.c['compile_loss']=True
                for fields,(expected,want) in zip(minibatches,references):
                    actual=eager_minibatch(trainer,fields).clone()
                    torch.testing.assert_close(actual,expected,atol=2e-6,rtol=2e-5)
                    maximum_metric=max(maximum_metric,float((actual-expected).abs().max().cpu()))
                    maximum_model=max(maximum_model,_check_values(trainer,want,atol=2e-6,rtol=2e-5))
                    for p,g in zip(trainer.model.parameters(),want['gradients']):
                        if g is not None:maximum_gradient=max(maximum_gradient,float((p.grad-g).abs().max().cpu()))
                return {'status':'passed','steps':len(minibatches),'reference_compile_loss':False,'candidate_compile_loss':True,'atol':2e-6,'rtol':2e-5,'maximum_metric_difference':maximum_metric,'maximum_post_adam_model_difference':maximum_model,'maximum_clipped_gradient_difference':maximum_gradient,'model_buffers_gradients_adam_compared_each_step':True}
            finally:saved.restore();trainer.c['compile_loss']=original;torch.cuda.synchronize()
        reports={}
        reports['fresh-eager']=run('fresh-eager',eager_control)
        reports['fresh-graph']=run('fresh-graph',lambda:qualify_learner_graph(trainer,minibatches,atol=0.,rtol=0.))
        reports['fresh-bucket']=run('fresh-bucket',lambda:qualify_bucket_learner(trainer,bucket_batches,atol=2e-6,rtol=2e-5))
        assert_static_entries(cache_summary(trainer),full_minibatch,tail_minibatch)
        # These are real updates, outside any constructor/warmup. All later
        # controls start from nonzero moments and step2, using the SAME Adam.
        eager_minibatch(trainer,minibatches[1]);eager_minibatch(trainer,minibatches[2]);torch.cuda.synchronize()
        for state in trainer.optimizer.state.values():assert float(state['step'].cpu())==2.
        state_addresses={(id(p),name):value.data_ptr() for p,state in trainer.optimizer.state.items() for name,value in state.items() if torch.is_tensor(value)}
        reports['existing-eager']=run('existing-eager',eager_control)
        reports['existing-graph']=run('existing-graph',lambda:qualify_learner_graph(trainer,minibatches,atol=0.,rtol=0.))
        reports['existing-bucket']=run('existing-bucket',lambda:qualify_bucket_learner(trainer,bucket_batches,atol=2e-6,rtol=2e-5))
        assert_static_entries(cache_summary(trainer),full_minibatch,tail_minibatch)
        for p,state in trainer.optimizer.state.items():
            for name,value in state.items():
                if torch.is_tensor(value):assert state_addresses[(id(p),name)]==value.data_ptr()
        assert tuple(id(p) for p in trainer.model.parameters())==parameter_ids and id(trainer.optimizer)==optimizer_id
        assert tuple(trainer.model.state_dict())==state_names
        result={'status':'passed','cuda_executed':True,'gpu':torch.cuda.get_device_name(),'compact_embedding_grad':compact,'model_width':config['width'],'model_embedding':config['embedding'],'full_minibatch':full_minibatch,'tail_minibatch':tail_minibatch,'recipe':compile_recipe(),'phases':reports,'real_minibatches':sources,'bucket_boundary_populations':populations,'same_parameter_optimizer_and_state_dict_identity':True,'existing_adam_addresses_preserved':True,'cache':cache_summary(trainer),'dynamo_cache_limit_modified':False,'dynamo_diagnostics':compiler_diagnostics(),'binding':trainer.binding,'execution_runtime':trainer.execution_runtime,'setup_compile_qualification_seconds':time.perf_counter()-started,'timing_comparison':False,'independent_process_full_trainer_resume':'separate test_training --compile-loss required'}
        dump(out/'result.json',result);log.close();return result
    except Exception as exc:
        dump(out/'failure.json',{'status':'failed','phase':phase,'exception_type':type(exc).__name__,'error':str(exc),'traceback':traceback.format_exc(),'dynamo_diagnostics':compiler_diagnostics(),'cache':cache_summary(trainer) if trainer is not None else None,'timing_comparison':False})
        raise
    finally:
        if trainer is not None:
            try:trainer.close()
            except Exception:
                if not (out/'failure.json').exists():raise


def parse_args(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--config',type=Path,default=HERE.parent/'configs/cuda-learning.json')
    parser.add_argument('--data',type=Path,default=HERE/'generated-v2/task-data.json')
    parser.add_argument('--fixture',type=Path)
    parser.add_argument('--cpu-only',action='store_true')
    parser.add_argument('--embedding',choices=['both','dense','compact'],default='both')
    parser.add_argument('--width',type=int,default=32,help='CUDA policy width; embedding dimension remains from config')
    parser.add_argument('--full-minibatch',type=int,default=48,help='CUDA full minibatch rows; repeated public rows are allowed')
    parser.add_argument('--tail-minibatch',type=int,default=16,help='CUDA smaller minibatch rows')
    parser.add_argument('--timeout-seconds',type=float,default=600.)
    parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main():
    args=parse_args()
    minibatch_cases(args.full_minibatch,args.tail_minibatch)
    if args.width<1:raise ValueError('positive policy width required')
    if args.cpu_only and (args.width,args.full_minibatch,args.tail_minibatch)!=(32,48,16):raise ValueError('production shape flags require actual CUDA qualification; --cpu-only retains its separate small historical control')
    if args.timeout_seconds<=0:raise ValueError('positive bounded qualification window required')
    if args.worker and args.embedding=='both':raise ValueError('internal worker requires one embedding mode')
    if not args.worker:
        args.out.mkdir(parents=True,exist_ok=False);started=time.monotonic();results={}
        for mode in (('dense','compact') if args.embedding=='both' else (args.embedding,)):
            remaining=args.timeout_seconds-(time.monotonic()-started)
            if remaining<=0:
                dump(args.out/'failure.json',{'status':'failed','embedding':mode,'exception_type':'TimeoutError','error':'bounded qualifier exhausted its allocation window'})
                raise TimeoutError('bounded qualifier exhausted its allocation window')
            cmd=[sys.executable,str(Path(__file__).resolve()),'--out',str(args.out/mode),'--embedding',mode,'--config',str(args.config),'--data',str(args.data),'--width',str(args.width),'--full-minibatch',str(args.full_minibatch),'--tail-minibatch',str(args.tail_minibatch),'--timeout-seconds',str(remaining),'--worker']
            if args.cpu_only:cmd+=['--cpu-only']
            if args.fixture:cmd+=['--fixture',str(args.fixture)]
            try:
                with (args.out/(mode+'.log')).open('w') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=remaining)
            except Exception as exc:
                dump(args.out/'failure.json',{'status':'failed','embedding':mode,'exception_type':type(exc).__name__,'error':str(exc),'log':mode+'.log'});raise
            results[mode]=json.loads((args.out/mode/'result.json').read_text())
        summary={'status':'passed','cuda_executed':not args.cpu_only,'separate_embedding_processes':True,'results':results,'timing_comparison':False}
        dump(args.out/'result.json',summary);print(json.dumps({'status':'passed','embedding_modes':list(results),'cuda_executed':not args.cpu_only}),flush=True)
    elif args.cpu_only:
        from compiled_loss import cpu_qualification
        cpu_qualification(args.out,fixture=args.fixture,compact=args.embedding=='compact',integrated=True)
    else:
        result=qualify_cuda(args.out,args.config,args.data,compact=args.embedding=='compact',width=args.width,full_minibatch=args.full_minibatch,tail_minibatch=args.tail_minibatch)
        print(json.dumps({'status':result['status'],'compact_embedding_grad':result['compact_embedding_grad'],'phases':list(result['phases'])}),flush=True)
if __name__=='__main__':main()
