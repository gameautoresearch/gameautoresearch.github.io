"""Exact fast-training CUDA-engine/CPU-engine qualification on ONE allocated GPU.

Both arms use the same GPU Policy64/e12/FP32, compact embeddings, device buckets,
Triton GAE, seed and PPO recipe. Only the integer simulator backend changes.
Arms run in separate, sequential subprocesses. No Modal/cloud job is launched;
this is correctness evidence, never a timing/throughput comparison.

Opaque World bytes (including all BattleRandom state/unused capacity/padding)
are retained losslessly PER UPDATE. They are not normalized or declared equal
across host/device compilers. Complete public rollouts and 32 fixed-action public
continuations MUST match exactly, as must the shared neural/optimizer/RNG state.
"""
from __future__ import annotations
from pathlib import Path
import argparse,copy,hashlib,json,os,signal,subprocess,sys,time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE));sys.path.insert(0,str(HERE.parent))
BACKENDS=('native-cuda','native-cpu-cuda')


def cpu(value):
    import numpy as np,torch
    if isinstance(value,torch.Tensor):return value.detach().cpu().clone()
    if isinstance(value,np.ndarray):return torch.from_numpy(value.copy())
    if isinstance(value,dict):return {key:cpu(v) for key,v in value.items()}
    if isinstance(value,list):return [cpu(v) for v in value]
    if isinstance(value,tuple):return tuple(cpu(v) for v in value)
    return copy.deepcopy(value)


def rng():
    import numpy as np,random,torch
    return {'python':random.getstate(),'numpy':np.random.get_state(),'torch_cpu':torch.get_rng_state().clone(),'cuda_all':[x.clone() for x in torch.cuda.get_rng_state_all()]}


def outputs(env):return {name:cpu(getattr(env,name)) for name in ('observations','masks','infos','rewards')}


def engine_image(env,workers):
    from snapshots import encode_engine_state
    # Exact lossless encoding of EVERY lane byte. No pointer/padding exclusions.
    before=rng();encoded=encode_engine_state(env.states if hasattr(env,'states') else env.state(),workers=workers)
    assert_exact(rng(),before,'opaque engine snapshot/compression must not consume training RNG')
    return encoded


def save(path,value):
    import torch
    temporary=path.with_suffix('.tmp');torch.save(value,temporary);temporary.replace(path)


def check_public(value,s):
    info=value['infos']
    for name in ('fault','not_modelled','illegal_0','illegal_1'):
        assert not info[:,s.info[name]].any(),('invalid public transition',name)


def continuation(trainer):
    import numpy as np,torch
    env=trainer.env;s=trainer.schema;frames=[];before=rng()
    for step in range(32):
        masks=cpu(env.masks).numpy();actions=np.zeros((trainer.c['envs'],2),np.int32)
        if step%3==0:
            for lane in range(len(actions)):
                for seat in range(2):
                    legal=np.flatnonzero(masks[lane,seat]);assert len(legal),'empty legal mask'
                    actions[lane,seat]=legal[(step*11+lane*7+seat*5)%len(legal)]
        env.step(actions,trainer.c['max_ticks']);public=outputs(env);check_public(public,s)
        info=public['infos'];done=(info[:,s.info['winner']]>=0)|(info[:,s.info['truncated']]!=0)
        reset=None;seeds=None
        if done.any():
            seeds=np.arange(len(actions),dtype=np.int32)+71000+step*101
            env.reset(seeds,done.numpy());reset=outputs(env);check_public(reset,s)
        frames.append({'step':step,'actions':torch.from_numpy(actions.copy()),'public_outputs':public,'done':done,'reset_seeds':None if seeds is None else torch.from_numpy(seeds),'reset_outputs':reset})
    assert_exact(rng(),before,'fixed-action continuation does not consume Python/NumPy/Torch/CUDA RNG')
    return frames


def worker(args):
    import torch
    from train import Trainer,binding
    from snapshots import available_cpu_workers
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise RuntimeError('one already allocated GPU required; no CPU-only fallback')
    args.out.mkdir(parents=True,exist_ok=False);config=json.loads(args.config.read_text());initial_binding=binding(args.data)
    workers=min(config['envs'],config['threads'],available_cpu_workers())
    # Observe the actual initial seed tensor, with NO extra draws and NO edits to
    # production Trainer. The same synchronization is added to BOTH arms.
    draws=[];original=torch.randint
    def observe_randint(*positional,**named):
        value=original(*positional,**named)
        if value.is_cuda and value.dtype==torch.int32 and tuple(value.shape)==(config['envs'],):draws.append(cpu(value))
        return value
    torch.randint=observe_randint
    try:trainer=Trainer(config,args.data)
    finally:torch.randint=original
    try:
        assert len(draws)==1,('expected exactly one actual initial engine seed tensor',len(draws))
        manifest={'format':1,'config':trainer.c,'source_binding':trainer.binding,'metadata':trainer.data['metadata'],'backend_build':trainer.backend_build,'auxiliary_builds':trainer.auxiliary_builds,'execution_runtime':trainer.execution_runtime,'gpu':torch.cuda.get_device_name(),'initial_seed_observation':'read-only torch.randint result observer only during Trainer construction; same copy/synchronization in both arms; no extra RNG draw'}
        (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        initial={'engine_seeds':draws[0],'model':cpu(trainer.model.state_dict()),'optimizer':cpu(trainer.optimizer.state_dict()),'public_outputs':outputs(trainer.env),'actor_inputs':cpu(trainer.actor),'rng':rng(),'counters':cpu(trainer.counters),'bucket_usage':cpu(trainer.bucket_usage),'engine_state':engine_image(trainer.env,workers)}
        check_public(initial['public_outputs'],trainer.schema);save(args.out/'initial.pt',initial)
        for index in range(config['updates']):
            before_rng=rng();data=trainer.collect()
            rollout={key:cpu(value) for key,value in trainer.store.items()}
            derived={key:cpu(data[key]) for key in ('advantage','return')};after_collect=rng();collected_public=outputs(trainer.env)
            metrics,steps=trainer.learn(data)
            record={'update':trainer.update,'samples':trainer.samples,'rollout_store':rollout,'derived':derived,'public_after_collection':collected_public,'public_after_learning':outputs(trainer.env),'actor_inputs':cpu(trainer.actor),'model':cpu(trainer.model.state_dict()),'optimizer':cpu(trainer.optimizer.state_dict()),'metrics':cpu(metrics),'optimizer_steps':steps,'counters':cpu(trainer.counters),'bucket_usage':cpu(trainer.bucket_usage),'rng_before_collection':before_rng,'rng_after_collection':after_collect,'rng_after_learning':rng(),'engine_state':engine_image(trainer.env,workers)}
            check_public(record['public_after_learning'],trainer.schema)
            save(args.out/f'update-{index+1:02d}.pt',record)
            assert binding(args.data)==initial_binding,'source/data/rules changed during backend training qualification'
            print(json.dumps({'backend':config['backend'],'control':'truncated' if config['max_ticks'] else 'live','update':trainer.update,'samples':trainer.samples,'optimizer_steps':steps,'counters':record['counters'].tolist()}),flush=True)
        frames=continuation(trainer);save(args.out/'continuation.pt',{'frames':frames,'rng':rng(),'final_public_outputs':outputs(trainer.env),'final_engine_state':engine_image(trainer.env,workers)})
        assert binding(args.data)==initial_binding,'source changed during public continuation'
        c=cpu(trainer.counters).tolist()
        if config['max_ticks']:assert c[4]>0,'truncation control did not actually truncate'
        else:assert c[4]==0,'unbounded control produced artificial truncation'
        summary={'status':'completed','backend':config['backend'],'updates':trainer.update,'samples':trainer.samples,'counters':c,'bucket_usage':cpu(trainer.bucket_usage).tolist(),'source_stable':True,'snapshot_encoding':'lossless all bytes; no normalization','fixed_action_continuation_decisions':32,'timing_comparison':False}
        (args.out/'worker-result.json').write_text(json.dumps(summary,indent=2)+'\n')
    finally:trainer.env.close()


class ComparisonError(AssertionError):
    def __init__(self,details):self.details=details;super().__init__(json.dumps(details))


def assert_exact(actual,expected,path='root'):
    import numpy as np,torch
    def fail(**extra):raise ComparisonError(dict(path=path,**extra))
    if isinstance(actual,torch.Tensor):
        if not isinstance(expected,torch.Tensor) or actual.dtype!=expected.dtype or actual.shape!=expected.shape:fail(reason='tensor schema differs',actual_dtype=str(actual.dtype),actual_shape=list(actual.shape),expected_dtype=str(getattr(expected,'dtype',None)),expected_shape=list(getattr(expected,'shape',())))
        if not torch.equal(actual,expected):
            where=torch.nonzero(actual!=expected);details={'reason':'tensor values differ','dtype':str(actual.dtype),'shape':list(actual.shape),'different_elements':len(where),'first_indices':where[:12].tolist()}
            if actual.is_floating_point():details['maximum_absolute_difference']=float((actual.double()-expected.double()).abs().max())
            fail(**details)
    elif isinstance(actual,np.ndarray):
        if not isinstance(expected,np.ndarray) or actual.dtype!=expected.dtype or actual.shape!=expected.shape or not np.array_equal(actual,expected):fail(reason='NumPy RNG array differs')
    elif isinstance(actual,dict):
        if not isinstance(expected,dict) or actual.keys()!=expected.keys():fail(reason='dictionary keys differ',actual_keys=list(actual),expected_keys=list(expected) if isinstance(expected,dict) else None)
        for key in actual:assert_exact(actual[key],expected[key],path+'.'+str(key))
    elif isinstance(actual,(list,tuple)):
        if type(actual)!=type(expected) or len(actual)!=len(expected):fail(reason='sequence schema differs')
        for index,(a,b) in enumerate(zip(actual,expected)):assert_exact(a,b,path+'.'+str(index))
    elif actual!=expected:fail(reason='scalar differs',actual=repr(actual),expected=repr(expected))


def shared_manifest(actual,expected):
    assert_exact(actual['source_binding'],expected['source_binding'],'shared source/data/rules binding')
    assert_exact(actual['metadata'],expected['metadata'],'shared frozen metadata')
    assert_exact(actual['auxiliary_builds'],expected['auxiliary_builds'],'shared actual GPU neural helper builds')
    a=dict(actual['execution_runtime']);b=dict(expected['execution_runtime'])
    # Only the native CUDA engine has a launch/render policy. All neural/runtime
    # precision flags, framework/library versions and GPU declarations MUST match.
    a_policy=a.pop('engine_policy',None);b_policy=b.pop('engine_policy',None)
    assert_exact(a,b,'common execution runtime including actual precision flags')
    ac={k:v for k,v in actual['config'].items() if k not in ('backend','cuda_launch_threads','cuda_render_mode')}
    bc={k:v for k,v in expected['config'].items() if k not in ('backend','cuda_launch_threads','cuda_render_mode')}
    assert_exact(ac,bc,'common full PPO configuration')
    assert actual['config']['backend']!=expected['config']['backend']
    return {'cuda_engine_policy':a_policy if actual['config']['backend']=='native-cuda' else b_policy,'backend_builds_retained_separately':True,'backend_library_equality_claimed':False}


def raw_difference_report(a,b):
    """Diagnostic only; never normalizes bytes or grants a parity exemption."""
    import numpy as np
    from snapshots import decode_engine_state
    aa=decode_engine_state(a);bb=decode_engine_state(b)
    if len(aa)!=len(bb) or any(len(x)!=len(y) for x,y in zip(aa,bb)):
        return {'same_raw_shape':False,'normalization_applied':False,'raw_equality_required':False}
    differing=[int(np.count_nonzero(np.frombuffer(l,dtype=np.uint8)!=np.frombuffer(r,dtype=np.uint8))) for l,r in zip(aa,bb)]
    return {'same_raw_shape':True,'raw_differing_bytes_per_lane':differing,'normalization_applied':False,'raw_equality_required':False,'scope':'All bytes and embedded engine RNG retained. No unknown padding/state bytes excluded; raw cross-compiler equality is not asserted.'}


def compare_control(folder,updates):
    import torch
    cuda=folder/'native-cuda';host=folder/'native-cpu-cuda'
    cm=json.loads((cuda/'manifest.json').read_text());hm=json.loads((host/'manifest.json').read_text());manifest=shared_manifest(cm,hm)
    reports=[]
    for stage in ['initial']+[f'update-{i+1:02d}' for i in range(updates)]:
        a=torch.load(cuda/(stage+'.pt'),map_location='cpu',weights_only=False);b=torch.load(host/(stage+'.pt'),map_location='cpu',weights_only=False)
        ai=dict(a);bi=dict(b);image_a=ai.pop('engine_state');image_b=bi.pop('engine_state')
        assert_exact(ai,bi,stage)
        reports.append({'stage':stage,'all_shared_training_fields_exact':True,'engine_image_diagnostic':raw_difference_report(image_a,image_b)})
    a=torch.load(cuda/'continuation.pt',map_location='cpu',weights_only=False);b=torch.load(host/'continuation.pt',map_location='cpu',weights_only=False)
    aa=dict(a);bb=dict(b);ia=aa.pop('final_engine_state');ib=bb.pop('final_engine_state');assert_exact(aa,bb,'32 fixed-action public continuation')
    ca=json.loads((cuda/'worker-result.json').read_text());cb=json.loads((host/'worker-result.json').read_text())
    assert_exact({k:v for k,v in ca.items() if k!='backend'},{k:v for k,v in cb.items() if k!='backend'},'worker counters/summary')
    return {'status':'passed','manifest':manifest,'updates':reports,'fixed_action_continuation_decisions':32,'continuation_all_public_outputs_actions_done_resets_rng_exact':True,'continuation_final_engine_diagnostic':raw_difference_report(ia,ib),'actual_counters':ca['counters'],'actual_bucket_usage':ca['bucket_usage'],'raw_world_normalization_applied':False,'natural_matches':ca['counters'][0],'artificial_truncations':ca['counters'][4]}


def configuration(base,backend,args,gpu,limit):
    c=copy.deepcopy(base)
    c.update(backend=backend,device='cuda',seed=args.seed,envs=args.envs,threads=args.cpu_threads,torch_threads=1,width=64,embedding=12,updates=args.updates,rollout_steps=args.rollout_steps,minibatch=args.minibatch,epochs=2,budget_seconds=600,max_ticks=limit,opponent_mode='selfplay',warmstart_steps=0,reward_shaping=0.,target_kl=0.,precision='fp32',compact_embedding_grad=True,bucket_actor=True,bucket_learner=True,learner_graph=False,rollout_graph=False,graph_sampling=True,optimizer_fused=True,gae_kernel='triton',compile_loss=False)
    for name in ('cuda_launch_threads','cuda_render_mode'):c.pop(name,None)
    if backend=='native-cuda':c.update(cuda_launch_threads=1,cuda_render_mode=4)
    c['hardware_allocation']={'gpu_model':gpu,'gpu_count':1,'cpu_threads':args.cpu_threads,'ram_gib':int(os.getenv('CLASH_RAM_GIB','64')),'qualification':'same allocated GPU and attached CPU/RAM; sequential correctness subprocesses, no timing comparison'}
    return c


def main(args):
    started=time.monotonic()
    from train import binding
    from snapshots import available_cpu_workers
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise RuntimeError('exactly one already allocated CUDA GPU required; no worker/cloud launch performed')
    if args.envs<1 or args.updates<1 or args.rollout_steps not in (16,32) or args.minibatch<1 or args.budget_seconds<1 or args.subprocess_timeout<1 or not 1<=args.cpu_threads<=available_cpu_workers():raise ValueError('positive bounded workload/budget, T16/32 and CPU threads within attached affinity/quota required')
    args.out=args.out.resolve()
    if args.out.is_relative_to(HERE.parent):raise ValueError('qualification outputs must be outside integration source to preserve source-bound checkpoints')
    args.out.mkdir(parents=True,exist_ok=False);source=binding(args.data);gpu=torch.cuda.get_device_name();base=json.loads(args.base_config.read_text());results=[];failure=None
    try:
        for name,limit in [('live',0),('truncated',125)]:
            folder=args.out/name;folder.mkdir()
            for backend in BACKENDS:
                c=configuration(base,backend,args,gpu,limit);config=folder/(backend+'-config.json');config.write_text(json.dumps(c,indent=2)+'\n');target=folder/backend
                command=[sys.executable,str(Path(__file__).resolve()),'--worker','--config',str(config),'--data',str(args.data),'--out',str(target)]
                remaining=args.budget_seconds-(time.monotonic()-started)
                if remaining<1:raise TimeoutError('bounded whole qualification budget exhausted')
                with (folder/(backend+'.log')).open('w') as log:
                    process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONDONTWRITEBYTECODE='1'))
                    try:process.wait(timeout=min(args.subprocess_timeout,remaining))
                    except subprocess.TimeoutExpired:
                        try:os.killpg(process.pid,signal.SIGKILL)
                        except ProcessLookupError:pass
                        process.wait();raise TimeoutError('backend qualifier child/process group exceeded bounded remaining budget')
                if process.returncode:raise RuntimeError(backend+' '+name+' worker failed; see retained log')
                assert binding(args.data)==source,'source/data changed between backend workers'
            result=compare_control(folder,args.updates);result['control']=name;results.append(result);(folder/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
        assert binding(args.data)==source,'qualification source changed'
    except BaseException as exc:
        failure={'type':type(exc).__name__,'message':str(exc),'details':exc.details if isinstance(exc,ComparisonError) else None}
    report={'status':'failed' if failure else 'passed','error':failure,'controls':results,'source_binding':source,'gpu':gpu,'workload':{'envs':args.envs,'rollout_steps':args.rollout_steps,'updates':args.updates,'width':64,'embedding':12,'cpu_threads':args.cpu_threads,'modes':['live','max_ticks125'],'precision':'fp32','fast_flags':['compact embeddings','bucket actor','bucket learner','Triton GAE','CUDA1/render4']},'scope':'Exact shared neural/optimizer/all exposed RNG and full public-state training qualification across simulators; all opaque engine/RNG bytes retained losslessly without cross-compiler normalization','full_simulator_natural_outcome_qualification_separate':True,'cloud_or_GPU_worker_launched':False,'timing_comparison':False}
    (args.out/'result.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k not in ('source_binding','controls')}),flush=True)
    if failure:raise RuntimeError(failure['message'])


def offline_self_test(out):
    """CPU-only negative controls for the comparator, never training evidence."""
    import tempfile,torch
    from snapshots import encode_engine_state
    out.mkdir(parents=True,exist_ok=False);checks=[]
    manifest={'source_binding':{'source':'same','data':'same'},'metadata':{'task':'fixed8'},'auxiliary_builds':{'neural':'same'},'execution_runtime':{'precision':{'mode':'fp32','cuda_matmul_allow_tf32':False}},'config':{'backend':'native-cuda','width':64,'embedding':12,'cuda_launch_threads':1,'cuda_render_mode':4}}
    other=copy.deepcopy(manifest);other['config']={'backend':'native-cpu-cuda','width':64,'embedding':12};manifest['execution_runtime']['engine_policy']={'cuda_launch_threads':1,'cuda_render_mode':4}
    shared_manifest(manifest,other);checks.append('only explicit backend/engine-policy differences accepted')
    for field in ('precision','source_binding','auxiliary_builds'):
        changed=copy.deepcopy(other)
        if field=='precision':changed['execution_runtime']['precision']['cuda_matmul_allow_tf32']=True
        else:changed[field]['extra']='different'
        try:shared_manifest(manifest,changed)
        except ComparisonError:checks.append(field+' mismatch rejected exactly')
        else:raise AssertionError('accepted '+field)
    try:assert_exact(torch.tensor([1.]),torch.tensor([1.0000001]),'tiny model drift')
    except ComparisonError:checks.append('single-FP32-ULP drift rejected; no numerical tolerance')
    else:raise AssertionError('accepted floating drift')
    images=[encode_engine_state([value],workers=1) for value in (b'\x00\x01\x02\x03',b'\x00\x01\xff\x03')]
    report=raw_difference_report(*images);assert report['raw_differing_bytes_per_lane']==[1] and not report['normalization_applied'];checks.append('unknown raw byte retained/reported without normalization')
    with tempfile.TemporaryDirectory(prefix='clash-backend-comparator-') as folder:
        root=Path(folder)
        for backend,m,image in zip(BACKENDS,(manifest,other),images):
            target=root/backend;target.mkdir();(target/'manifest.json').write_text(json.dumps(m))
            for name in ('initial','update-01'):save(target/(name+'.pt'),{'model':{'w':torch.tensor([1.])},'engine_state':image})
            save(target/'continuation.pt',{'frames':[{'public':torch.tensor([3])}]*32,'rng':{'state':torch.tensor([1],dtype=torch.uint8)},'final_engine_state':image})
            (target/'worker-result.json').write_text(json.dumps({'backend':backend,'counters':[0]*11,'bucket_usage':[0]*8}))
        assert compare_control(root,1)['status']=='passed';checks.append('complete fixture permits distinct opaque engine representations only')
        target=root/'native-cpu-cuda/update-01.pt';record=torch.load(target,weights_only=False);record['model']['w'][0]=2.;save(target,record)
        try:compare_control(root,1)
        except ComparisonError:checks.append('complete fixture rejects changed model despite matching public continuation')
        else:raise AssertionError('accepted changed model')
    result={'status':'passed','scope':'CPU comparator/retention logic only; no simulator, CUDA execution, training or performance evidence','checks':checks,'actual_backend_training_executed':False,'GPU_launched':False,'timing_comparison':False}
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);p.add_argument('--data',type=Path,default=HERE/'generated-v2/task-data.json');p.add_argument('--base-config',type=Path,default=HERE.parent/'configs/cuda-learning.json');p.add_argument('--cpu-threads',type=int,default=16);p.add_argument('--envs',type=int,default=4);p.add_argument('--rollout-steps',type=int,choices=[16,32],default=32);p.add_argument('--updates',type=int,default=4);p.add_argument('--minibatch',type=int,default=32);p.add_argument('--seed',type=int,default=81108);p.add_argument('--budget-seconds',type=int,default=600);p.add_argument('--subprocess-timeout',type=int,default=600);p.add_argument('--self-test',action='store_true');p.add_argument('--worker',action='store_true');p.add_argument('--config',type=Path);a=p.parse_args()
    if a.worker:
        if a.config is None:p.error('--worker requires --config')
        worker(a)
    elif a.self_test:offline_self_test(a.out)
    else:main(a)
