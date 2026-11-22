"""Independent GAE, source binding and exact CPU/CUDA PPO resume qualification.

Run .venv-clash/bin/python integrations/clash/native/test_training.py --out NEW_DIR.
Only new evidence/checkpoints are written. Never changes native trainer/oracle files.
"""
from pathlib import Path
import argparse,copy,ctypes,hashlib,json,os,subprocess,sys,time
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE));sys.path.insert(0,str(HERE.parent))

def gae_checks(device="cpu"):
    import numpy as np,torch
    from train import device_gae
    from runner import advantages
    rng=np.random.default_rng(8112026);cases=[];worst=0.
    for length,lanes,gamma,lam in [(1,3,.9,.95),(5,4,.99,.95),(17,6,1.,1.),(31,8,.95,0.),(8,5,0.,.95)]:
        reward=rng.normal(size=(length,lanes)).astype(np.float32);value=rng.normal(size=reward.shape).astype(np.float32);last=rng.normal(size=lanes).astype(np.float32)
        done=rng.random(reward.shape)<.3;trunc=np.where(done&(rng.random(reward.shape)<.5),rng.normal(size=reward.shape),0).astype(np.float32)
        got=device_gae(*(torch.from_numpy(v).to(device) for v in [reward,value,done,trunc,last]),gamma,lam)
        expected=advantages(reward,value,done,trunc,last,np.ones_like(done),gamma,lam)
        for actual,want in zip(got,expected):
            error=float(np.max(np.abs(actual.cpu().numpy()-want)));worst=max(worst,error);np.testing.assert_allclose(actual.cpu().numpy(),want,rtol=2e-6,atol=2e-6)
        cases.append(dict(ticks=length,seats=lanes,natural_terminals=int((done&(trunc==0)).sum()),truncations=int((done&(trunc!=0)).sum()),rollout_bootstraps=int((~done[-1]).sum()),gamma=gamma,gae_lambda=lam))
    # Transparent terminal/truncation/rollout examples independent of either helper.
    r=torch.tensor([[1.,2.,0.]]);v=torch.tensor([[.2,.3,.4]]);done=torch.tensor([[True,True,False]]);tv=torch.tensor([[0.,.8,0.]]);last=torch.tensor([9.,9.,.5])
    _,ret=device_gae(*(t.to(device) for t in (r,v,done,tv,last)),.9,.95);torch.testing.assert_close(ret.cpu(),torch.tensor([[1.,2.72,.45]]),rtol=1e-6,atol=1e-6)
    return dict(status='passed',device=str(device),cases=cases,maximum_absolute_error=worst,explicit_return_example=ret.tolist(),fault_handling='Trainer asserts invalid transitions before GAE; no invalid-transition bootstrap accepted')

def cpu_array(value):
    import torch
    return value.detach().cpu().numpy().copy() if isinstance(value,torch.Tensor) else value.copy()

def state_images(value):
    import torch
    if isinstance(value,dict):
        from snapshots import decode_engine_state
        return decode_engine_state(value)
    if isinstance(value,torch.Tensor):return [row.numpy().tobytes() for row in value.detach().cpu().contiguous()]
    return list(value)

def read_cache_fixture(env):
    import numpy as np
    before=state_images(env.state());outputs=[cpu_array(getattr(env,k)) for k in ('observations','masks','infos','rewards')]
    env.read();env.read();env.reset(np.arange(len(before),dtype=np.int32)+24680,np.zeros(len(before),dtype=bool))
    assert state_images(env.state())==before,'read/reset(false) mutated simulator or first-sight state'
    for key,want in zip(('observations','masks','infos','rewards'),outputs):np.testing.assert_array_equal(cpu_array(getattr(env,key)),want,err_msg='read cache '+key)
    return {'status':'passed','repeated_reads':2,'no_selected_reset':True}

def graph_usage_probe(trainer,c):
    """Observe real replay state; warmup/capture must contribute no usage."""
    import torch
    total=c['envs']*2*c['rollout_steps'];minibatch=c['minibatch']
    graph_mode=c.get('learner_graph',False) or c.get('bucket_learner',False)
    expected=sorted({min(minibatch,total-start) for start in range(0,total,minibatch)}) if graph_mode else []
    actual=sorted(getattr(trainer,'_learner_graphs',{}))
    assert actual==expected,('learner graph-cache keys',actual,expected)
    if graph_mode and minibatch==48 and total==64:assert actual==[16,48]
    usage=trainer.bucket_usage.detach().cpu().clone()
    assert usage.dtype==torch.int64 and usage.shape==(8,),('bucket_usage schema',usage.dtype,usage.shape)
    assert bool((usage>=0).all()),'negative bucket usage'
    actor_calls=trainer.update*c['rollout_steps'] if c.get('bucket_actor',False) else 0
    assert int(usage[:4].sum())==actor_calls,('actor bucket call count',usage.tolist(),actor_calls)
    if not c.get('bucket_learner',False):assert int(usage[4:].sum())==0
    elif c.get('target_kl',0)==0:
        calls=trainer.update*c['epochs']*((total+minibatch-1)//minibatch)
        assert int(usage[4:].sum())==calls,('learner bucket call count',usage.tolist(),calls)
    return actual,usage

def worker(args):
    import numpy as np,torch
    from train import Trainer
    c=json.loads(args.config.read_text());trainer=Trainer(c,args.data,args.resume);losses=[]
    cache=read_cache_fixture(trainer.env)
    try:
        while trainer.update<args.stop:
            batch=trainer.collect();metrics,steps=trainer.learn(batch);losses.append(dict(update=trainer.update,metrics=metrics.tolist(),steps=steps))
        graph_keys,bucket_usage=graph_usage_probe(trainer,c)
        trainer.checkpoint(args.out/'checkpoint.pt',0.)
        # Clone the actual saved state before the continuation probe. The fixed
        # action sequence needs no sampling and does not mutate the checkpoint.
        debug=[]
        if c['backend']!='native-cuda':
            lib=trainer.env.lib;ip=np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS');lib.cn_world_debug.argtypes=[ctypes.c_void_p,ip,ctypes.c_int]
            for handle in trainer.env.handles:
                values=np.zeros(100000,np.int32);count=lib.cn_world_debug(handle,values,len(values));debug.append(values[:count].copy())
        outputs=[]
        for step in range(32):
            actions=np.zeros((c['envs'],2),np.int32)
            if step%3==0:
                for lane in range(c['envs']):
                    for side in range(2):actions[lane,side]=np.flatnonzero(cpu_array(trainer.env.masks[lane,side]))[-1]
            trainer.env.step(actions,c.get('max_ticks',0));outputs.append(tuple(cpu_array(getattr(trainer.env,key)) for key in ('observations','masks','infos','rewards')))
            info=cpu_array(trainer.env.infos);done=(info[:,trainer.schema.info['winner']]>=0)|(info[:,trainer.schema.info['truncated']]!=0)
            trainer.env.reset(np.arange(c['envs'])+12300+step,done)
        torch.save(dict(debug=debug,continuation=outputs,losses=losses,read_cache=cache,learner_graph_keys=graph_keys,bucket_usage=bucket_usage),args.out/'probe.pt')
    finally:trainer.close()

def equal(a,b,path='root'):
    import numpy as np,torch
    if isinstance(a,torch.Tensor):
        assert isinstance(b,torch.Tensor) and a.dtype==b.dtype and a.shape==b.shape and torch.equal(a,b),path
    elif isinstance(a,np.ndarray):np.testing.assert_array_equal(a,b,err_msg=path)
    elif isinstance(a,dict):
        assert a.keys()==b.keys(),path+' keys'
        for key in a:equal(a[key],b[key],path+'.'+str(key))
    elif isinstance(a,(list,tuple)):
        assert type(a)==type(b) and len(a)==len(b),path
        for i,(x,y) in enumerate(zip(a,b)):equal(x,y,path+f'[{i}]')
    else:assert a==b,(path,a,b)

def main(args):
    import torch
    from train import binding
    if min(args.minibatch,args.envs,args.width,args.embedding)<1:raise ValueError('minibatch, envs, width and embedding must be positive')
    out=args.out;out.mkdir(parents=True,exist_ok=False);started=time.monotonic();initial_binding=binding(args.data)
    gpu=args.backend!='native-cpu'
    if gpu and torch.cuda.device_count()!=1:raise ValueError('CUDA qualification requires exactly one visible GPU')
    gae=gae_checks('cuda' if gpu else 'cpu');(out/'gae.json').write_text(json.dumps(gae,indent=2)+'\n')
    # The only pointers in NativeWorld are its immutable spec and the grid's
    # tiles/current/previous views. Normalize those addresses, not mutable data.
    layout_source=out/'layout.cpp';layout_binary=out/'layout'
    layout_source.write_text('#include <cstddef>\n#include <cstdio>\n#include "'+str(HERE/'world.hpp')+'"\nint main(){using namespace clash_native;std::printf("%zu %zu %zu %zu %zu %zu",sizeof(NativeWorld),sizeof(void*),offsetof(NativeWorld,spec),offsetof(NativeWorld,arenaGrid)+offsetof(GridView,tiles),offsetof(NativeWorld,arenaGrid)+offsetof(GridView,current),offsetof(NativeWorld,arenaGrid)+offsetof(GridView,previous));}\n')
    import shutil
    compiler=shutil.which('clang++') or shutil.which('g++')
    if compiler is None:raise RuntimeError('C++ compiler required for independent pointer layout')
    subprocess.run([compiler,'-std=c++20',str(layout_source),'-o',str(layout_binary)],check=True)
    layout=list(map(int,subprocess.check_output([str(layout_binary.resolve())],text=True).split()));world_bytes,pointer_bytes,*pointer_offsets=layout
    (out/'pointer-layout.json').write_text(json.dumps(dict(world_bytes=world_bytes,pointer_bytes=pointer_bytes,pointer_offsets=pointer_offsets),indent=2)+'\n')
    def normalized_image(image):
        assert len(image)==world_bytes
        result=bytearray(image)
        for offset in pointer_offsets:result[offset:offset+pointer_bytes]=b'\0'*pointer_bytes
        return bytes(result)
    config=qualification_config(args,gpu)
    config['precision']=args.precision;config['compile_loss']=args.compile_loss;config['packed_head']=args.packed_head
    if args.compact_embedding_grad:config['compact_embedding_grad']=True
    if args.rollout_graph:config['rollout_graph']=True
    if args.learner_graph:config['learner_graph']=True
    if args.bucket_actor:config['bucket_actor']=True
    if args.bucket_learner:config['bucket_learner']=True
    if args.gae_kernel is not None:config['gae_kernel']=args.gae_kernel
    results=[]
    for name,limit in [('live',0),('truncated',125)]:
        family=out/name;family.mkdir();c=config|{'max_ticks':limit};configpath=family/'config.json';configpath.write_text(json.dumps(c,indent=2)+'\n')
        for stage,stop,resume in [('full',4,None),('prefix',2,None),('resumed',4,family/'prefix/checkpoint.pt')]:
            target=family/stage;target.mkdir();stageconfig=configpath
            if stage=='prefix':
                stageconfig=target/'config.json';stageconfig.write_text(json.dumps(c|{'updates':2,'budget_seconds':150},indent=2)+'\n')
            cmd=[sys.executable,str(Path(__file__).resolve()),'--worker','--config',str(stageconfig),'--data',str(args.data),'--out',str(target),'--stop',str(stop)]
            if resume:cmd+=['--resume',str(resume)]
            with (target/'process.log').open('w') as log:
                process=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
            if process.returncode:raise RuntimeError(f'{stage} failed: {(target/"process.log").read_text()[-4000:]}')
        full=torch.load(family/'full/checkpoint.pt',map_location='cpu',weights_only=False);resumed=torch.load(family/'resumed/checkpoint.pt',map_location='cpu',weights_only=False)
        keys=['format','config','binding','backend_build','auxiliary_builds','execution_runtime','metadata','model','optimizer','engine_outputs','update','samples','counters','bucket_usage','python_rng','numpy_rng','torch_rng','cuda_rng']
        for key in keys:equal(full[key],resumed[key],key)
        pf=torch.load(family/'full/probe.pt',weights_only=False);pr=torch.load(family/'resumed/probe.pt',weights_only=False)
        equal(pf['debug'],pr['debug'],'complete_native_debug_state');equal(pf['read_cache'],pr['read_cache'],'read-cache fixture');equal(pf['continuation'],pr['continuation'],'32-step deterministic continuation');equal(pf['losses'][2:],pr['losses'],'continued PPO metrics')
        equal(pf['learner_graph_keys'],pr['learner_graph_keys'],'recreated learner graph-cache keys');equal(pf['bucket_usage'],pr['bucket_usage'],'probe bucket usage')
        equal(pf['bucket_usage'],full['bucket_usage'],'full checkpoint/probe bucket usage');equal(pr['bucket_usage'],resumed['bucket_usage'],'resumed checkpoint/probe bucket usage')
        full_images=state_images(full['engine_state']);resumed_images=state_images(resumed['engine_state'])
        raw_differing=sum(sum(a!=b for a,b in zip(x,y)) for x,y in zip(full_images,resumed_images))
        for lane,(actual,expected) in enumerate(zip(full_images,resumed_images)):assert normalized_image(actual)==normalized_image(expected),f'normalized full NativeWorld bytes lane{lane}'
        results.append(dict(mode=name,status='passed',updates=full['update'],learner_samples=full['samples'],counters=full['counters'].tolist(),bucket_usage=full['bucket_usage'].tolist(),learner_graph_keys=pf['learner_graph_keys'],minibatch=c['minibatch'],rollout_rows=c['envs']*2*c['rollout_steps'],read_cache=pf['read_cache'],exact_fields=keys+(['native_debug_state'] if args.backend!='native-cuda' else [])+['recreated learner graph-cache keys','probe bucket usage','read-cache fixture','32-step continuation','continued PPO metrics','entire NativeWorld image excluding four rebound pointer addresses'],raw_snapshot_differing_bytes=raw_differing,raw_snapshot_note='Only four documented pointer addresses normalized; every other serialized byte, including padding, exactly matched.'))
    # Corrupted binding/recipe artifacts are separate NEW files. Rejecting them
    # exercises the production validation path without editing source or data.
    checkpoint=out/'live/prefix/checkpoint.pt';base=torch.load(checkpoint,weights_only=False);rejections=[]
    for kind in ['source','data','rules','recipe','backend','runtime','auxiliary']:
        changed=copy.deepcopy(base)
        if kind=='source':changed['binding']['source']['native/train.py']='0'*64
        elif kind=='data':changed['binding']['task_data_sha256']='0'*64
        elif kind=='rules':changed['binding']['rules_lock_sha256']='0'*64
        elif kind=='recipe':changed['config']['gamma']=.7
        elif kind=='backend':changed['backend_build']['library_sha256']='0'*64
        elif kind=='runtime':changed['execution_runtime']['torch']='incompatible-test-runtime'
        else:changed['auxiliary_builds']={'incompatible':'test'}
        corrupt=out/f'incompatible-{kind}.pt';torch.save(changed,corrupt);target=out/f'reject-{kind}';target.mkdir();cmd=[sys.executable,str(Path(__file__).resolve()),'--worker','--config',str(out/'live/config.json'),'--data',str(args.data),'--out',str(target),'--stop','4','--resume',str(corrupt)]
        result=subprocess.run(cmd,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT);(target/'process.log').write_text(result.stdout)
        assert result.returncode!=0 and ('binding differs' in result.stdout if kind!='recipe' else 'recipe differs' in result.stdout),(kind,result.stdout)
        assert not (target/'checkpoint.pt').exists(),kind;rejections.append(kind)
    assert binding(args.data)==initial_binding,'source changed during exact-resume qualification'
    summary=dict(status='passed',gae=gae,resume=results,rejected_bindings=rejections,cuda_executed=gpu,gpu_name=torch.cuda.get_device_name() if gpu else None,backend=args.backend,compact_embedding_grad=args.compact_embedding_grad,packed_head=args.packed_head,workload={k:config[k] for k in ('envs','width','embedding','rollout_steps','minibatch')},elapsed_seconds=time.monotonic()-started,source_binding=initial_binding,timing_comparison=False)
    (out/'result.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps({k:v for k,v in summary.items() if k!='source_binding'}))

def qualification_config(args,gpu):
    """Explicit shape controls; historical default remains env2/width32/embed8."""
    return dict(backend=args.backend,seed=81108,envs=args.envs,torch_threads=1,width=args.width,embedding=args.embedding,learning_rate=.0003,rollout_steps=16,epochs=2,minibatch=args.minibatch,gamma=.99,gae_lambda=.95,clip=.2,value_coef=.5,entropy_coef=.001,max_grad_norm=.5,target_kl=0.,active_policy_loss=True,graph_sampling=gpu,updates=4,budget_seconds=300,max_ticks=0)


def parse_args(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--data',type=Path,default=HERE/'generated-v2/task-data.json');p.add_argument('--backend',choices=['native-cpu','native-cpu-cuda','native-cuda'],default='native-cpu');p.add_argument('--cuda',action='store_const',const='native-cuda',dest='backend');p.add_argument('--envs',type=int,default=2);p.add_argument('--width',type=int,default=32);p.add_argument('--embedding',type=int,default=8);p.add_argument('--minibatch',type=int,default=32);p.add_argument('--precision',choices=['fp32','tf32','bf16'],default='fp32');p.add_argument('--compile-loss',action='store_true');p.add_argument('--packed-head',action='store_true');p.add_argument('--compact-embedding-grad',action='store_true');p.add_argument('--rollout-graph',action='store_true');p.add_argument('--learner-graph',action='store_true');p.add_argument('--bucket-actor',action='store_true');p.add_argument('--bucket-learner',action='store_true');p.add_argument('--gae-kernel',choices=['torch','triton']);p.add_argument('--worker',action='store_true');p.add_argument('--config',type=Path);p.add_argument('--resume',type=Path);p.add_argument('--stop',type=int,default=4);return p.parse_args(argv)


if __name__=='__main__':
    args=parse_args();worker(args) if args.worker else main(args)
