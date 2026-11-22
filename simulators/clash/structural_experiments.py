"""Bounded, receipted experiments for structural Clash execution changes."""
from pathlib import Path
import sys,json,time,subprocess,statistics,os,re
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
EVIDENCE=ROOT/'evidence/clash-implementation/structural-20261108'
BASELINE=EVIDENCE/'baseline/integrations/clash/runner.py'

def run(out,name,c,*,baseline=False,command='profile',extra=(),init=True):
    path=out/(name+'.json');path.write_text(json.dumps(c,indent=2)+'\n')
    script=BASELINE if baseline else HERE/'runner.py'
    args=[sys.executable,str(script),command,'--config',str(path),'--out',str(out/name),*extra]
    if init and command!='evaluate':args+=['--init-checkpoint',str(ROOT/'evidence/clash-implementation/cuda-campaign/learning2/train/checkpoint.pt')]
    if Path('/usr/bin/time').exists() and sys.platform=='linux':args=['/usr/bin/time','-v','-o',str(out/(name+'-memory.txt')),*args]
    start=time.perf_counter()
    with (out/(name+'.log')).open('w') as log:
        result=subprocess.run(args,stdout=log,stderr=subprocess.STDOUT,timeout=1500)
    elapsed=time.perf_counter()-start
    receipt={'subprocess_wall_seconds':elapsed,'returncode':result.returncode,'baseline_snapshot':baseline}
    if (out/name/'result.json').exists():receipt.update(json.loads((out/name/'result.json').read_text()))
    metrics=out/name/'metrics.jsonl'
    if metrics.exists():
        rows=[json.loads(x) for x in metrics.read_text().splitlines()];warm=rows[2:]
        receipt['warm_iteration_median_seconds']=statistics.median(x['iteration_seconds'] for x in warm) if warm else None
        receipt['cold_first_iteration_seconds']=rows[0]['iteration_seconds'] if rows else None
        receipt['optimizer_steps']=sum(x.get('optimizer_steps',0) for x in rows)
        receipt['rollout_nonforced_rows']=sum(x.get('active_policy_samples',0) for x in rows)
        receipt['rollout_nonforced_rows_per_second']=receipt['rollout_nonforced_rows']/elapsed
        receipt['warm_phase_median_seconds']={k:statistics.median(x['iteration_timing'].get(k,0) for x in warm) for k in receipt['timing']} if warm else {}
    (out/(name+'-receipt.json')).write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({'run':name,'wall':elapsed,'status':receipt.get('status'),'returncode':result.returncode,'warm':receipt.get('warm_iteration_median_seconds'),'learner_samples':receipt.get('counts',{}).get('learner_decisions')}),flush=True)
    if result.returncode:
        print((out/(name+'.log')).read_text()[-7000:],flush=True)
        raise RuntimeError(name+' failed; retained failure receipt')
    return receipt

def qualified_native_engine(out):
    # Reuse only a byte-identical engine proof. A new ABI-preserving source edit
    # still produces a new build identity and must execute its own full qualifier.
    import hashlib
    sys.path.insert(0,str(HERE/'native'))
    from runtime import build_cuda
    actual=hashlib.sha256(build_cuda().read_bytes()).hexdigest()
    for name in ('native-h100-4','native-completion-1','native-optimized-3'):
        previous=out.parent/name/'engine-qualification/result.json'
        if previous.exists():
            retained=json.loads(previous.read_text())
            if retained['cuda_library_sha256']==actual:return retained
    subprocess.run([sys.executable,str(HERE/'native/qualify_cuda.py'),str(out/'engine-qualification')],check=True)
    current=json.loads((out/'engine-qualification/result.json').read_text())
    if current['cuda_library_sha256']!=actual:raise ValueError('engine changed during its own qualifier')
    return current


def main():
    import torch
    stage=sys.argv[1];out=Path(sys.argv[2]);results={}
    if stage=='native-kernel-variants':
        subprocess.run([sys.executable,str(HERE/'native/probe_kernel_variants.py'),'--out',str(out/'variants')],check=True)
        (out/'summary.json').write_text((out/'variants/result.json').read_text());return
    if stage=='native-packed-study':
        subprocess.run([sys.executable,str(HERE/'native/export_data.py')],check=True)
        subprocess.run([sys.executable,str(HERE/'native/data_codegen.py')],check=True)
        qualification=qualified_native_engine(out)
        env=dict(os.environ,CLASH_CUDA_LAUNCH_THREADS='1',CLASH_CUDA_RENDER_MODE='4')
        sys.path.insert(0,str(HERE/'native'))
        from benchmark_training import configurations,run_stage
        configs=configurations(ROOT,256,32,1701,32,128,torch.cuda.get_device_name())
        profiles={}
        def profile(name):
            import fcntl
            path=out/(name+'.json');path.write_text(json.dumps(configs[name],indent=2)+'\n')
            with (ROOT/'evidence/local-performance.lock').open('a+') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX)
                subprocess.run([sys.executable,str(HERE/'native/profile_detailed.py'),'--config',str(path),'--out',str(out/(name+'-profile')),'--chrome-trace'],check=True,env=env,timeout=400)
            profiles[name]=json.loads((out/(name+'-profile')/'result.json').read_text())
        profile('native-cuda-fast')
        checks={}
        def check(name,script,args):
            with (out/(name+'.log')).open('w') as log:
                subprocess.run([sys.executable,str(HERE/'native'/script),*args,'--out',str(out/name)],check=True,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=700)
            checks[name]=json.loads((out/name/'result.json').read_text())
            (out/'qualification-progress.json').write_text(json.dumps(checks,indent=2)+'\n')
            print(json.dumps({'qualification':name,'status':'passed'}),flush=True)
        check('packed-head','packed_head_qualify.py',['--device','cuda','--compact','--flatten-entities'])
        common=['--cuda','--compact-embedding-grad','--bucket-actor','--bucket-learner','--packed-head','--width','64','--embedding','12','--gae-kernel','triton']
        check('packed-resume-production','test_training.py',[*common,'--envs','32','--minibatch','1024'])
        check('packed-resume-tails','test_training.py',[*common,'--minibatch','48'])
        profile('native-cuda-packed')
        comparison=run_stage(out/'training-comparison',env_counts=(2048,),updates=32,variants=('original-first-smoke','native-cuda-fast','native-cuda-packed'),cores=32,ram_gib=128,gpu_model=torch.cuda.get_device_name(),stage_budget_seconds=2600)
        (out/'summary.json').write_text(json.dumps({'engine_qualification':qualification,'checks':checks,'profiles':profiles,'comparison':comparison,'strength_equivalence_claimed':False},indent=2)+'\n');return
    if stage=='native-detailed-profile':
        subprocess.run([sys.executable,str(HERE/'native/export_data.py')],check=True)
        subprocess.run([sys.executable,str(HERE/'native/data_codegen.py')],check=True)
        qualification=qualified_native_engine(out)
        sys.path.insert(0,str(HERE/'native'))
        from benchmark_training import configurations
        configurations_=configurations(ROOT,256,32,1701,int(os.getenv('CLASH_CPU_CORES','16')),int(os.getenv('CLASH_RAM_GIB','64')),torch.cuda.get_device_name())
        env=dict(os.environ,CLASH_CUDA_LAUNCH_THREADS='1',CLASH_CUDA_RENDER_MODE='4')
        import fcntl
        profiles={}
        with (ROOT/'evidence/local-performance.lock').open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            for name in ('native-cuda-fast','native-cpu-cuda-fast'):
                path=out/(name+'.json');path.write_text(json.dumps(configurations_[name],indent=2)+'\n')
                subprocess.run([sys.executable,str(HERE/'native/profile_detailed.py'),'--config',str(path),'--out',str(out/name),'--chrome-trace'],check=True,env=env,timeout=600)
                profiles[name]=json.loads((out/name/'result.json').read_text())
        (out/'summary.json').write_text(json.dumps({'engine_qualification':qualification,'profiles':profiles,'scope':'diagnostic phase/kernel profiles, not uninstrumented throughput comparisons'},indent=2)+'\n');return
    if stage=='native-completion':
        subprocess.run([sys.executable,str(HERE/'native/export_data.py')],check=True)
        subprocess.run([sys.executable,str(HERE/'native/data_codegen.py')],check=True)
        qualification=qualified_native_engine(out)
        checks={};env=dict(os.environ,CLASH_CUDA_LAUNCH_THREADS='1',CLASH_CUDA_RENDER_MODE='4',TORCHINDUCTOR_COMPILE_THREADS='4')
        def check(name,script,args,result_name='result.json'):
            with (out/(name+'.log')).open('w') as log:
                subprocess.run([sys.executable,str(HERE/'native'/script),*args,'--out',str(out/name)],stdout=log,stderr=subprocess.STDOUT,check=True,env=env)
            result=json.loads((out/name/result_name).read_text());checks[name]=result
            (out/'qualification-progress.json').write_text(json.dumps(checks,indent=2)+'\n');print(json.dumps({'qualification':name,'status':'passed'}),flush=True)
        check('render','qualify_cuda_render.py',['--steps','128'])
        common=['--compact-embedding-grad','--bucket-actor','--bucket-learner','--minibatch','1024','--envs','32','--width','64','--embedding','12','--gae-kernel','triton']
        check('cuda-resume','test_training.py',['--cuda',*common])
        check('cpu-cuda-resume','test_training.py',['--backend','native-cpu-cuda',*common])
        check('backend-parity','qualify_backend_training.py',['--cpu-threads',os.getenv('CLASH_CPU_CORES','16')])
        check('evaluation','qualify_cuda_evaluation.py',['--launch-threads','1','--render-mode','4'],result_name='summary.json')
        check('compiled','qualify_compiled_loss.py',['--embedding','compact','--width','64','--full-minibatch','1024','--tail-minibatch','48'])
        check('compiled-resume','test_training.py',['--cuda',*common,'--compile-loss'])
        for mode in ('tf32','bf16'):
            check(mode,'qualify_precision.py',['--mode',mode,'--compact-embedding-grad','--bucket-actor','--bucket-learner'])
            check(mode+'-resume','test_training.py',['--cuda',*common,'--precision',mode])
        from benchmark_training import run_stage
        summary=run_stage(out/'training-comparison',env_counts=(256,),updates=32,variants=('original-first-smoke','native-cuda-fast','native-cpu-cuda-fast','native-cuda-compiled','native-cuda-tf32','native-cuda-bf16'),cores=int(os.getenv('CLASH_CPU_CORES','4')),ram_gib=int(os.getenv('CLASH_RAM_GIB','16')),gpu_model=torch.cuda.get_device_name(),stage_budget_seconds=1100)
        (out/'summary.json').write_text(json.dumps({'engine_qualification':qualification,'checks':checks,'comparison':summary,'precision_strength_equivalence_claimed':False},indent=2)+'\n');return
    if stage in ('native-training-benchmark','native-batch-scaling','native-hardware'):
        subprocess.run([sys.executable,str(HERE/'native/export_data.py')],check=True)
        subprocess.run([sys.executable,str(HERE/'native/data_codegen.py')],check=True)
        sys.path.insert(0,str(HERE/'native'));from runtime import build_cuda
        import hashlib
        if stage=='native-hardware':
            subprocess.run([sys.executable,str(HERE/'native/qualify_cuda.py'),str(out/'engine-qualification')],check=True)
            qualification=json.loads((out/'engine-qualification/result.json').read_text())
        else:qualification=qualified_native_engine(out)
        if hashlib.sha256(build_cuda().read_bytes()).hexdigest()!=qualification['cuda_library_sha256']:raise ValueError('current engine differs from whole-world GPU qualifier')
        env=dict(os.environ,CLASH_CUDA_LAUNCH_THREADS='1',CLASH_CUDA_RENDER_MODE='4',TORCHINDUCTOR_COMPILE_THREADS='4')
        subprocess.run([sys.executable,str(HERE/'native/test_training.py'),'--cuda','--compact-embedding-grad','--bucket-actor','--bucket-learner','--minibatch','48','--gae-kernel','triton','--out',str(out/'current-source-resume')],check=True,env=env)
        variants=['original-first-smoke','native-cuda-fast']
        if stage=='native-hardware':
            common=['--envs','32','--width','64','--embedding','12','--minibatch','1024','--compact-embedding-grad','--bucket-actor','--bucket-learner','--gae-kernel','triton']
            subprocess.run([sys.executable,str(HERE/'native/qualify_precision.py'),'--mode','bf16','--compact-embedding-grad','--bucket-actor','--bucket-learner','--out',str(out/'bf16-qualification')],check=True,env=env)
            subprocess.run([sys.executable,str(HERE/'native/test_training.py'),'--cuda',*common,'--precision','bf16','--out',str(out/'bf16-resume')],check=True,env=env)
            subprocess.run([sys.executable,str(HERE/'native/test_training.py'),'--backend','native-cpu-cuda',*common,'--out',str(out/'cpu-cuda-resume')],check=True,env=env)
            subprocess.run([sys.executable,str(HERE/'native/qualify_backend_training.py'),'--cpu-threads',os.getenv('CLASH_CPU_CORES','32'),'--out',str(out/'backend-parity')],check=True,env=env)
            variants.extend(('native-cuda-bf16','native-cpu-cuda-fast'))
        if stage=='native-training-benchmark':
            subprocess.run([sys.executable,str(HERE/'native/qualify_backend_training.py'),'--out',str(out/'backend-parity')],check=True,env=env)
            modes=json.loads((out.parent/'native-learner-modes-1/summary.json').read_text())['checks']
            if modes.get('cpu-cuda-resume',{}).get('qualified'):variants.append('native-cpu-cuda-fast')
            for mode in ('compiled','tf32','bf16'):
                if modes.get(mode,{}).get('qualified') and modes.get(mode+'-resume',{}).get('qualified'):variants.append('native-cuda-'+mode)
        from benchmark_training import run_stage
        counts=(1024,) if stage=='native-batch-scaling' else (2048,) if stage=='native-hardware' else (256,)
        summary=run_stage(out/'training-comparison',env_counts=counts,updates=32,variants=tuple(variants),cores=int(os.getenv('CLASH_CPU_CORES','4')),ram_gib=int(os.getenv('CLASH_RAM_GIB','16')),gpu_model=torch.cuda.get_device_name(),stage_budget_seconds=1450)
        (out/'summary.json').write_text(json.dumps({'engine_qualification':qualification,'resume':json.loads((out/'current-source-resume/result.json').read_text()),'comparison':summary},indent=2)+'\n');return
    if stage=='native-learner-modes':
        subprocess.run([sys.executable,str(HERE/'native/export_data.py')],check=True)
        subprocess.run([sys.executable,str(HERE/'native/data_codegen.py')],check=True)
        sys.path.insert(0,str(HERE/'native'));from runtime import build_cuda
        import hashlib,signal
        qualification=qualified_native_engine(out)
        if hashlib.sha256(build_cuda().read_bytes()).hexdigest()!=qualification['cuda_library_sha256']:raise ValueError('current engine differs from retained whole-world GPU qualifier')
        env=dict(os.environ,CLASH_CUDA_LAUNCH_THREADS='1',CLASH_CUDA_RENDER_MODE='4',TORCHINDUCTOR_COMPILE_THREADS='4')
        checks={}
        def check(name,script,args,timeout=180,result_name='result.json'):
            command=[sys.executable,str(HERE/'native'/script),*args,'--out',str(out/name)]
            with (out/(name+'.log')).open('w') as log:
                process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
                try:code=process.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGKILL);process.wait();code=-9
            record={'exit_code':code,'qualified':code==0,'command':command}
            result_path=out/name/result_name
            if code==0 and result_path.exists():record['result']=json.loads(result_path.read_text())
            elif code:record['failure_tail']=(out/(name+'.log')).read_text()[-6000:]
            checks[name]=record;(out/'qualification-progress.json').write_text(json.dumps(checks,indent=2)+'\n')
            print(json.dumps({'qualification':name,'exit_code':code,'qualified':code==0}),flush=True)
            return code==0
        if not check('evaluation','qualify_cuda_evaluation.py',['--launch-threads','1','--render-mode','4'],result_name='summary.json'):raise RuntimeError('native CUDA evaluator qualification failed')
        compiled=check('compiled','qualify_compiled_loss.py',['--timeout-seconds','600'],timeout=620)
        common=['--cuda','--compact-embedding-grad','--bucket-actor','--bucket-learner','--minibatch','48','--gae-kernel','triton']
        if compiled:check('compiled-resume','test_training.py',common+['--compile-loss'],timeout=400)
        for mode in ('tf32','bf16'):
            if check(mode,'qualify_precision.py',['--mode',mode,'--compact-embedding-grad','--bucket-actor','--bucket-learner'],timeout=180):
                check(mode+'-resume','test_training.py',common+['--precision',mode],timeout=150)
        check('cpu-cuda-resume','test_training.py',['--backend','native-cpu-cuda','--compact-embedding-grad','--bucket-actor','--bucket-learner','--minibatch','48','--gae-kernel','triton'],timeout=150)
        (out/'summary.json').write_text(json.dumps({'engine_qualification':qualification,'checks':checks,'experimental_modes_are_optional':True,'performance_claimed':False},indent=2)+'\n');return
    if stage in ('native-bucket','native-codec'):
        subprocess.run([sys.executable,str(HERE/'native/export_data.py')],check=True)
        subprocess.run([sys.executable,str(HERE/'native/data_codegen.py')],check=True)
        sys.path.insert(0,str(HERE/'native'));from runtime import build_cuda
        import hashlib
        qualification=qualified_native_engine(out)
        if hashlib.sha256(build_cuda().read_bytes()).hexdigest()!=qualification['cuda_library_sha256']:raise ValueError('current engine differs from retained whole-world GPU qualifier')
        subprocess.run([sys.executable,str(HERE/'native/test_training.py'),'--cuda','--compact-embedding-grad','--bucket-actor','--bucket-learner','--minibatch','48','--gae-kernel','triton','--out',str(out/'bucket-resume')],check=True,env=dict(os.environ,CLASH_CUDA_LAUNCH_THREADS='1',CLASH_CUDA_RENDER_MODE='4'))
        from benchmark_training import run_stage
        summary=run_stage(out/'training-comparison',env_counts=(256,) if stage=='native-codec' else (32,256),updates=32,variants=('original-first-smoke','native-cuda-fast-raw','native-cuda-fast') if stage=='native-codec' else ('original-first-smoke','optimized-java','native-cuda-fast'),cores=int(os.getenv('CLASH_CPU_CORES','4')),ram_gib=int(os.getenv('CLASH_RAM_GIB','16')),gpu_model=torch.cuda.get_device_name())
        (out/'summary.json').write_text(json.dumps({'engine_qualification':qualification,'resume':json.loads((out/'bucket-resume/result.json').read_text()),'comparison':summary},indent=2)+'\n');return
    if stage=='native-kernel-sweep':
        subprocess.run([sys.executable,str(HERE/'native/export_data.py')],check=True)
        subprocess.run([sys.executable,str(HERE/'native/data_codegen.py')],check=True)
        sys.path.insert(0,str(HERE/'native'));from runtime import build_cuda
        import hashlib
        library_sha=hashlib.sha256(build_cuda().read_bytes()).hexdigest()
        # This stage qualifies its OWN binary against the pinned-Java-trained
        # trace before any timings. A previous binary's proof is never reused.
        (out/'current-binary.json').write_text(json.dumps({'cuda_library_sha256':library_sha,'qualification':'mandatory complete untimed CPU/CUDA trained-trace parity inside benchmark_cuda.py; no old binary qualification inherited'},indent=2)+'\n')
        subprocess.run([sys.executable,str(HERE/'native/qualify_cuda_render.py'),'--steps','128','--out',str(out/'render-qualification')],check=True)
        tape=ROOT/'evidence/clash-implementation/native-performance-20261108/cpu-baseline-1/generation/tape.npz'
        subprocess.run([sys.executable,str(HERE/'native/benchmark_cuda.py'),'--out',str(out/'kernel-sweep'),'--tape',str(tape),'--threads','1,32','--render-modes','1,4'],check=True)
        (out/'summary.json').write_text((out/'kernel-sweep/result.json').read_text());return
    if stage=='native-baselines':
        subprocess.run([sys.executable,str(HERE/'native/export_data.py')],check=True)
        subprocess.run([sys.executable,str(HERE/'native/data_codegen.py')],check=True)
        sys.path.insert(0,str(HERE/'native'));from benchmark_training import run_stage
        summary=run_stage(out/'training-comparison',updates=32,cores=int(os.getenv('CLASH_CPU_CORES','4')),ram_gib=int(os.getenv('CLASH_RAM_GIB','16')),gpu_model=torch.cuda.get_device_name())
        (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');return
    if stage=='native-optimized':
        subprocess.run([sys.executable,str(HERE/'native/export_data.py')],check=True)
        subprocess.run([sys.executable,str(HERE/'native/data_codegen.py')],check=True)
        checks={}
        for name,script,args in [
            ('engine','qualify_cuda.py',[str(out/'engine-qualification')]),
            ('render','qualify_cuda_render.py',['--steps','128','--out',str(out/'render-qualification')]),
            ('arithmetic','qualify_kernels.py',[str(out/'arithmetic-qualification')]),
            ('embedding','test_compact_embeddings.py',['--cuda','--out',str(out/'embedding-qualification')]),
            ('learner','qualify_learner_graph.py',['--out',str(out/'learner-qualification')]),
            ('compact-learner','qualify_learner_graph.py',['--compact-embedding-grad','--out',str(out/'compact-learner-qualification')]),
            ('resume','test_training.py',['--cuda','--compact-embedding-grad','--rollout-graph','--learner-graph','--gae-kernel','triton','--out',str(out/'resume-qualification')])]:
            subprocess.run([sys.executable,str(HERE/'native'/script),*args],check=True)
            checks[name]=True
        config=json.loads((HERE/'configs/cuda-throughput.json').read_text());config.update(backend='native-cuda',envs=256,torch_threads=1,threads=int(os.getenv('CLASH_CPU_CORES','4')),rollout_steps=32,minibatch=1024,epochs=2,updates=24,budget_seconds=1200,active_policy_loss=True,graph_sampling=False,optimizer_fused=True,target_kl=0,compact_embedding_grad=True,rollout_graph=True,learner_graph=True,gae_kernel='triton')
        config['hardware_allocation'].update(gpu_model=torch.cuda.get_device_name(),cpu_threads=int(os.getenv('CLASH_CPU_CORES','4')),ram_gib=int(os.getenv('CLASH_RAM_GIB','16')))
        path=out/'optimized-config.json';path.write_text(json.dumps(config,indent=2)+'\n')
        profiles={}
        for name,mode in [('optimized-profile',1),('cooperative-profile',4)]:
            env=dict(os.environ,CLASH_CUDA_LAUNCH_THREADS='1',CLASH_CUDA_RENDER_MODE=str(mode))
            subprocess.run([sys.executable,str(HERE/'native/profile_training.py'),'--config',str(path),'--out',str(out/name)],check=True,env=env)
            profiles[name]=json.loads((out/name/'result.json').read_text())
        experimental={}
        fixture=ROOT/'evidence/clash-implementation/native-pathfinding-20261108/gpu-features-2/java-public-observations.pt'
        for name,script,args in [
            ('device-branch','device_branch.py',['--out',str(out/'device-branch')]),
            ('bucket-actor','bucket_actor.py',['--fixture',str(fixture),'--out',str(out/'bucket-actor')]),
            ('bucket-learner','qualify_learner_graph.py',['--bucket','--compact-embedding-grad','--out',str(out/'bucket-learner')])]:
            with (out/(name+'.log')).open('w') as log:
                test=subprocess.run([sys.executable,str(HERE/'native'/script),*args],stdout=log,stderr=subprocess.STDOUT)
            experimental[name]={'exit_code':test.returncode}
            if test.returncode:experimental[name]['failure_tail']=(out/(name+'.log')).read_text()[-5000:];break
            experimental[name]['result']=json.loads((out/name/'result.json').read_text())
        (out/'summary.json').write_text(json.dumps({'checks':checks,'profiles':profiles,'experimental':experimental},indent=2)+'\n');return
    if stage in ('native-rng','native-rng-world'):
        subprocess.run([sys.executable,str(HERE/'native/diagnose_rng.py'),str(out/'rng')]+(['--cached-world-probe'] if stage=='native-rng-world' else []),check=True)
        (out/'summary.json').write_text((out/'rng/result.json').read_text())
        return
    if stage in ('native-cuda','native-training','native-profile'):
        subprocess.run([sys.executable,str(HERE/'native/export_data.py')],check=True)
        subprocess.run([sys.executable,str(HERE/'native/data_codegen.py')],check=True)
        if stage in ('native-cuda','native-profile'):
            subprocess.run([sys.executable,str(HERE/'native/qualify_cuda.py'),str(out/'qualification')],check=True)
            qualification=json.loads((out/'qualification/result.json').read_text())
        else:
            sys.path.insert(0,str(HERE/'native'));from runtime import build_cuda
            qualification=json.loads((out.parent/'native-cuda-4/qualification/result.json').read_text());import hashlib
            if hashlib.sha256(build_cuda().read_bytes()).hexdigest()!=qualification['cuda_library_sha256']:raise ValueError('retained CUDA qualification binary differs')
            (out/'retained-engine-qualification.json').write_text(json.dumps(qualification,indent=2)+'\n')
        if stage!='native-profile':subprocess.run([sys.executable,str(HERE/'native/test_training.py'),'--cuda','--out',str(out/'cuda-resume')],check=True)
        native_config=json.loads((HERE/'configs/cuda-throughput.json').read_text());native_config.update(backend='native-cuda',device='cuda',envs=256,threads=int(os.getenv('CLASH_CPU_CORES','4')),torch_threads=1,rollout_steps=32,minibatch=1024,epochs=2,updates=8,budget_seconds=1200,active_policy_loss=True,graph_sampling=True,optimizer_fused=True)
        native_config['hardware_allocation'].update(gpu_model=torch.cuda.get_device_name(),cpu_threads=int(os.getenv('CLASH_CPU_CORES','4')),ram_gib=int(os.getenv('CLASH_RAM_GIB','16')))
        (out/'native-training-config.json').write_text(json.dumps(native_config,indent=2)+'\n')
        if stage=='native-profile':
            profiles={}
            subprocess.run([sys.executable,str(HERE/'native/qualify_cuda_render.py'),'--out',str(out/'render-qualification')],check=True)
            graph_check=subprocess.run([sys.executable,str(HERE/'native/qualify_graph.py'),str(out/'native-training-config.json'),str(out/'graph-qualification')])
            for name,threads,render,graph in [('baseline',32,0,False),('spread',1,1,False)]+([('rollout-graph',1,1,True)] if graph_check.returncode==0 else []):
                config=dict(native_config,rollout_graph=graph,target_kl=0)
                path=out/(name+'-config.json');path.write_text(json.dumps(config,indent=2)+'\n')
                env=dict(os.environ,CLASH_CUDA_LAUNCH_THREADS=str(threads),CLASH_CUDA_RENDER_MODE=str(render))
                subprocess.run([sys.executable,str(HERE/'native/profile_training.py'),'--config',str(path),'--out',str(out/name)],check=True,env=env)
                profiles[name]=json.loads((out/name/'result.json').read_text())
            (out/'summary.json').write_text(json.dumps({'qualification':qualification,'graph_qualification_exit':graph_check.returncode,'profiles':profiles},indent=2)+'\n');return
        subprocess.run([sys.executable,str(HERE/'native/train.py'),'--config',str(out/'native-training-config.json'),'--out',str(out/'native-training')],check=True)
        summary={'qualification':qualification,'cuda_resume':json.loads((out/'cuda-resume/result.json').read_text()),'native_training':json.loads((out/'native-training/result.json').read_text())};(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        return
    c=json.loads((HERE/'configs/cuda-throughput.json').read_text())
    c.update(envs=32,updates=24,budget_seconds=1200,entity_bucket=0,inference_entity_bucket=200,checkpoint_interval=1000)
    c['hardware_allocation']['gpu_model']=torch.cuda.get_device_name()
    cores=int(os.getenv('CLASH_CPU_CORES','4'));ram=int(os.getenv('CLASH_RAM_GIB','16'))
    c.update(threads=cores);c['hardware_allocation'].update(cpu_threads=cores,ram_gib=ram)
    optimized=dict(preallocated_rollout=True,deferred_values=True,sparse_ppo=True,eval_skip_forced=True,actor_batch_bucket=8,inference_graph_cache_size=8,value_batch_size=1024,source_compresslevel=1)
    if stage in ('probe','validate','graphs'):
        with (out/'efficiency-tests.log').open('w') as log:
            check=subprocess.run([sys.executable,str(HERE/'tests/test_efficiency.py')],stdout=log,stderr=subprocess.STDOUT,timeout=120)
        if check.returncode:
            print((out/'efficiency-tests.log').read_text()[-7000:],flush=True);raise RuntimeError('efficiency tests failed')
    if stage=='probe':
        variants={'baseline':{},'sparse':{'sparse_ppo':True},'deferred':{k:v for k,v in optimized.items() if k!='sparse_ppo'},'both':optimized}
        for trial in range(2):
            for name in (variants if trial==0 else reversed(variants)):
                results[f'{name}-{trial}']=run(out,f'{name}-{trial}',dict(c,**variants[name]),baseline=name=='baseline')
        for envs in (128,256):
            results[f'both-env{envs}']=run(out,f'both-env{envs}',dict(c,envs=envs,updates=16,**optimized))
    elif stage=='validate':
        for script,arguments in (('validate_cuda.py',[str(out/'cuda-parity')]),('resume_check.py',[str(out/'cuda-resume'),'--cuda','--structural','--graph-native'])):
            with (out/(script+'.log')).open('w') as log:
                check=subprocess.run([sys.executable,str(HERE/'tests'/script),*arguments],stdout=log,stderr=subprocess.STDOUT,timeout=180)
            if check.returncode:
                print((out/(script+'.log')).read_text()[-7000:],flush=True);raise RuntimeError(script+' failed')
        for name,flags in (('baseline',{}),('both',optimized)):
            config=dict(c,device='cpu',envs=16,minibatch=256,updates=16,graph_inference=False,pinned_transfer=False,cache_rollout_device=False,**flags)
            config['hardware_allocation']={'gpu_model':'none (CPU comparison on allocated L4 worker)','gpu_count':1,'cpu_threads':4,'ram_gib':16,'qualification':'CPU execution charged to full L4 worker'}
            config['hardware_allocation']['gpu_model']=torch.cuda.get_device_name()
            # Declared allocation remains an L4 while computation deliberately uses CPU.
            if name=='both':config.update(ragged_entities=True,sparse_policy_head=True)
            results[name]=run(out,name,config,baseline=name=='baseline')
    elif stage=='graphs':
        # First qualify complete graph RNG before collecting timings.
        with (out/'cuda-parity.log').open('w') as log:
            check=subprocess.run([sys.executable,str(HERE/'tests/validate_cuda.py'),str(out/'cuda-parity')],stdout=log,stderr=subprocess.STDOUT,timeout=180)
        if check.returncode:
            print((out/'cuda-parity.log').read_text()[-7000:],flush=True);raise RuntimeError('CUDA graph qualification failed')
        variants={'baseline':{},'graph':dict(graph_sampling=True,preallocated_rollout=True,borrow_rollout_cache=True,source_compresslevel=1),'graph-fused':dict(graph_sampling=True,optimizer_fused=True,cache_rollout_fields=True,combined_learner_metrics=True,preallocated_rollout=True,borrow_rollout_cache=True,source_compresslevel=1)}
        for trial in range(3):
            for name in (variants if trial%2==0 else reversed(variants)):
                results[f'{name}-{trial}']=run(out,f'{name}-{trial}',dict(c,updates=32,**variants[name]),baseline=name=='baseline')
    elif stage in ('allocation-baseline','allocation-graph'):
        baseline=stage=='allocation-baseline'
        flags={} if baseline else dict(graph_sampling=True,preallocated_rollout=True,borrow_rollout_cache=True,eval_skip_forced=True,actor_batch_bucket=8,inference_graph_cache_size=8,source_compresslevel=1)
        config=dict(c,envs=128,updates=64,**flags)
        results['train']=run(out,'train',config,baseline=baseline)
        results['practice']=run(out,'practice',config,baseline=baseline,command='evaluate',init=False,extra=('--checkpoint',str(out/'train/checkpoint.pt'),'--schedule',str(HERE/'configs/eval-practice.json')))
    elif stage=='scaling':
        flags=dict(graph_sampling=True,preallocated_rollout=True,borrow_rollout_cache=True,source_compresslevel=1)
        for trial in range(2):
            for envs in ((32,128,512,1024) if trial==0 else (1024,512,128,32)):
                results[f'env{envs}-{trial}']=run(out,f'env{envs}-{trial}',dict(c,envs=envs,updates=16,**flags))
    elif stage=='native-scaling':
        with (out/'native-parity.log').open('w') as log:
            check=subprocess.run([sys.executable,str(HERE/'tests/native_parity.py'),str(out/'native-parity')],stdout=log,stderr=subprocess.STDOUT,timeout=180)
        if check.returncode:
            print((out/'native-parity.log').read_text()[-7000:],flush=True);raise RuntimeError('native thread/vector bookkeeping parity failed')
        # Retain the previously recommended 128-env/four-thread recipe on this
        # exact allocation, before changing vector width or native concurrency.
        results['baseline-env128']=run(out,'baseline-env128',dict(c,envs=128,threads=4,updates=16),baseline=True)
        flags=dict(graph_sampling=True,optimizer_fused=True,cache_rollout_fields=True,combined_learner_metrics=True,preallocated_rollout=True,borrow_rollout_cache=True,vector_bookkeeping=True,actor_chunk_size=512,inference_entity_bucket=64,source_compresslevel=1)
        for envs in ((256,1024,4096) if 'L4' in torch.cuda.get_device_name() else (1024,4096,8192)):
            results[f'env{envs}']=run(out,f'env{envs}',dict(c,envs=envs,updates=16,**flags))
    elif stage=='sim':
        with (out/'simulator.log').open('w') as log:
            subprocess.run([sys.executable,str(HERE/'simulator_profile.py'),'--out',str(out/'simulator'),'--threads',str(cores),'--jfr'],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1400)
    elif stage=='learning':
        learning_flags=dict(graph_sampling=True,optimizer_fused=True,cache_rollout_fields=True,combined_learner_metrics=True,preallocated_rollout=True,borrow_rollout_cache=True,vector_bookkeeping=True,eval_skip_forced=True,actor_batch_bucket=8,inference_graph_cache_size=8,source_compresslevel=1)
        for seed in (1701,2718):
            for name,flags in (('baseline',{}),('graph',learning_flags)):
                config=json.loads((HERE/'configs/cuda-learning-no-entropy.json').read_text())
                config.update(seed=seed,updates=200,budget_seconds=1200,**flags)
                config['hardware_allocation'].update(gpu_model=torch.cuda.get_device_name(),cpu_threads=cores,ram_gib=ram)
                key=f'{name}-{seed}'
                results[key]=run(out,key,config,baseline=name=='baseline',init=False,command='train')
                for phase,checkpoint in (('initial','initial.pt'),('teacher','post-warmstart.pt'),('final','checkpoint.pt')):
                    results[f'{key}-{phase}']=run(out,f'{key}-{phase}',config,baseline=name=='baseline',command='evaluate',init=False,extra=('--checkpoint',str(out/key/checkpoint),'--schedule',str(HERE/'configs/eval-practice.json')))
    (out/'summary.json').write_text(json.dumps(results,indent=2)+'\n')

if __name__=='__main__':main()
