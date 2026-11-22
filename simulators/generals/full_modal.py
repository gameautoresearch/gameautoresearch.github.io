"""Full-v2 qualification on fixed single-GPU workers; prior ledgers unchanged."""
import json
import time
import modal
from integrations.generals.modal_efficiency import image,ROOT

app=modal.App('autoresearch-generals-fullsize-v2-20261108')


def execute(mode,batches,seed,seconds,variant,minibatch_size):
    import io,os,subprocess,sys,tarfile
    from pathlib import Path
    start=time.monotonic();start_unix=time.time();out=Path('/artifacts');out.mkdir()
    with tarfile.open(out/'executed-source.tgz','w:gz') as tar:
        for p in Path('/workspace/integrations/generals').rglob('*'):
            if "autoresearch" in p.relative_to(Path('/workspace/integrations/generals')).parts:
                continue
            if p.is_file() and p.suffix in ('.py','.json','.lock'):
                tar.add(p,arcname=str(p.relative_to('/workspace')))
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name','--format=csv,noheader'],text=True).strip()
    template='full-h100' if 'H100' in gpu else 'full-l4'
    cfg=json.loads(Path(f'/workspace/integrations/generals/recipes/{template}.json').read_text())
    cfg['hardware']['gpu_model']=gpu
    env=dict(os.environ,GENERALS_ALLOCATION_CONFIRMED='1',GENERALS_ALLOCATION_START_UNIX=str(start_unix),
        JAX_COMPILATION_CACHE_DIR='/tmp/full-v2-cache',JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS='0')
    recipe=out/'recipe.json';checks=[]
    def command(args,name,timeout):
        t=time.monotonic()
        with (out/(name+'.log')).open('w') as log:
            try:r=subprocess.run([sys.executable]+args,cwd='/workspace',env=env,stdout=log,stderr=subprocess.STDOUT,timeout=timeout);code=r.returncode
            except subprocess.TimeoutExpired:code=124
        checks.append(dict(command=args,name=name,seconds=time.monotonic()-t,exit_code=code))
        (out/'commands.json').write_text(json.dumps(checks,indent=2)+'\n')
        print(json.dumps(checks[-1]),flush=True)
        return code==0
    if mode=='learn':
        b=int(batches);cfg.update(seed=seed,num_envs=b,minibatches=max(4,b*32//minibatch_size),
            allocated_worker_seconds=seconds,checkpoint_interval=1000000,checkpoint_reserve_seconds=20,
            evaluation_reserve_seconds=45,development_eval_seeds=4,max_updates=1000000)
        if variant=='legacy':
            cfg.update(execution='baseline',compute_dtype='float32',packed_rollout=False,
                historical_convolution='direct',compact_opponent_inference=False,sparse_reset_capacity=0)
        recipe.write_text(json.dumps(cfg,indent=2)+'\n')
        ok=command(['-m','integrations.generals.lease','--recipe',str(recipe),'--ledger',str(out/'allocation.json'),'--',
            sys.executable,'-m','integrations.generals','train','--device','gpu','--recipe',str(recipe),'--output',str(out/'train')],
            'train',seconds+45)
        if ok:
            ok=command(['-m','integrations.generals','evaluate','--device','gpu','--recipe',str(recipe),
                '--checkpoint',str(out/'train/latest.msgpack'),'--opponents','random,expansion','--split','practice',
                '--output',str(out/'evaluation')],'evaluate',240)
        if ok:
            command(['-m','integrations.generals.replay',str(out/'evaluation/replay.npz'),
                '--output',str(out/'replay-verification.json')],'replay',120)
    else:
        cfg.update(allocated_worker_seconds=1800)
        recipe.write_text(json.dumps(cfg,indent=2)+'\n')
        if mode=='validate':env['JAX_PLATFORMS']='cuda,cpu'
        for b in (batches.split(',') if mode=='profile' else ['128']):
            ok=command(['-m','integrations.generals.full_benchmark','--mode',mode,'--recipe',str(recipe),
                '--batch',b,'--output',str(out/('batch-'+b if mode=='profile' else 'validation'))],mode+'-'+b,1200)
            if not ok:break
        if mode=='validate' and ok:
            cfg.update(num_envs=32,minibatches=4,evaluate_before_after=False,allocated_worker_seconds=1800)
            recipe.write_text(json.dumps(cfg,indent=2)+'\n')
            folder=out/'cli';ckpt=folder/'train/latest.msgpack'
            cli=[['train','--max-updates','2','--output',str(folder/'train')],
                 ['train','--max-updates','4','--resume',str(ckpt),'--output',str(folder/'train')],
                 ['evaluate','--checkpoint',str(ckpt),'--split','practice','--opponents','random,expansion','--output',str(folder/'evaluation')],
                 ['profile','--output',str(folder/'profile')]]
            for i,args in enumerate(cli):
                ok=command(['-m','integrations.generals',args[0],'--device','gpu','--recipe',str(recipe)]+args[1:],f'cli-{i}',300)
                if not ok:break
            if ok:
                for backend in ('cuda','cpu'):
                    env['JAX_PLATFORMS']=backend
                    if not command(['-m','integrations.generals.replay',str(folder/'evaluation/replay.npz'),
                        '--output',str(folder/(backend+'-replay.json'))],backend+'-replay',180):break
    receipt=dict(mode=mode,variant=variant,seed=seed,recipe=cfg,checks=checks,
        function_seconds_before_archive=time.monotonic()-start,
        allocation=dict(gpu=gpu,gpu_count=1,cpu_physical_cores=4,ram_gib=16),
        accounting='Entire worker includes map setup/JIT, imports, all simulations, training, evaluation, checks, transfers, archives and idle; provider startup/shutdown reconciled separately.')
    (out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    data=io.BytesIO()
    with tarfile.open(fileobj=data,mode='w:gz') as tar:
        for p in out.rglob('*'):
            if p.is_file():tar.add(p,arcname=str(p.relative_to(out)))
    receipt['function_seconds']=time.monotonic()-start
    return data.getvalue(),receipt


@app.function(image=image,gpu='L4',cpu=(4,4),memory=(16384,16384),max_containers=3,
    min_containers=0,buffer_containers=0,retries=0,timeout=3600,startup_timeout=180,
    scaledown_window=2,single_use_containers=True)
def l4(mode:str,batches:str,seed:int,seconds:int,variant:str,minibatch_size:int):
    return execute(mode,batches,seed,seconds,variant,minibatch_size)


@app.function(image=image,gpu='H100!',cpu=(4,4),memory=(16384,16384),max_containers=3,
    min_containers=0,buffer_containers=0,retries=0,timeout=3600,startup_timeout=180,
    scaledown_window=2,single_use_containers=True)
def h100(mode:str,batches:str,seed:int,seconds:int,variant:str,minibatch_size:int):
    return execute(mode,batches,seed,seconds,variant,minibatch_size)


@app.local_entrypoint()
def main(tag:str,mode:str='profile',gpu:str='H100',batches:str='128,512,2048',
         seeds:str='42',seconds:int=600,variant:str='optimized',minibatch_size:int=4096):
    import hashlib,io,tarfile
    from modal.config import config
    if (config.get('environment')!='staging' or mode not in ('profile','validate','learn') or
        gpu not in ('L4','H100') or variant not in ('optimized','legacy') or not tag.replace('-','').isalnum()):raise ValueError('Invalid full-v2 launch')
    out=ROOT/'evidence/generals-implementation/fullsize-20261108'/mode/tag;out.mkdir(parents=True,exist_ok=False)
    row=dict(tag=tag,mode=mode,gpu=gpu,batches=batches,seeds=seeds,seconds=seconds,variant=variant,
        minibatch_size=minibatch_size,app_id=app.app_id,status='starting',start_unix=time.time(),calls=[],
        authorization='User explicitly requested full-size17..23 fog1v1 cap2048 standard, full validation and renewed throughput measurement; existing no-old-cap compute authorization preserved.')
    path=out/'launch.json';path.write_text(json.dumps(row,indent=2)+'\n');start=time.monotonic();calls=[]
    for seed in map(int,seeds.split(',')):
        call=(l4 if gpu=='L4' else h100).spawn(mode,batches,seed,seconds,variant,minibatch_size)
        record=dict(seed=seed,call_id=call.object_id,status='submitted');row['calls'].append(record);calls.append((call,record))
        path.write_text(json.dumps(row,indent=2)+'\n')
    try:
        for call,record in calls:
            binary,receipt=call.get();folder=out/f"seed-{record['seed']}";folder.mkdir()
            (folder/'artifacts.tar.gz').write_bytes(binary)
            with tarfile.open(fileobj=io.BytesIO(binary),mode='r:gz') as tar:tar.extractall(folder,filter='data')
            record.update(status='succeeded' if all(c['exit_code']==0 for c in receipt['checks']) else 'failed',receipt=receipt,
                archive_bytes=len(binary),archive_sha256=hashlib.sha256(binary).hexdigest())
            path.write_text(json.dumps(row,indent=2)+'\n')
        row['status']='succeeded' if all(c['status']=='succeeded' for c in row['calls']) else 'partial_failed'
    except BaseException as exc:
        row.update(status='client_error_inspect_before_retry',error_type=type(exc).__name__);raise
    finally:
        row['cold_launch_seconds']=time.monotonic()-start;path.write_text(json.dumps(row,indent=2)+'\n')
    print(json.dumps(row,indent=2),flush=True)
