"""One isolated, cumulatively bounded staging reference qualification on Modal.

Run from repository root, in the Generals host venv. Image build and immutable
uploads happen before any GPU allocation. Every attempt/receipt is retained.
"""
from pathlib import Path
import argparse,fcntl,hashlib,json,os,shutil,sys,time,subprocess,threading
from .bootstrap import ROOT,HERE,UPSTREAM,ENGINE,sha,binding


def write(path,value):
    path=Path(path);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT/'evidence/generals-averagejoe-20261109');p.add_argument('--inputs',type=Path,required=True);p.add_argument('--tag',default='smoke-01');p.add_argument('--grant-seconds',type=int,default=600);a=p.parse_args()
    if not a.tag.replace('-','').isalnum() or not 120<=a.grant_seconds<=600:raise ValueError('Bounded tag and120..600seconds required')
    a.root.mkdir(parents=True,exist_ok=True);folder=a.root/a.tag;folder.mkdir(exist_ok=False)
    from simulators.generals.autoresearch.provider import configure_sandbox_waits,allocation_info,read_rpc
    configure_sandbox_waits()
    import modal
    from modal.config import config
    authorization={'scope':'Actual AverageJoe transformer/memory/PPO/curriculum reference pilot on Modal, separately from all old campaigns','user_instruction':'use the actual AverageJoe recipe with the curriculum. run the small like 10 minute check or something to make sure it works well on modal','gpu_seconds_cap':600,'gpu_count':1,'gpu':'H100!','cpu_physical_cores':4,'memory_gib':16,'model_calls':0,'no_automatic_gpu_retries':True,'accounting':'Full provider GPU allocation including setup/loading/compilation/learning/evaluation/saving/transfer/idle, across all attempts. CPU preprocessing separately recorded. No old ledger renewed.'}
    auth=a.root/'authorization.json'
    if not auth.exists():write(auth,authorization)
    elif json.loads(auth.read_text())!=authorization:raise ValueError('Authorization drift')
    # Immutable source image ensures editing another game cannot disrupt upload.
    snap=folder/'source';snap.mkdir();(snap/'simulators/generals').mkdir(parents=True)
    shutil.copytree(HERE,snap/'simulators/generals/averagejoe',ignore=shutil.ignore_patterns('__pycache__','.pytest_cache'))
    for n in ['simulators/__init__.py','simulators/generals/__init__.py']:
        source=ROOT/n
        if source.exists():shutil.copy2(source,snap/n)
    (snap/'vendor/engines').mkdir(parents=True)
    shutil.copytree(UPSTREAM,snap/'vendor/engines/AverageJoe',ignore=shutil.ignore_patterns('.git','__pycache__','.pytest_cache'))
    shutil.copytree(ENGINE,snap/'vendor/engines/generals-bots',ignore=shutil.ignore_patterns('.git','__pycache__','.pytest_cache'))
    inputs=folder/'inputs';shutil.copytree(a.inputs,inputs)
    write(folder/'source-binding.json',binding())
    source_files={str(f.relative_to(snap)):sha(f) for f in snap.rglob('*') if f.is_file()};write(folder/'source-files.json',source_files)
    image=(modal.Image.debian_slim(python_version='3.12').apt_install('git')
        .pip_install('jax[cuda12]==0.12.0.1','flax==0.12.12','optax==0.2.8','numpy==2.5.4','pygame==2.6.1','python-socketio[client]==5.18.1','pytest==9.2.0','psutil==7.2.2')
        .pip_install('equinox==0.13.8','ruamel.yaml==0.19.1')
        .add_local_dir(snap,'/workspace',copy=True).add_local_dir(inputs,'/inputs',copy=True)
        .env({'PYTHONPATH':'/workspace','JAX_PLATFORMS':'cuda','JAX_DEFAULT_MATMUL_PRECISION':'tensorfloat32','XLA_PYTHON_CLIENT_PREALLOCATE':'false','OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4'})
        .workdir('/workspace'))
    app=modal.App('generals-averagejoe-curriculum-smoke-20261109')
    ledger_path=a.root/'gpu-budget.json';lock=(a.root/'gpu-budget.lock').open('a+')
    sb=None;proc=None;row=None
    with modal.enable_output(),app.run(environment_name='staging'):
        write(folder/'app.json',{'app_id':app.app_id,'environment':'staging'})
        t=time.monotonic();image.build(app);write(folder/'image-build.json',{'elapsed_seconds':time.monotonic()-t,'gpu_allocations':0,'reserve_usd':.5})
        fcntl.flock(lock,fcntl.LOCK_EX)
        ledger=json.loads(ledger_path.read_text()) if ledger_path.exists() else {'cap_gpu_seconds':600,'attempts':[]}
        used=sum(v.get('allocated_seconds',v['reserved_seconds']) for v in ledger['attempts'])
        if any(v['status'] not in ('settled','failed_settled') for v in ledger['attempts']):raise ValueError('Existing unresolved worker blocks another allocation')
        if used+a.grant_seconds>600:raise ValueError('Cumulative fresh600second GPU budget exhausted')
        row={'tag':a.tag,'status':'reserved','reserved_seconds':a.grant_seconds,'created_unix':time.time(),'app_id':app.app_id,'rate_gpu_usd_per_second':.001097,'rate_sandbox_cpu_usd_per_second_per_core':.00003942,'rate_sandbox_ram_usd_per_second_per_gib':.00000667}
        ledger['attempts'].append(row);write(ledger_path,ledger);fcntl.flock(lock,fcntl.LOCK_UN)
        try:
            sb=modal.Sandbox.create('/bin/sleep','86400',app=app,image=image,gpu='H100!',cpu=(4,4),memory=(16384,16384),timeout=a.grant_seconds-10,block_network=True,secrets=[],volumes={},environment_name='staging',include_oidc_identity_token=False)
            row.update(status='running',sandbox_id=sb.object_id);write(ledger_path,ledger);write(folder/'provider.json',dict(row))
            info=None
            for _ in range(120):
                info=allocation_info(app.app_id,sb.object_id)
                if info and info['started_at']:break
                time.sleep(.5)
            if not info or not info['started_at']:raise RuntimeError('Cannot verify provider allocation start after bounded startup wait')
            if 'H100' not in info['gpu_type'].upper():raise RuntimeError('Wrong GPU allocation: '+info['gpu_type'])
            deadline=info['started_at']+a.grant_seconds-15
            remaining=deadline-time.time();process_seconds=max(30,min(a.grant_seconds-70,remaining-55))
            command=['python','-u','-m','simulators.generals.averagejoe.train','--inputs','/inputs','--output','/out','--seconds',str(process_seconds)]
            # Shell redirects keep output durable even if the controller reconnects.
            import shlex
            shell=shlex.join(command)+' > /trainer.log 2>&1'
            proc=sb.exec('/bin/sh','-c',shell,timeout=max(30,int(remaining-20)))
            write(folder/'command.json',{'command':command,'provider_started_at':info['started_at'],'provider_deadline':deadline,'process_seconds':process_seconds})
            last=''
            while read_rpc(proc.poll) is None and time.time()<deadline-25:
                try:
                    value=read_rpc(lambda:sb.filesystem.read_text('/trainer.log'))
                    (folder/'trainer.log').write_text(value)
                    tail=value.splitlines()[-1] if value else ''
                    if tail and tail!=last:print('REMOTE:',tail,flush=True);last=tail
                except Exception as e:print('Log poll:',type(e).__name__,flush=True)
                time.sleep(5)
            rc=read_rpc(proc.poll);row['trainer_returncode']=rc;write(ledger_path,ledger)
            if rc is None:
                print('Original provider deadline near; retaining partial attempt, no retry.',flush=True)
                kill=sb.exec('/bin/sh','-c','pkill -TERM -f simulators.generals.averagejoe.train || true',timeout=2)
                time.sleep(1)
            try:(folder/'trainer.log').write_text(read_rpc(lambda:sb.filesystem.read_text('/trainer.log')))
            except Exception:pass
            # Fetch small evidence first, then final state/models if available.
            artifacts=folder/'artifacts';artifacts.mkdir()
            files=['result.json','progress.jsonl','events.jsonl','runtime.json','source.json','config.json','config.yaml','initial.json','after-first-update.json','final-state.json','final-state.eqx','final.eqx','final-ema.eqx']
            failures={}
            for name in files:
                if time.time()>deadline-3:failures[name]='provider deadline reached';continue
                try:read_rpc(lambda name=name:sb.filesystem.copy_to_local('/out/'+name,artifacts/name))
                except Exception as e:failures[name]=type(e).__name__+': '+str(e)[:300]
            write(folder/'export-errors.json',failures)
            row['result_received']=(artifacts/'result.json').exists();row['exported_files']={p.name:sha(p) for p in artifacts.iterdir() if p.is_file()}
        except BaseException as exc:
            row.update(error=type(exc).__name__+': '+str(exc),status='failed_pending_reconciliation')
            import traceback
            (folder/'host-error.txt').write_text(traceback.format_exc());raise
        finally:
            if sb is not None:
                try:sb.terminate(wait=True)
                except Exception as e:row['termination_error']=str(e)
                final=None
                for _ in range(30):
                    final=allocation_info(app.app_id,sb.object_id)
                    if final and final.get('started_at') and final.get('finished_at'):break
                    time.sleep(.5)
                if final and final.get('finished_at'):
                    row.update(status='settled' if row.get('trainer_returncode')==0 and row.get('result_received') else 'failed_settled',provider=final,allocated_seconds=final['finished_at']-final['started_at'])
                    rate=.001097+4*.00003942+16*.00000667
                    row['estimated_compute_usd']=row['allocated_seconds']*rate;row['invoice_verified']=False
                else:row['status']='allocation_unresolved_hold_retained'
            write(ledger_path,ledger);write(folder/'allocation.json',row)
    print(json.dumps(row,indent=2),flush=True)
    if row['status']!='settled':raise RuntimeError('Pilot incomplete; inspect preserved attempt before any further allocation')

if __name__=='__main__':main()
