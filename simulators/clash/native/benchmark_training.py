"""Matched full-loop original-Java/current-Java/native training benchmark setup.

No jobs are launched on import. --plan-only validates and records configurations
without training. Actual execution is called explicitly by the authorized worker;
all variants run serially on its same attached GPU/CPU/RAM allocation.
"""
from pathlib import Path
import argparse,contextlib,hashlib,json,math,os,signal,statistics,subprocess,sys,time
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path.insert(0,str(HERE))
from original_baseline_bundle import prepare_original,ARCHIVE_SHA256,RUNNER_SHA256,MODEL_SHA256
VARIANTS=('original-first-smoke','optimized-java','native-cpu-cuda','native-cuda')
SUPPORTED_VARIANTS=(*VARIANTS,'native-cuda-fast','native-cuda-fast-raw','native-cpu-cuda-fast','native-cuda-compiled','native-cuda-tf32','native-cuda-bf16','native-cuda-packed')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path,value):Path(path).write_text(json.dumps(value,sort_keys=True,indent=2)+'\n')


def configurations(root,envs,updates,seed,cores,ram_gib,gpu_name,*,rollout_steps=32,minibatch=1024):
    original=prepare_original(root);base=dict(original['manifest']['config'])
    base.update(device='cuda',seed=seed,envs=envs,threads=cores,torch_threads=1,updates=updates,budget_seconds=1200,rollout_steps=rollout_steps,minibatch=minibatch,epochs=2,width=64,embedding=12,opponent_mode='selfplay',reward_shaping=0.,active_policy_loss=False,warmstart_steps=0,snapshot_interval=1000000,checkpoint_interval=1000000,pool_size=8,target_kl=1e9,eval_greedy=False)
    # Preserve the actual earliest retained model's trim setting, rather than
    # deliberately changing its already-implemented optimization to a dense path.
    base['trim_entities']=True
    base['hardware_allocation']=dict(gpu_model=gpu_name,gpu_count=1,cpu_threads=cores,ram_gib=ram_gib,qualification='same exclusive allocated worker for all full-loop comparison arms')
    current=dict(base,pinned_transfer=True,combined_readback=True,cache_rollout_device=True,compile_forward=False,device_indices=True,graph_inference=False,graph_sampling=True,entity_bucket=0,inference_entity_bucket=200,optimizer_fused=True,cache_rollout_fields=True,combined_learner_metrics=True,preallocated_rollout=True,borrow_rollout_cache=True,vector_bookkeeping=True,actor_chunk_size=512,source_compresslevel=1,synchronize_timing=False)
    native=dict(base,backend='native-cuda',graph_sampling=True,optimizer_fused=True,max_ticks=0)
    fast=dict(native,compact_embedding_grad=True,learner_graph=True,bucket_actor=True,bucket_learner=True,rollout_graph=False,gae_kernel='triton',cuda_launch_threads=1,cuda_render_mode=4)
    return {'original-first-smoke':base,'optimized-java':current,'native-cpu-cuda':dict(native,backend='native-cpu-cuda'),'native-cuda':native,'native-cuda-fast':fast,'native-cuda-fast-raw':dict(fast,checkpoint_codec='raw'),'native-cpu-cuda-fast':dict(fast,backend='native-cpu-cuda'),'native-cuda-compiled':dict(fast,compile_loss=True),'native-cuda-tf32':dict(fast,precision='tf32'),'native-cuda-bf16':dict(fast,precision='bf16'),'native-cuda-packed':dict(fast,packed_head=True)}


def make_plan(out,*,root=ROOT,env_counts=(32,256),updates=8,trials=2,cores=4,ram_gib=16,gpu_model='MATCHED_ALLOCATED_GPU',variants=VARIANTS,rollout_steps=32,minibatch=1024):
    out=Path(out).resolve();root=Path(root).resolve();out.mkdir(parents=True,exist_ok=True)
    if (out/'comparison-plan.json').exists():raise FileExistsError('comparison plan already exists')
    if cores<1 or ram_gib<1 or updates<1 or trials<1 or rollout_steps<1 or minibatch<1 or any(n<1 for n in env_counts):raise ValueError('positive explicit workload/resource values required')
    if not variants or len(set(variants))!=len(variants) or any(v not in SUPPORTED_VARIANTS for v in variants):raise ValueError('unknown/duplicate comparison variant')
    original=prepare_original(root);runs=[]
    for trial in range(trials):
        for envs in (env_counts if trial%2==0 else tuple(reversed(env_counts))):
            configs=configurations(root,envs,updates,1701+101*trial,cores,ram_gib,gpu_model,rollout_steps=rollout_steps,minibatch=minibatch)
            for variant in (variants if trial%2==0 else tuple(reversed(variants))):
                name=f'{variant}-env{envs}-trial{trial+1}';config_path=out/(name+'.json');dump(config_path,configs[variant]);run_out=out/name
                script=original['runner'] if variant=='original-first-smoke' else root/'integrations/clash/runner.py' if variant=='optimized-java' else root/'integrations/clash/native/train.py'
                command=[sys.executable,str(script)]+(['profile'] if variant in VARIANTS[:2] else [])+['--config',str(config_path),'--out',str(run_out)]
                expected_samples=envs*2*rollout_steps*updates;expected_steps=updates*2*math.ceil((envs*2*rollout_steps)/minibatch)
                runs.append(dict(name=name,variant=variant,trial=trial,envs=envs,seed=configs[variant]['seed'],config=str(config_path),config_sha256=sha(config_path),command=command,script_sha256=sha(script),out=str(run_out),expected_updates=updates,expected_learner_samples=expected_samples,expected_optimizer_steps=expected_steps))
    plan=dict(format=1,status='prepared',runs=runs,resource=dict(gpu=gpu_model,gpu_count=1,cpu_cores=cores,ram_gib=ram_gib),source=dict(original_archive_sha256=ARCHIVE_SHA256,original_runner_sha256=RUNNER_SHA256,original_model_sha256=MODEL_SHA256,current_java_runner_sha256=sha(root/'integrations/clash/runner.py'),native_trainer_sha256=sha(root/'integrations/clash/native/train.py')),workload=dict(model_width=64,embedding=12,rollout_steps=rollout_steps,epochs=2,minibatch=minibatch,rewards='sparse terminal selfplay, no shaping/teacher',policy_loss='all valid rows, including forced waits',target_kl=1e9,initialization='fresh seeded random policies; no pretrained checkpoint'),trajectory_identity_claimed=False,trajectory_scope='same frozen game/model shape/configured PPO data volume; samplers, initial-engine seed generators and minibatch RNG layouts differ across historical/current/native implementations',upstream_learner_claimed=False,pre_all_optimization_snapshot_claimed=False,original_features_preserved=['pinned upstream persistent vector threadpool','upstream direct buffers and vector bridge','earliest retained live-entity trimming'],normalization_notes=['Original integration is not an upstream PPO learner.','Original source is exact; performance instrumentation/sampling differences remain part of historical implementation.','Matched architecture/hyperparameters are not proof of trajectory identity or equal policy strength.','Provider allocation receipt must include stage setup, idle gaps and artifact finalization beyond per-process timing.'])
    dump(out/'comparison-plan.json',plan);return plan


def _stop(process):
    if process.poll() is not None:return
    try:os.killpg(process.pid,signal.SIGTERM)
    except ProcessLookupError:return
    try:process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:os.killpg(process.pid,signal.SIGKILL)
        except ProcessLookupError:pass
        process.wait()


def _metrics(run):
    folder=Path(run['out']);result=json.loads((folder/'result.json').read_text());metrics_path=folder/'metrics.jsonl';rows=[json.loads(line) for line in metrics_path.read_text().splitlines()] if metrics_path.exists() else []
    count=result.get('counts',{});native=run['variant'].startswith('native-');samples=result.get('learner_samples') if native else count.get('learner_decisions')
    invalid=sum(count.get(k,0) for k in ('illegal_actions','unsupported_actions','truncation_episodes'))+sum(v for k,v in count.items() if k.startswith('fault_'))
    if native:
        c=result['counters'];invalid=int(c[4])+int(c[5]);activity=dict(natural_matches=int(c[0]),truncations=int(c[4]),ticks=int(c[6]),nonforced_player_rows=int(c[7]),late_game_player_rows=int(c[8]),maximum_exposed_entities=int(c[10]));peak=result.get('peak_cuda_bytes')
    else:
        activity=dict(natural_matches=count.get('natural_terminals',0),truncations=count.get('truncation_episodes',0),ticks=count.get('ticks'),nonforced_player_rows=sum(r.get('active_policy_samples',0) for r in rows) if any('active_policy_samples' in r for r in rows) else None,late_game_player_rows=count.get('late_game_player_decisions'),maximum_exposed_entities=count.get('entity_count_max'));peak=result.get('cuda_peak_memory_allocated')
    steps=sum(r.get('optimizer_steps',0) for r in rows)
    iterations=[]
    for i,row in enumerate(rows):
        if 'iteration_seconds' in row:iterations.append(row['iteration_seconds'])
        elif i and 'allocated_worker_seconds' in row:iterations.append(row['allocated_worker_seconds']-rows[i-1]['allocated_worker_seconds'])
    complete=result.get('status')=='completed' and result.get('updates')==run['expected_updates'] and samples==run['expected_learner_samples'] and steps==run['expected_optimizer_steps'] and invalid==0 and not result.get('fault_signatures')
    return dict(status='qualified-workload' if complete else 'invalid-or-incomplete',updates=result.get('updates'),learner_samples=samples,optimizer_steps=steps,invalid_transitions=invalid,activity=activity,peak_torch_cuda_bytes=peak,warm_iteration_median_seconds=statistics.median(iterations[2:]) if len(iterations)>2 else None,phase_timing=result.get('timing'),raw_result=result)


def _run(run,out,deadline):
    name=run['name'];start=time.perf_counter();command=run['command'];env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8');monitor=None;monitor_log=None;process=None;error=None;returncode=None
    if run['variant'] in ('native-cuda-fast','native-cuda-fast-raw','native-cpu-cuda-fast','native-cuda-compiled','native-cuda-tf32','native-cuda-bf16','native-cuda-packed'):env.update(CLASH_CUDA_LAUNCH_THREADS='1',CLASH_CUDA_RENDER_MODE='4')
    memory=out/(name+'-cpu-memory.txt')
    if Path('/usr/bin/time').exists() and sys.platform=='linux':command=['/usr/bin/time','-v','-o',str(memory),*command]
    try:
        remaining=deadline-time.perf_counter()
        if remaining<30:raise TimeoutError('stage budget cannot fit another full-loop process')
        import shutil
        if shutil.which('nvidia-smi'):
            monitor_log=(out/(name+'-gpu-memory.csv')).open('w');monitor=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,memory.used,utilization.gpu','--format=csv,noheader,nounits','--loop-ms=500'],stdout=monitor_log,stderr=subprocess.STDOUT,start_new_session=True)
        with (out/(name+'.log')).open('w') as log:
            process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
            try:returncode=process.wait(timeout=min(1500,remaining))
            except subprocess.TimeoutExpired as exc:_stop(process);raise TimeoutError('full-loop process exceeded bounded stage time') from exc
        if returncode:raise RuntimeError(f'{name} returned{ returncode }; see retained log')
        record=_metrics(run)
    except BaseException as exc:error=f'{type(exc).__name__}: {exc}';record=dict(status='failed',error=error)
    finally:
        if process is not None:_stop(process)
        if monitor is not None:_stop(monitor)
        if monitor_log is not None:monitor_log.close()
    record.update(name=name,variant=run['variant'],trial=run['trial'],envs=run['envs'],seed=run['seed'],command=command,config_sha256=run['config_sha256'],script_sha256=run['script_sha256'],subprocess_returncode=returncode,allocated_arm_span_seconds=time.perf_counter()-start,scope='full subprocess initialization, archive/checkpoint overhead, transfers, rollout, GAE, learner and reset on same live allocation')
    if record.get('learner_samples'):record['learner_samples_per_arm_span_second']=record['learner_samples']/record['allocated_arm_span_seconds']
    dump(out/(name+'-receipt.json'),record);print(json.dumps({k:record.get(k) for k in ('name','status','allocated_arm_span_seconds','learner_samples','optimizer_steps')}),flush=True)
    if error or record['status']!='qualified-workload':raise RuntimeError(error or name+' did not complete the matched workload')
    return record


def run_stage(out,*,root=ROOT,env_counts=(32,256),updates=8,trials=2,cores=None,ram_gib=None,gpu_model=None,variants=VARIANTS,rollout_steps=32,minibatch=1024,stage_budget_seconds=1500):
    start=time.perf_counter();root=Path(root).resolve();out=Path(out).resolve()
    import torch
    if torch.cuda.device_count()!=1:raise ValueError('exactly one visible GPU required; no CPU-only fallback')
    actual=torch.cuda.get_device_name();gpu_model=actual if gpu_model is None else gpu_model
    if actual!=gpu_model:raise ValueError('allocated GPU differs from requested comparison GPU')
    cores=int(os.getenv('CLASH_CPU_CORES','4')) if cores is None else cores;ram_gib=int(os.getenv('CLASH_RAM_GIB','16')) if ram_gib is None else ram_gib
    plan=make_plan(out,root=root,env_counts=tuple(env_counts),updates=updates,trials=trials,cores=cores,ram_gib=ram_gib,gpu_model=gpu_model,variants=tuple(variants),rollout_steps=rollout_steps,minibatch=minibatch)
    results=[];failure=None;setup=time.perf_counter()-start;lock_path=root/'evidence/local-performance.lock';lock_path.parent.mkdir(parents=True,exist_ok=True)
    try:
        import fcntl
        with lock_path.open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            for run in plan['runs']:
                results.append(_run(run,out,start+stage_budget_seconds));dump(out/'comparison-progress.json',results)
    except BaseException as exc:failure=f'{type(exc).__name__}: {exc}'
    groups={}
    for run in results:groups.setdefault((run['variant'],run['envs']),[]).append(run)
    medians=[dict(variant=v,envs=n,trials=len(items),median_full_arm_seconds=statistics.median(r['allocated_arm_span_seconds'] for r in items),median_learner_samples_per_second=statistics.median(r['learner_samples_per_arm_span_second'] for r in items)) for (v,n),items in groups.items()]
    summary=dict(status='failed' if failure else 'completed',error=failure,resource=plan['resource'],setup_seconds=setup,allocated_stage_seconds=time.perf_counter()-start,completed_runs=len(results),planned_runs=len(plan['runs']),runs=results,medians=medians,trajectory_identity_claimed=False,strength_gain_claimed=False,provider_allocation_receipt_required=True,source=plan['source'])
    dump(out/'summary.json',summary)
    if failure:raise RuntimeError(failure)
    return summary


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--envs',default='32,256');p.add_argument('--updates',type=int,default=8);p.add_argument('--trials',type=int,default=2);p.add_argument('--cpu-cores',type=int,default=int(os.getenv('CLASH_CPU_CORES','4')));p.add_argument('--ram-gib',type=int,default=int(os.getenv('CLASH_RAM_GIB','16')));p.add_argument('--gpu-model');p.add_argument('--variants',default=','.join(VARIANTS));p.add_argument('--rollout-steps',type=int,default=32);p.add_argument('--minibatch',type=int,default=1024);p.add_argument('--stage-budget-seconds',type=float,default=1500);p.add_argument('--plan-only',action='store_true');a=p.parse_args()
    kwargs=dict(root=a.root,env_counts=tuple(map(int,a.envs.split(','))),updates=a.updates,trials=a.trials,cores=a.cpu_cores,ram_gib=a.ram_gib,gpu_model=a.gpu_model or 'MATCHED_ALLOCATED_GPU',variants=tuple(a.variants.split(',')),rollout_steps=a.rollout_steps,minibatch=a.minibatch)
    if a.plan_only:result=make_plan(a.out,**kwargs)
    else:
        kwargs['gpu_model']=a.gpu_model;result=run_stage(a.out,stage_budget_seconds=a.stage_budget_seconds,**kwargs)
    print(json.dumps({k:v for k,v in result.items() if k not in ('runs','source')}))
if __name__=='__main__':main()
