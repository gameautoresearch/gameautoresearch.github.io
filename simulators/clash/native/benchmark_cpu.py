"""Locked, matched trained-policy simulator tape: pinned Java4 vs native CPU batch4.

The legacy checkpoint initializes weights only. No optimizer, engine state, private
state or sampler state is resumed. Timings are raw simulation/observation/mask/reset
throughput, not learner or allocated-GPU-hour measurements.
"""
from __future__ import annotations
import time
SCRIPT_START=time.perf_counter()
from pathlib import Path
import argparse,fcntl,hashlib,importlib.util,json,os,platform,resource,shutil,statistics,subprocess,sys
import numpy as np
HERE=Path(__file__).resolve().parent
INTEGRATION=HERE.parent
ROOT=INTEGRATION.parent.parent
sys.path.insert(0,str(INTEGRATION))

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path,value):Path(path).write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
def progress(event,**fields):print(json.dumps(dict(event=event,**fields)),flush=True)
def snapshot_processes():
 try:return subprocess.check_output(['ps','-Ao','pid,pcpu,comm'],text=True).splitlines()[:1]+sorted(subprocess.check_output(['ps','-Ao','pid,pcpu,comm'],text=True).splitlines()[1:],key=lambda row:float(row.split(None,2)[1]),reverse=True)[:12]
 except Exception:return []
def resources():
 result={'system':platform.system(),'release':platform.release(),'machine':platform.machine(),'python':sys.version,'cpu_count':os.cpu_count(),'maxrss_platform_units':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'cpu_affinity_enforced':False}
 if sys.platform=='darwin':
  for name in ['machdep.cpu.brand_string','hw.physicalcpu','hw.logicalcpu','hw.memsize']:
   try:result[name]=subprocess.check_output(['sysctl','-n',name],text=True).strip()
   except Exception:pass
 else:
  try:result['cpuinfo']=Path('/proc/cpuinfo').read_text().split('\n\n')[0]
  except Exception:pass
 return result

def arrays(env):return env.observations,env.masks,env.infos,env.rewards

def done_flags(env):
 if hasattr(env,'dones'):return np.asarray(env.dones,dtype=np.uint8)
 return ((env.infos[:,2]>=0)|(env.infos[:,9]!=0)|(env.infos[:,5]!=0)).astype(np.uint8)
def digest(env):
 h=hashlib.sha256()
 for a in (*arrays(env),done_flags(env)):h.update(a.tobytes())
 return h.hexdigest()

def native_module(snapshot):
 spec=importlib.util.spec_from_file_location('benchmark_static_cpu_batch',snapshot/'cpu_batch.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def make_backend(kind,config,seeds,snapshot,library):
 if kind=='java4':
  from game import make_vector
  env=make_vector(config);env.reset(seeds.tolist(),config['rules']['deck'],config['rules']['deck']);return env
 module=native_module(snapshot);module.build=lambda:library
 return module.NativeCPUBatch(seeds,threads=4 if kind=='native4' else 1)

def reset_all(env,kind,seeds,deck):
 if kind=='java4':env.reset(seeds.tolist(),deck,deck)
 else:env.reset(seeds)

def reset_selected(env,kind,reset_seeds):
 selected=reset_seeds>=0
 if not selected.any():return
 if kind=='java4':
  for lane in np.flatnonzero(selected):env.reset_env(int(lane),int(reset_seeds[lane]))
 else:env.reset(np.where(selected,reset_seeds,0).astype(np.int32),selected.astype(np.int32))

def assert_valid(env,label):
 if (env.infos[:,5]!=0).any() or (env.infos[:,11]!=0).any() or (env.infos[:,7:9]!=0).any() or (env.infos[:,9]!=0).any():raise AssertionError(label+' invalid/fault/unsupported/illegal/truncated transition')

def phase_generate(args,config):
 phase_start=time.perf_counter();args.out.mkdir(parents=True,exist_ok=False)
 from game import make_vector,Session,canonical
 from runner import setup,network,sample,Timer
 import torch
 config=dict(config);config.update(seed=args.policy_seed,device='cpu',torch_threads=1,envs=args.envs,threads=4,compile_forward=False,graph_inference=False,graph_sampling=False)
 rng=setup(config);cp=Path(args.checkpoint);saved=torch.load(cp,map_location='cpu',weights_only=False)
 if saved['rules_sha256']!=canonical(json.loads((INTEGRATION/'rules.lock.json').read_text())):raise ValueError('initial weights rules differ')
 for name in ('width','embedding'):config[name]=int(saved['config'][name])
 counts={'environment_decisions':args.envs*args.decisions,'seat_decisions':2*args.envs*args.decisions,'natural_terminals':0,'truncations':0,'faults':0,'illegal_actions':0,'unsupported':0,'late_seat_decisions':0,'nonforced_seat_decisions':0,'entity_overflow_observations':0,'actual_ticks':0,'reset_lanes':0};casts={name:0 for name in config['rules']['deck']};requested={name:0 for name in config['rules']['deck']};occupancy=[];terminal_winners={};digests={}
 with make_vector(config) as v:
  v.check_metadata(saved['metadata']);model=network(v,config);model.load_state_dict(saved['model']);model.eval();timer=Timer('cpu');session=Session(v,config,rng);initial=np.array(session.seeds,np.int32);tape=np.empty((args.decisions,args.envs,2),np.int32);resets=np.full((args.decisions,args.envs),-1,np.int32);dones=np.zeros((args.decisions,args.envs),np.uint8)
  for t in range(args.decisions):
   obs=v.observations;occupancy.extend(obs[...,48].reshape(-1).tolist());counts['late_seat_decisions']+=int((obs[...,0]>=2400).sum());counts['nonforced_seat_decisions']+=int((v.masks.sum(-1)>1).sum());counts['entity_overflow_observations']+=int((obs[...,49]>0).sum());before_tick=v.infos[:,0].copy();before_placed=obs[:,[1,0],114].copy()
   actions=sample(model,obs.reshape(-1,v.schema.seat_ints),v.masks.reshape(-1,v.schema.actions),config,timer)[0].reshape(args.envs,2).astype(np.int32)
   if not v.masks[np.arange(args.envs)[:,None],np.arange(2)[None,:],actions].all():raise AssertionError('masked generated action')
   for lane,side in np.argwhere((actions>0)&(actions<2305)):
    card=v.vocabulary[int(obs[lane,side,7+(int(actions[lane,side])-1)//576])];requested[card]+=1
   tape[t]=actions;session.step(actions);assert_valid(v,'generation');dones[t]=v.dones
   counts['actual_ticks']+=int((v.infos[:,0]-before_tick).sum());delta=v.observations[:,[1,0],114]-before_placed
   for lane,side in np.argwhere(delta>0):
    n=int(delta[lane,side]);assert 0<n<=8
    for r in range(n):
     card=v.vocabulary[int(v.observations[lane,1-side,116+r*9])];casts[card]+=1
   for lane in np.flatnonzero(v.dones):
    winner=str(int(v.infos[lane,2]));terminal_winners[winner]=terminal_winners.get(winner,0)+1
   counts['natural_terminals']+=int(v.dones.sum());finished=session.reset_done();counts['reset_lanes']+=len(finished)
   for lane in finished:resets[t,lane]=session.seeds[lane]
   if (v.infos[:,6]!=0).any():raise AssertionError('manual VectorEnv reset unexpectedly wrote transport reset flag')
   if (t+1)%128==0 or t+1==args.decisions:digests[str(t)]=digest(v)
   if (t+1)%512==0:progress('tape_generation',decisions=t+1,envs=args.envs,natural_terminals=counts['natural_terminals'],max_entities=max(occupancy))
  assert all(casts.values()),casts;assert counts['late_seat_decisions']>0;assert counts['natural_terminals']>0;assert counts['entity_overflow_observations']==0
  np.savez_compressed(args.out/'tape.npz',actions=tape,reset_seeds=resets,initial_seeds=initial,dones=dones)
  metadata=v.metadata
 result={'status':'generated','checkpoint':str(cp),'checkpoint_sha256':sha(cp),'checkpoint_usage':'model weights only as initialization; no engine/optimizer/sampler resume','config':config,'metadata':metadata,'tape_sha256':sha(args.out/'tape.npz'),'counts':counts,'actual_casts':casts,'card_requests':requested,'terminal_winners':terminal_winners,'occupancy':{'mean':float(np.mean(occupancy)),'median':float(np.median(occupancy)),'p95':float(np.percentile(occupancy,95)),'max':int(max(occupancy))},'checkpoint_digests_after_manual_reset':digests,'reset_semantics':'Java VectorEnv.resetEnv(auto=false): I_RESET=0, matching LadderEnv.observe and NativeWorld. Autoreset transport I_RESET=1 is not used or normalized.','training_seed_domain':config['train_seed_range'],'generation_seconds':time.perf_counter()-phase_start,'script_seconds':time.perf_counter()-SCRIPT_START,'policy_timing':timer.report(),'resources':resources()};dump(args.out/'result.json',result);progress('generated',seconds=result['generation_seconds'],counts=counts,actual_casts=casts,occupancy=result['occupancy']);return result

def phase_validate(args,config):
 args.out.mkdir(parents=True,exist_ok=False);tape=np.load(args.tape);reference=json.loads(args.tape.with_name('result.json').read_text());seeds=tape['initial_seeds'];envs=[];frames=0
 try:
  java=make_backend('java4',config,seeds,args.snapshot,args.library);envs.append(java);native=make_backend('native4',config,seeds,args.snapshot,args.library);envs.append(native)
  step_index=-1
  def check(label):
   nonlocal frames
   for name in ('observations','masks','infos','rewards'):
    a,b=getattr(native,name),getattr(java,name)
    if not np.array_equal(a,b):
     positions=np.argwhere(a!=b);lane=int(positions[0,0]);np.savez(args.out/'first-mismatch.npz',native=a,java=b,native_observations=native.observations,java_observations=java.observations,native_infos=native.infos,java_infos=java.infos,native_masks=native.masks,java_masks=java.masks,native_rewards=native.rewards,java_rewards=java.rewards)
     old=np.flatnonzero(tape['reset_seeds'][:max(step_index,0),lane]>=0);start=int(old[-1])+1 if len(old) else 0;seed=int(tape['reset_seeds'][old[-1],lane]) if len(old) else int(seeds[lane]);history={'seed':seed,'history':tape['actions'][start:step_index+1,lane].tolist()};dump(args.out/f'lane{lane}-divergence-history.json',history)
     detail={'label':label,'field':name,'lane':lane,'episode_seed':seed,'episode_decision':step_index-start+1,'differences':positions[:100].tolist(),'native_values':[a[tuple(q)].item() for q in positions[:100]],'java_values':[b[tuple(q)].item() for q in positions[:100]]}
     if name=='observations':
      column=int(positions[0,2]);detail['observation_location']='header '+str(column) if column<224 else {'entity_row':(column-224)//34,'field_offset':(column-224)%34}
     dump(args.out/'first-mismatch.json',detail);raise AssertionError(label+' '+name+' differs')
   np.testing.assert_array_equal(done_flags(native),done_flags(java));frames+=1
  check('initial')
  for t,actions in enumerate(tape['actions']):
   step_index=t
   java.step(actions);native.step(actions);assert_valid(java,'java validation');assert_valid(native,'native validation');check(f'after step {t}');np.testing.assert_array_equal(java.dones,tape['dones'][t]);row=tape['reset_seeds'][t]
   reset_selected(java,'java4',row);reset_selected(native,'native4',row)
   if (row>=0).any():check(f'after reset {t}')
   if str(t) in reference['checkpoint_digests_after_manual_reset']:
    want=reference['checkpoint_digests_after_manual_reset'][str(t)];assert digest(java)==want;assert digest(native)==want
   if (t+1)%512==0:progress('untimed_validation',decisions=t+1,frames=frames)
  result={'status':'passed','frames_compared':frames,'environment_frames_compared':frames*args.envs,'all_observations_masks_infos_rewards_dones_exact':True,'reset_flag_normalization':False,'reset_semantics':reference['reset_semantics'],'final_digest':digest(native),'library_sha256':sha(args.library),'script_seconds':time.perf_counter()-SCRIPT_START};dump(args.out/'result.json',result);progress('validation_passed',**result);return result
 finally:
  for env in envs:env.close()

def compilation_ms():
 try:
  import jpype
  return int(jpype.JClass('java.lang.management.ManagementFactory').getCompilationMXBean().getTotalCompilationTime())
 except Exception:return None

def phase_replay(args,config):
 args.out.mkdir(parents=True,exist_ok=False);load_start=time.perf_counter();tape=np.load(args.tape);actions=tape['actions'];resets=tape['reset_seeds'];initial=tape['initial_seeds'];reference=json.loads(args.tape.with_name('result.json').read_text());load_seconds=time.perf_counter()-load_start
 setup_start=time.perf_counter();env=make_backend(args.backend,config,initial,args.snapshot,args.library);setup_seconds=time.perf_counter()-setup_start;warmups=[];cpu_before=snapshot_processes()
 def replay(checkpoints=False):
  reset_all(env,args.backend,initial,config['rules']['deck']);begin=time.perf_counter()
  for t,row in enumerate(actions):
   env.step(row);reset_selected(env,args.backend,resets[t])
   if checkpoints and str(t) in reference['checkpoint_digests_after_manual_reset']:assert digest(env)==reference['checkpoint_digests_after_manual_reset'][str(t)]
  seconds=time.perf_counter()-begin
  # Deliberately outside the timed loop: complete untimed equality already passed.
  got=digest(env);assert got==reference['checkpoint_digests_after_manual_reset'][str(len(actions)-1)],got
  return seconds,got
 try:
  jit_initial=compilation_ms() if args.backend=='java4' else None;jit_warmups=[jit_initial]
  for _ in range(args.warmups):
   seconds,_=replay(True);warmups.append(seconds)
   if args.backend=='java4':jit_warmups.append(compilation_ms())
  compiler_before=compilation_ms() if args.backend=='java4' else None
  seconds,final=replay(False)
  compiler_after=compilation_ms() if args.backend=='java4' else None
  result={'status':'passed','backend':args.backend,'threads':4 if args.backend!='native1' else 1,'environment_decisions':args.envs*args.decisions,'seat_samples':2*args.envs*args.decisions,'warm_seconds':seconds,'environment_decisions_per_second':args.envs*args.decisions/seconds,'seat_samples_per_second':2*args.envs*args.decisions/seconds,'warmups_seconds':warmups,'backend_initialization_seconds':setup_seconds,'tape_load_seconds':load_seconds,'module_import_and_argument_seconds':load_start-SCRIPT_START,'script_before_teardown_seconds':time.perf_counter()-SCRIPT_START,'final_digest':final,'library_sha256':sha(args.library) if args.backend!='java4' else None,'jit_compilation_ms_before_warmups':jit_initial,'jit_compilation_ms_after_each_warmup':jit_warmups,'jit_compilation_ms_before_timed':compiler_before,'jit_compilation_ms_after_timed':compiler_after,'timed_scope':'vector step, complete observations/legal masks/readback, Python tape dispatch and natural manual resets; no policy/learner/digest comparisons in stopwatch','full_setup_compile_scope':'fresh compile separately; worker initializes backend from scratch, executes complete warmup tape, then complete timed tape','cpu_process_snapshot_before':cpu_before,'cpu_process_snapshot_after':snapshot_processes(),'resources':resources()};dump(args.out/'result.json',result);progress('replay_passed',backend=args.backend,warm_seconds=seconds,environment_decisions_per_second=result['environment_decisions_per_second'],seat_samples_per_second=result['seat_samples_per_second']);return result
 finally:env.close()

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);p.add_argument('--envs',type=int,default=32);p.add_argument('--decisions',type=int,default=2048);p.add_argument('--repeats',type=int,default=3);p.add_argument('--warmups',type=int,default=20);p.add_argument('--policy-seed',type=int,default=170108);p.add_argument('--checkpoint',type=Path,default=ROOT/'evidence/clash-implementation/cuda-campaign/learning2/train/checkpoint.pt');p.add_argument('--tape-source',type=Path);p.add_argument('--source-snapshot',type=Path);p.add_argument('--phase',choices=['generate','validate','replay']);p.add_argument('--config',type=Path);p.add_argument('--tape',type=Path);p.add_argument('--snapshot',type=Path);p.add_argument('--library',type=Path);p.add_argument('--backend',choices=['java4','native4','native1']);p.add_argument('--lock-fd',type=int);args=p.parse_args()
 if args.envs not in (16,32) or args.decisions<2048 or args.repeats<3 or args.warmups<1:raise ValueError('use16or32envs,atleast2048decisions,3pairedrepeats,and1fullwarmup')
 if args.phase:
  if args.lock_fd is None:raise ValueError('internal phase requires inherited performance lock')
  fcntl.flock(args.lock_fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
  config=json.loads(args.config.read_text())
  return {'generate':phase_generate,'validate':phase_validate,'replay':phase_replay}[args.phase](args,config)
 args.out=args.out.resolve();args.out.mkdir(parents=True,exist_ok=False)
 with (ROOT/'evidence/local-performance.lock').open('a') as lock:
  progress('waiting_performance_lock',path=str(ROOT/'evidence/local-performance.lock'));fcntl.flock(lock,fcntl.LOCK_EX);progress('performance_lock_acquired',pid=os.getpid())
  config=json.loads((INTEGRATION/'configs/profile-trimmed.json').read_text());config.update(envs=args.envs,threads=4,torch_threads=1,device='cpu');config['hardware_allocation'].update(cpu_threads=4,gpu_count=0,gpu_model='none(localCPUcomparison)');dump(args.out/'config.json',config)
  source_root=args.out/'snapshot/native';source_root.mkdir(parents=True);source_hashes={};source_origin=args.source_snapshot.resolve() if args.source_snapshot else HERE
  for path in sorted(source_origin.rglob('*')):
   if path.is_file() and '.build' not in path.parts and '__pycache__' not in path.parts and (path.suffix in ('.hpp','.h','.cpp','.py') or path.name in ('LICENSE','NOTICE') or path==source_origin/'generated-v2/task-data.json'):
    relative=path.relative_to(source_origin);target=source_root/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target);source_hashes[str(relative)]=sha(target)
  dump(args.out/'source-manifest.json',{'source_sha256':source_hashes,'source_origin':str(source_origin),'benchmark_script_sha256':sha(__file__),'engine_pin':'f0659dafabfb9d2a4a8d3b22a9d14de306f5e4de','resources':resources(),'lock_path':str(ROOT/'evidence/local-performance.lock'),'cpus_enforced':'4 simulator workers/JVM ActiveProcessorCount4, no OS affinity enforcement','rules_lock_sha256':sha(INTEGRATION/'rules.lock.json')})
  def run(phase,target,backend=None,library=None):
   cmd=[sys.executable,str(Path(__file__).resolve()),'--phase',phase,'--out',str(target),'--config',str(args.out/'config.json'),'--envs',str(args.envs),'--decisions',str(args.decisions),'--repeats',str(args.repeats),'--warmups',str(args.warmups),'--policy-seed',str(args.policy_seed),'--checkpoint',str(args.checkpoint),'--lock-fd',str(lock.fileno())]
   if phase!='generate':cmd+=['--tape',str(args.out/'generation/tape.npz'),'--snapshot',str(source_root),'--library',str(library)]
   if backend:cmd+=['--backend',backend]
   log=target.parent/(target.name+'.log');begin=time.perf_counter()
   with log.open('w') as output:subprocess.run(cmd,stdout=output,stderr=subprocess.STDOUT,pass_fds=(lock.fileno(),),check=True)
   result=json.loads((target/'result.json').read_text());result['full_worker_process_seconds']=time.perf_counter()-begin;dump(target/'process-receipt.json',result);progress('phase_finished',phase=phase,backend=backend,full_worker_process_seconds=result['full_worker_process_seconds']);return result
  if args.tape_source:
   origin=args.tape_source.resolve();generated=json.loads((origin/'result.json').read_text());tape=np.load(origin/'tape.npz');assert tape['actions'].shape==(args.decisions,args.envs,2);assert generated['checkpoint_sha256']==sha(args.checkpoint)
   target=args.out/'generation';target.mkdir();shutil.copyfile(origin/'tape.npz',target/'tape.npz');generated=dict(generated,reused_from=str(origin),original_result_sha256=sha(origin/'result.json'));dump(target/'result.json',generated);progress('existing_tape_reused',path=str(origin),tape_sha256=generated['tape_sha256'])
  else:generated=run('generate',args.out/'generation')
  progress('fresh_compile_start');compiler=shutil.which('clang++') or shutil.which('g++');library=args.out/('libclash-static.dylib' if sys.platform=='darwin' else 'libclash-static.so');command=[compiler,'-std=c++20','-O3','-pthread','-fPIC','-fno-fast-math','-ffp-contract=off','-dynamiclib' if sys.platform=='darwin' else '-shared',str(source_root/'primitives.cpp'),str(source_root/'engine.cpp'),str(source_root/'cpu_batch.cpp'),'-o',str(library)];begin=time.perf_counter()
  with (args.out/'compile.log').open('w') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
  compile_result={'seconds':time.perf_counter()-begin,'command':command,'compiler':subprocess.check_output([compiler,'--version'],text=True),'library_sha256':sha(library),'source_manifest_sha256':sha(args.out/'source-manifest.json')};dump(args.out/'compile.json',compile_result)
  validation=run('validate',args.out/'validation',library=library);progress('timed_window_start',repeats=args.repeats)
  trials=[]
  for repeat in range(args.repeats):
   order=['java4','native4'] if repeat%2==0 else ['native4','java4']
   for backend in order:trials.append(dict(repeat=repeat+1,**run('replay',args.out/f'trial-{repeat+1}-{backend}',backend,library)))
  medians={backend:{'warm_seconds':statistics.median(x['warm_seconds'] for x in trials if x['backend']==backend),'environment_decisions_per_second':statistics.median(x['environment_decisions_per_second'] for x in trials if x['backend']==backend),'seat_samples_per_second':statistics.median(x['seat_samples_per_second'] for x in trials if x['backend']==backend),'backend_initialization_seconds':statistics.median(x['backend_initialization_seconds'] for x in trials if x['backend']==backend),'full_worker_process_seconds':statistics.median(x['full_worker_process_seconds'] for x in trials if x['backend']==backend)} for backend in ['java4','native4']}
  result={'status':'passed','comparison_scope':'matched raw simulator+observation+mask+reset CPU throughput; not full PPO or GPU allocated-hour efficiency','envs':args.envs,'decisions_per_vector_lane':args.decisions,'repeats':args.repeats,'order':'java4/native4, native4/java4, java4/native4','checkpoint_sha256':generated['checkpoint_sha256'],'checkpoint_usage':generated['checkpoint_usage'],'tape_sha256':generated['tape_sha256'],'native_library_sha256':sha(library),'benchmark_script_sha256':sha(__file__),'source_manifest_sha256':sha(args.out/'source-manifest.json'),'fresh_native_compile':compile_result,'generation':generated,'validation':validation,'trials':trials,'medians':medians,'native_to_java_warm_speedup':medians['java4']['warm_seconds']/medians['native4']['warm_seconds'],'lock_held_for_entire_comparison':True,'worker_wall_seconds_before_final_report':time.perf_counter()-SCRIPT_START,'resources':resources()};dump(args.out/'result.json',result);progress('benchmark_complete',medians=medians,native_to_java_warm_speedup=result['native_to_java_warm_speedup']);return result

if __name__=='__main__':main()
