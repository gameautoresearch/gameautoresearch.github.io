"""Device-resident CUDA launch/render sweep on the retained trained-policy tape.

No policy inference, learning, JVM runtime, cloud launch or paid-job submission.
Every requested configuration passes complete untimed native-CPU output parity
before timing. Timed replays use preloaded device actions and manual reset seeds.
"""
from __future__ import annotations
from pathlib import Path
import argparse,ctypes as C,fcntl,hashlib,json,os,platform,statistics,sys,time
import numpy as np
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(HERE))

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path,value):Path(path).write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
def digest(arrays):
    h=hashlib.sha256()
    for value in arrays:h.update(np.ascontiguousarray(value).tobytes())
    return h.hexdigest()
def outputs(env):return tuple(getattr(env,name) for name in ('observations','masks','infos','rewards'))
def cpu_reference(actions,resets,seeds,out,metadata,expected_dones):
    from cpu_batch import NativeCPUBatch
    env=NativeCPUBatch(seeds,threads=4);before=[];after=[]
    try:
        initial=digest(outputs(env))
        for step,row in enumerate(actions):
            env.step(row);before.append(digest(outputs(env)))
            dones=((env.infos[:,2]>=0)|(env.infos[:,9]!=0)|(env.infos[:,5]!=0)).astype(np.uint8)
            np.testing.assert_array_equal(dones,expected_dones[step])
            selected=resets[step]>=0
            if selected.any():env.reset(np.where(selected,resets[step],0).astype(np.int32),selected.astype(np.int32))
            after.append(digest(outputs(env)))
            if str(step) in metadata['checkpoint_digests_after_manual_reset']:
                done=((env.infos[:,2]>=0)|(env.infos[:,9]!=0)|(env.infos[:,5]!=0)).astype(np.uint8)
                if digest((*outputs(env),done))!=metadata['checkpoint_digests_after_manual_reset'][str(step)]:raise AssertionError('CPU reference differs from retained originalJava tape checkpoint')
        result={'initial':initial,'before_reset':before,'after_reset':after,'library_sha256':sha(env.library) if hasattr(env,'library') else None}
        dump(out/'cpu-reference.json',result);return result
    finally:env.close()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--tape',type=Path,required=True,help='tape.npz from benchmark_cpu.py generation')
    p.add_argument('--replicas',type=int,default=8,help='explicit repetition of32trained lanes; default256total environments')
    p.add_argument('--threads',default='1,4,8,16,32')
    p.add_argument('--render-modes',default='0,1,2')
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--warmups',type=int,default=1)
    args=p.parse_args()
    threads=[int(v) for v in args.threads.split(',')];modes=[int(v) for v in args.render_modes.split(',')]
    if not set(threads)<=set((1,4,8,16,32)) or not set(modes)<=set((0,1,2,3,4)):raise ValueError('invalid launch/render configuration')
    if args.replicas<1 or args.repeats<3 or args.warmups<1:raise ValueError('positive replicas, >=3repeats and >=1warmup required')
    import torch
    from runtime import NativeCUDA
    if not torch.cuda.is_available():raise RuntimeError('CUDA hardware required; no cloud launch is performed')
    args.out.mkdir(parents=True,exist_ok=False);lock_path=ROOT/'evidence/local-performance.lock';lock_path.parent.mkdir(exist_ok=True)
    with lock_path.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        tape=np.load(args.tape);metadata=json.loads(args.tape.with_name('result.json').read_text())
        if sha(args.tape)!=metadata['tape_sha256']:raise ValueError('retained tape hash differs')
        actions=tape['actions'];resets=tape['reset_seeds'];seeds=tape['initial_seeds'];steps,base_n,_=actions.shape
        if steps<2048 or base_n not in (16,32):raise ValueError('full trained tape requires>=2048decisions and16or32unique lanes')
        reference=cpu_reference(actions,resets,seeds,args.out,metadata,tape['dones'])
        config_list=[(t,m) for t in threads for m in modes]
        repeated_seeds=np.tile(seeds,args.replicas)
        repeated_actions=np.tile(actions,(1,args.replicas,1));repeated_resets=np.tile(resets,(1,args.replicas))
        n=len(repeated_seeds);selected=(repeated_resets>=0).astype(np.int32)
        reset_rows=np.flatnonzero(selected.any(axis=1));reset_set=set(map(int,reset_rows))
        device_actions=torch.as_tensor(repeated_actions,device='cuda',dtype=torch.int32)
        device_reset_seeds=torch.as_tensor(np.maximum(repeated_resets,0),device='cuda',dtype=torch.int32)
        device_reset_selected=torch.as_tensor(selected,device='cuda',dtype=torch.int32)
        initial=torch.as_tensor(repeated_seeds,device='cuda',dtype=torch.int32)
        env=NativeCUDA(initial);lib=env.lib
        for name in ('cn_cuda_set_launch_threads','cn_cuda_set_render_mode'):
            fn=getattr(lib,name);fn.argtypes=[C.c_int];fn.restype=C.c_int
        def policy(t,m):
            env.check(lib.cn_cuda_set_launch_threads(t));env.check(lib.cn_cuda_set_render_mode(m))
        def reset_row(step):env.reset(device_reset_seeds[step],device_reset_selected[step])
        def compare(want,label):
            # Replicated lanes must ALL match the unique CPU trace. Compare each
            # device replica against replica0 before hashing that32-lane slice.
            base=[]
            for value in outputs(env):
                repeated=value.reshape(args.replicas,base_n,*value.shape[1:])
                if not torch.equal(repeated,repeated[0:1].expand_as(repeated)):raise AssertionError(label+' replicated lane outputs differ')
                base.append(repeated[0].cpu().numpy())
            if digest(base)!=want:
                np.savez_compressed(args.out/'first-divergence.npz',**{name:value for name,value in zip(('observations','masks','infos','rewards'),base)})
                raise AssertionError(label+' nativeCPU/CUDA complete output digest differs')
        validations=[];trials=[]
        try:
            for t,m in config_list:
                policy(t,m);env.reset(initial);compare(reference['initial'],f'{t}/{m} initial')
                for step,row in enumerate(device_actions):
                    env.step(row);compare(reference['before_reset'][step],f'{t}/{m} step{step}')
                    if step in reset_set:reset_row(step);compare(reference['after_reset'][step],f'{t}/{m} reset{step}')
                validations.append({'threads':t,'render_mode':m,'environment_frames_compared':n*(1+steps+len(reset_rows))})
                print(json.dumps({'event':'parity_passed',**validations[-1]}),flush=True)
            # Only after every requested configuration passed parity do timings
            # begin. This is full raw simulator+output GPU residency throughput.
            def replay(measure):
                env.reset(initial);torch.cuda.synchronize()
                begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                phase_events=[tuple(torch.cuda.Event(enable_timing=True) for _ in range(4 if step in reset_set else 3)) for step in range(steps)] if measure else None
                start=time.perf_counter();begin.record()
                for step,row in enumerate(device_actions):
                    events=phase_events[step] if measure else None
                    if events:events[0].record()
                    env.check(lib.cn_cuda_step(env.ptr(env.states),env.ptr(row),n,0,env.stream()))
                    if events:events[1].record()
                    env.read()
                    if events:events[2].record()
                    if step in reset_set:
                        reset_row(step)
                        if events:events[3].record()
                end.record();torch.cuda.synchronize();wall=time.perf_counter()-start
                result={'wall_seconds':wall,'cuda_span_seconds':begin.elapsed_time(end)/1000}
                if measure:
                    result.update(step_cuda_seconds=sum(e[0].elapsed_time(e[1]) for e in phase_events)/1000,read_cuda_seconds=sum(e[1].elapsed_time(e[2]) for e in phase_events)/1000,reset_cuda_seconds=sum(e[2].elapsed_time(e[3]) for e in phase_events if len(e)==4)/1000)
                compare(reference['after_reset'][-1],'timed final outside stopwatch')
                return result
            for repeat in range(args.repeats):
                order=config_list if repeat%2==0 else list(reversed(config_list))
                for t,m in order:
                    policy(t,m);warm=[replay(False) for _ in range(args.warmups)];trial=replay(False);phases=replay(True)
                    trial.update({('phase_'+k):v for k,v in phases.items()})
                    trial.update(repeat=repeat+1,threads=t,render_mode=m,warmups=warm,environment_decisions=n*steps,seat_samples=2*n*steps,environment_decisions_per_second=n*steps/trial['wall_seconds'],seat_samples_per_second=2*n*steps/trial['wall_seconds'])
                    trials.append(trial);dump(args.out/f'trial-{repeat+1}-{t}-{m}.json',trial);print(json.dumps({'event':'timed_trial',**{k:v for k,v in trial.items() if k!='warmups'}}),flush=True)
            medians=[]
            for t,m in config_list:
                cases=[r for r in trials if r['threads']==t and r['render_mode']==m]
                medians.append({'threads':t,'render_mode':m,**{name:statistics.median(r[name] for r in cases) for name in ['wall_seconds','cuda_span_seconds','phase_step_cuda_seconds','phase_read_cuda_seconds','phase_reset_cuda_seconds','environment_decisions_per_second','seat_samples_per_second']}})
            result={'status':'passed','scope':'raw device-resident simulator,observation,legal mask,info,reward and natural reset;preloaded actions/seeds;no policy/learner/host-output-transfer timing','cuda_event_scope':'full stream span and per-step/read/reset intervals; includes device idle gaps within those host-dispatched intervals; wall includes synchronization','event_instrumentation':'Headline wall throughput comes from a separate uninstrumented full replay. Another exact full replay records three events perdecision plus one perresetvector for phase_* diagnostics, with its distinct phase_wall_seconds and phase_cuda_span_seconds.','unique_trained_lanes':base_n,'replicas':args.replicas,'environments':n,'decisions_per_lane':steps,'training_tape_sha256':sha(args.tape),'checkpoint_sha256':metadata['checkpoint_sha256'],'checkpoint_usage':metadata['checkpoint_usage'],'workload':dict(metadata['counts'],natural_terminals_replayed=metadata['counts']['natural_terminals']*args.replicas,actual_ticks_replayed=metadata['counts']['actual_ticks']*args.replicas),'occupancy':metadata['occupancy'],'actual_casts_unique_tape':metadata['actual_casts'],'validation':validations,'cpu_reference_library_sha256':reference['library_sha256'],'cuda_library_sha256':sha(env.library),'cuda_build_receipt':json.loads(env.library.with_suffix('.json').read_text()),'benchmark_script_sha256':sha(__file__),'cuda_device':torch.cuda.get_device_name(),'cuda_device_properties':str(torch.cuda.get_device_properties(0)),'python':sys.version,'platform':platform.platform(),'repeats':args.repeats,'warmups_per_trial':args.warmups,'trials':trials,'medians':medians,'lock_path':str(lock_path),'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated()}
            dump(args.out/'result.json',result);print(json.dumps({'status':'passed','medians':medians}),flush=True)
        finally:
            policy(32,0);env.close()
if __name__=='__main__':main()
