"""Same-source full-game crossover for process warmth and synchronized timing.

Executed against the immutable qualified runtime. Observation hooks write only
sidecars, never training rows, tensors, RNG or checkpoint identity. Every arm
imports the same parent and resets the complete training state.
"""
import argparse
from collections import Counter
import contextlib
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time


def write(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def system():
    def read(path):
        try:return Path(path).read_text()
        except OSError:return None
    def command(args):
        try:return subprocess.run(args,capture_output=True,text=True,timeout=10).stdout
        except (OSError,subprocess.TimeoutExpired):return None
    return dict(unix=time.time(),cpu_affinity=sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else None,
        cpuinfo=read('/proc/cpuinfo'),status=read('/proc/self/status'),
        cgroup={f:read('/sys/fs/cgroup/'+f) for f in ['cpu.max','cpu.stat','cpuset.cpus.effective','cpuset.mems.effective','memory.max','memory.current']},
        environment={k:os.environ.get(k) for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','CUBLAS_WORKSPACE_CONFIG','CUDA_MODULE_LOADING','PYTHONPATH']},
        gpu=command(['nvidia-smi','--query-gpu=name,uuid,driver_version,memory.total,pci.bus_id,pstate,clocks.sm,clocks.mem,power.draw,power.limit,utilization.gpu','--format=csv,noheader,nounits']))


def arm(root, name, config, inputs, deadline):
    root=Path(root);out=root/name
    wall=time.perf_counter();before=system();usage=resource.getrusage(resource.RUSAGE_SELF)
    # CUDA is deliberately not initialized before the first actual run.
    import torch
    import numpy as np
    import taverngym as tg
    from simulators.tavern.reference_training.strong import train
    from simulators.tavern.reference_training import collection
    import model as actor_module
    graphs=Counter();shapes=Counter();timers=[];updates=[]
    old_graph=actor_module.graph_action;old_init=collection.Timer.__init__;old_write=train.write
    def graph(policy, values, deterministic):
        key=(len(values[0]),deterministic);cache=getattr(policy,'_actor_graphs',{})
        kind='hit' if key in cache else 'capture' if len(cache)<getattr(policy,'actor_graph_max_shapes',64) else 'fallback'
        graphs[kind]+=1;shapes[str((kind,len(values[0])))]+=1
        return old_graph(policy,values,deterministic)
    def timer_init(self,*args,**kwargs):
        old_init(self,*args,**kwargs);timers.append(self)
    def observe_write(path,value):
        old_write(path,value)
        if Path(path).name=='progress.json':
            timer=timers[-1];updates.append(dict(update=value['update'],session_elapsed=value['session_elapsed'],
                stage_seconds=dict(timer.seconds),diagnostics=dict(timer.diagnostics),graphs=dict(graphs),unix=time.time()))
            write(root/(name+'-updates.json'),updates)
    actor_module.graph_action=graph;collection.Timer.__init__=timer_init;train.write=observe_write
    try:
        with (root/(name+'.log')).open('w',buffering=1) as log,contextlib.redirect_stdout(log):
            checkpoint=train.run(config,out,inputs,max(1,deadline-time.time()-12),limit_updates=config['updates'])
    finally:
        actor_module.graph_action=old_graph;collection.Timer.__init__=old_init;train.write=old_write
    elapsed=time.perf_counter()-wall;after=system();endusage=resource.getrusage(resource.RUSAGE_SELF)
    result=json.loads((out/'result.json').read_text())
    import ctypes
    blas=[]
    for library in (Path(np.__file__).parent.parent/'numpy.libs').glob('*openblas*'):
        handle=ctypes.CDLL(str(library))
        for symbol in ['scipy_openblas_get_num_threads64_','scipy_openblas_get_num_threads','openblas_get_num_threads64_','openblas_get_num_threads']:
            if hasattr(handle,symbol):
                get=getattr(handle,symbol);get.restype=ctypes.c_int
                blas.append(dict(library=library.name,symbol=symbol,threads=get()));break
    metadata=dict(name=name,wall_seconds_including_import_and_metadata=elapsed,config=config,result=result,numpy_blas=blas,
        before=before,after=after,cpu_user_seconds=endusage.ru_utime-usage.ru_utime,cpu_system_seconds=endusage.ru_stime-usage.ru_stime,
        torch=torch.__version__,numpy=np.__version__,native=tg.build_info(),
        torch_threads=torch.get_num_threads(),torch_interop_threads=torch.get_num_interop_threads(),
        float32_matmul_precision=torch.get_float32_matmul_precision(),cuda_initialized=torch.cuda.is_initialized(),
        gpu_properties=str(torch.cuda.get_device_properties(0)) if config['device'].startswith('cuda') else None,
        graph_calls=dict(graphs),shape_counts=dict(shapes),updates=updates)
    write(root/(name+'-metadata.json'),metadata)
    if result['termination']!='update_limit' or result['update']!=config['updates']:
        raise RuntimeError('Incomplete attribution arm, preserved and charged: '+name)
    return checkpoint


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--inputs',default='/inputs');p.add_argument('--config',type=Path,required=True)
    p.add_argument('--cap',type=int,default=570);p.add_argument('--one');p.add_argument('--sync',type=int,default=0)
    p.add_argument('--local',action='store_true');p.add_argument('--cache',type=int);p.add_argument('--threads',type=int);p.add_argument('--plan',choices=['crossover','cache','threads'],default='crossover');a=p.parse_args()
    deadline=time.time()+a.cap;a.root.mkdir(parents=True,exist_ok=True)
    c=json.loads(a.config.read_text());c.update(updates=2)
    if a.local:c.update(device='cpu',n_games=2)
    if a.cache is not None:c['actor_graph_max_shapes']=a.cache
    if a.threads is not None:
        expected={k:str(a.threads) for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']}
        if any(os.environ.get(k)!=v for k,v in expected.items()):raise ValueError('Thread environment must precede Python/NumPy initialization')
        c['host_thread_environment']=expected
    if a.one:
        c['profile_synchronized']=bool(a.sync)
        state=dict(status='started',unix=time.time())
        try:
            arm(a.root,a.one,c,a.inputs,deadline);state.update(status='completed',finished_unix=time.time())
        except BaseException:
            import traceback
            state.update(status='failed',finished_unix=time.time(),traceback=traceback.format_exc());raise
        finally:
            write(a.root/(a.one+'-lifecycle.json'),state)
            if not a.local:subprocess.run(['sync',str(a.root)],check=True,timeout=10)
        return
    status=dict(status='started',unix=time.time(),automatic_retries=0);write(a.root/'lifecycle.json',status)
    monitor=None
    try:
        if not a.local:
            monitor=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,uuid,utilization.gpu,pstate,clocks.sm,clocks.mem,power.draw,power.limit','--format=csv','-l','1'],stdout=(a.root/'gpu-monitor.csv').open('w'),stderr=subprocess.STDOUT)
        paths=[]
        if a.plan=='crossover':
            for name,sync in [('shared-prod-cold',False),('shared-sync-warm',True),('shared-prod-warm',False)]:
                live=dict(c,profile_synchronized=sync);paths.append(arm(a.root,name,live,a.inputs,deadline))
        sequence=([('fresh-sync',True,None),('fresh-prod',False,None)] if a.plan=='crossover' else
                  [('fresh-cache4-before',False,4),('fresh-cache16',False,16),('fresh-cache4-after',False,4)] if a.plan=='cache' else
                  [('fresh-threads4-before',False,4),('fresh-threads1',False,1),('fresh-threads4-after',False,4)])
        for name,sync,cache in sequence:
            # A separate CUDA process cannot reuse this process's cached arena.
            # Release unused parent allocations for both cold-child arms equally.
            if 'torch' in sys.modules:
                import gc,torch
                gc.collect()
                if torch.cuda.is_initialized():
                    torch.cuda.synchronize();torch.cuda.empty_cache()
                    write(a.root/(name+'-parent-memory.json'),dict(allocated=torch.cuda.memory_allocated(),reserved=torch.cuda.memory_reserved()))
            command=[sys.executable,str(Path(__file__).resolve()),'--root',str(a.root),'--inputs',a.inputs,'--config',str(a.config),'--cap',str(max(1,int(deadline-time.time()-10))),'--one',name,'--sync',str(int(sync))]
            if a.local:command+=['--local']
            child_env=None
            if cache is not None:
                if a.plan=='threads':
                    command+=['--threads',str(cache)]
                    child_env={**os.environ,**{k:str(cache) for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']}}
                else:command+=['--cache',str(cache)]
            with (a.root/(name+'-entry.log')).open('w') as log:
                child=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=max(1,deadline-time.time()-8),env=child_env)
            if child.returncode:raise RuntimeError('Fresh process failed: '+name)
            paths.append(a.root/name/'latest.pt')
        if a.plan=='threads':
            pinned={k:'1' for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']};os.environ.update(pinned);c['host_thread_environment']=pinned
        from simulators.tavern.reference_training.strong.throughput import compare
        checks={path.parent.name:compare(paths[0],path) for path in paths[1:]}
        write(a.root/'parity.json',checks)
        if a.plan in ('cache','threads'):
            import torch
            from simulators.tavern.reference_training.strong.proof import validate
            # Exercise real next-update resume, not a completed/no-op horizon.
            validate(dict(c,updates=4,actor_graph_max_shapes=16 if a.plan=='cache' else c['actor_graph_max_shapes']),a.inputs,a.root/'resume-proof',seconds=max(10,deadline-time.time()-10))
            from simulators.tavern.reference_training.strong.export import export
            export(a.root/'resume-proof/full/latest.pt',a.root/'proof-export')
        status.update(status='completed',finished_unix=time.time(),arms=len(paths))
    except BaseException:
        import traceback
        status.update(status='failed',finished_unix=time.time(),traceback=traceback.format_exc());raise
    finally:
        if monitor is not None:monitor.terminate();monitor.wait(timeout=10)
        write(a.root/'lifecycle.json',status)
        if not a.local:subprocess.run(['sync',str(a.root)],check=True,timeout=10)


if __name__=='__main__':main()
