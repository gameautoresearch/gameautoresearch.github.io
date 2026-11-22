"""Finite fresh-process H100 population measurement/proof; no model selection."""
import argparse,contextlib,json,os,resource,subprocess,sys,time,traceback
from pathlib import Path

def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2)+'\n');t.replace(p)

def system():
    def read(p):
        try:return Path(p).read_text()
        except OSError:return None
    try:gpu=subprocess.run(['nvidia-smi','--query-gpu=name,uuid,driver_version,memory.total,clocks.sm,power.limit','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10).stdout
    except OSError:gpu=None
    return dict(unix=time.time(),gpu=gpu,cpuinfo=read('/proc/cpuinfo'),affinity=sorted(os.sched_getaffinity(0)),environment={k:os.environ.get(k) for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','CUBLAS_WORKSPACE_CONFIG','PYTHONPATH']})

def one(root,name,c,seconds,proof=False):
    root=Path(root);started=time.perf_counter();before=system();usage=resource.getrusage(resource.RUSAGE_SELF)
    import torch,numpy as np
    from simulators.tavern.reference_training.strong.train import run
    with (root/(name+'.log')).open('w',buffering=1) as log,contextlib.redirect_stdout(log):
        checkpoint=run(c,root/name,'/inputs',seconds,limit_updates=c['updates'])
        if proof:
            from simulators.tavern.reference_training.strong.proof import validate
            from simulators.tavern.reference_training.strong.export import export
            validate(dict(c,updates=4),'/inputs',root/(name+'-proof'),seconds=90)
            export(checkpoint,root/(name+'-export'))
    result=json.loads((checkpoint.parent/'result.json').read_text());after=system();end=resource.getrusage(resource.RUSAGE_SELF)
    write(root/(name+'-process.json'),dict(result=result,process_window_seconds=time.perf_counter()-started,before=before,after=after,torch=torch.__version__,numpy=np.__version__,torch_threads=torch.get_num_threads(),cpu_user=end.ru_utime-usage.ru_utime,cpu_system=end.ru_stime-usage.ru_stime,max_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    if result['termination']!='update_limit' or result['update']!=c['updates']:raise RuntimeError('Incomplete natural cohort measurement')

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--config',type=Path,required=True);p.add_argument('--cap',type=int,required=True);p.add_argument('--mode',choices=['single','pair','crossover'],default='single');p.add_argument('--variant',choices=['baseline','candidate'],default='baseline');p.add_argument('--reverse',action='store_true');p.add_argument('--proof',action='store_true');p.add_argument('--candidate-cache',type=int,default=16);p.add_argument('--validated-sampler',action='store_true');a=p.parse_args()
    a.root.mkdir(parents=True,exist_ok=True);start=time.time();status=dict(status='started',entry_unix=start);write(a.root/'lifecycle.json',status)
    c=json.loads(a.config.read_text());monitor=None
    try:
        monitor=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,utilization.gpu,memory.used,clocks.sm,power.draw','--format=csv','-l','1'],stdout=(a.root/'gpu.csv').open('w'),stderr=subprocess.STDOUT)
        if a.mode=='single':one(a.root,a.variant,c,max(1,a.cap-65),a.proof)
        else:
            names=['candidate','baseline'] if a.reverse else ['baseline','candidate']
            arms=[(name,name) for name in names]
            if a.mode=='crossover':arms=[(names[0]+'-a',names[0]),(names[1],names[1]),(names[0]+'-b',names[0])]
            paths=[]
            for label,name in arms:
                live=dict(c);live.update(batched_policy_dispatch=name=='candidate',ppo_backward_graph=name=='candidate')
                live['actor_graph_max_shapes']=a.candidate_cache if name=='candidate' else 16
                live['validated_sampler']=a.validated_sampler and name=='candidate'
                path=a.root/(label+'-config.json');write(path,live)
                env={**os.environ,'PYTHONPATH':'/baseline' if name=='baseline' else '/opt/task'}
                child=a.root/(label+'-child')
                args=[sys.executable,str(Path(__file__).resolve()),'--root',str(child),'--config',str(path),'--cap',str(max(1,int(a.cap-(time.time()-start)-12))),'--variant',name]
                if a.proof and name=='candidate':args+=['--proof']
                with (a.root/(label+'-child.log')).open('w') as log:
                    r=subprocess.run(args,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=max(1,a.cap-(time.time()-start)-8))
                if r.returncode:raise RuntimeError('Failed matched child '+name)
                paths.append(child/name/'latest.pt')
            from simulators.tavern.reference_training.strong.throughput import compare
            checks=[compare(paths[0],path) for path in paths[1:]]
            write(a.root/'parity.json',checks[0] if a.mode=='pair' else checks)
        status.update(status='completed',finished_unix=time.time())
    except BaseException:status.update(status='failed',finished_unix=time.time(),traceback=traceback.format_exc());raise
    finally:
        if monitor:monitor.terminate();monitor.wait(timeout=10)
        write(a.root/'lifecycle.json',status);subprocess.run(['sync',str(a.root)],check=True,timeout=10)

if __name__=='__main__':main()
