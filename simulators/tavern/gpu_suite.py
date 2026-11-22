"""Bounded actual-CUDA experiment driver, invoked inside the game-specific container."""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent


def run_command(out, config, command, checkpoint=None):
    cfg=out.parent/(out.name+'-recipe.json')
    cfg.write_text(json.dumps(config,indent=2)+'\n')
    cmd=[sys.executable,str(HERE/'run.py'),command,'--config',str(cfg),'--out',str(out)]
    if checkpoint: cmd+=['--checkpoint',str(checkpoint)]
    start=time.perf_counter()
    with out.with_suffix('.log').open('w') as log:
        result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=400)
    row={'run':out.name,'returncode':result.returncode,'full_process_seconds':time.perf_counter()-start}
    for name in ['accounting','profile','training','evaluation']:
        path=out/(name+'.json')
        if path.exists(): row[name]=json.loads(path.read_text())
    if result.returncode:
        print(out.with_suffix('.log').read_text()[-5000:],flush=True)
        raise RuntimeError(f'{out.name} failed')
    return row


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',required=True,type=Path)
    parser.add_argument('--phase',choices=['baseline','optimized','learning','sampling'],default='baseline')
    parser.add_argument('--seed',type=int,default=12345)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter()
    import torch
    import taverngym as tg
    assert torch.cuda.is_available(),'CUDA required'
    assert torch.cuda.get_device_name() == 'NVIDIA L4', 'fixed L4 allocation required'
    metadata={'torch':torch.__version__,'python':sys.version,'gpu':torch.cuda.get_device_name(),
      'cuda':torch.version.cuda,'native':tg.build_info(),'phase':args.phase,
      'nvidia_device_and_driver':subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total,compute_cap','--format=csv,noheader'],text=True).strip(),
      'rates':{'L4_second':.000222,'cpu_core_second':.0000131,'ram_GiB_second':.00000222},
      'cpu_physical_limit':4,'ram_GiB_limit':16,
      'cpu_model':next((line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')),'unknown'),
      'rustc':subprocess.check_output(['rustc','--version'],cwd=ROOT/'candidates/AutoTavern',text=True).strip()}
    (args.out/'device.json').write_text(json.dumps(metadata,indent=2)+'\n')
    monitor=open(args.out/'nvidia-smi.csv','w')
    sampler=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.used,power.draw','--format=csv','--loop-ms=200'],stdout=monitor,stderr=subprocess.DEVNULL)
    rows=[]
    try:
        c=json.loads((HERE/'recipes/profile-t4-copy1.json').read_text())
        c.update(device='cuda:0',profile_updates=2,updates=2,snapshot_every=0)
        c['machine']={'label':'Modal-staging-fixed-L4-4core-16GiB','attached_cpu_threads':4,'attached_ram_gib':16,'gpu_model':'NVIDIA L4'}
        c['evaluation'].update(seeds=[1000000007],games_per_seat=1)
        if args.phase == 'sampling':
            from sampling_experiment import experiment
            rows.append({'sampling':experiment(args.out/'sampling')})
            return
        if args.phase == 'optimized':
            from paired import paired
            paired(args.out/'cuda-pairs','cuda:0','/workspace/pilot.pt',trials=3)
            # Same dedicated host and allocation: CPU-only learner is also charged GPU allocation time.
            paired(args.out/'same-host-cpu-pairs','cpu','/workspace/pilot.pt',trials=2)
            return
        if args.phase == 'learning':
            from learning_experiment import experiment
            results=experiment(args.out/'learning',device='cuda:0',seeds=[args.seed])
            rows.append({'learning_results':results})
            if args.seed == 78901:
                subprocess.run([sys.executable,str(HERE/'validate_resume.py'),'--device','cuda:0','--out',str(args.out/'cuda-resume-check')],check=True,timeout=90)
            return
        for name,pilot in [('cold-initial',None),('warm-initial',None),('warm-trained','/workspace/pilot.pt')]:
            rows.append(run_command(args.out/name,c,'profile',pilot))
            print(json.dumps({'completed':name,'seconds':rows[-1]['full_process_seconds']}),flush=True)
        rows.append(run_command(args.out/'matched-eval',c,'evaluate',args.out/'warm-trained/latest.pt'))
    finally:
        sampler.terminate();sampler.wait();monitor.close()
        receipt={'metadata':metadata,'rows':rows,'function_worker_seconds':time.perf_counter()-started,
                 'completed':len(rows),'cuda_executed':True}
        (args.out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({'finished':args.phase,'seconds':time.perf_counter()-started}),flush=True)


if __name__=='__main__': main()
