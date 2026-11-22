#!/usr/bin/env python3
"""Bounded new-authorized structural experiment on fixed L4/4-core/16-GiB allocation."""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from gpu_suite import HERE, run_command
from provenance import write_json


def compare(a,b):
    import torch
    import numpy as np
    x=torch.load(Path(a)/'latest.pt',map_location='cpu',weights_only=False)
    y=torch.load(Path(b)/'latest.pt',map_location='cpu',weights_only=False)
    def same(u,v):
        if isinstance(u,torch.Tensor):return torch.equal(u,v)
        if isinstance(u,np.ndarray):return np.array_equal(u,v)
        if isinstance(u,dict):return u.keys()==v.keys() and all(same(u[k],v[k]) for k in u)
        if isinstance(u,(list,tuple)):return len(u)==len(v) and all(same(i,j) for i,j in zip(u,v))
        return u==v
    keys=['policy','optimizer','torch_rng','cuda_rng','env_states','stats']
    return {k:same(x[k],y[k]) for k in keys}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',required=True,type=Path)
    p.add_argument('--phase',choices=['probe','learning','refine','massive','final','packed','wall'],default='probe')
    p.add_argument('--seed',type=int,default=12345)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    import torch
    assert torch.cuda.is_available() and os.environ.get('TAVERN_GPU','L4') in torch.cuda.get_device_name()
    started=time.perf_counter();rows=[]
    metadata={'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(),
        'cpu_limit':int(os.environ.get('TAVERN_CPU_CORES','4')),'memory_gib_limit':int(os.environ.get('TAVERN_RAM_GIB','16')),
        'cpu_model':next((line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')),'unknown'),
        'cpu_quota':Path('/sys/fs/cgroup/cpu.max').read_text().strip() if Path('/sys/fs/cgroup/cpu.max').exists() else None,
        'memory_limit':Path('/sys/fs/cgroup/memory.max').read_text().strip() if Path('/sys/fs/cgroup/memory.max').exists() else None,
        'resource_enforcement':'Exact request/limit tuples in Modal function configuration; cgroup paths may be hidden by the container runtime.',
        'python':sys.version,'phase':a.phase}
    write_json(a.out/'device.json',metadata)
    with (a.out/'tests.log').open('w') as log:
        test=subprocess.run([sys.executable,'-m','pytest','-q',str(HERE/'tests')],stdout=log,stderr=subprocess.STDOUT)
    write_json(a.out/'tests.json',{'returncode':test.returncode})
    if test.returncode:
        raise RuntimeError('contract tests failed; retain tests.log and allocation before further experiments')
    if a.phase=='wall':
        from wall_structural import wall
        wall(a.out,a.seed)
        return
    if a.phase=='packed':
        from packed_structural import packed
        packed(a.out)
        return
    if a.phase=='final':
        from final_structural import final
        final(a.out)
        return
    if a.phase=='massive':
        from massive_structural import massive
        massive(a.out)
        return
    if a.phase=='refine':
        from refine_structural import refine
        refine(a.out)
        return
    if a.phase=='learning':
        from qualify_structural import qualify
        qualify(a.out)
        return
    from migrate_performance_checkpoint import migrate
    pilot=migrate('/workspace/pilot.pt',a.out/'pilot-compatible.pt')
    c=json.loads((HERE/'recipes/gpu-small.json').read_text())
    c.update(seed=a.seed,allocated_worker_hours=None,save_initial=False,updates=4,profile_updates=4)
    modes={'current':{},'encoded':{'preencode_learner':True},
           'dynamic':{'preencode_learner':True,'native_scheduler':'dynamic'},
           'graph':{'preencode_learner':True,'native_scheduler':'dynamic','graph_inference':True},
           'graphs':{'preencode_learner':True,'native_scheduler':'dynamic','graph_inference':True,'graph_learner':True},
           'cpu-actor':{'preencode_learner':True,'native_scheduler':'dynamic','actor_device':'cpu'}}
    errors=[]
    for workload in ['initial','busy-trained']:
        for trial in range(2):
            order=list(modes) if trial%2==0 else list(reversed(modes))
            for mode in order:
                cc=copy.deepcopy(c);cc.update(modes[mode],start_turn=8 if workload=='busy-trained' else 1)
                name=f'{workload}-t{trial}-{mode}'
                try:
                    row=run_command(a.out/name,cc,'profile',pilot if workload=='busy-trained' else None)
                except Exception as exc:
                    errors.append({'run':name,'error':repr(exc)});continue
                row.update(mode=mode,trial=trial,workload=workload)
                rows.append(row)
                write_json(a.out/'progress.json',{'rows':rows,'errors':errors})
                print(json.dumps({'completed':name,'seconds':row['full_process_seconds']}),flush=True)
    checks=[]
    for workload in ['initial','busy-trained']:
        for trial in range(2):
            base=a.out/f'{workload}-t{trial}-current'
            for mode in ['encoded','dynamic','graph','graphs']:
                path=a.out/f'{workload}-t{trial}-{mode}'
                if (path/'latest.pt').exists():checks.append(dict(workload=workload,trial=trial,mode=mode,checks=compare(base,path)))
    scaling=[]
    for n_games in [128,512]:
        for mode in ['graphs','cpu-actor']:
            cc=copy.deepcopy(c);cc.update(modes[mode],n_games=n_games)
            name=f'scale-{n_games}-{mode}'
            try:
                row=run_command(a.out/name,cc,'profile')
                row.update(mode=mode,n_games=n_games);scaling.append(row)
            except Exception as exc:errors.append({'run':name,'error':repr(exc)})
            print(json.dumps({'completed':name}),flush=True)
    write_json(a.out/'scaling.json',scaling)
    write_json(a.out/'summary.json',{'rows':rows,'errors':errors,'parity':checks,'allocated_function_seconds':time.perf_counter()-started})


if __name__=='__main__':main()
