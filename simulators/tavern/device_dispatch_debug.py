"""Locate launch-width/context-reuse differences against the independent CPU oracle."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import torch
import taverngym as oracle
from device_env import DeviceEnv
from device_state_schema import native_episode


def differences(a,b,path='',limit=12):
    out=[]
    if type(a)!=type(b):return [(path,'types',type(a).__name__,type(b).__name__)]
    if isinstance(a,dict):
        for k in sorted(set(a)|set(b)):
            if k not in a or k not in b:out.append((path+'.'+k,'missing'))
            else:out+=differences(a[k],b[k],path+'.'+k,limit)
            if len(out)>=limit:break
    elif isinstance(a,list):
        if len(a)!=len(b):out.append((path,'length',len(a),len(b)))
        for i,(x,y) in enumerate(zip(a,b)):
            out+=differences(x,y,path+f'[{i}]',limit)
            if len(out)>=limit:break
    elif a!=b:out.append((path,a,b))
    return out[:limit]


def run_case(out,width,watch=None):
    out.mkdir(parents=True,exist_ok=False);started=time.perf_counter()
    trace=json.loads(Path('/results/native-state-schema-v29/scripted-128.json').read_text())
    cpu=oracle.BatchEnv(trace['n_games'],seed=trace['seed'],controlled=[0],config=trace['rules'],threads=4)
    env=DeviceEnv(trace['n_games'],seed=trace['seed'],controlled=(0,),config=trace['rules'],block_threads=width)
    def snapshot(stage):
        a=[json.loads(cpu.save(g)) for g in range(cpu.n_games)]
        b=[native_episode(json.loads(s),seed=trace['seed'],controlled=(0,)) for s in env.save()]
        diff=differences(a,b)
        if diff:
            (out/'cpu.states.json').write_text(json.dumps(a,separators=(',',':'))+'\n')
            (out/'gpu.states.json').write_text(json.dumps(b,separators=(',',':'))+'\n')
        return {'stage':stage,'state_differences':diff}
    result={'block_threads':width,'passed':False,'native_source':oracle.build_info()}
    try:
        initial=snapshot('initial')
        if initial['state_differences']:result.update(initial)
        else:
            actions=torch.tensor(trace['actions'],dtype=torch.int64,device='cuda')
            for step,(host_action,gpu_action) in enumerate(zip(trace['actions'],actions)):
                x=cpu.observe();y={k:v.cpu().numpy() for k,v in env.observe().items()}
                ready=np.flatnonzero(x['ready']);bad=[]
                if not np.array_equal(x['ready'],y['ready']):bad.append('ready')
                for key in ['features','cards','action_mask']:
                    if not np.array_equal(x[key][ready],y[key][ready]):bad.append(key)
                if bad:
                    result.update(step=step,view_mismatches=bad,**snapshot('before_action'));break
                ar,ad,ai=cpu.step(np.array(host_action,np.int64));br,bd,bi=env.step(gpu_action)
                bad=[]
                if not np.array_equal(ar,br.cpu().numpy()):bad.append('rewards')
                if not np.array_equal(ad,bd.cpu().numpy()):bad.append('dones')
                for key in ai:
                    if not np.array_equal(ai[key],bi[key].cpu().numpy()):bad.append(key)
                if bad:
                    result.update(step=step,result_mismatches=bad,**snapshot('after_action'));break
                if watch:
                    a=[json.loads(cpu.save(gi)) for gi in watch]
                    b=[native_episode(json.loads(s),seed=trace['seed'],controlled=(0,)) for s in env.save(indices=watch)]
                    diff=differences(a,b)
                    if diff:
                        (out/'first-hidden-cpu.states.json').write_text(json.dumps(a,separators=(',',':'))+'\n')
                        (out/'first-hidden-gpu.states.json').write_text(json.dumps(b,separators=(',',':'))+'\n')
                        result.update(step=step,watched_games=watch,hidden_state_differences=diff,**snapshot('after_action_hidden_state'));break
            else:
                final=snapshot('final');result.update(final,steps=len(trace['actions']),passed=not final['state_differences'])
    finally:
        env.close();result['wall_seconds']=time.perf_counter()-started
        (out/'case.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps({'dispatch_debug_case':str(out),'result':result}),flush=True)
    return result


def qualify(out):
    rows=[]
    for name,width in [('width1-first',1),('width1-repeat',1),('width2-reused-context',2),('width32-reused-context',32)]:
        rows.append(run_case(out/name,width))
        (out/'isolation-progress.json').write_text(json.dumps(rows,indent=2)+'\n')
    child=out/'width2-fresh-context'
    proc=subprocess.run([sys.executable,'-u',str(Path(__file__)),'--out',str(child),'--width','2'],timeout=1200)
    if proc.returncode:raise RuntimeError('fresh-context diagnostic failed; retain outputs')
    rows.append(json.loads((child/'case.json').read_text()))
    return {'rows':rows,'all_cases_passed':all(r['passed'] for r in rows),
        'scope':'Diagnostic only: full native/GPU initial state, every view/mask/result and final state on identical frozen trace. First divergence retained; no performance or learning qualification claim.'}


def watch_hidden(out):
    rows=[run_case(out/'width2-state-watch',2,watch=[98]),run_case(out/'width32-state-watch',32,watch=[98])]
    return {'rows':rows,'all_cases_passed':all(r['passed'] for r in rows),
            'scope':'Independent full hidden game/RNG state checked after every action for recorded lobby98; complete rules/other lobbies remain active. Diagnostic, not learning or throughput qualification.'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--width',type=int,required=True)
    a=p.parse_args();run_case(a.out,a.width)
