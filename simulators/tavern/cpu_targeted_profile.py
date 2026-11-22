"""Profile the qualified compact CPU simulator on a frozen full-game trace.

Timing runs hold the shared local performance lock. Sampling is diagnostic only;
its timings are never used as speedup evidence.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]


def workload(trace,seconds,ready_path):
    import numpy as np
    import taverngym as tg
    t=json.loads(trace.read_text());actions=[np.array(a,np.int64) for a in t['actions']]
    ready_path.write_text(json.dumps({'pid':os.getpid(),'native':tg.build_info()})+'\n')
    started=time.perf_counter();trials=decisions=0;observe=step=0.
    while time.perf_counter()-started<seconds:
        env=tg.BatchEnv(t['n_games'],seed=t['seed'],controlled=[0],config=t['rules'],threads=4)
        for action in actions:
            begin=time.perf_counter();o=env.observe_ready(copy=False);observe+=time.perf_counter()-begin
            decisions+=len(o['slots'])
            begin=time.perf_counter();_,_,info=env.step(action);step+=time.perf_counter()-begin
            if info['rejected'].any():raise AssertionError('frozen profile action rejected')
        trials+=1
    print(json.dumps({'trials':trials,'decisions':decisions,'observation_seconds':observe,
        'step_seconds':step,'wall_seconds':time.perf_counter()-started,'native':tg.build_info()}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path);p.add_argument('--trace',type=Path,required=True)
    p.add_argument('--seconds',type=int,default=12)
    p.add_argument('--worker-ready',type=Path)
    a=p.parse_args()
    if a.worker_ready:return workload(a.trace,a.seconds,a.worker_ready)
    if not a.out or a.out.exists():raise ValueError('new output directory required')
    a.out.mkdir(parents=True)
    import fcntl,shutil,taverngym as tg
    package=Path(tg.__file__).parent
    shutil.copytree(package,a.out/'baseline-package/taverngym',ignore=shutil.ignore_patterns('__pycache__'))
    source={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted((ROOT/'candidates/AutoTavern/crates').rglob('*.rs'))}
    binary=Path(tg.build_info()['file'])
    (a.out/'baseline.json').write_text(json.dumps({'native':tg.build_info(),
        'native_sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),'source_sha256':source,
        'trace_sha256':hashlib.sha256(a.trace.read_bytes()).hexdigest()},indent=2)+'\n')
    with (ROOT/'evidence/local-performance.lock').open('a+') as lock:
        print('waiting for local performance lock',flush=True);fcntl.flock(lock,fcntl.LOCK_EX)
        ready=a.out/'worker-ready.json'
        with (a.out/'workload.log').open('x') as log:
            child=subprocess.Popen([sys.executable,'-u',str(Path(__file__)),
                '--trace',str(a.trace.resolve()),'--seconds',str(a.seconds),'--worker-ready',str(ready.resolve())],
                stdout=log,stderr=subprocess.STDOUT,
                env=dict(os.environ,TAVERNGYM_WORKERS='pooled',TAVERNGYM_SCHEDULER='dynamic'))
            deadline=time.monotonic()+30
            while not ready.exists():
                if child.poll() is not None:raise RuntimeError('profile worker failed; inspect workload.log')
                if time.monotonic()>deadline:child.terminate();raise TimeoutError('profile worker startup')
                time.sleep(.05)
            if sys.platform=='darwin':
                sample=subprocess.run(['/usr/bin/sample',str(child.pid),str(max(1,a.seconds-3)),'1',
                    '-file',str((a.out/'native-sample.txt').resolve())],capture_output=True,text=True)
                (a.out/'sampler.json').write_text(json.dumps({'returncode':sample.returncode,'stderr':sample.stderr})+'\n')
            if child.wait(timeout=a.seconds+30):raise RuntimeError('profile workload failed')
    print(json.dumps({'completed':str(a.out)}),flush=True)


if __name__=='__main__':main()
