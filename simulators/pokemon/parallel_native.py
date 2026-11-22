"""Simulator+visible-encoder scaling, separate from neural/PPO throughput."""
import argparse,fcntl,json,time,subprocess,statistics
from pathlib import Path
from .provenance import ROOT,write_json,manifest
from .cuda_suite import config

def run(out,cpus=4,repeats=3,games=256):
    out=Path(out);cfg=config('cpu',cpus-1,1);cfg['hardware']['attached_cpu_count']=cpus
    manifest(out,cfg,'parallel-native',{'games_per_process':games,'repeats':repeats,'seeds':'23000 + worker*10000 + game','policy':'public power/LCG diagnostic'})
    rows=[];counts=[1,3] if cpus==4 else [1,3,7,15,31]
    for repeat in range(repeats):
        for workers in counts if repeat%2==0 else list(reversed(counts)):
            start=time.perf_counter()
            jobs=[subprocess.Popen(['node','integrations/pokemon/simulator_profile.cjs',f'--games={games}',f'--seed-start={23000+i*10000}'],cwd=ROOT,stdout=subprocess.PIPE,text=True) for i in range(workers)]
            results=[]
            try:
                for job in jobs:
                    raw,_=job.communicate(timeout=120)
                    if job.returncode:raise RuntimeError(f'Native worker exited {job.returncode}')
                    results.append(json.loads(raw))
            finally:
                for job in jobs:
                    if job.poll() is None:job.kill();job.wait()
            elapsed=time.perf_counter()-start
            row=dict(repeat=repeat,workers=workers,seconds=elapsed,games=sum(r['count'] for r in results),
                decisions=sum(r['decisions'] for r in results),late_decisions=sum(r['late'] for r in results),
                natural=sum(r['natural'] for r in results),truncations=sum(r['truncated'] for r in results),faults=0,
                decisions_per_second=sum(r['decisions'] for r in results)/elapsed,processes=results)
            rows.append(row);write_json(out/'partial.json',{'rows':rows});print(json.dumps({k:v for k,v in row.items() if k!='processes'}),flush=True)
    result=dict(rows=rows,cpus=cpus,median_decisions_per_second={str(w):statistics.median(r['decisions_per_second'] for r in rows if r['workers']==w) for w in counts})
    write_json(out/'results.json',result);return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--cpus',type=int,default=4);p.add_argument('--repeats',type=int,default=3)
    args=p.parse_args()
    with (ROOT/'evidence/local-performance.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);run(args.output,args.cpus,args.repeats)
if __name__=='__main__':main()
