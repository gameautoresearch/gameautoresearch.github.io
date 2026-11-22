"""Exclusive local benchmark; all other games share this advisory flock."""
import fcntl,json,time,statistics,argparse,platform
from pathlib import Path
from .cuda_suite import config,profile_case
from .provenance import manifest,write_json,ROOT

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args()
 with (ROOT/'evidence/local-performance.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  cfg=config('cpu',2,4);cfg['hardware']['machine']='local-'+platform.machine()
  out=Path(args.output);manifest(out,cfg,'exclusive-cpu-paired',{'lock':str(ROOT/'evidence/local-performance.lock')})
  rows={};cp=ROOT/'evidence/pokemon-implementation/learning-verified/checkpoint.pt'
  for variant in ('baseline','packed'):rows[variant]=profile_case(out/variant,cfg,variant=variant,repeats=3,checkpoint=cp)
  reference=rows['baseline'][0]['traces']
  assert all(r['traces']==reference for group in rows.values() for r in group)
  median={k:statistics.median(r['seconds'] for r in group) for k,group in rows.items()}
  result={'same_trajectories':True,'lock_exclusive':True,'rows':rows,'median_seconds':median,'speedup':median['baseline']/median['packed']}
  write_json(out/'results.json',result);print(json.dumps({'median_seconds':median,'speedup':result['speedup']},indent=2))
if __name__=='__main__':main()
