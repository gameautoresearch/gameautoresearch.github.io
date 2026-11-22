"""Serialized simulator-only timing, with separate V8 sampling diagnostics."""
import fcntl,argparse,json,subprocess,statistics,time
from collections import Counter
from pathlib import Path
from .provenance import ROOT,write_json,manifest
from .cuda_suite import config

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args();out=Path(args.output)
 with (ROOT/'evidence/local-performance.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);manifest(out,config('cpu'),'native-profile',{'seeds':'23000..23127','choices':'visible power/LCG'})
  rows=[]
  for repeat in range(5):
   modes=['baseline','token','memo','engine']
   for mode in (modes if repeat%2==0 else list(reversed(modes))):
    flags=[] if mode=='baseline' else ['--cache'] if mode=='token' else ['--cache','--memo'] if mode=='memo' else ['--cache','--memo','--callbacks']
    args=['node','integrations/pokemon/simulator_profile.cjs','--games=256',*flags]
    start=time.perf_counter();r=json.loads(subprocess.check_output(args,cwd=ROOT,text=True));r['process_seconds']=time.perf_counter()-start;r['repeat']=repeat;r['mode']=mode;rows.append(r)
  assert len({r['sha256'] for r in rows})==1
  diagnostic=json.loads(subprocess.check_output(['node','--cpu-prof',f'--cpu-prof-dir={out.resolve()}','--cpu-prof-name=simulator.cpuprofile','integrations/pokemon/simulator_profile.cjs','--games=256','--cache','--memo','--callbacks'],cwd=ROOT,text=True))
  profile=json.loads((out/'simulator.cpuprofile').read_text());nodes={n['id']:n for n in profile['nodes']};hits=Counter(profile['samples'])
  functions=Counter();categories=Counter()
  for i,n in hits.items():
   frame=nodes[i]['callFrame'];url=frame['url'];functions[(url,frame['functionName'])]+=n
   category='showdown' if '/candidates/pokemon-showdown/' in url else 'visible_encoder' if url.endswith('/view.cjs') else 'adapter' if url.endswith('/worker.cjs') else 'runtime/driver'
   categories[category]+=n
  result=dict(rows=rows,exact_trace=True,median_seconds={c:statistics.median(r['process_seconds'] for r in rows if r['mode']==c) for c in modes},
    diagnostic=diagnostic,cpu_sample_categories=dict(categories),hot_functions=[dict(url=k[0],function=k[1],samples=n) for k,n in functions.most_common(30)])
  write_json(out/'results.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ('rows','hot_functions')},indent=2))
if __name__=='__main__':main()
