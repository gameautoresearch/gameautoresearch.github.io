"""Bounded independent pair processes within the same8-core CPU allocation."""
import concurrent.futures,json,multiprocessing,time
from pathlib import Path
from .common import dump,digest
from .bundle import verify
from .arena import summarize

def _one_pair(payload):
 from .evaluation_v2 import screen
 return screen(payload['specs'],payload['out'],seeds=payload['seeds'],development=payload['development'],max_seconds=max(1,int(payload['deadline']-time.time()-10)),diagnostics=True,interface=payload['interface'],simulation_threads=1,uid_base=65480-payload['index']*2,trajectory_digest=payload['trajectory_digest'])

def evaluate(bundle,schedule,out,*,development=False,split='public-development',max_seconds=900,interface=None,pair_workers=3,trajectory_digest=False):
 from .release_v3 import INTERFACE_V3
 if pair_workers not in (1,2,3):raise ValueError('at most3 fixed CPU pair workers')
 value=json.loads(Path(schedule).read_text());manifest=verify(bundle);out=Path(out);details=out.with_suffix('.details');details.mkdir(parents=True,exist_ok=False);started=time.monotonic();deadline=time.time()+max_seconds;rows=[];results=[];failures=[];interface=interface or INTERFACE_V3
 if not value['opponents'] or any(p['name']=='candidate' for p in value['opponents']) or len({p['name'] for p in value['opponents']})!=len(value['opponents']):raise ValueError('invalid opponent names')
 candidate={'name':'candidate','kind':'checkpoint','bundle':str(bundle),'identity':manifest['identity']}
 try:
  with concurrent.futures.ProcessPoolExecutor(max_workers=pair_workers,mp_context=multiprocessing.get_context('spawn')) as pool:
   futures={pool.submit(_one_pair,{'specs':[candidate,p],'out':str(details/('pair-'+p['name'])),'seeds':value['engine_seeds'],'development':development,'deadline':deadline,'interface':interface,'index':i,'trajectory_digest':trajectory_digest}):p['name'] for i,p in enumerate(value['opponents'])}
   for future in concurrent.futures.as_completed(futures):
    name=futures[future]
    try:
     result=future.result();results.append({'opponent':name,'result':result})
     if not result['complete']:failures.extend(result['failures'] or [{'kind':'incomplete_pair','opponent':name}])
     for r in result['rows']:
      row={'opponent':r['right'],'seed_index':r['public_seed_index'],'seat':r['left_seat'],'winner':r['winner'],'points':r['left_points'],'decisions':r['decisions'],'natural':True,'truncated':False,'illegal':False,'fault':False,'execution_diagnostics':r['execution_diagnostics']}
      if 'public_trajectory_sha256' in r:row['public_trajectory_sha256']=r['public_trajectory_sha256']
      rows.append(row)
    except Exception as exc:failures.append({'kind':'pair_worker_failure','opponent':name,'error':type(exc).__name__+': '+str(exc)})
    dump(details/'progress.json',{'completed_pairs':len(results),'completed_natural_games':len(rows),'failures':failures})
  expected=len(value['opponents'])*2*len(value['engine_seeds'])
  if failures or len(rows)!=expected:raise ValueError('incomplete/invalid parallel grade')
  rows.sort(key=lambda r:(r['opponent'],r['seed_index'],r['seat']))
  grade={**summarize(rows),'split':split,'seconds':time.monotonic()-started,'development_only':development,'submission_identity':manifest['identity'],'schedule_commitment':digest(value),'interface_sha256':digest(interface),'rows':rows,'entrant_probes_by_pair':{r['opponent']:r['result']['entrant_probes'] for r in results},'latencies_by_pair':{r['opponent']:r['result']['latencies'] for r in results},'execution_diagnostics':True,'grading_pair_workers':pair_workers,'simulation_threads_per_pair':1,'strong_panel_qualified':False,'superhuman_evidence':False}
  dump(out,grade);return grade
 except Exception as exc:
  dump(out.with_suffix('.failure.json'),{'valid':False,'raw_score_0_100':None,'error':type(exc).__name__+': '+str(exc),'failures':failures,'rows':rows,'completed_natural_games':len(rows),'submission_identity':manifest['identity'],'schedule_commitment':digest(value)});raise
