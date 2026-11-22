"""Qualified3-pair evaluator on the previously published32-unit PUBLIC bank."""
from pathlib import Path
import argparse,json,subprocess,time
from simulators.clash.autoresearch.common import dump,sha,digest,engine_binding,GAME

def run(root):
 root=Path(root);root.chmod(0o700);inputs=root/'inputs';inputs.chmod(0o700);plan=json.loads((inputs/'plan.json').read_text())
 from simulators.clash.autoresearch.release_v3 import PUBLIC_CAPACITY_SEEDS,PUBLIC_SEEDS,INTERFACE_V31
 from simulators.clash.autoresearch.bundle import verify
 from simulators.clash.autoresearch.evaluation_v3_parallel import evaluate
 seeds=PUBLIC_CAPACITY_SEEDS if plan['public_units']==32 else PUBLIC_SEEDS
 if plan['public_units'] not in (8,32):raise ValueError('Published8/32unitpublicschedule required')
 if plan['purpose']!='PUBLIC DEVELOPMENT ONLY' or plan['engine_sha256']!=digest(engine_binding()) or plan['public_engine_seeds']!=seeds:raise ValueError('Explicit published public32-unit schedule required')
 if any(sha(GAME/'autoresearch'/n)!=v for n,v in plan['qualified_parallel_runtime_files'].items()):raise ValueError('Qualified runtime source differs')
 for item in plan['candidates']+plan['opponents']:
  if item['kind']=='checkpoint':verify(inputs/item['bundle'],item['identity']);item['bundle']=str(inputs/item['bundle'])
 schedule=root/'public-schedule.json';dump(schedule,{'engine_seeds':seeds,'policy_seed':20261109,'opponents':plan['opponents'],'schedule_kind':'public-32-unit-capacity-qualification'})
 started=time.time();deadline=started+plan['worker_seconds'];summaries={};grades=[]
 for candidate in plan['candidates']:
  remaining=int(deadline-time.time())
  if remaining<30:raise ValueError('Finite public worker deadline reached before next candidate')
  out=root/'grades'/(candidate['name']+'.json');result=evaluate(candidate['bundle'],schedule,out,development=False,split='public-32-unit-development',max_seconds=remaining,interface=INTERFACE_V31,pair_workers=3)
  if result['games']!=len(seeds)*2*len(plan['opponents']) or any(result[k] for k in ('truncations','illegal_actions','simulator_faults')):raise ValueError('Incomplete/invalid public32-unit candidate')
  summaries[candidate['name']]={'complete':True,'public_development_only':True,'mean_points':result['raw_score_0_100'],'by_opponent':result['by_opponent'],'games':result['games'],'identity':candidate['identity']}
  grades.append({'candidate':candidate['name'],'relative_path':str(out.relative_to(root)),'sha256':sha(out)})
  dump(root/'progress.json',{'public_development_only':True,'completed_candidates':len(grades),'candidate':candidate['name'],'elapsed_seconds':time.time()-started});subprocess.run(['sync',str(root)],check=True,timeout=20)
 dump(root/'result.json',{'complete':True,'public_development_only':True,'public_units':len(seeds),'pair_workers':3,'summaries':summaries,'grades':grades,'elapsed_seconds':time.time()-started,'plan_sha256':sha(inputs/'plan.json')});subprocess.run(['sync',str(root)],check=True,timeout=20)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);args=p.parse_args();run(args.root)
