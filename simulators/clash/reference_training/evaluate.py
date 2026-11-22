"""Focused public development pair assessment with canonical evaluator-v2 actors."""
from pathlib import Path
import argparse,json,fcntl
from simulators.clash.autoresearch.common import ROOT,dump,sha
from .export_adapter import derive
from simulators.clash.autoresearch.bundle import seal,verify
from simulators.clash.autoresearch.readiness_contract import admit

DEVELOPMENT_SEEDS=[1000101,1000102,1000103,1000104,1000105,1000106,1000107,1000108]

def bundle(original,target,provenance):
 original=Path(original);target=Path(target)
 if not target.exists():
  derived=target.parent/(target.name+'-derived');receipt=derive(original,derived)
  manifest=seal(derived,target,{**provenance,'derived_head':receipt})
 else:manifest=verify(target)
 admission=admit(target)
 return {'kind':'checkpoint','bundle':str(target.resolve()),'identity':manifest['identity'],'parameters':manifest['parameters'],'admission_interface_sha256':admission['interface_sha256']}

def assess(spec,output,seeds=DEVELOPMENT_SEEDS,opponents=None):
 from simulators.clash.autoresearch.evaluation_v2 import screen
 output=Path(output);output.mkdir(parents=True,exist_ok=False)
 opponents=opponents or [{'name':'standard','kind':'scripted'},{'name':'pressure','kind':'scripted'},{'name':'intermediate','kind':'intermediate'},{'name':'teacher','kind':'teacher'}]
 dump(output/'plan.json',{'candidate':spec,'opponents':opponents,'public_development_seeds':seeds,'teacher_source_sha256':sha(ROOT/'simulators/clash/autoresearch/visible_teacher.py'),'selection_bank':'public development; reusable for iteration'})
 results=[]
 with open(ROOT/'evidence/local-performance.lock','a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  for opponent in opponents:
   r=screen([spec,opponent],output/opponent['name'],seeds=seeds,development=True,max_seconds=300)
   results.append({'opponent':opponent['name'],'complete':r['complete'],'pair':r['pairs'][0],'failures':r['failures'],'rows':r['rows'],'entrant_probes':r['entrant_probes']})
 complete=all(r['complete'] for r in results)
 value={'complete':complete,'games':sum(len(r['rows']) for r in results),'mean_points':sum(r['pair']['left_points_0_100'] for r in results)/len(results) if complete else None,'by_opponent':{r['opponent']:r['pair']['left_points_0_100'] for r in results},'results':results,'strong_panel_qualified':False,'superhuman_evidence':False}
 dump(output/'result.json',value);return value

def main():
 p=argparse.ArgumentParser();p.add_argument('--original',required=True);p.add_argument('--bundle',required=True);p.add_argument('--out',required=True);p.add_argument('--name',default='candidate');a=p.parse_args();spec={'name':a.name,**bundle(a.original,a.bundle,{'track':'clash-reference-development'})};r=assess(spec,a.out);print(json.dumps({k:v for k,v in r.items() if k!='results'}))
if __name__=='__main__':main()
