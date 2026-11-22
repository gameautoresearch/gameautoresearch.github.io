"""New public-only development panel; never reads either campaign's private bank."""
from pathlib import Path
import argparse,json
from .provider import OWN,CAMPAIGN
from simulators.clash.reference_training.evaluate import bundle,assess
from simulators.clash.autoresearch.bundle import verify
from simulators.clash.autoresearch.common import ROOT,dump,sha,engine_binding

def execution_observer():
 from simulators.clash.autoresearch.common import setup_native
 setup_native();import cpu_batch
 from simulators.clash.autoresearch.refusal_diagnostics import DecisionCounters,visible_own_plays
 original=cpu_batch.NativeCPUBatch;recorders=[]
 class Observed(original):
  def __init__(self,*args,**kwargs):
   super().__init__(*args,**kwargs);self.execution=[[DecisionCounters(),DecisionCounters()] for _ in range(self.n)];recorders.append(self)
  def step(self,actions,*args,**kwargs):
   before=self.observations.copy();masks=self.masks.copy();result=super().step(actions,*args,**kwargs)
   for lane in range(self.n):
    for seat in (0,1):
     old,new,cards=visible_own_plays(before[lane,1-seat],self.observations[lane,1-seat])
     self.execution[lane][seat].observe(before[lane,seat],masks[lane,seat],int(actions[lane,seat]),self.observations[lane,seat],own_placed_before=old,own_placed_after=new,own_placed_cards=cards)
   return result
 cpu_batch.NativeCPUBatch=Observed;return original,recorders

def plan():
 target=OWN/'development-plan-v1.json'
 if target.exists():
  value=json.loads(target.read_text())
  if sha(ROOT/'simulators/clash/autoresearch/visible_teacher.py')!=value['teacher_sha256']:raise ValueError('Public development teacher source drift; a new qualified plan is required')
  for x in value['opponents']:
   if x['kind']=='checkpoint':verify(x['bundle'],x['identity'])
  return value
 parent=ROOT/'evidence/reference-readiness-20261109-0251Z/clash/rl/paired-v4-49192/teacher-final-bundle'
 retained=ROOT/'evidence/overnight-20261108-1533Z/clash/C/deepseek-v41-flash/submissions/c2691202f6bbfbe97a1ac949b28239003b3bd3842a0bdb9d67697201731be724'
 opponents=[{'name':'standard','kind':'scripted'},{'name':'pressure','kind':'scripted'},{'name':'intermediate','kind':'intermediate'},{'name':'teacher','kind':'teacher'}]
 for name,path in [('retained-deepseek',retained),('incumbent-v4',parent)]:
  identity=verify(path)['identity'];opponents.append({'name':name,'kind':'checkpoint','bundle':str(path),'identity':identity})
 value={'purpose':'reusable NEW PUBLIC development schedule for iteration; never private final feedback','public_engine_seeds':list(range(2100101,2100109)),'opponents':opponents,'teacher_sha256':sha(ROOT/'simulators/clash/autoresearch/visible_teacher.py'),'selection_cutoff_utc':'2026-11-09T09:36:36Z'}
 dump(target,value);return value

def evaluate(original,name):
 if json.loads((Path(original)/'lineage.json').read_text())['binding']['engine']!=engine_binding():raise ValueError('Original training engine differs from development release')
 p=plan();folder=OWN/'development'/name;folder.mkdir(parents=True,exist_ok=False)
 spec={'name':name,**bundle(original,folder/'bundle',{'track':'new-five-hour-public-development','plan':str(OWN/'development-plan-v1.json')})}
 original,recorders=execution_observer()
 try:result=assess(spec,folder/'assessment',seeds=p['public_engine_seeds'],opponents=p['opponents'])
 finally:
  import cpu_batch
  cpu_batch.NativeCPUBatch=original
 execution={}
 if len(recorders)!=len(result['results']):raise ValueError('Unexpected public development observer lane layout')
 from collections import Counter
 for record,env in zip(result['results'],recorders):
  rows=[];counts=Counter();request=Counter();run=Counter();attempted=Counter();placed=Counter()
  for row in record['rows']:
   lane=2*row['public_seed_index']+row['left_seat'];detail=env.execution[lane][row['left_seat']].summary();rows.append({'public_seed_index':row['public_seed_index'],'seat':row['left_seat'],'counters':detail})
   counts.update({k:v for k,v in detail.items() if isinstance(v,int)});request.update(detail['request_refusal_codes']);run.update(detail['run_refusal_codes']);attempted.update(detail['attempted_card_ids']);placed.update(detail['placed_card_ids'])
  execution[record['opponent']]={'counts':dict(counts),'request_codes':dict(request),'run_codes':dict(run),'attempted_cards':dict(attempted),'placed_cards':dict(placed),'rows':rows}
 summary={k:v for k,v in result.items() if k!='results'};summary['candidate']=spec;summary['public_execution_diagnostics']=execution;summary['observer_source_sha256']=sha(ROOT/'simulators/clash/autoresearch/refusal_diagnostics.py')
 dump(folder/'summary.json',summary);return summary

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--original',required=True);p.add_argument('--name',required=True);a=p.parse_args();print(json.dumps(evaluate(a.original,a.name),indent=2))
