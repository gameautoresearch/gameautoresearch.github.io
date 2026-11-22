"""Paired fresh-bank summaries; preserve training seeds, seats and fixed opponents."""
from pathlib import Path
import argparse,json
import numpy as np
from simulators.clash.autoresearch.common import dump
from .launch import CAMPAIGN

def analyze(path):
 result=json.loads(Path(path).read_text())
 if not result['complete']:raise ValueError('invalid assessment has no aggregate score')
 families=sorted({x['family'] for x in result['results']});opponents=sorted({x['opponent'] for x in result['results']});units=result['engine_units'];arrays={};summaries={}
 for family in families:
  records=[x for x in result['results'] if x['family']==family];seeds=sorted({x['training_seed'] for x in records});stages=sorted({x['stage'] for x in records})
  if len(stages)!=1:raise ValueError('A family must use one selected stage for every seed')
  a=np.full((len(seeds),units,len(opponents),2),np.nan);request=run=0
  for record in records:
   for row in record['rows']:
    if not row['natural'] or row['administrative_cap'] or row['illegal_action'] or row['simulator_fault'] or row['infrastructure_fault']:raise ValueError('invalid game in assessment')
    pos=(seeds.index(record['training_seed']),row['public_seed_index'],opponents.index(record['opponent']),row['left_seat'])
    if np.isfinite(a[pos]):raise ValueError('duplicate assessment unit')
    a[pos]=row['left_points'];side=row['left_seat']
    request+=row.get('visible_request_refusal_nonzero_advanced_decisions_by_seat',[0,0])[side]
    run+=row.get('visible_run_refusal_nonzero_advanced_decisions_by_seat',[0,0])[side]
  if not np.isfinite(a).all():raise ValueError('incomplete paired bank')
  arrays[family]=(seeds,a)
  summaries[family]={'training_seeds':seeds,'stage':stages[0],'games':int(a.size),'mean_points_0_100':float(a.mean()*100),'natural_wins':int((a==1).sum()),'natural_ties':int((a==.5).sum()),'natural_losses':int((a==0).sum()),'by_opponent':{name:float(a[:,:,i,:].mean()*100) for i,name in enumerate(opponents)},'by_seat':{str(i):float(a[:,:,:,i].mean()*100) for i in (0,1)},'by_training_seed':{str(seed):float(a[i].mean()*100) for i,seed in enumerate(seeds)},'visible_request_refusal_nonzero_decisions':request,'visible_run_refusal_nonzero_decisions':run}
 comparisons={}
 for reference,control in [('reference','control'),('reference','untrained'),('spatial-reference','spatial-control'),('spatial-reference','spatial-untrained')]:
  if reference not in arrays or control not in arrays:continue
  seeds,a=arrays[reference];other,b=arrays[control]
  if seeds!=other:raise ValueError('comparison training seeds are not matched')
  delta=(a-b).mean((2,3));rng=np.random.default_rng(7092026);boot=[]
  for _ in range(5000):
   t=rng.integers(0,len(seeds),len(seeds));u=rng.integers(0,units,units);boot.append(float(delta[np.ix_(t,u)].mean()*100))
  comparisons[reference+'-minus-'+control]={'mean_delta_points':float(delta.mean()*100),'interval95':np.quantile(boot,[.025,.975]).tolist(),'method':'crossed training-seed/evaluation-unit percentile bootstrap; fixed opponent panel; small training replication' if len(seeds)>1 else 'paired evaluation-unit percentile bootstrap, conditional on ONE training seed; not a seed-generalized method result','training_replicates':len(seeds)}
 value={'complete':True,'natural_games':result['natural_games'],'families':summaries,'paired_comparisons':comparisons,'harness_qualified':True,'strong_panel_qualified':False,'superhuman_evidence':False,'reference_normalized_score':None,'qualification_limit':'Panel opponents lack independent expert calibration; a fixed-panel score is not a human/superhuman win rate.'}
 for prefix in ('','spatial-'):
  keys=[prefix+'reference-minus-'+prefix+x for x in ('control','untrained')]
  value[prefix+'learned_reference_improvement_qualified']=all(k in comparisons and comparisons[k]['interval95'][0]>0 for k in keys)
 dump(CAMPAIGN/'clash/rl/fresh-assessment/analysis.json',value);return value

def main():
 p=argparse.ArgumentParser();p.add_argument('--result',required=True);a=p.parse_args();print(json.dumps(analyze(a.result),indent=2))
if __name__=='__main__':main()
