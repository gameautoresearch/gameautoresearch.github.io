"""Compare only the explicitly published32-seed CPU development bank."""
import argparse,json,re
import numpy as np
from .provider import OWN,CAMPAIGN
from simulators.clash.autoresearch.common import dump,sha

def compare(batch,candidate):
 if not all(re.fullmatch('[a-zA-Z0-9_-]+',s) for s in (batch,candidate)):raise ValueError('Simple public batch/candidate names required')
 root=OWN/'cpu-development'/batch;plan=json.loads((root/'inputs/plan.json').read_text());base=CAMPAIGN/'clash/evaluator/jobs/public-v31-capacity-01';schedule=json.loads((base/'inputs/schedule.json').read_text())
 if plan['purpose']!='PUBLIC DEVELOPMENT ONLY' or plan['public_units']!=32 or plan['public_engine_seeds']!=list(range(2100401,2100433)) or schedule['engine_seeds']!=plan['public_engine_seeds']:raise ValueError('PublishedPUBLIC32 schedule required')
 semantic=lambda panel:[(x['name'],x['kind'],x.get('identity')) for x in panel]
 if semantic(plan['opponents'])!=semantic(schedule['opponents']):raise ValueError('Public opponent registries differ')
 path=next((p for p in (root/'collected/grades'/(candidate+'.json'),root/'observed-complete-candidates'/(candidate+'.json')) if p.exists()),None)
 if path is None:raise FileNotFoundError('Complete individual public grade not collected')
 candidate_grade=json.loads(path.read_text());baseline_path=base/'retained/grade.json';baseline=json.loads(baseline_path.read_text());identity=next(x['identity'] for x in plan['candidates'] if x['name']==candidate)
 if candidate_grade['submission_identity']!=identity or baseline['submission_identity']!='79783463c9a2afb8be871de100448d6a5b3a802bd7c76c3cf2758a77bb760613' or candidate_grade['interface_sha256']!=baseline['interface_sha256']:raise ValueError('Public candidate/baseline interface identity differs')
 def cells(grade):
  if grade['games']!=384 or grade['natural_games']!=384 or any(grade[k] for k in ('truncations','illegal_actions','simulator_faults')):raise ValueError('Incomplete/invalid public grade')
  values={}
  for r in grade['rows']:
   k=(r['opponent'],r['seed_index'],r['seat'])
   if k in values or not r['natural'] or any(r.get(n) for n in ('truncated','illegal','fault')) or r['points'] not in (0.,.5,1.):raise ValueError('Invalid public cell')
   values[k]=r['points']
  if set(values)!={(p['name'],i,s) for p in plan['opponents'] for i in range(32) for s in (0,1)}:raise ValueError('Missing public cells')
  return values
 a,b=cells(candidate_grade),cells(baseline);differences=np.array([100*np.mean([a[k]-b[k] for k in a if k[1]==i]) for i in range(32)]);rng=np.random.default_rng(900073);boot=differences[rng.integers(0,32,size=(100000,32))].mean(1)
 value={'purpose':'complete individual candidate on publishedPUBLIC32 bank; no privatefeedback','candidate':candidate,'candidate_identity':identity,'candidate_points':candidate_grade['raw_score_0_100'],'baseline_points':baseline['raw_score_0_100'],'games_each':384,'public_seed_clusters':32,'paired_change':float(differences.mean()),'paired_seed_bootstrap_interval95':np.percentile(boot,[2.5,97.5]).tolist(),'by_opponent':candidate_grade['by_opponent'],'candidate_grading_seconds':candidate_grade['seconds'],'baseline_grading_seconds':baseline['seconds'],'candidate_grade_sha256':sha(path),'baseline_grade_sha256':sha(baseline_path),'all_required_candidate_cells_complete':True,'repeated_selection_corrected':False,'strong_or_expert_evidence':False}
 dump(root/(candidate+'-paired-public32.json'),value);return value

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--batch',required=True);p.add_argument('--candidate',required=True);a=p.parse_args();print(json.dumps(compare(a.batch,a.candidate),indent=2))
