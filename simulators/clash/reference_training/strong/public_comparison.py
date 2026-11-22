"""Paired public seed-cluster diagnostic; repeated selection remains exploratory."""
from pathlib import Path
import argparse,json
import numpy as np
from simulators.clash.autoresearch.common import dump,sha

def compare(candidate,baseline,out):
 def read(root):
  root=Path(root);p=root/'assessment';plan=json.loads((p/'plan.json').read_text());result=json.loads((p/'result.json').read_text())
  if not result['complete']:raise ValueError('No aggregate/interval for an incomplete public panel')
  rows={}
  for pair in result['results']:
   for row in pair['rows']:
    if not row['natural'] or any(row.get(k) for k in ('illegal_action','simulator_fault','infrastructure_fault','administrative_cap')):raise ValueError('Invalid public outcome')
    key=(pair['opponent'],row['public_seed_index'],row['left_seat'])
    if key in rows:raise ValueError('Duplicate public cell')
    rows[key]=float(row['left_points'])
  spec=[(x['name'],x['kind'],x.get('identity')) for x in plan['opponents']]
  return plan,spec,rows,sha(p/'result.json')
 a,ao,ar,ah=read(candidate);b,bo,br,bh=read(baseline)
 if a['public_development_seeds']!=b['public_development_seeds'] or ao!=bo or set(ar)!=set(br):raise ValueError('Public panels differ')
 n=len(a['public_development_seeds']);expected=len(ao)*n*2
 if len(ar)!=expected:raise ValueError('Missing required public cells')
 differences=np.array([np.mean([ar[k]-br[k] for k in ar if k[1]==seed])*100 for seed in range(n)])
 rng=np.random.default_rng(900071);boot=differences[rng.integers(0,n,size=(100000,n))].mean(1)
 value={'purpose':'PUBLIC exploratory paired diagnostic; no private feedback','candidate':str(Path(candidate).resolve()),'baseline':str(Path(baseline).resolve()),'games_each':len(ar),'independent_public_seed_clusters':n,'candidate_points':100*np.mean(list(ar.values())),'baseline_points':100*np.mean(list(br.values())),'paired_change':float(differences.mean()),'seed_cluster_differences':differences.tolist(),'paired_seed_bootstrap_interval95':np.percentile(boot,[2.5,97.5]).tolist(),'resamples':100000,'bootstrap_rng_seed':900071,'candidate_result_sha256':ah,'baseline_result_sha256':bh,'repeated_selection_corrected':False,'method_replication_or_expert_strength_established':False}
 dump(out,value);return value

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--candidate',required=True);p.add_argument('--baseline',required=True);p.add_argument('--out',required=True);a=p.parse_args();print(json.dumps(compare(a.candidate,a.baseline,a.out),indent=2))
