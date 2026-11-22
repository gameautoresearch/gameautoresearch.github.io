"""V3 candidate-only grading through the qualified isolated cross-play path."""
import json,time
from pathlib import Path
from .common import dump,digest
from .bundle import verify
from .arena import summarize
from .evaluation_v2 import screen

def evaluate(bundle,schedule,out,*,development=False,split='public-development',max_seconds=900,interface=None,trajectory_digest=False):
 from .release_v3 import INTERFACE_V3
 value=json.loads(Path(schedule).read_text());manifest=verify(bundle);out=Path(out)
 specs=[{'name':'candidate','kind':'checkpoint','bundle':str(bundle),'identity':manifest['identity']},*value['opponents']]
 if any(s['name']=='candidate' for s in value['opponents']):raise ValueError('reserved candidate name')
 result=screen(specs,out.with_suffix('.details'),seeds=value['engine_seeds'],development=development,max_seconds=max_seconds,diagnostics=True,interface=interface or INTERFACE_V3,pair_names=[('candidate',s['name']) for s in value['opponents']],trajectory_digest=trajectory_digest)
 if not result['complete']:
  dump(out.with_suffix('.failure.json'),{'valid':False,'raw_score_0_100':None,'failures':result['failures'],'submission_identity':manifest['identity'],'schedule_commitment':digest(value),'completed_natural_games':result['natural_games']})
  raise ValueError('incomplete/invalid v3 grading; failure sidecar retained')
 rows=[{'opponent':r['right'],'seed_index':r['public_seed_index'],'seat':r['left_seat'],'winner':r['winner'],'points':r['left_points'],'decisions':r['decisions'],'natural':True,'truncated':False,'illegal':False,'fault':False,'execution_diagnostics':r['execution_diagnostics']} for r in result['rows']]
 if trajectory_digest:
  for row,original in zip(rows,result['rows']):row['public_trajectory_sha256']=original['public_trajectory_sha256']
 grade={**summarize(rows),'split':split,'seconds':result['seconds'],'development_only':development,'submission_identity':manifest['identity'],'schedule_commitment':digest(value),'interface_sha256':result['interface_sha256'],'rows':rows,'entrant_probes':result['entrant_probes'],'latencies':result['latencies'],'execution_diagnostics':True,'strong_panel_qualified':False,'superhuman_evidence':False}
 dump(out,grade);return grade
