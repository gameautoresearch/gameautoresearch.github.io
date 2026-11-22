"""Trainer-owned PUBLIC development only, using the unchanged qualified arena."""
import json,time
from pathlib import Path
from ..autoresearch.evaluation_v2.arena import run as assess
from .strong_campaign import write

def run(spec,job):
 if spec.get('split')!='public-development':raise ValueError('Trainer development must never open a private final bank')
 out=job/'public-assessment';out.mkdir();results=[]
 for item in spec['assessments']:
  if not item['seeds'] or any(not 900_000_000<=x<1_000_000_000 for x in item['seeds']):raise ValueError('Public development seeds required')
  if time.time()>spec['deadline_unix']-20:raise TimeoutError('Public cohort incomplete at finite allocation deadline')
  target=out/item['candidate']['name'];target.mkdir();assess(item,target,str(job/'request.json'))
  results.append({'candidate':item['candidate'],'summary':json.loads((target/'summary.json').read_text())})
  write(out/'screen.json',results)
 return {'public_development_only':True,'independent_expert_qualification':False,'candidates':results}
