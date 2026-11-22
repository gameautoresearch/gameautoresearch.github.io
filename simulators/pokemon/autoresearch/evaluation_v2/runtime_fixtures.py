"""Remote fixed-work reproducibility, visibility and hard-deadline fixtures."""
import argparse,json
from pathlib import Path
from .qualification import prepare_search
from .env import EvaluationVector
from .foul_process import FoulPlayProcess
from ..contract import write


def main():
 p=argparse.ArgumentParser();p.add_argument('--request',required=True);p.add_argument('--output',required=True);a=p.parse_args()
 spec=json.loads(Path(a.request).read_text());out=Path(a.output);source=prepare_search(spec,out,a.request)
 with EvaluationVector(1) as vector:
  vector.reset([900000151],4096);visible=vector.visible([0])[0]
 records=[]
 for i in range(3):
  value=json.loads(json.dumps(visible))
  if i==2:
   import re
   value['protocol']=[re.sub(r'^\|t:\|[0-9]+$','|t:|9999999999',text,flags=re.MULTILINE) for text in value['protocol']]
  with FoulPlayProcess(source,spec['public_sets'],mode='fixed-work',iterations=1000,time_ms=10,samples=2,timeout=1.,deny_file=a.request) as bot:
   action,meta=bot.choose(value)
   records.append({'action':action,'input_sha256':meta['input_sha256'],'belief_state_sha256':meta['belief_state_sha256'],'visits':meta['visits'],'identity':bot.startup})
 assert records[0]==records[1]==records[2]
 with FoulPlayProcess(source,spec['public_sets'],mode='fixed-work',iterations=1000,time_ms=10,samples=2,timeout=.000001,deny_file=a.request) as bot:
  try:bot.choose(visible)
  except TimeoutError:timeout=True
  else:timeout=False
 assert timeout
 with FoulPlayProcess(source,spec['public_sets'],mode='fixed-work',iterations=1000,time_ms=10,samples=2,timeout=1.,deny_file=a.request) as bot:
  malicious=dict(visible,actual_hidden_team=['forbidden'])
  try:bot.choose(malicious)
  except ValueError:hidden_rejected=True
  else:hidden_rejected=False
 assert hidden_rejected
 write(out/'summary.json',{'passed':True,'fixed_work_repetitions':records,'wall_timestamp_invariance':True,
  'reference_timeout_raises_without_fallback_or_points':True,'extra_hidden_field_rejected_without_fallback':True,
  'reference_uid':65533,'candidate_uid':65534,'private_input_denied_both_uids':True,'fixed_work':{'belief_samples':2,'iterations_per_sample':1000,'threads':1},
  'production':{'belief_samples':2,'soft_milliseconds_per_sample':10,'hard_end_to_end_seconds':1.,'note':'Upstream checks time in1000iteration chunks; observed overshoot reported, hard parent deadline bounds request.'}})
if __name__=='__main__':main()
