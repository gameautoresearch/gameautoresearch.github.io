"""Summarize private records without copying final seeds, weights or traces to reports."""
import argparse,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]

def usage(root):
 from tools.overnight_budget import CampaignBudget
 b=CampaignBudget('tavern',path=root/'campaign.sqlite3').snapshot()
 rows=[o for o in b['operations'] if o['game']=='tavern' and o['kind'].startswith('rl-')]
 def tally(settled):
  selected=[o for o in rows if (o['status']=='settled')==settled]
  return dict(gpu_seconds=sum(o['gpu_charged'] if settled else o['gpu_hold'] for o in selected),resource_usd=sum(o['usd_charged'] if settled else o['usd_hold'] for o in selected if o['bucket']=='resource'),model_usd=sum(o['usd_charged'] if settled else o['usd_hold'] for o in selected if o['bucket']=='model'))
 return dict(settled=tally(True),held=tally(False),role_subcaps=dict(gpu_seconds=12600,resource_usd=32,model_usd=2.5),operations=[{k:o[k] for k in ['id','kind','status','provider_id','gpu_hold','usd_hold','gpu_charged','usd_charged']} for o in rows])

def strip_assessment(path):
 result=json.loads(path.read_text());return {name:dict(summary=r['summary'],per_profile=r['per_profile'],per_seat=r['per_seat'],counts=r['counts'],per_opponent=r.get('per_opponent',{}),identity_sha256=r['identity_sha256']) for name,r in result.items()}

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT/'evidence/reference-readiness-20261109-0251Z');a=p.parse_args()
 print(json.dumps(usage(a.root),indent=2))
