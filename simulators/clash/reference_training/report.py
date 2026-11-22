"""Read-only campaign report inputs. Never creates a worker or a training seed."""
from pathlib import Path
import argparse,json
from .launch import CAMPAIGN,LEDGER,role_usage
from simulators.clash.autoresearch.common import dump,sha
from tools.overnight_budget import CampaignBudget

def collect():
 root=CAMPAIGN/'clash/rl';b=CampaignBudget('clash',path=LEDGER);snap=b.snapshot();attempts=[]
 for path in sorted(root.iterdir()):
  if not path.is_dir() or not (path/'reservations.json').exists():continue
  item={'name':path.name,'reservations':json.loads((path/'reservations.json').read_text())}
  for file in ('provider.json','settlement.json','prelaunch-recovery.json','no-gpu-created.json'):
   if (path/file).exists():item[file]=json.loads((path/file).read_text())
  out=path/'collected'
  for file in ('result.json','output/result.json','case-plan.json','lifecycle.json','allocation.json'):
   if (out/file).exists():item['collected/'+file]=json.loads((out/file).read_text())
  for file in ('final-development/result.json','early-development/result.json'):
   if (path/file).exists():
    result=json.loads((path/file).read_text());item[file]={k:v for k,v in result.items() if k not in ('rows','latencies','results','entrant_probes')}
  if (path/'source.tar.gz').exists():item['source_archive_sha256']=sha(path/'source.tar.gz')
  attempts.append(item)
 rows=[r for r in snap['operations'] if r['game']=='clash' and r['kind'].startswith('clash/rl/')]
 settled=[r for r in rows if r['status']=='settled'];held=[r for r in rows if r['status']!='settled']
 value={'campaign_id':snap['campaign_id'],'explicit_ledger':str(LEDGER),'attempts':attempts,'role_use':role_usage(b),'settled':{'gpu_seconds':sum(r['gpu_charged'] for r in settled),'resources_usd':sum(r['usd_charged'] for r in settled)},'held':{'gpu_seconds':sum(r['gpu_hold'] for r in held),'resources_usd':sum(r['usd_hold'] for r in held)},'resource_operations':rows,'external_model_calls':0,'model_cost_usd':0,'invoice_verified':False,'no_allowance_resets':True}
 dump(root/'report-inputs.json',value);return value

def main():
 p=argparse.ArgumentParser();p.parse_args();r=collect();print(json.dumps({k:v for k,v in r.items() if k not in ('attempts','resource_operations')},indent=2))
if __name__=='__main__':main()
