"""Read-only local status, full conservative accounting and public results."""
import json,datetime
from pathlib import Path
from .provider import OWN,LEDGER
from tools.open_credit_campaign import OpenCreditBudget
from simulators.clash.autoresearch.common import dump

def status():
 b=OpenCreditBudget('clash',LEDGER);rows=[x for x in b.snapshot()['operations'] if x['game']=='clash' and x['kind'].startswith('clash/rl/')];done=[x for x in rows if x['status']=='settled'];held=[x for x in rows if x['status']!='settled']
 metrics={'gpu_seconds_settled':sum(x['gpu_charged'] for x in done),'gpu_seconds_held':sum(x['gpu_hold'] for x in held),'resource_usd_settled':sum(x['usd_charged'] for x in done),'resource_usd_held':sum(x['usd_hold'] for x in held),'model_api_calls':0,'model_api_usd':0,'dollar_and_aggregate_gpu_caps':None,'invoice_verified':False}
 public=[]
 for path in sorted((OWN/'development').glob('*/summary.json')):
  x=json.loads(path.read_text());public.append({'name':path.parent.name,**{k:x[k] for k in ('complete','mean_points','by_opponent','games','candidate')}})
 cpu=[]
 for path in sorted((OWN/'cpu-development').glob('*/settlement.json')):
  x=json.loads(path.read_text());cpu.append({'name':path.parent.name,'complete':x['complete'],'summaries':x['summaries']})
 providers=[{'attempt':str(path.parent.relative_to(OWN)),**json.loads(path.read_text())} for path in OWN.glob('**/provider.json') if '/source/' not in str(path)]
 value={'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'explicit_ledger':str(LEDGER),'accounting':metrics,'public_development':public,'cpu_public_development':cpu,'providers':providers,'resource_operations':rows,'selection_close_utc':b.auth['selection_close_utc'],'worker_stop_utc':b.auth['worker_stop_utc'],'deadline_utc':b.auth['deadline_utc']};dump(OWN/'status-snapshot.json',value);return value

if __name__=='__main__':
 v=status();print(json.dumps({k:v[k] for k in ('at_utc','accounting','public_development','cpu_public_development')},indent=2))
