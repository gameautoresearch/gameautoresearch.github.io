"""Read/reconcile/collect exactly one evaluator attempt; never launches a worker."""
import argparse,json,os,sys
from pathlib import Path
os.environ.setdefault('MODAL_DISABLE_API_PROXY','1')
from .budget import ROOT,CAMPAIGN,Budget,LEDGER
sys.path.insert(0,str(ROOT/'simulators/tavern/autoresearch'))
from provider import configure_sandbox_waits,allocation_info
from contract import write

def main():
 import modal
 p=argparse.ArgumentParser();p.add_argument('--attempt',required=True);p.add_argument('--collect',action='store_true');p.add_argument('--collect-summary',action='store_true',help='Collect outcomes/receipts, retaining bulk native traces privately on the Volume');a=p.parse_args();out=CAMPAIGN/'tavern/evaluator'/a.attempt;launch=json.loads((out/'provider.json').read_text());configure_sandbox_waits();info=allocation_info(launch['app_id'],launch['sandbox_id']);volume=modal.Volume.from_name(launch['volume_name'],environment_name='staging');result=dict(provider=info)
 for f in ['lifecycle.json','qualification.json','qualification-privacy.json','qualification-terminal-replay.json','screen/progress.json','screen/result.json','control/allocation.json','failure.txt']:
  try:
   raw=b''.join(volume.read_file('/'+a.attempt+'/'+f)).decode();result[f]=json.loads(raw) if f.endswith('.json') else raw[-3000:]
  except Exception as exc:result[f]=dict(unavailable=type(exc).__name__)
 if info and info.get('started_at') and info.get('finished_at'):
  b=Budget();op=launch['reservations']['worker'];rows={r['id']:r for r in b.snapshot()['operations']}
  if rows[op]['status']!='settled':b.settle_worker(op,info)
  write(out/'allocation.json',dict(provider=info,allocated_seconds=info['finished_at']-info['started_at'],observer_and_build_holds_retained=True))
 write(out/'last-status.json',result)
 if a.collect_summary:
  if not info or not info.get('finished_at'):raise ValueError('Worker still active/unknown; do not collect partial state')
  dest=out/'summary';dest.mkdir(exist_ok=False)
  for name in ['lifecycle.json','qualification.json','qualification-privacy.json','qualification-terminal-replay.json','qualification.xml','qualification.log','screen/episodes.json','screen/result.json','screen/progress.json','control/allocation.json','control/provider-status.json','control/observer-started.json','failure.txt']:
   try:raw=b''.join(volume.read_file('/'+a.attempt+'/'+name))
   except FileNotFoundError:continue
   target=dest/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
  write(dest/'remote-traces.json',dict(volume=launch['volume_name'],prefix='/'+a.attempt+'/screen/traces',retained_private=True,downloaded=False))
  result['summary_collected']=str(dest)
 if a.collect:
  if not info or not info.get('finished_at'):raise ValueError('Worker still active/unknown; do not duplicate or collect partial state')
  dest=out/'collected';dest.mkdir(exist_ok=False);prefix=a.attempt+'/'
  for entry in volume.iterdir('/'+a.attempt,recursive=True):
   if entry.type==1:
    name=entry.path.lstrip('/')
    if not name.startswith(prefix):raise ValueError('Wrong volume prefix')
    target=dest/name[len(prefix):];target.parent.mkdir(parents=True,exist_ok=True)
    with target.open('wb') as f:
     for chunk in volume.read_file(entry.path):f.write(chunk)
  result['collected']=str(dest)
 write(out/'last-status.json',result)
 summary=dict(provider=info,lifecycle=result['lifecycle.json'],qualification=result['qualification.json'],progress=result['screen/progress.json'])
 if isinstance(result['screen/result.json'],dict):summary['screen']={k:result['screen/result.json'].get(k) for k in ('candidate_id','expected_cases','observed_cases','complete_valid_schedule','raw_score','failures')}
 print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
