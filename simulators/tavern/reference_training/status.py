"""Read/reconcile one owned allocation; this command never creates a GPU."""
import argparse,json,sys,os
os.environ.setdefault("MODAL_DISABLE_API_PROXY","1")
from pathlib import Path
from .bootstrap import ROOT,GAME,write
from .launch import CAMPAIGN,LEDGER

def main():
 import modal
 sys.path.insert(0,str(GAME/'autoresearch'))
 from provider import configure_sandbox_waits,allocation_info,read_rpc
 configure_sandbox_waits()
 p=argparse.ArgumentParser();p.add_argument('--attempt',required=True);p.add_argument('--collect',action='store_true');a=p.parse_args()
 root=CAMPAIGN/'tavern/rl'/a.attempt;launch=json.loads((root/'provider.json').read_text());info=allocation_info(launch['app_id'],launch['sandbox_id'])
 volume=modal.Volume.from_name(launch['volume_name'],environment_name='staging')
 def read(name):return b''.join(volume.read_file('/'+a.attempt+'/'+name)).decode()
 result=dict(provider=info)
 for name in ['lifecycle.json','training/progress.json','training/result.json','proof/validation.json','export/summary.json','control/allocation.json']:
  try:result[name]=json.loads(read(name))
  except Exception as exc:result[name]=dict(unavailable=type(exc).__name__)
 if info and not info.get('finished_at'):
  try:
   sb=modal.Sandbox.from_id(launch['sandbox_id']);result['log_tail']=read_rpc(lambda:sb.filesystem.read_text('/retained/'+a.attempt+'/trainer.log'))[-4000:];sb.detach()
  except Exception as exc:result['log_error']=str(exc)
 if info and info.get('started_at') and info.get('finished_at'):
  from tools.overnight_budget import CampaignBudget
  budget=CampaignBudget('tavern',path=LEDGER);operation=launch['reservations']['gpu'];rows={r['id']:r for r in budget.snapshot()['operations']}
  if rows[operation]['status']!='settled':budget.settle_worker(operation,info)
  write(root/'allocation.json',dict(provider=info,elapsed=info['finished_at']-info['started_at'],observer_and_build_holds_retained=True))
 if a.collect:
  if not info or not info.get('finished_at'):raise ValueError('Still active; wait before final collection')
  dest=root/'collected';dest.mkdir(exist_ok=False)
  prefix=a.attempt+'/'
  for e in volume.iterdir('/'+a.attempt,recursive=True):
   if e.type==1:
    name=e.path.lstrip('/')
    if not name.startswith(prefix):raise ValueError('Wrong volume prefix')
    target=dest/name[len(prefix):];target.parent.mkdir(parents=True,exist_ok=True)
    with target.open('wb') as f:
     for chunk in volume.read_file(e.path):f.write(chunk)
  result['collected']=str(dest)
 write(root/'last-status.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
