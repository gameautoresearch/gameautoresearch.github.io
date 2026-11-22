"""Read and reconcile this campaign only; optional priority/full durable collection."""
import argparse,json,os,sys,time,hashlib
from pathlib import Path
os.environ.setdefault('MODAL_DISABLE_API_PROXY','1')
from .common import ROOT,GAME,write,sha
CAMPAIGN=ROOT/'evidence/strong-bots-20261109-0516Z'
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--collect',action='store_true');p.add_argument('--critical-only',action='store_true');a=p.parse_args();root=CAMPAIGN/'tavern/rl'/a.name;launch=json.loads((root/'provider.json').read_text());sys.path.insert(0,str(GAME/'autoresearch'))
 from provider import configure_sandbox_waits,allocation_info,read_rpc
 from tools.open_credit_campaign import OpenCreditBudget
 import modal
 configure_sandbox_waits();info=allocation_info(launch['app_id'],launch['sandbox_id']);volume=modal.Volume.from_name(launch['volume_name'],environment_name='staging');result=dict(provider=info,checked_unix=time.time())
 def read(n):return b''.join(volume.read_file('/'+a.name+'/'+n))
 for n in ['lifecycle.json','training/progress.json','training/result.json','export/summary.json','proof/validation.json','control/allocation.json']:
  try:result[n]=json.loads(read(n))
  except Exception as e:result[n]=dict(unavailable=type(e).__name__)
 if info and info.get('finished_at') and info.get('started_at'):
  b=OpenCreditBudget('tavern',Path(CAMPAIGN/'campaign.sqlite3'));operation=launch['reservations']['gpu'];ops={o['id']:o for o in b.snapshot()['operations']}
  if ops[operation]['status']!='settled':b.settle_worker(operation,info)
  write(root/'allocation.json',dict(provider=info,allocated_GPU_seconds=info['finished_at']-info['started_at'],unknown_CPU_image_storage_holds_retained=True))
 elif info:
  try:
   sandbox=modal.Sandbox.from_id(launch['sandbox_id']);raw=read_rpc(lambda:sandbox.filesystem.read_text('/retained/'+a.name+'/training/progress.json'));result['live_progress']=json.loads(raw);sandbox.detach()
  except Exception as exc:result['live_read_error']=type(exc).__name__
 if a.collect or a.critical_only:
  if not info or not info.get('finished_at'):raise ValueError('Allocation still active')
  dest=root/'collected';dest.mkdir(exist_ok=True)
  def collect(n,optional=False):
   target=dest/n
   if target.exists():return
   target.parent.mkdir(parents=True,exist_ok=True);temp=target.with_suffix(target.suffix+'.download')
   try:
    with temp.open('wb') as f:
     for chunk in volume.read_file('/'+a.name+'/'+n):f.write(chunk)
    temp.replace(target)
   except Exception:
    temp.unlink(missing_ok=True)
    if not optional:raise
  critical=['lifecycle.json','config.json','proof/validation.json','proof/import-parity.json','export/policy.py','export/learner.py','export/train.py','export/identity.json','export/summary.json','export/weights.npz','training/result.json','training/manifest.json','training/history.jsonl','training/latest.pt','training/initial.pt']
  for n in critical:collect(n,optional=True)
  if (dest/'export/summary.json').exists():
   s=json.loads((dest/'export/summary.json').read_text());assert sha(dest/'export/weights.npz')==s['weights_sha256'] and sha(dest/'export/identity.json')==s['identity_sha256']
  if (dest/'training/result.json').exists():assert sha(dest/'training/latest.pt')==json.loads((dest/'training/result.json').read_text())['checkpoint_sha256']
  write(root/'critical-collection.json',dict(complete=True,available=[n for n in critical if (dest/n).exists()]));print('Critical collection complete',flush=True)
  if a.collect:
   prefix=a.name+'/'
   for entry in volume.iterdir('/'+a.name,recursive=True):
    if entry.type==1:
     n=entry.path.lstrip('/')
     if not n.startswith(prefix):raise ValueError('Volume prefix mismatch')
     collect(n[len(prefix):])
   write(root/'collection-complete.json',dict(complete=True,files=len(list(dest.rglob('*')))))
 write(root/'last-status.json',result);row=result.get('live_progress',result.get('training/progress.json',{}));print(json.dumps(dict(provider=info,update=row.get('update'),samples=row.get('lineage_samples'),learning=row.get('learning'),counts=row.get('counts'),lifecycle=result['lifecycle.json'],proof=result['proof/validation.json'],export=result['export/summary.json']),indent=2))
