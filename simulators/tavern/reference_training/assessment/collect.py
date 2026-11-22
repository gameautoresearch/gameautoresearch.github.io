"""Collect verified critical artifacts first, then every durable output. No new worker."""
import argparse,hashlib,json,os,sys,time
from pathlib import Path
os.environ.setdefault('MODAL_DISABLE_API_PROXY','1')
ROOT=Path(__file__).resolve().parents[4]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2)+'\n');t.replace(p)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--attempt',required=True);a=p.parse_args();campaign=ROOT/'evidence/reference-readiness-20261109-0251Z';root=campaign/'tavern/rl'/a.attempt;launch=json.loads((root/'provider.json').read_text())
 sys.path.insert(0,str(ROOT/'simulators/tavern/autoresearch'))
 from provider import configure_sandbox_waits,allocation_info
 from tools.overnight_budget import CampaignBudget
 import modal
 configure_sandbox_waits();info=allocation_info(launch['app_id'],launch['sandbox_id'])
 if not info or not info.get('started_at') or not info.get('finished_at'):raise ValueError('Wait for completed authoritative interval')
 budget=CampaignBudget('tavern',path=campaign/'campaign.sqlite3');op=launch['reservations']['gpu'];rows={r['id']:r for r in budget.snapshot()['operations']}
 if rows[op]['status']!='settled':budget.settle_worker(op,info)
 write(root/'allocation.json',dict(provider=info,elapsed=info['finished_at']-info['started_at'],observer_and_build_holds_retained=True))
 volume=modal.Volume.from_name(launch['volume_name'],environment_name='staging');dest=root/'collected';dest.mkdir(exist_ok=False);done=set()
 def collect(name,optional=False):
  target=dest/name;target.parent.mkdir(parents=True,exist_ok=True);temp=target.with_suffix(target.suffix+'.download')
  try:
   with temp.open('wb') as f:
    for chunk in volume.read_file('/'+a.attempt+'/'+name):f.write(chunk)
   temp.replace(target);done.add(name)
  except Exception:
   temp.unlink(missing_ok=True)
   if not optional:raise
 for name in ['lifecycle.json','recipe.json','training/result.json','training/latest.pt','training/initial.pt','training/history.jsonl','training/manifest.json','export/policy.py','export/learner.py','export/train.py','export/weights.npz','export/identity.json','export/summary.json']:
  collect(name)
 result=json.loads((dest/'training/result.json').read_text());summary=json.loads((dest/'export/summary.json').read_text())
 if sha(dest/'training/latest.pt')!=result['checkpoint_sha256'] or sha(dest/'export/weights.npz')!=summary['weights_sha256'] or sha(dest/'export/identity.json')!=summary['identity_sha256']:raise ValueError('Critical checkpoint/export hashes differ')
 write(root/'critical-artifacts-verified.json',dict(verified=True,provider=info,checkpoint_sha256=result['checkpoint_sha256'],final_update=result['update'],files={name:sha(dest/name) for name in sorted(done)}));print(json.dumps(dict(critical_artifacts_verified=True,final_update=result['update'],provider_seconds=info['finished_at']-info['started_at'])),flush=True)
 prefix=a.attempt+'/'
 for entry in volume.iterdir('/'+a.attempt,recursive=True):
  if entry.type==1:
   name=entry.path.lstrip('/')
   if not name.startswith(prefix):raise ValueError('Wrong Volume prefix')
   relative=name[len(prefix):]
   if relative not in done:collect(relative)
 write(root/'collection-complete.json',dict(complete=True,files=len(done),finished_unix=time.time()));print(json.dumps(dict(all_outputs_collected=True,files=len(done))),flush=True)
