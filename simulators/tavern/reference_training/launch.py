"""Explicit-ledger, finite detached worker; no automatic training/retry controller."""
import argparse,json,os,shutil,sys,time
from pathlib import Path
os.environ.setdefault('MODAL_DISABLE_API_PROXY','1')
import modal
from .bootstrap import ROOT,GAME,HERE,write,sha
CAMPAIGN=ROOT/'evidence/reference-readiness-20261109-0251Z'
LEDGER=CAMPAIGN/'campaign.sqlite3'
VOLUME='tavern-reference-readiness-20261109-0251z'

def observe(app_id,sandbox_id,volume_name,attempt,launched,cap,worker_stop=None):
 import json,time,sys
 sys.path.insert(0,"/opt/task/simulators/tavern/autoresearch")
 from pathlib import Path
 import modal
 from simulators.tavern.autoresearch.provider import configure_sandbox_waits,allocation_info
 configure_sandbox_waits();volume=modal.Volume.from_name(volume_name,environment_name='staging');root=Path('/retained')/attempt/'control';root.mkdir(parents=True,exist_ok=True)
 def save(name,obj):
  (root/name).write_text(json.dumps(obj,indent=2)+'\n');volume.commit()
 save('observer-started.json',dict(app_id=app_id,sandbox_id=sandbox_id,no_gpu_creation_capability=True))
 sb=modal.Sandbox.from_id(sandbox_id);info=None
 absolute_stop=worker_stop if worker_stop is not None else launched+cap+90
 while time.time()<min(launched+cap+95,absolute_stop+5):
  info=allocation_info(app_id,sandbox_id)
  save('provider-status.json',dict(provider=info,checked_unix=time.time()))
  if info and info.get('finished_at'):break
  allocated_start=info.get('started_at') if info else None
  cutoff=min((allocated_start or launched)+cap-30,absolute_stop-30,launched+cap+85)
  if time.time()>cutoff:sb.terminate(wait=True)
  time.sleep(5)
 if not info or not info.get('finished_at'):
  save('allocation.json',dict(status='unresolved_hold_retained',provider=info));return
 save('allocation.json',dict(status='settled',provider=info,allocated_seconds=info['finished_at']-info['started_at'],within_cap=info['finished_at']-info['started_at']<=cap,active_owned_gpu_workers=0))
 return info

def main():
 p=argparse.ArgumentParser();p.add_argument('--attempt',required=True);p.add_argument('--cap',type=int,default=600);p.add_argument('--proof',action='store_true');p.add_argument('--resume',type=str);p.add_argument('--source-release',type=Path);a=p.parse_args()
 if (CAMPAIGN/'tavern/rl/selection-closed.json').exists():raise ValueError('Reference selection is closed; no further training in this baseline')
 if a.resume and not a.source_release:raise ValueError('Resume needs explicit immutable source release')
 if a.cap> (600 if a.proof else 1800) or a.cap<180:raise ValueError('Finite staged cap required')
 counterpart=json.loads((CAMPAIGN/'tavern/coordination/evaluator.json').read_text())
 if counterpart['interface']['native_fingerprint']!='217577b83fc4525e' or counterpart['interface']['admission']!='admission-v2':raise ValueError('No matching evaluator handshake')
 from tools.overnight_budget import CampaignBudget
 budget=CampaignBudget('tavern',path=LEDGER)
 own=[o for o in budget.snapshot()['operations'] if o['game']=='tavern' and o['kind'].startswith('rl-')]
 gpu=sum(o['gpu_charged'] if o['status']=='settled' else o['gpu_hold'] for o in own)
 usd=sum(o['usd_charged'] if o['status']=='settled' else o['usd_hold'] for o in own if o['bucket']=='resource')
 if gpu+a.cap>12600 or usd+2>32:raise ValueError('RL role subcap')
 if not a.proof:
  proofs=list((CAMPAIGN/'tavern/rl').glob('*/collected/proof/validation.json'))
  def admitted(q):
   exported=q.parents[1]/'export/summary.json'
   if not json.loads(q.read_text())['all_passed'] or not exported.exists():return False
   summary=json.loads(exported.read_text())
   return summary['finite_losses'] and summary['changed_weights'] and summary['updates']>0 and summary['parameters']<=2000000
  if not any(admitted(q) for q in proofs):raise ValueError('First CUDA proof/export not admitted')
 out=CAMPAIGN/'tavern/rl'/a.attempt;out.mkdir(parents=True,exist_ok=False)
 source=out/'source'
 if a.source_release:
  expected=json.loads((a.source_release.parent/'source-files.json').read_text())
  if any(sha(a.source_release/f)!=h for f,h in expected.items()):raise ValueError('Source release mutated')
  shutil.copytree(a.source_release,source)
 else:
  game=source/'simulators/tavern';game.mkdir(parents=True)
  for f in GAME.glob('*.py'):shutil.copy2(f,game/f.name)
  shutil.copytree(HERE,game/'reference_training',ignore=shutil.ignore_patterns('__pycache__','.pytest_cache'))
  helper=game/'autoresearch';helper.mkdir()
  for name in ['__init__.py','provider.py','contract.py','benchmark.json','diagnostic_trace.py']:
   shutil.copy2(GAME/'autoresearch'/name,helper/name)
  shutil.copytree(GAME/'autoresearch/recipes',helper/'recipes')
 write(out/'source-files.json',{str(f.relative_to(source)):sha(f) for f in source.rglob('*') if f.is_file()})
 write(out/'counterpart-before-paid.json',counterpart)
 sys.path.insert(0,str(GAME/'autoresearch'))
 from provider import configure_sandbox_waits,image_for,allocation_info
 configure_sandbox_waits()
 base=ROOT/'evidence/overnight-20261108-1533Z/tavern/release-admission-v2'
 buildop=budget.reserve_worker('rl-image-'+a.attempt,900,None,4,16,sandbox=False,extra_usd=.5)
 write(out/'build-reservation.json',dict(operation=buildop,unknown_image_charge_hold_retained=True))
 app=modal.App('tavern-reference-'+a.attempt+'-20261109')
 volume=modal.Volume.from_name(VOLUME,environment_name='staging',create_if_missing=True,version=2)
 image=(image_for(base).add_local_dir(source,'/opt/task',copy=True).add_local_dir(source,'/reference-source',copy=True)
        .env({'PYTHONPATH':'/opt/task','TAVERNGYM_WORKERS':'pooled','TAVERNGYM_SCHEDULER':'dynamic'}).workdir('/opt/task'))
 observer=app.function(image=image,cpu=(.25,.25),memory=(1024,1024),volumes={'/retained':volume},timeout=a.cap+100,startup_timeout=120,retries=0,max_containers=1,min_containers=0,buffer_containers=0,single_use_containers=True,serialized=True,include_source=False,name='observe_allocation')(observe)
 with modal.enable_output(),app.run(detach=True,environment_name='staging'):
  budget.attach(buildop,app.app_id);image.build(app)
  write(out/'image-build.json',dict(app_id=app.app_id,hold_retained=True))
  gpuop=budget.reserve_worker('rl-gpu-'+a.attempt,a.cap,'L4',4,16)
  cpuop=budget.reserve_worker('rl-observer-'+a.attempt,a.cap+120,None,.25,1,sandbox=False)
  write(out/'reservations.json',dict(gpu=gpuop,observer=cpuop,build=buildop))
  launched=time.time();command=['python','-u','-m','simulators.tavern.reference_training.autonomous','--root','/retained/'+a.attempt,'--seconds',str(a.cap-60)]
  if a.proof:command+=['--proof']
  if a.resume:command+=['--resume','/retained/'+a.resume]
  sb=modal.Sandbox.create(*command,app=app,image=image,gpu='L4',cpu=(4,4),memory=(16384,16384),timeout=a.cap-30,block_network=True,secrets=[],volumes={'/retained':volume},environment_name='staging',include_oidc_identity_token=False,name='tavern-reference-'+a.attempt,tags={'campaign':'reference-readiness-20261109-0251Z','role':'rl','attempt':a.attempt})
  budget.attach(gpuop,sb.object_id)
  receipt=dict(app_id=app.app_id,sandbox_id=sb.object_id,volume_name=VOLUME,attempt=a.attempt,launched_unix=launched,cap=a.cap,reservations=dict(gpu=gpuop,observer=cpuop,build=buildop),automatic_retries=0)
  write(out/'provider.json',receipt)
  from datetime import datetime
  worker_stop=datetime.fromisoformat(budget.auth['worker_stop_utc'].replace('Z','+00:00')).timestamp()
  call=observer.spawn(app.app_id,sb.object_id,VOLUME,a.attempt,launched,a.cap,worker_stop);budget.attach(cpuop,call.object_id);receipt['observer_call_id']=call.object_id;write(out/'provider.json',receipt)
  sb.detach();print(json.dumps(receipt,indent=2),flush=True)
if __name__=='__main__':main()
