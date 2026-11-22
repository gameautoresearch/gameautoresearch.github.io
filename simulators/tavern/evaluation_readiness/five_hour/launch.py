"""One bounded detached CPU worker+receipt observer; no training or researcher calls."""
import argparse,hashlib,json,os,shutil,sys,time,subprocess
from datetime import datetime
from pathlib import Path
os.environ.setdefault('MODAL_DISABLE_API_PROXY','1')
from .budget import ROOT,CAMPAIGN,LEDGER,Budget
from .runtime import public_image,RECEIPT
HERE=Path(__file__).resolve().parents[1];VOLUME='tavern-evaluator-strong-bots-20261109-0516z'
sys.path.insert(0,str(HERE.parent/'autoresearch'))
from contract import write,sha,binding
from provider import configure_sandbox_waits,image_for


def observe(app_id,sandbox_id,volume_name,attempt,launched,cap,absolute_stop):
 import json,time,sys,modal
 from pathlib import Path
 sys.path.insert(0,'/opt/task/simulators/tavern/autoresearch')
 from provider import configure_sandbox_waits,allocation_info
 configure_sandbox_waits();volume=modal.Volume.from_name(volume_name,environment_name='staging');out=Path('/retained')/attempt/'control';out.mkdir(parents=True,exist_ok=True)
 def save(name,obj):(out/name).write_text(json.dumps(obj,indent=2)+'\n');volume.commit()
 sb=modal.Sandbox.from_id(sandbox_id);info=None
 save('observer-started.json',dict(app_id=app_id,sandbox_id=sandbox_id,can_create_worker=False,allocation_deadline_basis='authoritative provider started_at; conservative fallback only before metadata'))
 try:
  while time.time()<absolute_stop-10:
   info=allocation_info(app_id,sandbox_id);save('provider-status.json',dict(provider=info,checked_unix=time.time()))
   if info and info.get('finished_at'):break
   start=info.get('started_at') if info else None
   deadline=min(absolute_stop-10,(start+cap-10) if start else launched+cap+120)
   if time.time()>deadline:sb.terminate(wait=True)
   time.sleep(5)
 finally:
  if not info or not info.get('finished_at'):
   sb.terminate(wait=True)
   for _ in range(10):
    info=allocation_info(app_id,sandbox_id)
    if info and info.get('finished_at'):break
    time.sleep(1)
 if info and info.get('finished_at'):save('allocation.json',dict(status='settled',provider=info,allocated_seconds=info['finished_at']-info['started_at'],within_cap=info['finished_at']-info['started_at']<=cap))
 else:save('allocation.json',dict(status='unresolved_hold_retained',provider=info))
 return info


def main():
 import modal
 p=argparse.ArgumentParser();p.add_argument('--attempt',required=True);p.add_argument('--candidate',required=True);p.add_argument('--cap',type=int,default=600);p.add_argument('--proof',action='store_true');p.add_argument('--registry',type=Path,required=True);p.add_argument('--panel',type=Path,required=True);p.add_argument('--purpose',choices=['development','qualification','final'],default='development');p.add_argument('--execute',action='store_true');a=p.parse_args()
 if not a.execute:print(json.dumps(dict(plan_only=True,ledger=str(LEDGER),CPU_only=True,worker_cap=a.cap,candidate=a.candidate)));return
 if not a.attempt.replace('-','').isalnum() or not 300<=a.cap<=1800:raise ValueError('Unique finite CPU attempt required')
 safe_name='tavern-eval-'+hashlib.sha256(a.attempt.encode()).hexdigest()[:24]
 from modal._utils.name_utils import check_object_name
 check_object_name(safe_name,'Sandbox')
 b=Budget();counterpart=json.loads((CAMPAIGN/'tavern/coordination/rl.json').read_text())
 absolute_stop=datetime.fromisoformat(b.auth['worker_stop_utc'].replace('Z','+00:00')).timestamp()
 phase_stop=datetime.fromisoformat(b.auth['selection_close_utc'].replace('Z','+00:00')).timestamp() if a.purpose=='development' else absolute_stop
 # Final full288-cell assessments keep a longer CPU lease while still reserving
 # explicit bounded observer/startup/teardown time before the absolute stop.
 final=a.purpose=='final';margin=300 if final else 600;watch_extra=180 if final else 480;function_extra=120 if final else 300;startup_extra=60 if final else 120
 a.cap=min(a.cap,int(phase_stop-time.time()-margin))
 if a.cap<300:raise ValueError('Insufficient time for a bounded worker/observer before absolute deadline')
 if counterpart['interface']['native_fingerprint']!='217577b83fc4525e' or counterpart['interface']['admission']!='admission-v2':raise ValueError('Counterpart interface pin not matched')
 expected=json.loads((a.registry/'manifest.json').read_text())
 if any(sha(a.registry/f)!=h for f,h in expected.items()):raise ValueError('Public registry mutated')
 runtime_check=subprocess.run([str(ROOT/'.venv-tavern/bin/python'),'-c',"import json,sys;sys.path.insert(0,'simulators/tavern/autoresearch');from contract import binding;print(json.dumps(binding()))"],cwd=ROOT,capture_output=True,text=True,check=True,timeout=20)
 if json.loads((a.registry/'registry.json').read_text())['binding']!=json.loads(runtime_check.stdout):raise ValueError('Runtime binding drifted')
 out=CAMPAIGN/'tavern/evaluator'/a.attempt;out.mkdir(parents=True,exist_ok=False);source=out/'source/simulators/tavern';source.mkdir(parents=True)
 shutil.copytree(HERE,source/'evaluation_readiness',ignore=shutil.ignore_patterns('__pycache__'))
 # Snapshot the exact qualified host/runtime closure, not an old release's omitted tests.
 for f in HERE.parent.glob('*.py'):shutil.copy2(f,source/f.name)
 helper=source/'autoresearch'
 shutil.copytree(HERE.parent/'autoresearch',helper,ignore=shutil.ignore_patterns('__pycache__','.pytest_cache','.cache','*.pyc'))
 shutil.copytree(HERE.parent/'recipes',source/'recipes',ignore=shutil.ignore_patterns('__pycache__'))
 tests=source/'tests';tests.mkdir()
 for name in ('test_evaluation_readiness.py','test_admission_credit.py','test_contract.py'):shutil.copy2(HERE.parent/'tests'/name,tests/name)
 # The complete autoresearch tree includes its tests, public recipes and metadata.
 required=['tests/test_evaluation_readiness.py','tests/test_admission_credit.py','tests/test_contract.py','autoresearch/tests/test_harness.py','autoresearch/policy_worker.py','autoresearch/benchmark.json','autoresearch/recipes/smoke.json','autoresearch/recipes/starter.json','recipes/starter-local.json']
 if any(not (source/f).is_file() for f in required):raise ValueError('Incomplete immutable qualification/runtime source closure')
 assets=out/'assets';assets.mkdir()
 for rel,h in expected.items():
  dest=assets/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(a.registry/rel,dest)
  if sha(dest)!=h:raise ValueError('Staged inference artifact drift')
 shutil.copy2(a.registry/'manifest.json',assets/'manifest.json')
 tools_dir=out/'source/tools';tools_dir.mkdir(parents=True)
 for name in ('open_credit_campaign.py','overnight_budget.py'):shutil.copy2(ROOT/'tools'/name,tools_dir/name)
 write(out/'source-files.json',{str(f.relative_to(out/'source')):sha(f) for f in (out/'source').rglob('*') if f.is_file()});write(out/'counterpart-before-paid.json',counterpart);shutil.copy2(a.panel,out/'panel.json')
 configure_sandbox_waits();build=b.reserve('image-'+a.attempt,900,sandbox=False,extra=.25,purpose='build');write(out/'build-reservation.json',dict(operation=build,unknown_build_and_storage_hold_retained=True))
 app=modal.App(safe_name);volume=modal.Volume.from_name(VOLUME,environment_name='staging',create_if_missing=True,version=2)
 base=ROOT/'evidence/overnight-20261108-1533Z/tavern/release-admission-v2'
 foundation=public_image() if RECEIPT.exists() else image_for(base)
 image=(foundation.add_local_dir(out/'source','/opt/task',copy=True).add_local_dir(assets,'/opt/evaluator-assets',copy=True).add_local_file(a.panel,'/opt/evaluator-panel.json',copy=True).env({'PYTHONPATH':'/opt/task','TAVERNGYM_WORKERS':'pooled','TAVERNGYM_SCHEDULER':'dynamic'}).workdir('/opt/task'))
 observer=app.function(image=image,cpu=(.25,.25),memory=(1024,1024),volumes={'/retained':volume},timeout=a.cap+function_extra,startup_timeout=startup_extra,retries=0,max_containers=1,min_containers=0,buffer_containers=0,single_use_containers=True,serialized=True,include_source=False,name='observe_strong_evaluator_allocation')(observe)
 with modal.enable_output(),app.run(detach=True,environment_name='staging'):
  b.attach(build,app.app_id);image.build(app);write(out/'image-build.json',dict(app_id=app.app_id,image_id=image.object_id,public_runtime_receipt_sha256=sha(RECEIPT) if RECEIPT.exists() else None,hold_retained=True))
  worker=b.reserve('cpu-'+a.attempt,a.cap,purpose=a.purpose);watch=b.reserve('observer-'+a.attempt,a.cap+watch_extra,cpu=.25,memory=1,sandbox=False,purpose='observer');write(out/'reservations.json',dict(worker=worker,observer=watch,build=build))
  launched=time.time();cmd=['python','-u','-m','simulators.tavern.evaluation_readiness.five_hour.guest','--root','/retained/'+a.attempt,'--candidate',a.candidate,'--seconds',str(a.cap-70),'--absolute-stop',str(datetime.fromisoformat(b.auth['worker_stop_utc'].replace('Z','+00:00')).timestamp())]
  if a.proof:cmd+=['--proof']
  sb=modal.Sandbox.create(*cmd,app=app,image=image,cpu=(4,4),memory=(16384,16384),timeout=a.cap-30,block_network=True,secrets=[],volumes={'/retained':volume},include_oidc_identity_token=False,name=safe_name,tags={'campaign':'strong-bots-20261109-0516Z','role':'evaluator','attempt':a.attempt});b.attach(worker,sb.object_id)
  receipt=dict(app_id=app.app_id,sandbox_id=sb.object_id,volume_name=VOLUME,attempt=a.attempt,candidate=a.candidate,launched_unix=launched,cap=a.cap,purpose=a.purpose,reservations=dict(worker=worker,observer=watch,build=build),automatic_retries=0,panel_sha256=sha(a.panel),registry_manifest_sha256=sha(a.registry/'manifest.json'),runtime_protocol='private-paths-terminal-v2-all-lobbies',deadline_policy=dict(absolute_stop_unix=absolute_stop,prelaunch_margin_seconds=margin,observer_reserved_seconds=a.cap+watch_extra,observer_timeout_seconds=a.cap+function_extra,observer_startup_seconds=startup_extra))
  write(out/'provider.json',receipt);call=observer.spawn(app.app_id,sb.object_id,VOLUME,a.attempt,launched,a.cap,datetime.fromisoformat(b.auth['worker_stop_utc'].replace('Z','+00:00')).timestamp());b.attach(watch,call.object_id);receipt['observer_call_id']=call.object_id;write(out/'provider.json',receipt);sb.detach();print(json.dumps(receipt),flush=True)
if __name__=='__main__':main()
