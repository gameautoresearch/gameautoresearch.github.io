"""New uncapped/time-bounded campaign, explicit reservations, immutable stages."""
import argparse,json,os,shutil,sys,time
from pathlib import Path
from datetime import datetime
os.environ.setdefault('MODAL_DISABLE_API_PROXY','1')
import modal
from .common import ROOT,GAME,HERE,sha,write
CAMPAIGN=ROOT/'evidence/strong-bots-20261109-0516Z'
LEDGER=CAMPAIGN/'campaign.sqlite3'
VOLUME='tavern-strong-bots-20261109-0516z'

def observe(app_id,sandbox_id,volume_name,name,launched,cap,absolute_stop):
 import json,time,sys,os,threading
 if time.time()>=absolute_stop-5:return
 # Independent of status RPC progress, stop this single-use observer before
 # the absolute selection deadline. No retries or successor are created.
 watchdog=threading.Timer(max(1,absolute_stop-time.time()-5),lambda:os._exit(78));watchdog.daemon=True;watchdog.start()
 from pathlib import Path
 import modal
 sys.path.insert(0,'/opt/task/simulators/tavern/autoresearch')
 from simulators.tavern.autoresearch.provider import configure_sandbox_waits,allocation_info
 configure_sandbox_waits();volume=modal.Volume.from_name(volume_name,environment_name='staging');root=Path('/retained')/name/'control';root.mkdir(parents=True,exist_ok=True)
 def save(n,x):(root/n).write_text(json.dumps(x,indent=2)+'\n');volume.commit()
 save('observer-started.json',dict(sandbox_id=sandbox_id,launched_unix=launched,absolute_stop_unix=absolute_stop,can_launch_GPU=False))
 sandbox=modal.Sandbox.from_id(sandbox_id);info=None
 while time.time()<min(launched+cap+175,absolute_stop+10):
  info=allocation_info(app_id,sandbox_id);save('provider-status.json',dict(provider=info,checked_unix=time.time()))
  if info and info.get('finished_at'):break
  allocated_start=info.get('started_at') if info else None
  cutoff=min((allocated_start or launched)+cap-30,absolute_stop-30)
  if time.time()>=cutoff:sandbox.terminate(wait=True)
  time.sleep(5)
 if not info or not info.get('started_at') or not info.get('finished_at'):
  save('allocation.json',dict(status='unresolved_hold_retained',provider=info));watchdog.cancel();return
 save('allocation.json',dict(status='settled',provider=info,allocated_seconds=info['finished_at']-info['started_at'],within_cap=info['finished_at']-info['started_at']<=cap,active_owned_GPU_workers=0))
 watchdog.cancel()
 return info

def main():
 p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--arm',choices=['plain','curriculum'],required=True);p.add_argument('--seconds',type=int,default=3400);p.add_argument('--proof',action='store_true');p.add_argument('--resume',type=str);p.add_argument('--source-release',type=Path);p.add_argument('--inputs',type=Path);p.add_argument('--recipe',choices=['plain.json','curriculum.json','curriculum-quarter.json','curriculum-anchor.json']);a=p.parse_args()
 if (CAMPAIGN/'tavern/rl/selection-closed.json').exists():raise ValueError('This new campaign selection is closed')
 if a.seconds> (600 if a.proof else 3400) or a.seconds<180:raise ValueError('Finite stage and proof bounds')
 if a.resume and not a.source_release:raise ValueError('Exact source release required for resume')
 counterpart=json.loads((CAMPAIGN/'tavern/coordination/evaluator.json').read_text())
 if counterpart['interface']['native_fingerprint']!='217577b83fc4525e':raise ValueError('Evaluator/native handshake differs')
 from tools.open_credit_campaign import OpenCreditBudget
 budget=OpenCreditBudget('tavern',Path(LEDGER));selection=datetime.fromisoformat(budget.auth['selection_close_utc'].replace('Z','+00:00')).timestamp();stop=datetime.fromisoformat(budget.auth['worker_stop_utc'].replace('Z','+00:00')).timestamp()
 if not a.proof:
  gates=list((CAMPAIGN/'tavern/rl').glob('*/collected/proof/validation.json'))
  if not any(json.loads(g.read_text()).get('all_passed') and (g.parents[1]/'export/summary.json').exists() for g in gates):raise ValueError('First new-source CUDA proof/export not yet admitted')
 inputs=(a.inputs or CAMPAIGN/'tavern/rl/inputs-v1').resolve();expected=json.loads((inputs/'manifest.json').read_text())
 if any(sha(inputs/f)!=h for f,h in expected.items()):raise ValueError('Frozen warmstart/script inputs mutated')
 out=CAMPAIGN/'tavern/rl'/a.name;out.mkdir(parents=True,exist_ok=False);source=out/'source'
 if a.source_release:
  oldfiles=json.loads((a.source_release.parent/'source-files.json').read_text())
  if any(sha(a.source_release/f)!=h for f,h in oldfiles.items()):raise ValueError('Immutable runtime release mutated')
  shutil.copytree(a.source_release,source)
 else:
  game=source/'simulators/tavern';game.mkdir(parents=True)
  for f in GAME.glob('*.py'):shutil.copy2(f,game/f.name)
  shutil.copytree(GAME/'reference_training',game/'reference_training',ignore=shutil.ignore_patterns('__pycache__','.pytest_cache'))
  helpers=game/'autoresearch';helpers.mkdir()
  for f in ['__init__.py','provider.py','contract.py','benchmark.json','policy_worker.py']:shutil.copy2(GAME/'autoresearch'/f,helpers/f)
  shutil.copytree(GAME/'autoresearch/recipes',helpers/'recipes')
  tools=source/'tools';tools.mkdir()
  for f in ['open_credit_campaign.py','overnight_budget.py']:shutil.copy2(ROOT/'tools'/f,tools/f)
  if (ROOT/'tools/__init__.py').exists():shutil.copy2(ROOT/'tools/__init__.py',tools/'__init__.py')
 write(out/'source-files.json',{str(p.relative_to(source)):sha(p) for p in source.rglob('*') if p.is_file()})
 write(out/'input-binding.json',dict(path=str(inputs),manifest_sha256=sha(inputs/'manifest.json')))
 recipe=a.recipe or a.arm+'.json';recipe_path=source/'simulators/tavern/reference_training/strong'/recipe;config=json.loads(recipe_path.read_text())
 if config['mode']!=a.arm:raise ValueError('Declared arm and actual recipe mode differ')
 if not a.proof:
  qualified=False
  active_strong={str(p.relative_to(source)) for p in (source/'simulators/tavern/reference_training/strong').glob('*.py')}
  for g in gates:
   export=g.parents[1]/'export';identity=export/'identity.json'
   if not json.loads(g.read_text()).get('all_passed') or not identity.exists():continue
   ident=json.loads(identity.read_text());pin=ident['reference_lineage']['reference_binding'];files=pin['files']
   tested_strong={f for f in files if f.startswith('simulators/tavern/reference_training/strong/')}
   if active_strong!=tested_strong or ident['config']['network']!=config['network']:continue
   if any(not (ROOT/f if f.startswith('vendor/') else source/f).is_file() or sha(ROOT/f if f.startswith('vendor/') else source/f)!=h for f,h in files.items()):continue
   if (inputs/'fork-request.json').exists():
    parity=g.parent/'import-parity.json'
    if not parity.exists() or json.loads(parity.read_text()).get('kind')!='explicit-strong-parent-warmstart':continue
   qualified=True;break
  if not qualified:raise ValueError('Exact executable release/architecture/import branch needs a<=600s CUDA proof before scaling')
 write(out/'counterpart-before-paid.json',counterpart)
 host=out/'host-controller';host.mkdir()
 for f in [HERE/'launch.py',HERE/'common.py',ROOT/'tools/open_credit_campaign.py',ROOT/'tools/overnight_budget.py',GAME/'autoresearch/provider.py']:
  target=host/f.name;shutil.copy2(f,target)
 write(host/'manifest.json',{f.name:sha(f) for f in host.glob('*.py')})
 sys.path.insert(0,str(GAME/'autoresearch'))
 from provider import configure_sandbox_waits,image_for
 configure_sandbox_waits();build=budget.reserve_worker('rl:image:'+a.name,900,None,4,16,sandbox=False,extra_usd=.75,purpose='build');write(out/'build-reservation.json',dict(operation=build,unknown_image_storage_hold_retained=True))
 app=modal.App('tavern-strong-'+a.name+'-20261109');volume=modal.Volume.from_name(VOLUME,environment_name='staging',create_if_missing=True,version=2)
 base=ROOT/'evidence/overnight-20261108-1533Z/tavern/release-admission-v2'
 image=(image_for(base).add_local_dir(source,'/opt/task',copy=True).add_local_dir(source,'/reference-source',copy=True).add_local_dir(inputs,'/inputs',copy=True).env({'PYTHONPATH':'/opt/task','PYTHONDONTWRITEBYTECODE':'1','TAVERNGYM_WORKERS':'pooled','TAVERNGYM_SCHEDULER':'dynamic'}).workdir('/opt/task'))
 observer_seconds=min(3600,a.seconds+200)
 observer=app.function(image=image,cpu=(.25,.25),memory=(1024,1024),volumes={'/retained':volume},timeout=observer_seconds-10,startup_timeout=120,retries=0,max_containers=1,min_containers=0,buffer_containers=0,single_use_containers=True,serialized=True,include_source=False,name='observe_allocation')(observe)
 with modal.enable_output(),app.run(detach=True,environment_name='staging'):
  budget.attach(build,app.app_id);image.build(app);write(out/'image-build.json',dict(app_id=app.app_id,unpriced_hold_retained=True))
  cap=min(a.seconds,int(selection-time.time()-45))
  if cap<180:raise ValueError('Selection cutoff leaves no safe training stage')
  gpu=budget.reserve_worker('rl:GPU:'+a.name,cap,'L4',4,16,purpose='qualification' if a.proof else 'training')
  reserved_observer_seconds=min(observer_seconds,int(stop-time.time()-30))
  if reserved_observer_seconds<cap:raise ValueError('No safe observer interval remains')
  cpu=budget.reserve_worker('rl:observer:'+a.name,reserved_observer_seconds,None,.25,1,sandbox=False,purpose='observer')
  write(out/'reservations.json',dict(gpu=gpu,observer=cpu,build=build))
  launched=time.time();command=['python','-u','-m','simulators.tavern.reference_training.strong.autonomous','--root','/retained/'+a.name,'--config','/opt/task/simulators/tavern/reference_training/strong/'+recipe,'--cap',str(cap),'--selection-close',str(selection)]
  if a.proof:command+=['--proof']
  if a.resume:command+=['--resume','/retained/'+a.resume]
  sandbox=modal.Sandbox.create(*command,app=app,image=image,gpu='L4',cpu=(4,4),memory=(16384,16384),timeout=cap-25,block_network=True,secrets=[],volumes={'/retained':volume},environment_name='staging',include_oidc_identity_token=False,name='tavern-strong-'+a.name,tags={'campaign':'strong-bots-20261109-0516Z','role':'rl','stage':a.name,'arm':a.arm})
  budget.attach(gpu,sandbox.object_id);provider=dict(app_id=app.app_id,sandbox_id=sandbox.object_id,volume_name=VOLUME,name=a.name,arm=a.arm,recipe=recipe,recipe_sha256=sha(recipe_path),launched_unix=launched,cap=cap,selection_close_unix=selection,worker_stop_unix=stop,reservations=dict(gpu=gpu,observer=cpu,build=build),source_manifest_sha256=sha(out/'source-files.json'),input_manifest_sha256=sha(inputs/'manifest.json'),automatic_GPU_retries=0)
  write(out/'provider.json',provider);call=observer.spawn(app.app_id,sandbox.object_id,VOLUME,a.name,launched,cap,min(stop,selection));budget.attach(cpu,call.object_id);provider['observer_call_id']=call.object_id;write(out/'provider.json',provider);sandbox.detach();print(json.dumps(provider,indent=2),flush=True)
if __name__=='__main__':main()
