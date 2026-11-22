"""Explicit-new-ledger reservations and finite detached reference allocations."""
from pathlib import Path
import argparse,io,json,os,shutil,tarfile,tempfile,time,shlex
os.environ.setdefault('MODAL_DISABLE_API_PROXY','1')
from simulators.clash.autoresearch.common import dump,sha,engine_binding,ROOT,GAME
CAMPAIGN=ROOT/'evidence/reference-readiness-20261109-0251Z'
LEDGER=CAMPAIGN/'campaign.sqlite3'


def observe(app_id,sandbox_id,volume_name,seconds,launched):
 import modal
 from simulators.generals.autoresearch.provider import configure_sandbox_waits,allocation_info
 configure_sandbox_waits();volume=modal.Volume.from_name(volume_name,environment_name='staging');root=Path('/retained');receipt=None
 while time.time()<launched+seconds+45:
  receipt=allocation_info(app_id,sandbox_id)
  if receipt and receipt.get('finished_at'):break
  time.sleep(5)
 if not receipt or not receipt.get('finished_at'):
  modal.Sandbox.from_id(sandbox_id).terminate(wait=True)
  for _ in range(20):
   receipt=allocation_info(app_id,sandbox_id)
   if receipt and receipt.get('finished_at'):break
   time.sleep(.5)
 volume.reload();dump(root/'allocation.json',{'provider':receipt,'resolved':bool(receipt and receipt.get('finished_at')),'automatic_retries':0});volume.commit()
 return receipt


def snapshot(target):
 from harness.games.clash.provider import snapshot as old_snapshot
 archive=old_snapshot(target/'original');folder=target/'source';folder.mkdir(parents=True)
 with tarfile.open(archive) as t:t.extractall(folder,filter='data')
 dest=folder/'simulators/clash/reference_training'
 shutil.copytree(GAME/'reference_training',dest,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','.pytest_cache','tests'))
 helpers=folder/'simulators/generals/autoresearch';helpers.mkdir(parents=True)
 for n in ('__init__.py','provider.py','transport.py','contract.py','bundle.py'):shutil.copyfile(ROOT/'simulators/generals/autoresearch'/n,helpers/n)
 final=target/'source.tar.gz'
 with tarfile.open(final,'w:gz') as t:
  for p in sorted(folder.rglob('*')):
   if p.is_file():t.add(p,arcname=p.relative_to(folder))
 final.chmod(0o444);dump(target/'source-files.json',{str(p.relative_to(folder)):sha(p) for p in folder.rglob('*') if p.is_file()})
 return final


def role_usage(b):
 rows=[r for r in b.snapshot()['operations'] if r['game']=='clash' and r['kind'].startswith('clash/rl/')]
 return {'gpu_seconds':sum(r['gpu_charged'] if r['status']=='settled' else r['gpu_hold'] for r in rows),'resource_usd':sum(r['usd_charged'] if r['status']=='settled' else r['usd_hold'] for r in rows)}


def launch(name,seconds=570,proof=False,resume=None,paired_seed=None,variant="v2"):
 if (CAMPAIGN/'clash/rl/selection-close.json').exists():raise ValueError('Selection is closed; no further training')
 import modal
 from tools.overnight_budget import CampaignBudget
 from harness.games.clash.provider import image_for
 from simulators.generals.autoresearch.provider import configure_sandbox_waits
 configure_sandbox_waits();budget=CampaignBudget('clash',path=LEDGER)
 if (CAMPAIGN/'clash/rl/selection-close.json').exists():raise ValueError('Selection is closed; no further training after fresh assessment')
 counterpart=json.loads((CAMPAIGN/'clash/coordination/evaluator.json').read_text())
 if counterpart['runtime_pin']!=engine_binding():raise ValueError('counterpart engine binding differs')
 if seconds>570 and proof:raise ValueError('first proof total reservation <=600 including 30sec end margin')
 if not proof:
  proofs=list((CAMPAIGN/'clash/rl').glob('*/collected/output/proof.json'))
  if not proofs or not any(json.loads(p.read_text()).get('exact_next_update_resume') for p in proofs):raise ValueError('no collected proof passed')
  admission_path=CAMPAIGN/'clash/rl/admission-proof.json'
  if not admission_path.exists():raise ValueError('export admission proof required before scale')
  admission=json.loads(admission_path.read_text())
  required=('finite_actual_learning','weights_changed','actual_learned_history_games','exact_next_update_resume','deterministic_legal_inference','source_bound_checkpoint','export_admitted')
  if not all(admission.get(k) for k in required):raise ValueError('incomplete first learning/export/resume/admission gate')
 if seconds+30>1800:raise ValueError('allocation maximum 1800sec')
 if paired_seed is not None:
  if variant in ('v3','v4'):
   planned=json.loads((CAMPAIGN/('clash/rl/'+variant+'-precommit.json')).read_text());source=GAME/'reference_training'/variant
   if paired_seed not in planned['training_seeds'] or sha(source/'recipe.json')!=planned['recipe_sha256'] or any(sha(source/n)!=v for n,v in planned['source'].items()):raise ValueError('source/recipe/seed differs from precommit')
   controller=json.loads((CAMPAIGN/('clash/rl/'+variant+'-precommit-controller.json')).read_text())
   if sha(GAME/'reference_training'/('v4_cases.py' if variant=='v4' else 'cases.py'))!=controller['cases_entrypoint_sha256']:raise ValueError('paired controller differs from precommit')
  teacher=Path(counterpart['public_teacher_source']);want=json.loads((GAME/'reference_training'/variant/'recipe.json').read_text())['public_teacher_sha256']
  if sha(teacher)!=want or sha(GAME/'autoresearch/visible_teacher.py')!=want:raise ValueError('frozen evaluator teacher handshake differs')
 target=CAMPAIGN/'clash/rl'/name;target.mkdir(parents=True,exist_ok=False)
 archive=snapshot(target);usage=role_usage(budget)
 # Reserve CPU image build plus GPU worker plus conservative CPU observer BEFORE providers.
 quote=(seconds+30)*(.001097+32*.00003942+128*.00000667)*1.1
 if usage['gpu_seconds']+seconds+30>12600 or usage['resource_usd']+quote+.6+(1.7 if variant=='v4' else 0)>32:raise ValueError('RL role subcap reached')
 build=budget.reserve_worker('clash/rl/image/'+name,600,cpu_cores=32,memory_gib=128,sandbox=False)
 worker=budget.reserve_worker('clash/rl/train/'+name,seconds+30,gpu='H100!',cpu_cores=32,memory_gib=128)
 observer=budget.reserve_worker('clash/rl/observer/'+name,seconds+90,cpu_cores=.25,memory_gib=1,sandbox=False)
 dump(target/'reservations.json',{'image_build':build,'gpu':worker,'observer':observer,'role_usage_before':usage})
 volume_name='clash-reference-20261109-'+name;volume=modal.Volume.from_name(volume_name,environment_name='staging',create_if_missing=True,version=2)
 os.environ.setdefault('CLASH_AR_BASE_IMAGE_ID','im-8pfiKtMJEgLNOpUg3qFbSS')
 # Audited retained runtime: all native source hashes match the current engine pin.
 cached=ROOT/'evidence/overnight-20261108-1533Z/clash/provider/runtime-snapshots/d9f129bb3225b4fb2392073f02f81e704b7b65bf2012cca904a5d68f5be494d3.tar.gz'
 if sha(cached)!='d9f129bb3225b4fb2392073f02f81e704b7b65bf2012cca904a5d68f5be494d3':raise ValueError('retained runtime archive drift')
 with tarfile.open(cached) as t:
  hashes={m.name:__import__('hashlib').sha256(t.extractfile(m).read()).hexdigest() for m in t.getmembers() if m.isfile()}
 if any(hashes.get('simulators/clash/'+n)!=v for n,v in engine_binding()['source'].items()):raise ValueError('retained runtime native source differs')
 app=modal.App('clash-reference-20261109-'+name)
 image=(modal.Image.from_id('im-uOmXtJlDktZrpzpL4ZQBk6')
        .add_local_file(archive,'/opt/reference-source.tar.gz',copy=True)
        .run_commands('tar -xzf /opt/reference-source.tar.gz -C /opt/clash',
          'timeout 500s env PYTHONPATH=/opt/clash python -c '+shlex.quote("import sys;sys.path.insert(0,'/opt/clash/simulators/clash/native');from build import build;build();import torch;torch.cuda.get_device_capability=lambda *a:(9,0);from runtime import build_cuda;build_cuda()"))
        )
 monitor=app.function(image=image,cpu=(.25,.25),memory=(1024,1024),volumes={'/retained':volume},timeout=int(seconds+90),startup_timeout=120,retries=0,max_containers=1,min_containers=0,buffer_containers=0,single_use_containers=True,serialized=True,include_source=False,name='allocation_observer')(observe)
 with modal.enable_output(),app.run(detach=True,environment_name='staging'):
  dump(target/'app.json',{'app_id':app.app_id,'role':'clash-rl'})
  image.build(app);budget.attach(build,image.object_id)
  dump(target/'image.json',{'image_id':image.object_id,'source_sha256':sha(archive),'build_hold_retained':True})
  args=['python','-u','-m','simulators.clash.reference_training.autonomous','--root','/retained','--seconds',str(seconds-10),'--volume',volume_name]
  if proof:args+=['--proof']
  if paired_seed is not None:
   args=['python','-u','-m','simulators.clash.reference_training.'+('v4_cases' if variant=='v4' else 'cases'),'--root','/retained','--seconds',str(seconds-30),'--seed',str(paired_seed),'--variant',variant]
  if resume:
   # Parent Volume is immutable read-only; exact source/config/build is verified remotely.
   parent=json.loads(Path(resume).read_text());parent_volume=modal.Volume.from_name(parent['volume_name'],environment_name='staging');args+=['--resume','/parent/output/checkpoint.pt']
  launched=time.time();sb=modal.Sandbox.create(*args,app=app,image=image,gpu='H100!',cpu=(32,32),memory=(131072,131072),timeout=int(seconds),block_network=True,secrets=[],volumes={'/retained':volume,**({'/parent':parent_volume} if resume else {})},include_oidc_identity_token=False,environment_name='staging',tags={'campaign':'reference-readiness-20261109-0251Z','role':'clash-rl','attempt':name})
  budget.attach(worker,sb.object_id);value={'app_id':app.app_id,'sandbox_id':sb.object_id,'volume_name':volume_name,'operation':worker,'image_build_operation':build,'observer_operation':observer,'launched_unix':launched,'provider_timeout':seconds,'source_archive_sha256':sha(archive),'proof':proof,'resume_provider':resume,'paired_seed':paired_seed,'variant':variant}
  dump(target/'provider.json',value)
  call=monitor.spawn(app.app_id,sb.object_id,volume_name,seconds,launched);budget.attach(observer,call.object_id);value['observer_call_id']=call.object_id;dump(target/'provider.json',value);sb.detach()
 print(json.dumps(value,indent=2))


def collect(name):
 import modal
 from tools.overnight_budget import CampaignBudget
 from simulators.generals.autoresearch.provider import configure_sandbox_waits,allocation_info
 configure_sandbox_waits();target=CAMPAIGN/'clash/rl'/name;provider=json.loads((target/'provider.json').read_text());budget=CampaignBudget('clash',path=LEDGER)
 receipt=allocation_info(provider['app_id'],provider['sandbox_id'])
 dump(target/'provider-current.json',receipt)
 if not receipt or not receipt.get('finished_at'):print(json.dumps({'running_or_unresolved':True,'provider':receipt}));return
 row=next(r for r in budget.snapshot()['operations'] if r['id']==provider['operation'])
 if row['status']!='settled':budget.settle_worker(provider['operation'],receipt)
 volume=modal.Volume.from_name(provider['volume_name'],environment_name='staging');destination=target/'collected';destination.mkdir(exist_ok=True)
 manifest_path=target/'collection-files.json';downloaded=json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
 for entry in volume.iterdir('/',recursive=True):
  if entry.type.name!='FILE':continue
  if entry.path.endswith('/resume-probe.pt'):continue # retained remotely; final resume state is collected
  path=destination/entry.path.lstrip('/');path.parent.mkdir(parents=True,exist_ok=True)
  if path.exists() and downloaded.get(entry.path)==sha(path):continue
  partial=path.with_name(path.name+'.download-part')
  with partial.open('wb') as stream:
   for block in volume.read_file(entry.path):stream.write(block)
  partial.replace(path);downloaded[entry.path]=sha(path);dump(manifest_path,downloaded)
 result_path=destination/('result.json' if provider.get('paired_seed') is not None else 'output/result.json')
 result=json.loads(result_path.read_text()) if result_path.exists() else None
 dump(target/'settlement.json',{'provider':receipt,'result_present':result is not None,'owned_gpu_workers_active':0,'remaining_build_observer_holds':True})
 print(json.dumps({'seconds':receipt['finished_at']-receipt['started_at'],'updates':result.get('updates') if result else None,'role_usage':role_usage(budget)}))


def main():
 p=argparse.ArgumentParser();p.add_argument('action',choices=['launch','collect']);p.add_argument('--name',required=True);p.add_argument('--seconds',type=int,default=570);p.add_argument('--proof',action='store_true');p.add_argument('--resume');p.add_argument('--paired-seed',type=int);p.add_argument('--variant',choices=('v2','v3','v4'),default='v2');a=p.parse_args()
 if a.action=='launch':launch(a.name,a.seconds,a.proof,a.resume,a.paired_seed,a.variant)
 else:collect(a.name)
if __name__=='__main__':main()


def recover_prelaunch(name,old_app,image_id):
 """Continue SAME reservation after a build-stream failure, only with empty inventory."""
 if (CAMPAIGN/'clash/rl/selection-close.json').exists():raise ValueError('Selection closed; no training recovery')
 import modal
 from modal.client import _Client
 from modal_proto import api_pb2
 from modal._utils.async_utils import synchronizer
 from tools.overnight_budget import CampaignBudget
 from simulators.generals.autoresearch.provider import configure_sandbox_waits
 configure_sandbox_waits();target=CAMPAIGN/'clash/rl'/name
 if (CAMPAIGN/'clash/rl/selection-close.json').exists():raise ValueError('Selection closed; no training recovery launch')
 if (target/'provider.json').exists():raise ValueError('A known worker cannot be relaunched')
 async def inventory():
  client=await _Client.from_env();r=await client.stub.SandboxList(api_pb2.SandboxListRequest(app_id=old_app,environment_name='staging',include_finished=True));return [s.id for s in r.sandboxes]
 rows=synchronizer.create_blocking(inventory)()
 dump(target/'prelaunch-recovery.json',{'previous_app_id':old_app,'sandbox_ids':rows,'image_id':image_id,'same_original_reservations':True,'reason':'Build-log RPC connection reset after native image finished, before app body; reuse completed immutable native image, no duplicated GPU allocation.'})
 if rows:raise ValueError('Existing allocation found; reconcile instead')
 b=CampaignBudget('clash',path=LEDGER);reserved=json.loads((target/'reservations.json').read_text());image=modal.Image.from_id(image_id);app=modal.App('clash-reference-20261109-'+name+'-prelaunch-recovery')
 b.attach(reserved['image_build'],'im-xyJSlgJdZdtZSZZWQpEHWP')
 volume_name='clash-reference-20261109-'+name;volume=modal.Volume.from_name(volume_name,environment_name='staging')
 monitor=app.function(image=image,cpu=(.25,.25),memory=(1024,1024),volumes={'/retained':volume},timeout=660,startup_timeout=120,retries=0,max_containers=1,min_containers=0,buffer_containers=0,single_use_containers=True,serialized=True,include_source=False,name='allocation_observer')(observe)
 with app.run(detach=True,environment_name='staging'):
  dump(target/'image.json',{'image_id':image_id,'source_sha256':sha(target/'source.tar.gz'),'build_hold_retained':True,'recovery_from_completed_layer':True})
  launched=time.time();sb=modal.Sandbox.create('python','-u','-m','simulators.clash.reference_training.autonomous','--root','/retained','--seconds','560','--volume',volume_name,'--proof',app=app,image=image,gpu='H100!',cpu=(32,32),memory=(131072,131072),timeout=570,block_network=True,secrets=[],volumes={'/retained':volume},include_oidc_identity_token=False,environment_name='staging',tags={'campaign':'reference-readiness-20261109-0251Z','role':'clash-rl','attempt':name})
  b.attach(reserved['gpu'],sb.object_id);value={'app_id':app.app_id,'sandbox_id':sb.object_id,'volume_name':volume_name,'operation':reserved['gpu'],'image_build_operation':reserved['image_build'],'observer_operation':reserved['observer'],'launched_unix':launched,'provider_timeout':570,'source_archive_sha256':sha(target/'source.tar.gz'),'proof':True,'resume_provider':None};dump(target/'provider.json',value)
  call=monitor.spawn(app.app_id,sb.object_id,volume_name,570,launched);b.attach(reserved['observer'],call.object_id);value['observer_call_id']=call.object_id;dump(target/'provider.json',value);sb.detach()
 print(json.dumps(value,indent=2))
