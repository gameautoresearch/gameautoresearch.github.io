"""New explicit OpenCreditBudget campaign; old capped ledger is never mutated."""
from pathlib import Path
import argparse,json,os,tarfile,time,shlex
os.environ.setdefault('MODAL_DISABLE_API_PROXY','1')
from simulators.clash.autoresearch.common import ROOT,GAME,dump,sha,engine_binding
from tools.open_credit_campaign import OpenCreditBudget
from tools.overnight_budget import stamp
CAMPAIGN=ROOT/'evidence/strong-bots-20261109-0516Z';LEDGER=CAMPAIGN/'campaign.sqlite3';OWN=CAMPAIGN/'clash/rl'

def observe(app_id,sandbox_id,seconds,absolute_end,queued_at):
 import time,modal
 from pathlib import Path
 from simulators.generals.autoresearch.provider import configure_sandbox_waits,allocation_info
 from simulators.clash.autoresearch.common import dump
 configure_sandbox_waits();receipt=None;reason=None
 while True:
  receipt=allocation_info(app_id,sandbox_id);now=time.time()
  if receipt and receipt.get('finished_at'):break
  started=receipt.get('started_at',0) if receipt else 0
  if now>=absolute_end:reason='absolute selection/worker cutoff';break
  if started and now>=started+seconds+10:reason='full allocated lifetime exceeded';break
  # Leave enough of the finite observer lifetime to cover the entire GPU/CPU
  # allocation even after queueing. The historical247second queue is retained;
  # new stages cancel before queue+worker life can overrun this observer.
  if not started and now>=queued_at+min(60,seconds):reason='bounded queue cancellation, not a training-time discount';break
  time.sleep(5)
 if reason:
  modal.Sandbox.from_id(sandbox_id).terminate(wait=True)
  receipt=allocation_info(app_id,sandbox_id)
 dump(Path('/retained')/'allocation.json',{'provider':receipt,'termination_reason':reason,'resolved':bool(receipt and receipt.get('finished_at')),'automatic_retries':0})
 import subprocess
 subprocess.run(['sync','/retained'],check=True,timeout=20)
 return receipt

def snapshot(target):
 from simulators.clash.reference_training.launch import snapshot as base_snapshot
 base_snapshot(target);folder=target/'source';tools=folder/'tools';tools.mkdir(exist_ok=True)
 import shutil
 for n in ('open_credit_campaign.py','overnight_budget.py'):shutil.copyfile(ROOT/'tools'/n,tools/n)
 archive=target/'strong-source.tar.gz'
 with tarfile.open(archive,'w:gz') as tar:
  for p in sorted(folder.rglob('*')):
   if p.is_file():tar.add(p,arcname=p.relative_to(folder))
 archive.chmod(0o444);dump(target/'strong-source-files.json',{str(p.relative_to(folder)):sha(p) for p in folder.rglob('*') if p.is_file()});return archive

def launch(name,parent_provider,parent_checkpoint,parent_sha,seconds=570,variant='v4',teacher_coef=None,seed=None,mode='auto'):
 if (OWN/'user-stop-request.json').exists():raise ValueError('Stopped by explicit user order; no new workers')
 if mode not in ('auto','exact','warmstart') or variant not in ('v4','v5','v6','v7','v8','v9','v10','v11','v12','v13'):raise ValueError('Explicit supported source and resume mode required')
 if mode=='warmstart' and variant=='v4':raise ValueError('Historical v4 source supports exact resume only')
 import modal
 from simulators.generals.autoresearch.provider import configure_sandbox_waits
 configure_sandbox_waits();budget=OpenCreditBudget('clash',LEDGER);auth=budget.auth;cutoff=stamp(auth['selection_close_utc'])
 if seconds>3500 or seconds<60 or time.time()+seconds+40>=cutoff:raise ValueError('Finite stage must fit selection cutoff with teardown margin')
 if (OWN/'selection-close.json').exists():raise ValueError('New campaign selection already closed')
 peer=json.loads((CAMPAIGN/'clash/coordination/evaluator.json').read_text());pin=peer['source_interface_pin']
 if pin['engine_binding_sha256']!=__import__('simulators.clash.autoresearch.common',fromlist=['digest']).digest(engine_binding()):raise ValueError('Evaluator engine pin differs')
 if sha(GAME/'autoresearch/visible_teacher.py')!=pin['qualified_public_teacher_sha256']:raise ValueError('Teacher differs from evaluator handshake')
 local_parent=Path(parent_provider).parent/'collected'/parent_checkpoint.lstrip('/')
 if not local_parent.is_file() or sha(local_parent)!=parent_sha:raise ValueError('Host parent checkpoint checksum mismatch before any paid reservation')
 parent=json.loads(Path(parent_provider).read_text());target=OWN/name;target.mkdir(parents=True,exist_ok=False);archive=snapshot(target)
 build=budget.reserve_worker('clash/rl/image/'+name,600,cpu_cores=32,memory_gib=128,sandbox=False,purpose='build')
 gpu=budget.reserve_worker('clash/rl/train/'+name,seconds+30,gpu='H100!',cpu_cores=32,memory_gib=128,purpose='training')
 observer=budget.reserve_worker('clash/rl/observer/'+name,min(3600,seconds+90),cpu_cores=.25,memory_gib=1,sandbox=False,purpose='observer')
 dump(target/'reservations.json',{'image_build':build,'gpu':gpu,'observer':observer})
 volume_name='clash-strong-20261109-'+name;volume=modal.Volume.from_name(volume_name,environment_name='staging',create_if_missing=True,version=2);pv=modal.Volume.from_name(parent['volume_name'],environment_name='staging')
 image=(modal.Image.from_id('im-uOmXtJlDktZrpzpL4ZQBk6').add_local_file(archive,'/opt/strong-source.tar.gz',copy=True).run_commands('tar -xzf /opt/strong-source.tar.gz -C /opt/clash','timeout 500s env PYTHONPATH=/opt/clash python -c '+shlex.quote("import sys;sys.path.insert(0,'/opt/clash/simulators/clash/native');from build import build;build();import torch;torch.cuda.get_device_capability=lambda *a:(9,0);from runtime import build_cuda;build_cuda()")))
 app=modal.App('clash-strong-20261109-'+name);monitor=app.function(image=image,cpu=(.25,.25),memory=(1024,1024),volumes={'/retained':volume},timeout=min(3600,seconds+90),startup_timeout=120,retries=0,max_containers=1,min_containers=0,buffer_containers=0,single_use_containers=True,serialized=True,include_source=False,name='allocation_observer')(observe)
 with modal.enable_output(),app.run(detach=True,environment_name='staging'):
  dump(target/'app.json',{'app_id':app.app_id});image.build(app);budget.attach(build,image.object_id);dump(target/'image.json',{'image_id':image.object_id,'source_sha256':sha(archive),'build_hold_retained':True})
  queued=time.time();args=['python','-u','-m','simulators.clash.reference_training.strong.worker','--parent','/parent/'+parent_checkpoint.lstrip('/'),'--parent-sha',parent_sha,'--out','/retained/output','--seconds',str(seconds-25),'--absolute-end',str(cutoff-15),'--variant',variant]
  if teacher_coef is not None:args+=['--teacher-coef',str(teacher_coef)]
  if seed is not None:args+=['--seed',str(seed)]
  args+=['--mode',mode]
  sb=modal.Sandbox.create(*args,app=app,image=image,gpu='H100!',cpu=(32,32),memory=(131072,131072),timeout=seconds,block_network=True,secrets=[],volumes={'/retained':volume,'/parent':pv},include_oidc_identity_token=False,environment_name='staging',tags={'campaign':auth['campaign_id'],'role':'clash-rl','attempt':name})
  budget.attach(gpu,sb.object_id);value={'app_id':app.app_id,'sandbox_id':sb.object_id,'volume_name':volume_name,'operation':gpu,'image_build_operation':build,'observer_operation':observer,'provider_timeout':seconds,'source_archive_sha256':sha(archive),'variant':variant,'parent_provider':str(parent_provider),'parent_checkpoint_path':parent_checkpoint,'parent_checkpoint_sha256':parent_sha,'selection_cutoff_unix':cutoff}
  value.update(teacher_coef=teacher_coef,seed=seed,resume_mode=mode);dump(target/'provider.json',value);call=monitor.spawn(app.app_id,sb.object_id,seconds,cutoff-10,queued);budget.attach(observer,call.object_id);value['observer_call_id']=call.object_id;dump(target/'provider.json',value);sb.detach()
 return value

def _collect(name):
 import modal
 from simulators.generals.autoresearch.provider import configure_sandbox_waits,allocation_info
 configure_sandbox_waits();target=OWN/name;p=json.loads((target/'provider.json').read_text());receipt=allocation_info(p['app_id'],p['sandbox_id']);dump(target/'provider-current.json',receipt)
 if not receipt or not receipt.get('finished_at'):return {'running_or_unknown':True,'provider':receipt}
 b=OpenCreditBudget('clash',LEDGER);row=next(x for x in b.snapshot()['operations'] if x['id']==p['operation'])
 if row['status']!='settled':b.settle_worker(p['operation'],receipt)
 v=modal.Volume.from_name(p['volume_name'],environment_name='staging');dest=target/'collected';dest.mkdir(exist_ok=True);mp=target/'collection-files.json';files=json.loads(mp.read_text()) if mp.exists() else {}
 # Drain metadata before slow downloads: keeping VolumeListFiles2 open during
 # transfers can expire its stream and prevent later nested files collecting.
 entries=list(v.iterdir('/',recursive=True))
 for e in entries:
  if e.type.name!='FILE' or e.path.endswith('/resume-probe.pt'):continue
  out=dest/e.path.lstrip('/');out.parent.mkdir(parents=True,exist_ok=True)
  if out.exists() and files.get(e.path)==sha(out):continue
  part=out.with_name(out.name+'.collect-part-'+str(os.getpid()))
  with part.open('wb') as f:
   for block in v.read_file(e.path):f.write(block)
  part.replace(out);files[e.path]=sha(out);dump(mp,files)
 result_path=next((dest/n for n in ('output/training/result.json','output/result.json') if (dest/n).exists()),None);result=json.loads(result_path.read_text()) if result_path else None
 summary={'provider':receipt,'complete':bool(result),'new_updates':result.get('new_updates',result.get('updates')) if result else None,'new_samples':result.get('new_samples',result.get('samples')) if result else None,'actual_gpu_seconds':receipt['finished_at']-receipt['started_at']};dump(target/'settlement.json',summary);return summary

def collect(name):
 import fcntl
 with (OWN/name/'.collection.lock').open('a+') as lock:
  try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:return {'collection_in_progress':True,'name':name}
  return _collect(name)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['launch','collect']);p.add_argument('--name',required=True);p.add_argument('--parent-provider');p.add_argument('--parent-checkpoint');p.add_argument('--parent-sha');p.add_argument('--seconds',type=int,default=570);p.add_argument('--variant',default='v4');p.add_argument('--teacher-coef',type=float);p.add_argument('--seed',type=int);p.add_argument('--mode',choices=['auto','exact','warmstart'],default='auto');a=p.parse_args();print(json.dumps(collect(a.name) if a.action=='collect' else launch(a.name,a.parent_provider,a.parent_checkpoint,a.parent_sha,a.seconds,a.variant,a.teacher_coef,a.seed,a.mode),indent=2))
