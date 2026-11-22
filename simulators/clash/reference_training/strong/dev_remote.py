"""Accounted CPU development batches; no private final schedules or GPU creation."""
from pathlib import Path
import argparse,json,os,shutil,time
os.environ.setdefault('MODAL_DISABLE_API_PROXY','1')
from .provider import CAMPAIGN,OWN,LEDGER,snapshot,observe
from .evaluate import plan
from simulators.clash.autoresearch.common import dump,sha,digest,engine_binding
from tools.open_credit_campaign import OpenCreditBudget
from tools.overnight_budget import stamp

def launch(name,candidate_paths,seconds=900,units=8,parallel=False):
 if (OWN/'user-stop-request.json').exists():raise ValueError('Stopped by explicit user order; no new evaluation workers')
 import modal
 from simulators.generals.autoresearch.provider import configure_sandbox_waits
 from simulators.clash.autoresearch.bundle import verify
 configure_sandbox_waits();b=OpenCreditBudget('clash',LEDGER);cutoff=stamp(b.auth['selection_close_utc'])
 if units not in (8,32):raise ValueError('Explicit published8/32-unit public schedules only')
 if len(candidate_paths)>2 or not candidate_paths or seconds>1800 or time.time()+seconds+90>=cutoff:raise ValueError('Bounded one/two-policy public development batch required')
 peer=json.loads((CAMPAIGN/'clash/coordination/evaluator.json').read_text())
 if peer['source_interface_pin']['engine_binding_sha256']!=digest(engine_binding()):raise ValueError('Evaluator engine handshake changed')
 parallel=parallel or units==32
 runtime_pin=None
 if parallel:
  from simulators.clash.autoresearch.common import GAME
  from simulators.clash.autoresearch.release_v3 import PUBLIC_CAPACITY_SEEDS
  capacity=CAMPAIGN/'clash/evaluator/jobs/public-v31-capacity-01'
  runtime_pin=json.loads((capacity/'source.json').read_text())['runtime_files']
  if any(sha(GAME/'autoresearch'/n)!=v for n,v in runtime_pin.items()):raise ValueError('Qualified parallel runtime drift')
 target=OWN/'cpu-development'/name;target.mkdir(parents=True,exist_ok=False);inputs=target/'inputs';inputs.mkdir(mode=0o700);spec=plan()
 if units==32:
  capacity_schedule=json.loads((capacity/'inputs/schedule.json').read_text())
  if capacity_schedule['engine_seeds']!=PUBLIC_CAPACITY_SEEDS or capacity_schedule['schedule_kind']!='public-32-unit-capacity-qualification':raise ValueError('Not the published PUBLIC capacity schedule')
  spec={'public_engine_seeds':PUBLIC_CAPACITY_SEEDS,'opponents':[]}
  for item in capacity_schedule['opponents']:
   item=dict(item)
   if item['kind']=='checkpoint':item['bundle']=str(capacity/'inputs/bundles'/item['name'])
   spec['opponents'].append(item)
 candidates=[]
 for candidate_path in candidate_paths:
  source=Path(candidate_path);manifest=verify(source);candidate=source.parent.name;dest=inputs/'candidates'/candidate;shutil.copytree(source,dest);candidates.append({'name':candidate,'kind':'checkpoint','bundle':'candidates/'+candidate,'identity':manifest['identity']})
 opponents=[]
 for x in spec['opponents']:
  item=dict(x)
  if x['kind']=='checkpoint':
   dest=inputs/'opponents'/x['name'];shutil.copytree(x['bundle'],dest);item['bundle']='opponents/'+x['name']
  opponents.append(item)
 dump(inputs/'plan.json',{'purpose':'PUBLIC DEVELOPMENT ONLY','engine_sha256':digest(engine_binding()),'public_engine_seeds':spec['public_engine_seeds'],'candidates':candidates,'opponents':opponents,'counterpart_status_sha256':sha(CAMPAIGN/'clash/coordination/evaluator.json'),'public_units':units,'qualified_parallel_runtime_files':runtime_pin,'worker_seconds':seconds-40})
 archive=snapshot(target)
 if runtime_pin and any(sha(target/'source/simulators/clash/autoresearch'/n)!=v for n,v in runtime_pin.items()):raise ValueError('Parallel runtime changed during snapshot')
 build=b.reserve_worker('clash/rl/dev-image/'+name,600,cpu_cores=32,memory_gib=128,sandbox=False,purpose='build');worker=b.reserve_worker('clash/rl/dev-cpu/'+name,seconds+30,cpu_cores=8,memory_gib=32,purpose='development');observer=b.reserve_worker('clash/rl/dev-observer/'+name,seconds+90,cpu_cores=.25,memory_gib=1,sandbox=False,purpose='observer');dump(target/'reservations.json',{'image_build':build,'worker':worker,'observer':observer})
 volume_name='clash-strong-dev-20261109-'+name;volume=modal.Volume.from_name(volume_name,environment_name='staging',create_if_missing=True,version=2);image=modal.Image.from_id('im-uOmXtJlDktZrpzpL4ZQBk6').add_local_file(archive,'/opt/strong-dev.tar.gz',copy=True).run_commands('tar -xzf /opt/strong-dev.tar.gz -C /opt/clash')
 app=modal.App('clash-strong-dev-'+name);monitor=app.function(image=image,cpu=(.25,.25),memory=(1024,1024),volumes={'/retained':volume},timeout=seconds+90,startup_timeout=120,retries=0,max_containers=1,min_containers=0,buffer_containers=0,single_use_containers=True,serialized=True,include_source=False,name='allocation_observer')(observe)
 with modal.enable_output(),app.run(detach=True,environment_name='staging'):
  dump(target/'app.json',{'app_id':app.app_id});image.build(app);b.attach(build,image.object_id);dump(target/'image.json',{'image_id':image.object_id,'source_archive_sha256':sha(archive),'build_hold_retained':True})
  with volume.batch_upload() as upload:upload.put_directory(inputs,'/inputs')
  queued=time.time();entrypoint='simulators.clash.reference_training.strong.'+('dev_parallel_worker' if parallel else 'dev_worker');sb=modal.Sandbox.create('python','-u','-m',entrypoint,'--root','/retained',app=app,image=image,cpu=(8,8),memory=(32768,32768),timeout=seconds,block_network=True,secrets=[],volumes={'/retained':volume},include_oidc_identity_token=False,environment_name='staging',tags={'campaign':b.auth['campaign_id'],'role':'clash-rl-public-development'})
  b.attach(worker,sb.object_id);p={'app_id':app.app_id,'sandbox_id':sb.object_id,'volume_name':volume_name,'operation':worker,'observer_operation':observer,'image_build_operation':build,'cpu_only':True,'provider_timeout':seconds,'public_development_only':True,'public_units':units,'pair_workers':3 if parallel else 1};dump(target/'provider.json',p);call=monitor.spawn(app.app_id,sb.object_id,seconds,cutoff-10,queued);b.attach(observer,call.object_id);p['observer_call_id']=call.object_id;dump(target/'provider.json',p);sb.detach()
 return p

def _collect(name):
 import modal
 from simulators.generals.autoresearch.provider import configure_sandbox_waits,allocation_info
 configure_sandbox_waits();target=OWN/'cpu-development'/name;p=json.loads((target/'provider.json').read_text());r=allocation_info(p['app_id'],p['sandbox_id']);dump(target/'provider-current.json',r)
 if not r or not r.get('finished_at'):return {'running_or_unknown':True,'provider':r}
 b=OpenCreditBudget('clash',LEDGER);row=next(x for x in b.snapshot()['operations'] if x['id']==p['operation'])
 if row['status']!='settled':b.settle_worker(p['operation'],r)
 v=modal.Volume.from_name(p['volume_name'],environment_name='staging');dest=target/'collected';dest.mkdir(exist_ok=True)
 entries=list(v.iterdir('/',recursive=True))
 for e in entries:
  if e.type.name!='FILE' or e.path.lstrip('/').startswith('inputs/'):continue
  out=dest/e.path.lstrip('/');out.parent.mkdir(parents=True,exist_ok=True)
  if out.exists():continue
  part=out.with_name(out.name+'.collect-part-'+str(os.getpid()))
  with part.open('wb') as f:
   for block in v.read_file(e.path):f.write(block)
  part.replace(out)
 result=json.loads((dest/'result.json').read_text()) if (dest/'result.json').exists() else None;value={'cpu_only':True,'public_development_only':True,'complete':bool(result and result['complete']),'summaries':result['summaries'] if result else None,'provider':r};dump(target/'settlement.json',value);return value

def collect(name):
 import fcntl
 with (OWN/'cpu-development'/name/'.collection.lock').open('a+') as lock:
  try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:return {'collection_in_progress':True,'name':name}
  return _collect(name)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['launch','collect']);p.add_argument('--name',required=True);p.add_argument('--bundles',nargs='*');p.add_argument('--seconds',type=int,default=900);p.add_argument('--units',type=int,choices=[8,32],default=8);p.add_argument('--parallel',action='store_true');a=p.parse_args();print(json.dumps(collect(a.name) if a.action=='collect' else launch(a.name,a.bundles,a.seconds,a.units,a.parallel),indent=2))
