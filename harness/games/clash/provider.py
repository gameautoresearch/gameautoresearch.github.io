"""Clash Modal workers: the worker image, finite L40S training allocations, CPU-only practice.

Each allocation has its own sandbox, Volume and output folder and is reserved in the caller's ledger, which decides
how many may be unresolved at once (one in seconds mode; parallel in dollar mode). Nothing mutable is shared.

A reservation precedes every create. Actual provider start/end intervals include
setup/idle/transfers/checkpoints/cleanup. Bulk download occurs after GPU shutdown.
There are no retries: a failed allocation stays charged and an allocation whose
end is unknown keeps its full hold until the provider reports it. A created worker
the provider never starts (no GPU allocated) is terminated and settled at 0 seconds
from the provider's own record (harness.workers); queued time is never charged.
"""
from pathlib import Path
import gzip,io,json,math,os,shutil,tarfile,tempfile,time,uuid

from harness.policy import ENVIRONMENT
from harness import hardware, workers
from simulators.clash.autoresearch.common import GAME,MATCH_FORM,ROOT,STOP_REQUEST,dump,sha

STOPPED='stopped by request'


def snapshot(destination):
 """Content-addressed, game-only worker source: native engine, research runtime, pinned scripted opponents."""
 destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
 with tempfile.TemporaryDirectory() as temp:
  root=Path(temp)/'root';target=root/'simulators/clash';target.mkdir(parents=True)
  for p in GAME.rglob('*'):
   relative=p.relative_to(GAME)
   if relative.parts[0] not in ('native','autoresearch') and relative.name not in ('rules.lock.json','README.md','model.py'):continue
   if not p.is_file() or any(v in ('__pycache__','.build') for v in p.parts):continue
   if any(v.startswith('generated') for v in p.parts) and p.parent!=GAME/'native/generated-v2':continue
   if p.suffix not in ('.py','.json','.hpp','.cpp','.cu','.cuh','.md','.txt') and p.name not in ('LICENSE','NOTICE'):continue
   q=target/p.relative_to(GAME);q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,q)
  engine=ROOT/'vendor/engines/crforge/python/crforge_env'
  for p in engine.glob('*.py'):
   q=root/'candidates/crforge/python/crforge_env'/p.name;q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,q)
  archive=Path(temp)/'source.tar.gz'
  with archive.open('wb') as stream,gzip.GzipFile(fileobj=stream,mode='wb',mtime=0) as compressed:
   with tarfile.open(fileobj=compressed,mode='w') as tar:
    for p in sorted(root.rglob('*')):
     if p.is_file():
      payload=p.read_bytes();member=tarfile.TarInfo(str(p.relative_to(root)));member.size=len(payload);member.mode=0o444;member.mtime=0;tar.addfile(member,io.BytesIO(payload))
  identity=sha(archive);out=destination/(identity+'.tar.gz')
  if not out.exists():shutil.copyfile(archive,out);out.chmod(0o444)
  return out


def image_for(archive):
 import modal
 unpack="mkdir -p /opt/clash && tar -xzf /opt/clash-source.tar.gz -C /opt/clash"
 # Native kernels are compiled for the repository GPU's architecture (harness.hardware.CUDA_ARCH, e.g. sm_89).
 # The engine libraries are the benchmark's match form (common.MATCH_FORM), prebuilt so no worker compiles them.
 CAPABILITY=(int(hardware.CUDA_ARCH[3]),int(hardware.CUDA_ARCH[4:]))
 build=f"PYTHONPATH=/opt/clash python -c \"import sys;sys.path.insert(0,'/opt/clash/simulators/clash/native');from build import build;build({MATCH_FORM!r});import torch;torch.cuda.get_device_capability=lambda *a:{CAPABILITY};from runtime import build_cuda;build_cuda({MATCH_FORM!r});torch.cuda.is_available=lambda:True;from simulators.clash.autoresearch.graph_bridge import BRIDGES,build as bridge_build;[bridge_build(name) for name in BRIDGES]\""
 base=modal.Image.from_id(os.environ['CLASH_AR_BASE_IMAGE_ID']) if os.getenv('CLASH_AR_BASE_IMAGE_ID') else modal.Image.from_registry('nvidia/cuda:12.8.1-devel-ubuntu24.04',add_python='3.12').apt_install('g++','ca-certificates').pip_install('torch==2.9.1','numpy==2.5.4','JPype1==1.7.1','gymnasium==1.4.0','pettingzoo==1.27.0')
 return (base.env({'PYTHONPATH':'/opt/clash','CUBLAS_WORKSPACE_CONFIG':':4096:8','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}).add_local_file(archive,'/opt/clash-source.tar.gz',copy=True).run_commands(unpack,build,'chmod -R a-w /opt/clash; chmod a+rw /opt/clash/simulators/clash/native/.build/*.lock').workdir('/job'))


def download_atomic(volume,remote,target):
 """A disconnected download never leaves a file that can be treated as complete."""
 target=Path(target);target.parent.mkdir(parents=True,exist_ok=True)
 if target.exists():return
 temporary=target.with_name('.'+target.name+'.partial')
 try:
  with temporary.open('wb') as stream:
   for chunk in volume.read_file(remote):stream.write(chunk)
  temporary.replace(target)
 finally:
  if temporary.exists():temporary.unlink()


def halt(stop):
 if stop is not None and stop():raise RuntimeError(STOPPED)


class FastProvider:
 reservation_overhead_seconds=30
 minimum_run_seconds=90
 max_practice_evaluations=6
 interface=None  # the release's evaluation interface and pair topology; the controller sets both
 pair_workers=1
 start_poll_seconds=2.0  # provider-receipt polling while a created worker waits for a GPU (harness.workers)
 stop_grace_seconds=180.0  # how long a stopped training loop may take to finish its update and export
 abandon_poll_seconds=1.0
 def __init__(self,app,archive,root):
  os.environ.setdefault('MODAL_DISABLE_API_PROXY','1')
  # The pinned Modal transport and polling helpers are shared with the Generals provider.
  from simulators.generals.autoresearch.provider import configure_sandbox_waits,read_rpc
  configure_sandbox_waits();self.app=app;self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
  cache=self.root/'image.json';self.archive=Path(archive)
  if cache.exists():
   value=json.loads(cache.read_text())
   if value['source_sha256']!=sha(archive):raise ValueError('immutable provider image source differs')
   import modal
   self.image=modal.Image.from_id(value['image_id'])
  else:
   self.image=image_for(archive);read_rpc(lambda:self.image.build(app));dump(cache,{'image_id':self.image.object_id,'source_sha256':sha(archive),'gpu_seconds':0,'CPU_build_cost_not_invoice_reconciled':True})
 def train(self,ledger,name,source,out,seconds,stop=None):
  c=json.loads((Path(source)/'config.json').read_text());h=c['hardware_allocation']
  return self.allocate(ledger,name,out,seconds,h['cpu_threads'],h['ram_gib'],{'source':Path(source)},'train',stop)['result']
 def allocate(self,ledger,name,out,seconds,cpu,ram,inputs,mode,stop=None):
  import modal
  from simulators.generals.autoresearch.provider import allocation_info,read_rpc
  if (cpu,ram) not in ((4,16),(32,128)) or seconds<90 or not math.isfinite(seconds):raise ValueError('explicit supported finite profile required')
  # Reserve the hard timeout plus settlement/teardown margin, then reject any
  # other unresolved operation. Neither failure nor timeout renews allowance.
  reserved=seconds+self.reservation_overhead_seconds
  ledger.reserve(name,reserved);out=Path(out);out.mkdir(parents=True,exist_ok=False)
  vname='clash-fast-'+uuid.uuid4().hex;volume=modal.Volume.from_name(vname,environment_name=ENVIRONMENT,create_if_missing=True,version=2)
  with volume.batch_upload() as batch:
   for label,path in inputs.items():
    if Path(path).is_dir():batch.put_directory(path,'/inputs/'+label)
    else:batch.put_file(path,'/inputs/'+label)
  dump(out/'request.json',{'mode':mode,'hard_worker_seconds':seconds,'reserved_seconds':reserved,'cpu_cores':cpu,'memory_gib':ram,'gpu':hardware.GPU,'gpu_selection':'repository default GPU (harness.hardware)','volume':vname,'automatic_retries':0,'source_archive_sha256':sha(self.archive)})
  sb=None;receipt=None;error=None;result=None;never=None
  try:
   sb=modal.Sandbox.create('/bin/sleep','86400',app=self.app,image=self.image,gpu=hardware.GPU,cpu=(cpu,cpu),memory=(ram*1024,ram*1024),timeout=math.ceil(seconds),volumes={'/retained':volume},block_network=True,secrets=[],include_oidc_identity_token=False,environment_name=ENVIRONMENT,tags={'clash_fast_starter':name})
   dump(out/'provider.json',{'app_id':self.app.app_id,'sandbox_id':sb.object_id,'volume':vname});ledger.record(name,status='running',app_id=self.app.app_id,sandbox_id=sb.object_id)
   # A created worker can wait in the provider's queue for a GPU for many minutes. Wait for it (queued time is
   # never charged); if it never starts, terminate it and settle the allocation at 0 from the provider's record.
   info=lambda:allocation_info(self.app.app_id,sb.object_id)
   receipt=workers.await_start(info,stop=stop,poll=self.start_poll_seconds)
   if receipt is None:
    receipt=workers.abandon(lambda:sb.terminate(),info,poll=self.abandon_poll_seconds)
    if receipt.get('provider_id')==sb.object_id and workers.never_started(receipt):
     stopped=stop is not None and stop()
     reason=(STOPPED+' while queued; ' if stopped else '')+workers.NEVER_STARTED
     ledger.settle_never_started(name,receipt,status='stopped' if stopped else 'failed',error=reason)
     never=workers.NeverStarted(receipt,reason)
     raise never
    raise RuntimeError('the worker started only after it was abandoned; charged from its provider receipt')
   deadline=receipt['started_at']+seconds-12
   proc=sb.exec('python','-u','-m','simulators.clash.autoresearch.fast_worker','--root','/retained','--mode',mode,'--deadline-at',str(deadline),timeout=max(1,int(deadline-time.time())))
   # `ar stop` on a training run asks the trusted loop to end at its next update, so the network trained so far is
   # checkpointed, exported and collected like any finished run. Everything else ends the worker like any other
   # failure, terminated and charged for its real provider interval below: a session being closed (`stop()` is
   # 'now'), another worker mode, a loop that has not ended within the grace, a request that could not be sent.
   asked=None
   while read_rpc(proc.poll) is None and time.time()<deadline:
    request=stop() if stop is not None else None
    if request:
     if mode!='train' or request=='now' or (asked is not None and time.time()-asked>=self.stop_grace_seconds):halt(stop)
     if asked is None:
      asked=time.time()
      try:read_rpc(lambda:sb.filesystem.write_text('stop\n',STOP_REQUEST))
      except Exception:halt(stop)
    time.sleep(.5)
   if read_rpc(proc.poll) is None:raise RuntimeError('finite worker deadline reached')
   (out/'stdout.log').write_text(proc.stdout.read());(out/'stderr.log').write_text(proc.stderr.read())
   if proc.returncode:raise RuntimeError('worker failed: '+(out/'stderr.log').read_text()[-4000:])
  except BaseException as exc:error=type(exc).__name__+': '+str(exc)
  finally:
   if never is not None:
    # Already terminated and settled at 0 from the provider's record; nothing ran, nothing to collect.
    dump(out/'settlement.json',{'provider':never.receipt,'error':error,'never_started':True,'charged_seconds':0,'termination_requested':True})
   elif sb:
    try:sb.terminate(wait=True)
    except BaseException as exc:error=(error or '')+' termination '+str(exc)
    for _ in range(40):
     receipt=allocation_info(self.app.app_id,sb.object_id)
     if receipt and receipt.get('finished_at'):break
     time.sleep(.25)
   if never is None:
    if receipt and receipt.get('started_at') and receipt.get('finished_at'):ledger.settle(name,receipt,status='failed' if error else 'complete',error=error)
    else:ledger.record(name,status='uncertain',error=error or 'provider end unavailable')
    dump(out/'settlement.json',{'provider':receipt,'error':error,'termination_requested':sb is not None,'bulk_download_after_gpu_shutdown':True})
  if never is not None:raise never
  # Collect failed outputs too. An unknown worker end retains its full hold.
  if receipt and receipt.get('finished_at'):
   try:
    target=out/'retained';target.mkdir()
    # Large full-world checkpoints stay durable on the private Volume. The
    # control plane needs weights/source plus their worker-computed commitments;
    # it must not wait on a desktop bulk transfer to keep using a finished run.
    download_atomic(volume,'/output-manifest.json',target/'output-manifest.json')
    m=json.loads((target/'output-manifest.json').read_text());remote={}
    for entry in volume.iterdir('/',recursive=True):
     name=entry.path.lstrip('/')
     if entry.type.name!='FILE' or name not in m['files']:continue
     if entry.size>8*1024**2:
      remote[name]={'volume':vname,'path':entry.path,'sha256':m['files'][name],'bytes':entry.size};continue
     download_atomic(volume,entry.path,target/name)
    if any(sha(target/n)!=h for n,h in m['files'].items() if n not in remote):raise ValueError('collected artifact hash differs')
    dump(out/'remote-artifacts.json',remote)
    dump(out/'collection.json',{'local_small_artifact_hashes_valid':True,'large_artifacts_retained_remotely':remote,'large_hashes_computed_by_completed_worker':True,'worker_complete':m['complete'],'provider_finished':True})
    if mode=='train' and (target/'output/artifact/result.json').exists():
     shutil.copytree(target/'output/artifact',out/'artifact');result=json.loads((out/'artifact/result.json').read_text())
     checkpoint='output/artifact/checkpoint.pt'
     if checkpoint in remote:
      result.update(checkpoint_sha256=m['files'][checkpoint],checkpoint_retained_remotely=remote[checkpoint]);dump(out/'artifact/result.json',result)
   except BaseException as exc:
    # A stopped or crashed worker may leave no manifest. Its own failure is the one to report.
    if error:raise RuntimeError(error+'; worker output not collected ('+type(exc).__name__+')') from None
    raise
  if error:raise RuntimeError(error)
  if not receipt or not receipt.get('finished_at'):raise RuntimeError('unresolved allocation; hold remains')
  return {'result':result,'provider':receipt,'retained':str(out/'retained')}
 def recover(self,ledger,name):
  """Settle an allocation whose end was unknown when its worker thread left. Its artifact is never adopted.

  The recorded sandbox is terminated now. One the provider never started is settled at 0 from its record; one
  that started is charged from its start/finish receipt. Without a provider record the full hold stays.
  """
  import modal
  from simulators.generals.autoresearch.provider import allocation_info
  row=next(r for r in ledger.read()['allocations'] if r['name']==name)
  if 'charged_seconds' in row:return {'resolved':True,'adopted':False}
  if not row.get('sandbox_id') or not row.get('app_id'):return {'resolved':False,'hold_retained':True,'reason':'no provider identity was recorded; the full hold stays'}
  receipt=workers.abandon(lambda:modal.Sandbox.from_id(row['sandbox_id']).terminate(),lambda:allocation_info(row['app_id'],row['sandbox_id']),poll=self.abandon_poll_seconds)
  prior=row.get('error') or 'provider end unavailable'
  if receipt.get('provider_id')==row['sandbox_id'] and workers.never_started(receipt):
   ledger.settle_never_started(name,receipt,error=prior+'; '+workers.NEVER_STARTED)
   return {'resolved':True,'adopted':False,'never_started':True,'charged_seconds':0,'provider':receipt}
  if not receipt.get('started_at') or not receipt.get('finished_at'):return {'resolved':False,'hold_retained':True}
  ledger.settle(name,receipt,status='failed',error=prior+'; settled later from the provider receipt, artifact not adopted')
  return {'resolved':True,'adopted':False,'provider':receipt}
 def cpu_grade(self,bundle,schedule,opponents,out,seconds=600,split='public-development',diagnostics=True,stop=None):
  import modal
  from simulators.generals.autoresearch.provider import wait_process,read_rpc,allocation_info
  sb=None;out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
  try:
   halt(stop)
   sb=modal.Sandbox.create('/bin/sleep','86400',app=self.app,image=self.image,cpu=(8,8),memory=(32768,32768),timeout=seconds,block_network=True,secrets=[],include_oidc_identity_token=False,environment_name=ENVIRONMENT,tags={'clash_fast_starter':'public-practice'})
   dump(out.with_suffix('.provider-start.json'),{'app_id':self.app.app_id,'sandbox_id':sb.object_id,'cpu_only':True})
   sb.filesystem.make_directory('/private');wait_process(sb.exec('chmod','700','/private',timeout=10))
   for label,root in [('candidate',Path(bundle)),*[(n,Path(p)) for n,p in opponents.items()]]:
    sb.filesystem.make_directory('/private/'+label)
    for f in root.iterdir():
     if f.is_file():sb.filesystem.write_bytes(f.read_bytes(),'/private/'+label+'/'+f.name)
   value=json.loads(Path(schedule).read_text())
   for actor in value['opponents']:
    if actor['kind']=='checkpoint':actor['bundle']='/private/'+actor['name']
   sb.filesystem.write_text(json.dumps(value),'/private/schedule.json')
   module='evaluation_v3_parallel' if self.pair_workers==3 else 'evaluation_v3'
   extra=',pair_workers=3' if self.pair_workers==3 else ''
   code=f"from simulators.clash.autoresearch.{module} import evaluate;evaluate('/private/candidate','/private/schedule.json','/private/result.json',development=False,split={split!r},interface={self.interface!r},max_seconds={seconds-40}{extra})"
   proc=sb.exec('python','-u','-c',code,timeout=seconds-20)
   while read_rpc(proc.poll) is None:
    halt(stop);time.sleep(.05)
   if proc.returncode:raise RuntimeError('public practice failed: '+proc.stderr.read()[-4000:])
   value=json.loads(read_rpc(lambda:sb.filesystem.read_text('/private/result.json')));dump(out,value);return value
  finally:
   if sb:
    sb.terminate(wait=True);receipt=allocation_info(self.app.app_id,sb.object_id)
    for _ in range(40):
     if receipt and receipt.get('finished_at'):break
     time.sleep(.25);receipt=allocation_info(self.app.app_id,sb.object_id)
    dump(out.with_suffix('.provider.json'),{'cpu_only':True,'provider':receipt})
