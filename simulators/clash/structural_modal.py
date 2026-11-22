"""Clash structural research, newly authorized on 2026-11-08.

Historical caps/attempts remain immutable in cuda-campaign. Each new call is
bounded, zero-retry, single-container and fully receipted without an obsolete
campaign spend ceiling. Only this app's jobs are managed here.
"""
from pathlib import Path
import json,os,time,tarfile,io,subprocess,re
import modal
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent.parent
OUT=ROOT/'evidence/clash-implementation/structural-20261108'
BUNDLE=OUT/'upload.tar.gz'
STATIC=OUT/'static-base.tar.gz'
APP_NAME='clash-structural-20261108'
RATE=.000222+4*.0000131+16*.00000222
def bundle():
 import hashlib,shutil,tempfile
 OUT.mkdir(parents=True,exist_ok=True)
 # The first complete upload is immutable. Reuse its provider blob and overlay
 # only game-owned code on later stages, instead of retransmitting pinned jars,
 # tables, Git metadata and starting checkpoints for every source change.
 if not STATIC.exists():
  legacy=OUT/'upload.tar.gz'
  if not legacy.exists():raise RuntimeError('restore the retained first full upload before creating the immutable static base')
  shutil.copyfile(legacy,STATIC);STATIC.chmod(0o444)
  (OUT/'static-base-receipt.json').write_text(json.dumps({'sha256':hashlib.sha256(STATIC.read_bytes()).hexdigest(),'bytes':STATIC.stat().st_size,'origin':'retained first structural full upload; pinned runtime/data/source/start checkpoints'},indent=2)+'\n')
 lock=json.loads((HERE/'rules.lock.json').read_text());engine=ROOT/'candidates/crforge'
 for name,want in lock['engine_source_sha256'].items():
  if hashlib.sha256((engine/name).read_bytes()).hexdigest()!=want:raise ValueError('engine drift requires a separately qualified static artifact: '+name)
 uploads=OUT/'uploads';uploads.mkdir(exist_ok=True)
 descriptor,temporary=tempfile.mkstemp(prefix='overlay-',suffix='.tar.gz',dir=uploads);os.close(descriptor)
 with tarfile.open(temporary,'w:gz',compresslevel=1) as tar:
  for path in HERE.rglob('*'):
   if path.is_file() and not any(part in ('__pycache__','.build') or part.startswith('generated') for part in path.parts):tar.add(path,arcname='integrations/clash/'+str(path.relative_to(HERE)))
  from native.original_baseline_bundle import add_original_to_tar
  add_original_to_tar(tar,ROOT)
  tape_root=ROOT/'evidence/clash-implementation/native-performance-20261108/cpu-baseline-1/generation'
  for name in ('tape.npz','result.json'):tar.add(tape_root/name,arcname=str((tape_root/name).relative_to(ROOT)))
  fixture=ROOT/'evidence/clash-implementation/native-pathfinding-20261108/gpu-features-2/java-public-observations.pt'
  tar.add(fixture,arcname=str(fixture.relative_to(ROOT)))
  neural_fixture=ROOT/'evidence/clash-implementation/native-training-20261108/compact-embedding-cpu-3/real-observations.npz'
  tar.add(neural_fixture,arcname=str(neural_fixture.relative_to(ROOT)))

 payload=Path(temporary).read_bytes();digest=hashlib.sha256(payload).hexdigest();target=uploads/(digest+'.tar.gz')
 if target.exists():
  if hashlib.sha256(target.read_bytes()).hexdigest()!=digest:raise ValueError('content-addressed transport archive drift')
  Path(temporary).unlink()
 else:os.replace(temporary,target);target.chmod(0o444)
 return target

# Freeze before Modal hydrates file mounts; each transport is immutable and content-addressed.
if modal.is_local():BUNDLE=bundle()
image=(modal.Image.from_registry('nvidia/cuda:12.8.1-devel-ubuntu24.04',add_python='3.12')
 .apt_install('curl','git','ca-certificates')
 .run_commands('curl -fL --retry 1 https://github.com/adoptium/temurin21-binaries/releases/download/jdk-21.0.9%2B10/OpenJDK21U-jre_x64_linux_hotspot_21.0.9_10.tar.gz -o /opt/java.tgz && mkdir /opt/java && tar -xzf /opt/java.tgz -C /opt/java --strip-components=1 && sha256sum /opt/java.tgz')
 .pip_install('torch==2.9.1','numpy==2.5.4','JPype1==1.7.1','gymnasium==1.4.0','pettingzoo==1.27.0')
 .env({'JAVA_HOME':'/opt/java','PYTHONPATH':'/workspace/candidates/crforge/python:/workspace/integrations/clash','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','CUBLAS_WORKSPACE_CONFIG':':4096:8'})
 .add_local_file(STATIC,'/opt/static-base.tar.gz')
 .add_local_file(BUNDLE,'/opt/upload.tar.gz'))
app=modal.App(APP_NAME)
VOLUME_NAME='clash-structural-artifacts-20261108'
volume=modal.Volume.from_name(VOLUME_NAME,create_if_missing=True)

def worker(stage:str,run_id:str,cpu_cores:int,ram_gib:int,gpu_model:str="L4",timeout_seconds:int=1800,expected_overlay_sha:str=""):
 import sys,traceback
 start=time.perf_counter();rate=({'L4':.000222,'H100':.001097}[gpu_model])+cpu_cores*.0000131+ram_gib*.00000222
 os.environ['CLASH_CPU_CORES']=str(cpu_cores);os.environ['CLASH_RAM_GIB']=str(ram_gib);os.environ['CLASH_NATIVE_BUILD_CACHE']='/artifacts/native-builds'
 import hashlib
 actual_overlay_sha=hashlib.sha256(Path('/opt/upload.tar.gz').read_bytes()).hexdigest()
 if not expected_overlay_sha or actual_overlay_sha!=expected_overlay_sha:raise RuntimeError('mounted source archive differs from the exact launcher binding')
 root=Path('/workspace');root.mkdir(exist_ok=True)
 for archive in ('/opt/static-base.tar.gz','/opt/upload.tar.gz'):
  with tarfile.open(archive) as tar:tar.extractall(root,filter='data')
 os.chdir(root)
 base=root/'evidence/clash-implementation/structural-20261108/baseline'
 for name in ('candidates','.cache-crforge','evidence'):(base/name).symlink_to(root/name,target_is_directory=True)
 folder=Path('/artifacts')/run_id;folder.mkdir(parents=True,exist_ok=False)
 # Preserve the exact mounted overlay for every attempt, including failures.
 # The changing local upload is transport state, never the historical record.
 import hashlib,shutil
 shutil.copyfile('/opt/upload.tar.gz',folder/'overlay-source.tar.gz')
 payload=(folder/'overlay-source.tar.gz').read_bytes()
 (folder/'overlay-source-receipt.json').write_text(json.dumps({'sha256':hashlib.sha256(payload).hexdigest(),'bytes':len(payload),'origin':'exact immutable overlay mounted in this allocation','static_base':'immutable static-base.tar.gz retained separately'},indent=2)+'\n')
 (folder/'started.json').write_text(json.dumps({'stage':stage,'run_id':run_id,'gpu':gpu_model,'cpu_physical_cores':cpu_cores,'ram_gib':ram_gib,'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())},indent=2)+'\n');volume.commit()
 (folder/'pip-freeze.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
 (folder/'cpuinfo.txt').write_text(Path('/proc/cpuinfo').read_text())
 (folder/'cgroup-limits.json').write_text(json.dumps({name:Path('/sys/fs/cgroup/'+name).read_text() if Path('/sys/fs/cgroup/'+name).exists() else None for name in ('cpu.max','memory.max')},indent=2))
 smi=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,driver_version,memory.total','--format=csv,noheader'],text=True).strip()
 monitor=(folder/'nvidia-smi.csv').open('w')
 mon=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,power.draw','--format=csv,noheader','--loop-ms=1000'],stdout=monitor)
 code=-1
 try:
  with (folder/'driver.log').open('w') as log:
   proc=subprocess.Popen([sys.executable,'-u',str(root/'integrations/clash/structural_experiments.py'),stage,str(folder)],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
   try:
    for line in proc.stdout:print(line,end='',flush=True);log.write(line);log.flush()
    code=proc.wait(timeout=timeout_seconds-150)
   finally:
    if proc.poll() is None:proc.kill();proc.wait()
 finally:
  mon.terminate();mon.wait();monitor.close()
  elapsed=time.perf_counter()-start
  (folder/'function-receipt.json').write_text(json.dumps({'stage':stage,'run_id':run_id,'function_body_seconds':elapsed,'conservative_load_headroom_seconds':60,'estimated_usd_with_headroom':(elapsed+60)*rate,'rate_usd_per_second':rate,'gpu':gpu_model,'cpu_physical_cores':cpu_cores,'ram_gib':ram_gib,'nvidia_smi':smi,'exit_code':code,'scope':'full allocated function estimate; provider billing separate'},indent=2))
  # Persist privately before returning a small control-plane result. Bulk source
  # archives/checkpoints must not hold the controller or a GPU open during a
  # slow desktop download. Volume commit and setup remain allocated-worker time.
  volume.commit()
 settlement={'allocated_body_seconds_including_commit':time.perf_counter()-start,'cpu_physical_cores':cpu_cores,'ram_gib':ram_gib,'gpu':gpu_model,'exit_code':code}
 (folder/'function-settlement.json').write_text(json.dumps(settlement,indent=2)+'\n');volume.commit()
 summary=json.loads((folder/'summary.json').read_text()) if (folder/'summary.json').exists() else {}
 return {'exit_code':code,'seconds':time.perf_counter()-start,'summary':summary,'volume':VOLUME_NAME,'remote_directory':run_id,'settlement':settlement,'overlay_sha256':actual_overlay_sha}

@app.function(image=image,gpu='L4',cpu=(4,4),memory=(16384,16384),volumes={'/artifacts':volume},max_containers=1,min_containers=0,buffer_containers=0,scaledown_window=2,retries=0,timeout=1800,startup_timeout=180)
def execute(stage:str,run_id:str,overlay_sha256:str):return worker(stage,run_id,4,16,expected_overlay_sha=overlay_sha256)

@app.function(image=image,gpu='L4',cpu=(16,16),memory=(65536,65536),volumes={'/artifacts':volume},max_containers=1,min_containers=0,buffer_containers=0,scaledown_window=2,retries=0,timeout=1800,startup_timeout=180)
def execute16(stage:str,run_id:str,overlay_sha256:str):return worker(stage,run_id,16,64,expected_overlay_sha=overlay_sha256)

@app.function(image=image,gpu='L4',cpu=(32,32),memory=(131072,131072),volumes={'/artifacts':volume},max_containers=1,min_containers=0,buffer_containers=0,scaledown_window=2,retries=0,timeout=1800,startup_timeout=180)
def execute32(stage:str,run_id:str,overlay_sha256:str):return worker(stage,run_id,32,128,expected_overlay_sha=overlay_sha256)

@app.function(image=image,gpu='H100',cpu=(32,32),memory=(131072,131072),volumes={'/artifacts':volume},max_containers=1,min_containers=0,buffer_containers=0,scaledown_window=2,retries=0,timeout=1800,startup_timeout=180)
def execute_h100(stage:str,run_id:str,overlay_sha256:str):return worker(stage,run_id,32,128,'H100',expected_overlay_sha=overlay_sha256)

@app.function(image=image,gpu='H100',cpu=(32,32),memory=(131072,131072),volumes={'/artifacts':volume},max_containers=1,min_containers=0,buffer_containers=0,scaledown_window=2,retries=0,timeout=3600,startup_timeout=180)
def execute_h100_research(stage:str,run_id:str,overlay_sha256:str):return worker(stage,run_id,32,128,'H100',3600,expected_overlay_sha=overlay_sha256)

@app.function(image=image,cpu=(1,1),memory=(4096,4096),volumes={'/artifacts':volume},max_containers=1,min_containers=0,scaledown_window=2,retries=0,timeout=600)
def analyze_retained(run_id:str,recover_call_id:str=''):
 import sys
 if recover_call_id:
  result=modal.FunctionCall.from_id(recover_call_id).get()
  folder=Path('/artifacts')/run_id
  if folder.exists():raise ValueError('recovered artifact directory already exists')
  with tarfile.open(fileobj=io.BytesIO(result['archive'])) as archive:archive.extractall('/artifacts',filter='data')
  (folder/'recovery-settlement.json').write_text(json.dumps({'original_function_seconds':result['seconds'],'original_exit_code':result['exit_code'],'recovery':'original successful result, no repeat training','function_call_id':recover_call_id},indent=2)+'\n')
  volume.commit()
 sys.path.insert(0,'/workspace/integrations/clash')
 # This helper runs without the research worker unpacking its source overlay.
 for archive in ('/opt/static-base.tar.gz','/opt/upload.tar.gz'):
  with tarfile.open(archive) as source:source.extractall('/workspace',filter='data')
 from artifact_analysis import analyze
 folder=Path('/artifacts')/run_id;summary=analyze(folder)
 target=folder/'analysis-v1.json'
 if not target.exists():target.write_text(json.dumps(summary,indent=2)+'\n');volume.commit()
 return summary

@app.local_entrypoint()
def main(stage:str='probe',run_id:str='probe-1',cpu_cores:int=4,gpu_model:str='L4',recover_call_id:str=''):
 from modal.config import config
 import fcntl
 if config.get('environment')!='staging':raise RuntimeError('staging only')
 if stage=='analyze':
  if not re.fullmatch('[a-z0-9-]+',run_id):raise ValueError('invalid artifact directory')
  target=OUT/run_id/'analysis-v1.json'
  if target.exists():raise ValueError('analysis receipt already retained')
  summary=analyze_retained.remote(run_id,recover_call_id)
  target.parent.mkdir(exist_ok=True);target.write_text(json.dumps(summary,indent=2)+'\n')
  print(json.dumps(summary),flush=True);return
 if stage not in ('probe','validate','sim','learning','graphs','scaling','native-scaling','allocation-baseline','allocation-graph','native-cuda','native-rng','native-rng-world','native-training','native-profile','native-baselines','native-optimized','native-kernel-sweep','native-bucket','native-codec','native-learner-modes','native-training-benchmark','native-batch-scaling','native-hardware','native-completion','native-kernel-variants','native-detailed-profile','native-packed-study'):raise ValueError('unknown bounded experiment stage')
 if gpu_model not in ('L4','H100') or (gpu_model=='H100' and cpu_cores!=32):raise ValueError('H100 comparison uses 32 physical CPU cores and 128 GiB RAM')
 if stage=='native-packed-study' and (gpu_model!='H100' or cpu_cores!=32):raise ValueError('packed-head qualification matrix requires H100/32-core/128-GiB')
 if cpu_cores not in (4,16,32):raise ValueError('supported allocations: 4/16 GiB, 16/64 GiB, 32/128 GiB CPU/RAM')
 if not re.fullmatch('[a-z0-9-]+',run_id):raise ValueError('invalid run id')
 launch_lock=(OUT/'launch.lock').open('a');fcntl.flock(launch_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 ledger=OUT/'cost-ledger.json'
 state=json.loads(ledger.read_text()) if ledger.exists() else {'authorization':'2026-11-08 user removed previous dollar/time/attempt ceilings; preserve old campaign unchanged','allocation':{'gpu':'L4','cpu_physical_cores':4,'ram_gib':16},'calls':[]}
 if any(x['status'] in ('submitted','reserved','needs_reconciliation') for x in state['calls']):raise RuntimeError('finish/reconcile the outstanding Clash call first')
 if (OUT/run_id).exists() or any(x['run_id']==run_id for x in state['calls']):raise ValueError('new run id required to preserve evidence')
 row={'stage':stage,'run_id':run_id,'submitted_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'app_id':app.app_id,'status':'reserved','overlay_sha256':BUNDLE.name.removesuffix('.tar.gz'),'function_timeout_seconds':3600 if stage=='native-packed-study' else 1800,'allocation':{'gpu':gpu_model,'cpu_physical_cores':cpu_cores,'ram_gib':{4:16,16:64,32:128}[cpu_cores]}}
 state['calls'].append(row);ledger.write_text(json.dumps(state,indent=2)+'\n')
 try:call=(execute_h100_research if stage=='native-packed-study' else execute_h100 if gpu_model=='H100' else {4:execute,16:execute16,32:execute32}[cpu_cores]).spawn(stage,run_id,BUNDLE.name.removesuffix('.tar.gz'))
 except Exception as exc:
  row.update(status='needs_reconciliation',error=f'{type(exc).__name__}: {exc}')
  ledger.write_text(json.dumps(state,indent=2)+'\n');raise
 row['function_call_id']=call.object_id;row['status']='submitted';ledger.write_text(json.dumps(state,indent=2)+'\n');print(json.dumps(row),flush=True)
 result=call.get();valid_result=bool(result.get('summary')) and result.get('overlay_sha256')==BUNDLE.name.removesuffix('.tar.gz')
 row['result_binding_valid']=valid_result
 row.update(status='returned',function_body_seconds=result['seconds'],exit_code=result['exit_code'])
 folder=OUT/run_id;folder.mkdir(exist_ok=False)
 (folder/'summary.json').write_text(json.dumps(result['summary'],indent=2)+'\n')
 (folder/'function-settlement.json').write_text(json.dumps(result['settlement'],indent=2)+'\n')
 row.update(artifact_volume=result['volume'],artifact_directory=result['remote_directory'],artifact_status='committed privately; bulk download independent of GPU allocation')
 rate={'L4':.000222,'H100':.001097}[gpu_model]+cpu_cores*.0000131+{4:16,16:64,32:128}[cpu_cores]*.00000222
 row['function_estimate_usd_with_load_headroom']=(result['seconds']+60)*rate
 # Merge with an independently completed artifact recovery rather than replacing it.
 with (OUT/'ledger-update.lock').open('a') as update_lock:
  fcntl.flock(update_lock,fcntl.LOCK_EX)
  state=json.loads(ledger.read_text());next(x for x in state['calls'] if x['run_id']==run_id).update(row)
  ledger.write_text(json.dumps(state,indent=2)+'\n')
 print(json.dumps(row),flush=True)
 if result['exit_code']==0 and not valid_result:raise RuntimeError('successful worker returned no stage result or a mismatched source binding; attempt retained as invalid')
