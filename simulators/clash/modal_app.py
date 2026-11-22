"""Bounded Clash-only Modal staging experiments. Never touches other apps/jobs."""
from pathlib import Path
import json,os,time,tarfile,io,subprocess
import modal
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent.parent
OUT=ROOT/'evidence/clash-implementation/cuda-campaign'
BUNDLE=OUT/'upload.tar.gz'
APP_NAME='clash-fixed8-cuda-pilot-20261108'
HARD_SECONDS=900
RESERVED_SECONDS=1200
RATE=.000222+4*.0000131+16*.00000222
CAP_USD=12.;CAP_GPU_SECONDS=7200.
image=(modal.Image.debian_slim(python_version='3.12')
 .apt_install('curl','git','ca-certificates')
 .run_commands('curl -fL --retry 1 https://github.com/adoptium/temurin21-binaries/releases/download/jdk-21.0.9%2B10/OpenJDK21U-jre_x64_linux_hotspot_21.0.9_10.tar.gz -o /opt/java.tgz && mkdir /opt/java && tar -xzf /opt/java.tgz -C /opt/java --strip-components=1 && sha256sum /opt/java.tgz')
 .pip_install('torch==2.9.1','numpy==2.5.4','JPype1==1.7.1','gymnasium==1.4.0','pettingzoo==1.27.0')
 .env({'JAVA_HOME':'/opt/java','PYTHONPATH':'/workspace/candidates/crforge/python:/workspace/integrations/clash','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','CUBLAS_WORKSPACE_CONFIG':':4096:8'})
 .add_local_file(BUNDLE,'/opt/upload.tar.gz'))
app=modal.App(APP_NAME)
@app.function(image=image,gpu='L4',cpu=(4,4),memory=(16384,16384),max_containers=1,min_containers=0,buffer_containers=0,scaledown_window=2,retries=0,timeout=HARD_SECONDS,startup_timeout=180)
def execute(stage:str):
 import time,os,subprocess,sys,tarfile,io,json
 start=time.perf_counter();root=Path('/workspace');root.mkdir(exist_ok=True)
 with tarfile.open('/opt/upload.tar.gz') as tar:tar.extractall(root,filter='data')
 os.chdir(root)
 folder=root/'evidence/clash-implementation/cuda-campaign'/stage;folder.mkdir(parents=True,exist_ok=False)
 smi=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,driver_version,memory.total','--format=csv,noheader'],text=True).strip()
 print('GPU',smi,flush=True)
 monitor=(folder/'nvidia-smi.csv').open('w')
 mon=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,power.draw','--format=csv,noheader','--loop-ms=1000'],stdout=monitor)
 code=0
 (folder/'pip-freeze.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
 (folder/'cpuinfo.txt').write_text(Path('/proc/cpuinfo').read_text())
 (folder/'cgroup-limits.json').write_text(json.dumps({name:Path('/sys/fs/cgroup/'+name).read_text() if Path('/sys/fs/cgroup/'+name).exists() else None for name in ('cpu.max','memory.max')},indent=2))
 try:
  with (folder/'driver.log').open('w') as log:
   proc=subprocess.Popen([sys.executable,'-u',str(root/'integrations/clash/cuda_experiments.py'),stage,str(folder)],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
   for line in proc.stdout:print(line,end='',flush=True);log.write(line);log.flush()
   code=proc.wait()
 finally:
  mon.terminate();mon.wait();monitor.close()
  elapsed=time.perf_counter()-start
  (folder/'function-receipt.json').write_text(json.dumps({'stage':stage,'function_body_seconds':elapsed,'conservative_load_headroom_seconds':60,'estimated_usd_with_headroom':(elapsed+60)*RATE,'rate_usd_per_second':RATE,'gpu':'L4','cpu_physical_cores':4,'ram_gib':16,'nvidia_smi':smi,'exit_code':code,'scope':'function duration estimate; billing API reconciliation separate'},indent=2))
  buf=io.BytesIO()
  with tarfile.open(fileobj=buf,mode='w:gz') as tar:tar.add(folder,arcname=stage)
 return {'exit_code':code,'seconds':elapsed,'archive':buf.getvalue()}

def bundle():
 OUT.mkdir(parents=True,exist_ok=True)
 with tarfile.open(BUNDLE,'w:gz') as tar:
  engine=ROOT/'candidates/crforge'
  names=subprocess.check_output(['git','ls-files','-z'],cwd=engine).decode().split('\0')
  for name in names:
   if name and (engine/name).is_file():tar.add(engine/name,arcname='candidates/crforge/'+name)
  tar.add(engine/'.git',arcname='candidates/crforge/.git')
  tar.add(engine/'env/build/install/env',arcname='candidates/crforge/env/build/install/env')
  tar.add(ROOT/'.cache-crforge/tables/16.635.8',arcname='.cache-crforge/tables/16.635.8')
  for p in HERE.rglob('*'):
   if p.is_file() and '__pycache__' not in p.parts:tar.add(p,arcname='integrations/clash/'+str(p.relative_to(HERE)))
  paired_checkpoint=OUT/'paired/trimmed-0/checkpoint.pt'
  if paired_checkpoint.exists():tar.add(paired_checkpoint,arcname='evidence/clash-implementation/cuda-campaign/paired/trimmed-0/checkpoint.pt')
  tar.add(ROOT/'evidence/clash-implementation/final-validation/training/checkpoint.pt',arcname='evidence/clash-implementation/final-validation/training/checkpoint.pt')
  tar.add(ROOT/'evidence/clash-implementation/pinned-env-dist.tar.gz',arcname='evidence/clash-implementation/pinned-env-dist.tar.gz')

@app.local_entrypoint()
def main(stage:str='probe'):
 from modal.config import config
 import fcntl
 launch_lock=(OUT/'launch.lock').open('a');fcntl.flock(launch_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if config.get('environment')!='staging':raise RuntimeError('staging only')
 if stage not in ('probe','paired','learning1','learning2','audit','learning-entropy'):raise ValueError('bounded stage matrix only')
 ledger=OUT/'cost-ledger.json'
 state=json.loads(ledger.read_text()) if ledger.exists() else {'cap_usd':CAP_USD,'cap_gpu_seconds':CAP_GPU_SECONDS,'rate_usd_per_second':RATE,'price_url':'https://modal.com/pricing','price_checked':'2026-11-08','calls':[],'build_cost_reserve_usd':.5}
 if any(x['status'] in ('submitted','reserved') for x in state['calls']):raise RuntimeError('another Clash stage is outstanding; finish/reconcile it before launching')
 reserved=sum(x['reserved_seconds'] for x in state['calls'])
 cost_reserved=state['build_cost_reserve_usd']+sum(x['reserved_usd'] for x in state['calls'])
 if reserved+RESERVED_SECONDS>CAP_GPU_SECONDS or cost_reserved+RESERVED_SECONDS*RATE+.05>CAP_USD:raise RuntimeError('Clash cap exhausted')
 if any(x['stage']==stage for x in state['calls']):raise RuntimeError('stage already attempted; bounded zero automatic retries')
 row={'stage':stage,'reserved_seconds':RESERVED_SECONDS,'reserved_usd':RESERVED_SECONDS*RATE+.05,'submitted_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'app_id':app.app_id,'status':'reserved'}
 state['calls'].append(row);ledger.write_text(json.dumps(state,indent=2)+'\n')
 call=execute.spawn(stage);row['function_call_id']=call.object_id;row['status']='submitted';ledger.write_text(json.dumps(state,indent=2)+'\n');print(json.dumps(row),flush=True)
 result=call.get();row.update(status='returned',function_body_seconds=result['seconds'],exit_code=result['exit_code'])
 archive=OUT/(stage+'.tar.gz');archive.write_bytes(result['archive'])
 with tarfile.open(archive) as tar:tar.extractall(OUT,filter='data')
 row['function_estimate_usd']=(result['seconds']+60)*RATE;ledger.write_text(json.dumps(state,indent=2)+'\n')
 print(json.dumps({k:v for k,v in row.items()}),flush=True)
