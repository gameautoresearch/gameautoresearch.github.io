"""Five-hour November9 campaign. Explicit uncapped ledger; finite detached stages."""
from __future__ import annotations
import argparse,hashlib,json,os,shutil,tarfile,time
from pathlib import Path
from datetime import datetime
from tools.open_credit_campaign import OpenCreditBudget
from .remote import provider_info,download_tree

ROOT=Path(__file__).resolve().parents[3]
CAMPAIGN=ROOT/'evidence/strong-bots-20261109-0516Z'
BASE=CAMPAIGN/'pokemon/rl'
APP=VOLUME='pokemon-strong-bots-20261109-0516Z'
IMAGE='im-JwNTde85zO3cAKA9qnqInr'
PRIOR=ROOT/'evidence/reference-readiness-20261109-0251Z/pokemon/rl'
INCUMBENT=PRIOR/'jobs/matched-selfplay-2718/collected/training/checkpoint.pt'
ORIGINAL_RELEASE=PRIOR/'releases/3691ae6d6dd3fa63ee768e8718bb812939de1bd296106ec3db7e267c349b4bee'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(path,value):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');t.replace(p)
def budget():return OpenCreditBudget('pokemon',CAMPAIGN/'campaign.sqlite3')
def epoch(v):return datetime.fromisoformat(v.replace('Z','+00:00')).timestamp()
def own_usage():
 rows=[x for x in budget().snapshot()['operations'] if x['game']=='pokemon' and x['kind'].startswith('pokemon/rl/')]
 return {'settled_gpu_seconds':sum(x['gpu_charged'] for x in rows if x['status']=='settled'),'held_gpu_seconds':sum(x['gpu_hold'] for x in rows if x['status']!='settled'),'settled_resource_usd':sum(x['usd_charged'] for x in rows if x['status']=='settled'),'held_resource_usd':sum(x['usd_hold'] for x in rows if x['status']!='settled'),'model_usd':0}
def coordinate(phase,**fields):
 p=CAMPAIGN/'pokemon/coordination/rl.json';v=json.loads(p.read_text());v.update(phase=phase,updated_at_unix=time.time(),budget_use=own_usage(),**fields);write(p,v)
def counterpart():
 p=CAMPAIGN/'pokemon/coordination/evaluator.json';v=json.loads(p.read_text());assert v['source_interface']['observations']==700 and v['source_interface']['actions']==14;return v

def noncompute_hold():
 path=BASE/'storage-hold.json'
 if not path.exists():
  op=budget()._reserve('pokemon/rl/noncompute-storage-and-egress','resource',0,.25,{'purpose':'build','max_seconds':0,'liability_kind':'unpriced-private-artifact-storage-egress','retain_until_invoice_reconciled':True})
  write(path,{'operation':op,'usd_hold':.25,'provider_interval':None,'status':'unknown noncompute cost; hold retained'})

def freeze_original():
 target=BASE/'original-continuation-release'
 if target.exists():return target
 target.mkdir(parents=True,exist_ok=False)
 shutil.copyfile(ORIGINAL_RELEASE/'source.tar',target/'original-source.tar')
 shutil.copyfile(ORIGINAL_RELEASE/'release.json',target/'original-release.json')
 shutil.copyfile(Path(__file__).with_name('strong_entry.py'),target/'strong_entry.py')
 for name in ('open_credit_campaign.py','overnight_budget.py'):
  (target/'tools').mkdir(exist_ok=True);shutil.copyfile(ROOT/'tools'/name,target/'tools'/name)
 write(target/'manifest.json',{'original_reference_release':str(ORIGINAL_RELEASE),'files':{str(p.relative_to(target)):sha(p) for p in target.rglob('*') if p.is_file()},'purpose':'Exact compatible full optimizer/RNG/population continuation; host guard modules retained; new campaign ledger only'})
 for p in target.rglob('*'):
  if p.is_file():p.chmod(0o444)
 return target

def launch_original(label,seconds=3600,parent=INCUMBENT):
 import modal
 counterpart();release=freeze_original();b=budget();seconds=min(seconds,int(epoch(b.auth['selection_close_utc'])-time.time())-40)
 if seconds<150:raise ValueError('Insufficient selection window')
 target=BASE/'jobs'/label;target.mkdir(parents=True,exist_ok=False)
 parent=Path(parent);request={'label':label,'mode':'original-continuation','max_seconds':seconds,'selection_close_unix':epoch(b.auth['selection_close_utc']),'worker_stop_unix':epoch(b.auth['worker_stop_utc']),'parent_sha256':sha(parent),'original_source_tar_sha256':sha(release/'original-source.tar'),'source_release':str(release),'campaign_id':b.auth['campaign_id']}
 write(target/'request.json',request);noncompute_hold()
 operation=b.reserve_worker('pokemon/rl/'+label,seconds,gpu='L4',cpu_cores=4,memory_gib=8,purpose='training');write(target/'reservation.json',{'operation':operation,'max_seconds':seconds})
 volume=modal.Volume.from_name(VOLUME,environment_name='staging',create_if_missing=True,version=2)
 with volume.batch_upload() as upload:
  for file in ('original-source.tar','original-release.json','strong_entry.py','manifest.json'):upload.put_file(release/file,'/jobs/'+label+'/input/'+file)
  for file in ('open_credit_campaign.py','overnight_budget.py'):upload.put_file(release/'tools'/file,'/jobs/'+label+'/input/tools/'+file)
  upload.put_file(parent,'/jobs/'+label+'/input/parent.pt');upload.put_file(target/'request.json','/jobs/'+label+'/request.json')
 app=modal.App.lookup(APP,environment_name='staging',create_if_missing=True)
 sb=modal.Sandbox.create('python','/runs/jobs/'+label+'/input/strong_entry.py','/runs/jobs/'+label+'/request.json',app=app,image=modal.Image.from_id(IMAGE),gpu='L4',cpu=(4,4),memory=(8192,8192),timeout=seconds-25,volumes={'/runs':volume},env={'CUBLAS_WORKSPACE_CONFIG':':4096:8','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'},block_network=True,secrets=[],include_oidc_identity_token=False,tags={'role':'pokemon-strong-rl','job':label,'campaign':b.auth['campaign_id']})
 b.attach(operation,sb.object_id);write(target/'provider.json',{'sandbox_id':sb.object_id,'app_id':app.app_id,'operation':operation,'gpu':'L4','source_release':str(release)})
 coordinate('original-compatible-continuation-running',active_jobs=[{'label':label,'provider_id':sb.object_id,'max_seconds':seconds,'source_release':str(release)}])
 return {'label':label,'provider_id':sb.object_id,'operation':operation,'max_seconds':seconds}

def collect(label):
 import modal
 target=BASE/'jobs'/label;meta=json.loads((target/'provider.json').read_text());info=provider_info(meta['app_id'],meta['sandbox_id'])
 if not info or not info.get('finished_at'):return {'status':'running-or-unknown','provider':info}
 op=next(x for x in budget().snapshot()['operations'] if x['id']==meta['operation'])
 if op['status']!='settled':budget().settle_worker(meta['operation'],info)
 write(target/'allocation.json',info)
 volume=modal.Volume.from_name(VOLUME,environment_name='staging');download_parallel(volume,'/jobs/'+label,target/'collected')
 result=json.loads((target/'collected/worker-result.json').read_text()) if (target/'collected/worker-result.json').exists() else {'status':'missing-result'}
 return {'provider':info,'result':result,'usage':own_usage()}

def main():
 p=argparse.ArgumentParser();p.add_argument('command',choices=['launch-original','collect','status']);p.add_argument('--label');p.add_argument('--seconds',type=int,default=3600);p.add_argument('--parent',type=Path,default=INCUMBENT);a=p.parse_args()
 r=launch_original(a.label,a.seconds,a.parent) if a.command=='launch-original' else collect(a.label) if a.command=='collect' else own_usage();print(json.dumps(r,indent=2))
if __name__=='__main__':main()


def freeze_current():
 """Full current trainer plus frozen qualified evaluator; canonical engines remain read-only."""
 here=ROOT/'simulators/pokemon';qualified=ROOT/'evidence/reference-readiness-20261109-0251Z/pokemon/evaluator/entrant-api-releases/6d08e5d926f2a58f6404fd8a633aea76365a2bb9cdf4757ac374521851596839'
 ack=counterpart();active=ack.get('active_evaluator_release')
 if active and active.get('qualification_status')=='qualified-new-public-regression':
  candidate=Path(active['path']);qualification=candidate/'qualification.json'
  if sha(qualification)!=active['qualification_sha256']:raise ValueError('Evaluator qualification attestation changed')
  q=json.loads(qualification.read_text())
  if q['release_sha256']!=active['release_sha256'] or q['faults'] or q['caps']:raise ValueError('Evaluator extension not qualified')
  qualified=candidate
 meta=json.loads((qualified/'release.json').read_text());files={}
 frozen=BASE/'cached-image-FROZEN.json'
 if frozen.exists():
  receipt=json.loads((BASE/'cached-image-FROZEN-provenance.json').read_text())
  if receipt['source_image_id']!=IMAGE or sha(frozen)!=receipt['sha256']:raise ValueError('Cached original runtime metadata changed')
  files['FROZEN.json']=frozen
 for p in [*here.glob('*.py'),*here.glob('*.cjs'),here/'requirements.txt']:files['integrations/pokemon/'+p.name]=p
 for p in (here/'reference_training').rglob('*'):
  if p.is_file() and p.suffix in ('.py','.json','.md','.cjs') and '__pycache__' not in p.parts:files['integrations/pokemon/reference_training/'+str(p.relative_to(here/'reference_training'))]=p
 for p in (here/'autoresearch').glob('*.py'):files['integrations/pokemon/autoresearch/'+p.name]=p
 for name,digest in meta['source_file_sha256'].items():
  p=qualified/'source'/name
  if sha(p)!=digest:raise ValueError('Qualified evaluator snapshot changed')
  files['integrations/pokemon/autoresearch/evaluation_v2/'+name]=p
 for name in ('open_credit_campaign.py','overnight_budget.py'):files['tools/'+name]=ROOT/'tools'/name
 for name in ('practice.json','heldout.json'):files['integrations/pokemon/recipes/'+name]=here/'recipes'/name
 identity=hashlib.sha256(json.dumps({n:sha(p) for n,p in files.items()},sort_keys=True).encode()).hexdigest();target=BASE/'releases'/identity
 if target.exists():return target
 source=target/'source';source.mkdir(parents=True,exist_ok=False)
 for name,p in files.items():
  dest=source/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
 for name in ('integrations/__init__.py','tools/__init__.py'):(source/name).write_text('')
 shutil.copyfile(ORIGINAL_RELEASE/'source/ENGINE-PIN.json',source/'ENGINE-PIN.json')
 write(target/'release.json',{'release_sha256':identity,'files':{str(p.relative_to(source)):sha(p) for p in source.rglob('*') if p.is_file()},'qualified_evaluator_release':meta['release_sha256'],'qualified_core_release':meta.get('qualified_core_release_sha256',meta['release_sha256']),'image':IMAGE,'guard_modules':['tools/open_credit_campaign.py','tools/overnight_budget.py']})
 with tarfile.open(target/'source.tar','w') as tar:tar.add(source,arcname='source')
 for p in target.rglob('*'):
  if p.is_file():p.chmod(0o444)
 return target

def launch_job(label,mode,spec,files=(),*,seconds=1200,gpu=None,purpose='development',release=None):
 import modal
 counterpart();b=budget();training_mode=mode in ('train','proof','teacher');cutoff=epoch(b.auth['selection_close_utc'] if training_mode or purpose=='training' else b.auth['worker_stop_utc']);cutoff=min(cutoff,float(spec.get('stage_stop_unix',cutoff)));seconds=min(seconds,int(cutoff-time.time())-40)
 if seconds<90:raise ValueError('Absolute cutoff leaves no bounded stage')
 release=Path(release) if release else freeze_current();target=BASE/'jobs'/label;target.mkdir(parents=True,exist_ok=False)
 entrypoint=release/'source/integrations/pokemon/reference_training/strong_worker.py'
 if not entrypoint.is_file():raise ValueError('Frozen generic worker entrypoint missing from release')
 request=dict(spec,label=label,mode=mode,max_seconds=seconds,deadline_unix=min(time.time()+seconds-60,cutoff-35),absolute_cutoff_unix=cutoff,release=release.name,runtime_archive_sha256=sha(release/'source.tar'),entrypoint_sha256=sha(entrypoint),host_launcher_sha256=sha(__file__))
 write(target/'request.json',request);noncompute_hold();op=b.reserve_worker('pokemon/rl/'+label,seconds,gpu=gpu,cpu_cores=4,memory_gib=8,purpose=purpose);write(target/'reservation.json',{'operation':op,'max_seconds':seconds,'source_release':str(release)})
 volume=modal.Volume.from_name(VOLUME,environment_name='staging',create_if_missing=True,version=2)
 try:entries=volume.listdir('/releases/'+release.name);exists=any(x.path.endswith('/source.tar') for x in entries)
 except modal.exception.NotFoundError:exists=False
 with volume.batch_upload() as upload:
  if not exists:upload.put_file(release/'source.tar','/releases/'+release.name+'/source.tar')
  upload.put_file(target/'request.json','/jobs/'+label+'/request.json')
  upload.put_file(__file__,'/jobs/'+label+'/input/host-controlplane-strong_campaign.py')
  upload.put_file(entrypoint,'/jobs/'+label+'/strong_worker.py')
  for local,name in files:upload.put_file(local,'/jobs/'+label+'/input/'+name)
 app=modal.App.lookup(APP,environment_name='staging',create_if_missing=True)
 sb=modal.Sandbox.create('python','/runs/jobs/'+label+'/strong_worker.py','/runs/jobs/'+label+'/request.json',app=app,image=modal.Image.from_id(IMAGE),gpu=gpu,cpu=(4,4),memory=(8192,8192),timeout=seconds-25,volumes={'/runs':volume},env={'CUBLAS_WORKSPACE_CONFIG':':4096:8','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1'},block_network=True,secrets=[],include_oidc_identity_token=False,tags={'role':'pokemon-strong-rl','job':label,'campaign':b.auth['campaign_id']})
 b.attach(op,sb.object_id);write(target/'provider.json',{'sandbox_id':sb.object_id,'app_id':app.app_id,'operation':op,'gpu':gpu,'source_release':str(release)})
 return {'label':label,'provider_id':sb.object_id,'operation':op,'release':str(release),'max_seconds':seconds}


def download_parallel(volume,remote_path,output):
 import concurrent.futures
 output=Path(output);output.mkdir(parents=True,exist_ok=True);marker=output/'collection-complete.json'
 if marker.exists():
  saved=json.loads(marker.read_text())
  if all((output/name).is_file() and sha(output/name)==digest for name,digest in saved['files'].items()):return saved
 entries=[e for e in volume.listdir(remote_path,recursive=True) if e.type.name=='FILE']
 def one(entry):
  relative=Path(entry.path).relative_to(remote_path.lstrip('/'));local=output/relative;local.parent.mkdir(parents=True,exist_ok=True);temporary=local.with_name(local.name+'.download-'+str(time.time_ns()))
  with temporary.open('xb') as f:
   for chunk in volume.read_file(entry.path):f.write(chunk)
  identity=sha(temporary)
  if local.exists() and sha(local)==identity:temporary.unlink()
  else:
   if local.exists():local.rename(local.with_name(local.name+'.incomplete-'+str(time.time_ns())))
   temporary.replace(local)
  return str(relative),identity
 with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:files=dict(pool.map(one,entries))
 result={'remote_path':remote_path,'files':files,'complete':True};write(marker,result);return result
