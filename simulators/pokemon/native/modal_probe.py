"""Bounded native CUDA component development; full task admission remains closed."""
from pathlib import Path
import io,json,time,tarfile,hashlib
import modal
ROOT=Path(__file__).resolve().parents[3] if modal.is_local() else Path('/workspace')
OUT=ROOT/'evidence/pokemon-implementation/native-rewrite-20261108'
app=modal.App('pokemon-native-rewrite-20261108')
volume=modal.Volume.from_name('pokemon-native-rewrite-20261108-receipts',environment_name='staging',create_if_missing=True)
image=(modal.Image.from_registry('nvidia/cuda:13.0.2-devel-ubuntu24.04',add_python='3.12')
 .apt_install('g++').pip_install('numpy==2.5.4','torch==2.15.0')
 .env({'PYTHONPATH':'/workspace','CUBLAS_WORKSPACE_CONFIG':':4096:8','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'})
 .add_local_dir(ROOT/'integrations/pokemon/native','/workspace/integrations/pokemon/native',ignore=['**/__pycache__/**'])
 .add_local_file(OUT/'oracle-gpu.json','/workspace/oracle-gpu.json')
 .add_local_file(OUT/'effects-4/oracle-fixtures.jsonl','/workspace/oracle-effects.jsonl')) if modal.is_local() else modal.Image.debian_slim()
@app.function(image=image,gpu='L4',cpu=(4,4),memory=(8192,8192),timeout=600,startup_timeout=300,
 max_containers=1,min_containers=0,buffer_containers=0,retries=0,single_use_containers=True,
 scaledown_window=2,volumes={'/receipts':volume})
def probe(label):
 start=time.perf_counter();out=Path('/receipts')/label
 try:
  from integrations.pokemon.native.cuda_probe import run
  result=run(out,Path('/workspace/oracle-gpu.json'))
  from integrations.pokemon.native.cuda_effects import run as check_effects
  result['effects']=check_effects(out/'effects',Path('/workspace/oracle-effects.jsonl'),out/'cpu-build/libpokemon_native.so',out/'libpokemon_cuda.so')
 except BaseException as e:
  import traceback
  out.mkdir(parents=True,exist_ok=True);result={'error':str(e),'traceback':traceback.format_exc(),'full_gen9_admission':False};print(result['traceback'],flush=True)
 result['allocated_body_seconds']=time.perf_counter()-start
 (out/'allocation.json').write_text(json.dumps(result,indent=2)+'\n');volume.commit()
 bio=io.BytesIO()
 with tarfile.open(fileobj=bio,mode='w:gz') as tar:tar.add(out,arcname='receipts')
 archive=bio.getvalue();(Path('/receipts')/(label+'.tar.gz')).write_bytes(archive);volume.commit()
 return {'result':result,'artifact':archive,'body_seconds_with_archive':time.perf_counter()-start}
@app.local_entrypoint()
def main(label:str='probe-1'):
 from modal.config import config
 if config.get('environment')!='staging':raise RuntimeError('Staging only')
 target=OUT/label
 if target.exists():raise ValueError('Use a unique label')
 t=time.perf_counter();call=probe.spawn(label)
 ledger=OUT/(label+'-attempt.json');ledger.write_text(json.dumps(dict(label=label,function_call_id=call.object_id,started_unix=time.time(),status='running',full_gen9_admission=False),indent=2)+'\n')
 result=call.get(timeout=650);target.mkdir(exist_ok=False)
 archive=target/'artifacts.tar.gz';archive.write_bytes(result.pop('artifact'))
 with tarfile.open(archive) as tar:tar.extractall(target,filter='data')
 result.update(client_seconds=time.perf_counter()-t,artifact_sha256=hashlib.sha256(archive.read_bytes()).hexdigest());(target/'receipt.json').write_text(json.dumps(result,indent=2)+'\n')
 ledger.write_text(json.dumps(dict(label=label,function_call_id=call.object_id,status='completed' if 'error' not in result['result'] else 'failed_with_artifacts',body_seconds=result['body_seconds_with_archive'],client_seconds=result['client_seconds'],full_gen9_admission=False),indent=2)+'\n')
 print(json.dumps(result['result'],indent=2))
