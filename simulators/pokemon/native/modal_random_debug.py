"""Single-use L4 fault diagnosis against an immutable failed-run source/binary."""
import gzip,hashlib,io,json,shutil,tarfile,tempfile,time
from pathlib import Path
import modal
ROOT=Path(__file__).resolve().parents[3]if modal.is_local()else Path('/workspace')
OUT=ROOT/'evidence/pokemon-implementation/native-rewrite-20261108'
if modal.is_local():
    staging=tempfile.TemporaryDirectory(prefix='pokemon-random-debug-');upload=Path(staging.name)
    frozen=OUT/'resident-7/receipts';shutil.copytree(frozen/'cpu-build/source',upload/'native');shutil.copytree(frozen/'cpu-build',upload/'cpu-build')
    shutil.copyfile(frozen/'libpokemon_cuda.so',upload/'libpokemon_cuda.so');shutil.copyfile(ROOT/'integrations/pokemon/native/random_debug.py',upload/'random_debug.py')
    for name in ('network.py','efficient.py'):shutil.copyfile(ROOT/'integrations/pokemon'/name,upload/name)
    input_hashes={}
    for name,path,keys in (('generator.json',OUT/'teamgen-full-1/oracle.json',('species','moves','status_moves','items')),('constructor.json',OUT/'materialize-1/oracle.json',('metadata',)),('task.json',OUT/'task-seeds-1/oracle.json',('specs','chart'))):
        original=path.read_bytes();source=json.loads(original);raw=json.dumps({key:source[key]for key in keys},separators=(',',':')).encode();(upload/(name+'.gz')).write_bytes(gzip.compress(raw,mtime=0));input_hashes[name]=dict(original_sha256=hashlib.sha256(original).hexdigest(),diagnostic_sha256=hashlib.sha256(raw).hexdigest(),retained_keys=keys)
    (upload/'input-manifest.json').write_text(json.dumps(input_hashes,indent=2)+'\n')
app=modal.App('pokemon-native-random-debug-20261108');volume=modal.Volume.from_name('pokemon-native-resident-20261108-receipts',environment_name='staging')
if modal.is_local():
    image=(modal.Image.from_registry('nvidia/cuda:13.0.2-devel-ubuntu24.04',add_python='3.12').pip_install('numpy==2.5.4','torch==2.15.0')
        .env({'PYTHONPATH':'/workspace','CUDA_LAUNCH_BLOCKING':'1','CUBLAS_WORKSPACE_CONFIG':':4096:8','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'})
        .add_local_dir(upload/'native','/workspace/integrations/pokemon/native')
        .add_local_dir(upload/'cpu-build','/workspace/cpu-build')
        .add_local_file(upload/'libpokemon_cuda.so','/workspace/libpokemon_cuda.so')
        .add_local_file(upload/'random_debug.py','/workspace/random_debug.py')
        .add_local_file(upload/'network.py','/workspace/integrations/pokemon/network.py')
        .add_local_file(upload/'efficient.py','/workspace/integrations/pokemon/efficient.py')
        .add_local_file(upload/'generator.json.gz','/workspace/generator.json.gz')
        .add_local_file(upload/'constructor.json.gz','/workspace/constructor.json.gz')
        .add_local_file(upload/'task.json.gz','/workspace/task.json.gz')
        .add_local_file(upload/'input-manifest.json','/workspace/input-manifest.json'))
else:image=modal.Image.debian_slim()
@app.function(image=image,gpu='L4',cpu=(4,4),memory=(8192,8192),timeout=900,startup_timeout=300,retries=0,max_containers=1,min_containers=0,buffer_containers=0,single_use_containers=True,scaledown_window=2,volumes={'/receipts':volume})
def experiment(label):
    started=time.perf_counter();path=Path('/receipts')/label;inputs=tempfile.TemporaryDirectory(prefix='random-debug-input-');root=Path(inputs.name)
    try:
        for name in ('generator.json','constructor.json','task.json'):(root/name).write_bytes(gzip.decompress((Path('/workspace')/(name+'.gz')).read_bytes()))
        from random_debug import run
        result=run(path,root/'generator.json',root/'constructor.json',root/'task.json','/workspace/cpu-build/libpokemon_native.so','/workspace/libpokemon_cuda.so')
    except BaseException as error:
        import traceback
        result=dict(error=str(error),traceback=traceback.format_exc(),full_gen9_admission=False);print(result['traceback'],flush=True)
    finally:inputs.cleanup()
    path.mkdir(parents=True,exist_ok=True);shutil.copyfile('/workspace/input-manifest.json',path/'input-manifest.json');result['body_before_archive_seconds']=time.perf_counter()-started;(path/'results.json').write_text(json.dumps(result,indent=2)+'\n');volume.commit()
    buffer=io.BytesIO()
    with tarfile.open(fileobj=buffer,mode='w:gz')as archive:archive.add(path,arcname='receipts')
    return dict(result=result,artifact=buffer.getvalue(),body_seconds=time.perf_counter()-started)
@app.local_entrypoint()
def main(label:str='random-debug-1'):
    from modal.config import config
    if config.get('environment')!='staging':raise RuntimeError('Staging only')
    path=OUT/label
    if path.exists():raise ValueError('Fresh output label required')
    started=time.perf_counter();call=experiment.spawn(label);ledger=OUT/(label+'-attempt.json');ledger.write_text(json.dumps(dict(status='running',call_id=call.object_id,frozen_source='resident-7',full_gen9_admission=False),indent=2)+'\n')
    result=call.get(timeout=950);path.mkdir(exist_ok=False);artifact=result.pop('artifact');(path/'artifacts.tar.gz').write_bytes(artifact)
    with tarfile.open(fileobj=io.BytesIO(artifact))as archive:archive.extractall(path,filter='data')
    result.update(client_seconds=time.perf_counter()-started,artifact_sha256=hashlib.sha256(artifact).hexdigest());(path/'receipt.json').write_text(json.dumps(result,indent=2)+'\n');ledger.write_text(json.dumps(dict(status='failed_with_artifacts'if'error'in result['result']else'completed',call_id=call.object_id,body_seconds=result['body_seconds'],full_gen9_admission=False),indent=2)+'\n');print(json.dumps(result['result'],indent=2))
