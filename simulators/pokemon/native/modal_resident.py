"""Bounded actual CUDA validation of the native environment/collector pipeline."""
import hashlib
import gzip
import io
import json
import os
import shutil
import tempfile
from pathlib import Path
import tarfile
import time

import modal

ROOT=Path(__file__).resolve().parents[3] if modal.is_local() else Path('/workspace')
OUT=ROOT/'evidence/pokemon-implementation/native-rewrite-20261108'
FIXTURES=Path(os.environ.get('POKEMON_NATIVE_FIXTURES',str(OUT/'gpu-corpus-2/oracle-fixtures.jsonl')))
GENERATOR_FIXTURES=Path(os.environ.get('POKEMON_NATIVE_GENERATOR_FIXTURES',str(OUT/'teamgen-full-1/oracle.json')))
CONSTRUCTOR_FIXTURES=Path(os.environ.get('POKEMON_NATIVE_CONSTRUCTOR_FIXTURES',str(OUT/'materialize-1/oracle.json')))
RESET_FIXTURES=Path(os.environ.get('POKEMON_NATIVE_RESET_FIXTURES',str(OUT/'generated-rollout-7/oracle.json')))
TASK_FIXTURES=Path(os.environ.get('POKEMON_NATIVE_TASK_FIXTURES',str(OUT/'task-seeds-1/oracle.json')))
if modal.is_local():
    _staging=tempfile.TemporaryDirectory(prefix='pokemon-native-upload-')
    UPLOAD=Path(_staging.name)
    shutil.copytree(Path(os.environ.get('POKEMON_NATIVE_SOURCE',str(ROOT/'integrations/pokemon/native'))),UPLOAD/'native',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for name in ('network.py','efficient.py'):shutil.copyfile(ROOT/'integrations/pokemon'/name,UPLOAD/name)
    INPUT_HASHES={};shards=UPLOAD/'fixture-shards';shards.mkdir();digest=hashlib.sha256();batch=[];part=0
    with FIXTURES.open('rb')as source:
        for line in source:
            digest.update(line);batch.append(line)
            if len(batch)==128:
                (shards/f'{part:05}.jsonl.gz').write_bytes(gzip.compress(b''.join(batch),mtime=0));part+=1;batch.clear()
    if batch:(shards/f'{part:05}.jsonl.gz').write_bytes(gzip.compress(b''.join(batch),mtime=0))
    INPUT_HASHES['fixtures.jsonl']=digest.hexdigest()
    for name,path in (('generator-fixtures.json',GENERATOR_FIXTURES),('constructor-fixtures.json',CONSTRUCTOR_FIXTURES),('reset-fixtures.json',RESET_FIXTURES),('task-fixtures.json',TASK_FIXTURES)):
        raw=path.read_bytes();INPUT_HASHES[name]=hashlib.sha256(raw).hexdigest();(UPLOAD/(name+'.gz')).write_bytes(gzip.compress(raw,mtime=0))
    (UPLOAD/'input-manifest.json').write_text(json.dumps(INPUT_HASHES,indent=2)+'\n')
app=modal.App('pokemon-native-resident-20261108')
volume=modal.Volume.from_name('pokemon-native-resident-20261108-receipts',environment_name='staging',create_if_missing=True)
image=(modal.Image.from_registry('nvidia/cuda:13.0.2-devel-ubuntu24.04',add_python='3.12')
    .apt_install('g++').pip_install('numpy==2.5.4','torch==2.15.0')
    .env({'PYTHONPATH':'/workspace','CUBLAS_WORKSPACE_CONFIG':':4096:8','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'})
    .add_local_dir(UPLOAD/'native','/workspace/integrations/pokemon/native',ignore=['**/__pycache__/**'])
    .add_local_file(UPLOAD/'network.py','/workspace/integrations/pokemon/network.py')
    .add_local_file(UPLOAD/'efficient.py','/workspace/integrations/pokemon/efficient.py')
    .add_local_dir(UPLOAD/'fixture-shards','/workspace/fixture-shards')
    .add_local_file(UPLOAD/'input-manifest.json','/workspace/input-manifest.json')
    .add_local_file(UPLOAD/'generator-fixtures.json.gz','/workspace/generator-fixtures.json.gz')
    .add_local_file(UPLOAD/'constructor-fixtures.json.gz','/workspace/constructor-fixtures.json.gz')
    .add_local_file(UPLOAD/'reset-fixtures.json.gz','/workspace/reset-fixtures.json.gz')
    .add_local_file(UPLOAD/'task-fixtures.json.gz','/workspace/task-fixtures.json.gz')) if modal.is_local() else modal.Image.debian_slim()


@app.function(image=image,gpu='L4',cpu=(4,4),memory=(8192,8192),timeout=1200,startup_timeout=300,
              max_containers=1,min_containers=0,buffer_containers=0,retries=0,single_use_containers=True,
              scaledown_window=2,volumes={'/receipts':volume})
def experiment(label):
    started=time.perf_counter();output=Path('/receipts')/label
    inputs=tempfile.TemporaryDirectory(prefix='native-inputs-');input_root=Path(inputs.name)
    try:
        for name in ('generator-fixtures.json','constructor-fixtures.json','reset-fixtures.json','task-fixtures.json'):(input_root/name).write_bytes(gzip.decompress((Path('/workspace')/(name+'.gz')).read_bytes()))
        with(input_root/'fixtures.jsonl').open('xb')as target:
            for shard in sorted(Path('/workspace/fixture-shards').glob('*.jsonl.gz')):
                with gzip.open(shard,'rb')as source:shutil.copyfileobj(source,target)
        expected=json.loads(Path('/workspace/input-manifest.json').read_text())
        for name,wanted in expected.items():
            digest=hashlib.sha256()
            with(input_root/name).open('rb')as source:
                for chunk in iter(lambda:source.read(1024*1024),b''):digest.update(chunk)
            if digest.hexdigest()!=wanted:raise ValueError(f'Remote input hash changed: {name}')
        from integrations.pokemon.native.resident_suite import run
        result=run(output,input_root/'fixtures.jsonl',input_root/'generator-fixtures.json',input_root/'constructor-fixtures.json',input_root/'reset-fixtures.json',input_root/'task-fixtures.json')
    except BaseException as error:
        import traceback
        output.mkdir(parents=True,exist_ok=True)
        result=dict(error=str(error),traceback=traceback.format_exc(),full_gen9_admission=False)
        print(result['traceback'],flush=True)
    finally:inputs.cleanup()
    output.mkdir(parents=True,exist_ok=True);shutil.copyfile('/workspace/input-manifest.json',output/'input-manifest.json')
    result['allocated_body_before_archive_seconds']=time.perf_counter()-started
    (output/'allocation.json').write_text(json.dumps(result,indent=2)+'\n');volume.commit()
    buffer=io.BytesIO()
    with tarfile.open(fileobj=buffer,mode='w:gz') as archive:archive.add(output,arcname='receipts')
    artifact=buffer.getvalue();(Path('/receipts')/(label+'.tar.gz')).write_bytes(artifact);volume.commit()
    return dict(result=result,artifact=artifact,body_seconds=time.perf_counter()-started)


@app.local_entrypoint()
def main(label:str='resident-1'):
    from modal.config import config
    if config.get('environment')!='staging':raise RuntimeError('Staging only')
    target=OUT/label
    if target.exists():raise ValueError('Use a fresh label')
    started=time.perf_counter();call=experiment.spawn(label)
    ledger=OUT/(label+'-attempt.json');ledger.write_text(json.dumps(dict(label=label,function_call_id=call.object_id,
        status='running',timeout_seconds=1200,fixture_sha256=INPUT_HASHES['fixtures.jsonl'],generator_fixture_sha256=INPUT_HASHES['generator-fixtures.json'],constructor_fixture_sha256=INPUT_HASHES['constructor-fixtures.json'],reset_fixture_sha256=INPUT_HASHES['reset-fixtures.json'],task_fixture_sha256=INPUT_HASHES['task-fixtures.json'],full_gen9_admission=False),indent=2)+'\n')
    response=call.get(timeout=1250);target.mkdir(exist_ok=False);artifact=target/'artifacts.tar.gz';artifact.write_bytes(response.pop('artifact'))
    with tarfile.open(artifact) as archive:archive.extractall(target,filter='data')
    response.update(client_seconds=time.perf_counter()-started,artifact_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest())
    (target/'receipt.json').write_text(json.dumps(response,indent=2)+'\n')
    ledger.write_text(json.dumps(dict(label=label,function_call_id=call.object_id,status='failed_with_artifacts' if 'error' in response['result'] else 'completed',
        body_seconds=response['body_seconds'],client_seconds=response['client_seconds'],full_gen9_admission=False),indent=2)+'\n')
    print(json.dumps(response['result'],indent=2))
