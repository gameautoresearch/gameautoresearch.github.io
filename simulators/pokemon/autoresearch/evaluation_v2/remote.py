"""Finite CPU Sandbox with complete inputs and durable Volume before launch, plus recovery."""
import argparse
import ast
import json
import os
from pathlib import Path
import time

from .campaign import BASE,guard,reserve_cpu,coordinate
from ..contract import write,sha,identity

VOLUME='pokemon-evaluator-readiness-20261109'
APP='pokemon-evaluator-readiness-20261109'
# Retained immutable v1 runtime; no new image build or historical ledger reuse.
IMAGE='im-JwNTde85zO3cAKA9qnqInr'

def allocation_info(app_id,sandbox_id):
    # This campaign's workers ran in their original environment; read-only lookups of them stay there.
    from harness.games.pokemon.provider import allocation_info as lookup
    return lookup(app_id,sandbox_id,environment='staging')

ENTRY=r'''
import json,os,pathlib,shutil,subprocess,time,traceback
os.umask(0o077)
job=pathlib.Path('/data')/os.environ['EVALUATOR_JOB'];job.chmod(0o700)
out=job/'output';out.mkdir(mode=0o700,exist_ok=True)
try:
 src=job/'input/evaluation_v2';dest=pathlib.Path('/opt/pk/integrations/pokemon/autoresearch/evaluation_v2')
 shutil.copytree(src,dest)
 dest.chmod(0o755)
 for path in dest.rglob('*'):path.chmod(0o755 if path.is_dir() else 0o444)
 os.environ['PYTHONPATH']='/opt/pk';os.chdir('/opt/pk')
 # Private mounted job/output directories remain0700. The unchanged legacy
 # inference launcher expects a normal umask for its public755 code directory.
 os.umask(0o022)
 with (out/'stdout.log').open('w') as stdout,(out/'stderr.log').open('w') as stderr:
  code=subprocess.run(['python','-m','integrations.pokemon.autoresearch.evaluation_v2.'+os.environ['EVALUATOR_MODULE'],'--request',str(job/'input/request.json'),'--output',str(out)],stdout=stdout,stderr=stderr,timeout=int(os.environ['EVALUATOR_SECONDS'])-35).returncode
 (out/'exit.json').write_text(json.dumps({'returncode':code,'finished_at_unix':time.time()}))
except BaseException:
 (out/'entry-error.log').write_text(traceback.format_exc())
finally:
 subprocess.run(['sync',str(job)],timeout=15)
'''


def collect(label):
    import modal
    target=BASE/'jobs'/label;provider=json.loads((target/'provider.json').read_text())
    receipt=allocation_info(provider['app_id'],provider['sandbox_id'])
    if not receipt or not receipt.get('finished_at'):return {'status':'running-or-unknown','provider':receipt}
    operation=json.loads((target/'reservation.json').read_text())['operation']
    row=next(r for r in guard().snapshot()['operations'] if r['id']==operation)
    if row['status']!='settled':guard().settle_worker(operation,receipt)
    write(target/'compute.json',{'provider':receipt,'campaign_operation':operation,'owned_worker_stopped':True,'gpu':None})
    volume=modal.Volume.from_name(VOLUME,environment_name='staging')
    for entry in volume.listdir(label+'/input/evaluation_v2',recursive=False):
        if entry.type.name!='FILE':continue
        destination=target/'source'/Path(entry.path).name;destination.parent.mkdir(exist_ok=True)
        content=b''.join(volume.read_file(entry.path))
        if destination.exists():
            if destination.read_bytes()!=content:raise ValueError('Previously collected immutable source changed')
        else:destination.write_bytes(content);destination.chmod(0o444)
    source_files={p.name:sha(p) for p in (target/'source').iterdir() if p.is_file()}
    write(target/'source-manifest.json',{'files':source_files,'source_sha256':identity(source_files),'origin':'Exact immutable uploaded worker inputs retrieved from durable Volume'})
    for entry in volume.listdir(label+'/output',recursive=True):
        if '/output/build/' in entry.path or entry.path.endswith('/output/build'):
            continue
        if entry.type.name!='FILE':
            continue
        name=entry.path.split('/output/',1)[-1]
        if '/output/' not in entry.path:continue
        destination=target/'output'/name
        try:
            content=b''.join(volume.read_file(entry.path))
        except Exception:
            if entry.type.name=='DIRECTORY':continue
            raise
        destination.parent.mkdir(parents=True,exist_ok=True)
        if destination.exists():
            if destination.read_bytes()!=content:raise ValueError('Previously collected output changed')
        else:destination.write_bytes(content)
    coordinate('cpu-job-collected',last_job=label)
    return {'status':'finished','provider':receipt,'output':str(target/'output')}


def launch(label,module,spec,files,seconds=300,network=False,source_release=None):
    import modal
    if 'wheel_sha256' in spec:
        spec=dict(spec,wheels='/data/'+label+'/input/wheels')
        files=list(files)+[(BASE/'jobs/pinned-search-build-002/output/wheels'/name,'wheels/'+name)
            for name in spec['wheel_sha256']]
    target=BASE/'jobs'/label;target.mkdir(parents=True,exist_ok=False)
    operation=reserve_cpu(label,seconds)
    write(target/'reservation.json',{'operation':operation,'max_seconds':seconds,'durable_volume':VOLUME,'no_new_build':True},exclusive=True)
    write(target/'request.json',spec,exclusive=True)
    source=Path(__file__).parent
    entry=ENTRY
    if source_release:
        release=Path(source_release);manifest=json.loads((release/'release.json').read_text())
        if identity({k:v for k,v in manifest.items() if k!='release_sha256'})!=manifest['release_sha256']:raise ValueError('Frozen evaluator release changed')
        source=release/'source'
        for name,digest in manifest['source_file_sha256'].items():
            if sha(source/name)!=digest:raise ValueError('Frozen evaluator source changed: '+name)
        tree=ast.parse((source/'remote.py').read_text())
        entry=ast.literal_eval(next(n.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='ENTRY' for t in n.targets)))
        write(target/'frozen-release-binding.json',{'path':str(release.resolve()),'release_sha256':manifest['release_sha256']},exclusive=True)
    volume=modal.Volume.from_name(VOLUME,create_if_missing=True,version=2,environment_name='staging')
    with volume.batch_upload() as upload:
        for path in source.iterdir():
            if path.suffix in ('.py','.cjs','.json','.patch'):upload.put_file(path,label+'/input/evaluation_v2/'+path.name)
        upload.put_file(target/'request.json',label+'/input/request.json')
        for src,name in files:upload.put_file(src,label+'/input/'+name)
    app=modal.App.lookup(APP,create_if_missing=True,environment_name='staging')
    mounts={'/data/'+label:volume.with_mount_options(sub_path='/'+label)}
    sandbox=modal.Sandbox.create('python','-c',entry,app=app,image=modal.Image.from_id(IMAGE),
        cpu=(4,4),memory=(8192,8192),timeout=seconds-10,block_network=not network,secrets=[],include_oidc_identity_token=False,
        volumes=mounts,env={'EVALUATOR_JOB':label,'EVALUATOR_MODULE':module,'EVALUATOR_SECONDS':str(seconds)},
        environment_name='staging',tags={'role':'pokemon-evaluator','operation':operation,'job':label})
    write(target/'provider.json',{'app_id':app.app_id,'sandbox_id':sandbox.object_id,'image_id':IMAGE},exclusive=True)
    guard().attach(operation,sandbox.object_id)
    coordinate('cpu-job-running',active_job=label,active_provider=sandbox.object_id)
    print(json.dumps({'launched':label,'provider':sandbox.object_id,'max_seconds':seconds}),flush=True)
    sandbox.wait()
    return collect(label)


def main():
    os.environ.setdefault('MODAL_DISABLE_API_PROXY','1')
    parser=argparse.ArgumentParser();parser.add_argument('--collect',required=True);a=parser.parse_args();print(json.dumps(collect(a.collect)))
if __name__=='__main__':main()
