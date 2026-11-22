"""Finite detached Sandbox launcher with explicit shared-ledger holds and immutable release."""
from __future__ import annotations
import argparse,hashlib,io,json,os,shutil,tarfile,time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'evidence/reference-readiness-20261109-0251Z/pokemon/rl'
CAMPAIGN=BASE.parents[1]
APP='pokemon-reference-20261109'
VOLUME=APP+'-private'
IMAGE='im-JwNTde85zO3cAKA9qnqInr'

def write(path,v):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');tmp.replace(path)

def budget():
    from tools.overnight_budget import CampaignBudget
    return CampaignBudget('pokemon',path=CAMPAIGN/'campaign.sqlite3')

def role_usage():
    state=budget().snapshot();own=[r for r in state['operations'] if r['game']=='pokemon' and r['kind'].startswith('pokemon/rl/')]
    return {'gpu_seconds':sum(r['gpu_charged'] if r['status']=='settled' else r['gpu_hold'] for r in own),'resource_usd':sum(r['usd_charged'] if r['status']=='settled' else r['usd_hold'] for r in own)}

def freeze():
    from ..cli import executable_signature
    from .trainer import source_signature
    here=ROOT/'simulators/pokemon';runtime={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted([*here.glob('*.py'),*here.glob('*.cjs'),here/'requirements.txt'])}
    counterpart=json.loads((CAMPAIGN/'pokemon/coordination/evaluator.json').read_text())
    interface=counterpart['runtime_interface']
    if interface.get('status')!='stable-training-interface' or interface.get('planned_canonical_runtime_mutations') is not False:raise ValueError('Stable training interface acknowledgment required')
    for name,sha in interface['runtime_file_sha256'].items():
        if runtime[name]!=sha:raise ValueError('Counterpart runtime pin changed')
    signature=hashlib.sha256(json.dumps({'runtime':runtime,'reference':source_signature()},sort_keys=True).encode()).hexdigest()
    target=BASE/'releases'/signature
    if target.exists():return target
    snapshot=target/'source';(snapshot/'integrations/pokemon').mkdir(parents=True,exist_ok=False)
    import subprocess
    engine=ROOT/'vendor/engines/pokemon-showdown'
    pin=subprocess.check_output(['git','-C',str(engine),'rev-parse','HEAD'],text=True).strip()
    if pin!='567c1cedddeaa2f35576005d3bfd526cf4f8a13e' or subprocess.check_output(['git','-C',str(engine),'diff','--name-only','HEAD'],text=True).strip():raise ValueError('Upstream pin/source changed')
    files=subprocess.check_output(['git','-C',str(engine),'ls-files','-z']).decode().split('\0')
    write(snapshot/'ENGINE-PIN.json',{'engine_pin':pin,'tracked_source_sha256':{name:hashlib.sha256((engine/name).read_bytes()).hexdigest() for name in files if name and (engine/name).is_file()},'provenance':'verified clean local tracked pin; remote exact file hash checks replace git metadata, not source validation'})
    (snapshot/'integrations/__init__.py').write_text('')
    for p in [*here.glob('*.py'),*here.glob('*.cjs'),here/'requirements.txt']:shutil.copyfile(p,snapshot/'integrations/pokemon'/p.name)
    shutil.copytree(here/'reference_training',snapshot/'integrations/pokemon/reference_training',ignore=shutil.ignore_patterns('__pycache__'))
    (snapshot/'integrations/pokemon/recipes').mkdir()
    for name in ('practice.json','heldout.json'):shutil.copyfile(here/'recipes'/name,snapshot/'integrations/pokemon/recipes'/name)
    metadata={'release_sha256':signature,'canonical_executable_signature':executable_signature(),'reference_source_signature':source_signature(),'runtime_files':runtime,'counterpart_interface':interface,'base_image_id':IMAGE,'engine_pin':'567c1cedddeaa2f35576005d3bfd526cf4f8a13e'}
    write(target/'release.json',metadata)
    with tarfile.open(target/'source.tar','w') as tar:tar.add(snapshot,arcname='source')
    for p in target.rglob('*'):
        if p.is_file():p.chmod(0o444)
    return target

def provider_info(app_id,sandbox_id):
    async def fetch():
        from modal.client import _Client
        from modal_proto import api_pb2
        client=await _Client.from_env();before=None
        while True:
            r=await client.stub.SandboxList(api_pb2.SandboxListRequest(app_id=app_id,environment_name='staging',include_finished=True,**({'before_timestamp':before} if before else {})))
            for s in r.sandboxes:
                if s.id==sandbox_id:
                    t=s.task_info;return {'provider_id':s.id,'task_id':t.id,'started_at':t.started_at,'finished_at':t.finished_at,'gpu_type':t.gpu_type,'created_at':s.created_at}
            if not r.sandboxes:return None
            before=r.sandboxes[-1].created_at
    from modal._utils.async_utils import synchronizer
    return synchronizer.create_blocking(fetch)()

def launch(label,mode='proof',seconds=600,recipe='selfplay',resume=None,seed=1729,release_path=None):
    import modal
    release=Path(release_path).resolve() if release_path else freeze();usage=role_usage()
    if not (release/'source.tar').is_file() or not (release/'release.json').is_file():raise ValueError('Complete immutable release required')
    if seconds>600 and not (BASE/'proof-admitted.json').is_file():raise ValueError('Longer allocation needs proof/export/resume/admission')
    if seconds>1800 or seconds<150 or usage['gpu_seconds']+seconds>12600:raise ValueError('RL role or per-worker time cap')
    if usage['resource_usd']+seconds*.0005+.03>32:raise ValueError('RL resource subcap')
    cp=json.loads((CAMPAIGN/'pokemon/coordination/evaluator.json').read_text())
    if cp['runtime_interface'].get('canonical_runtime_sha256')!=json.loads((release/'release.json').read_text())['counterpart_interface']['canonical_runtime_sha256']:raise ValueError('Counterpart source acknowledgment changed')
    target=BASE/'jobs'/label;target.mkdir(parents=True,exist_ok=False)
    request={'label':label,'mode':mode,'seconds':seconds,'recipe':recipe,'seed':seed,'resume':resume,'release':release.name}
    write(target/'request.json',request)
    from modal_proto import api_pb2
    vol=modal.Volume.from_name(VOLUME,environment_name='staging',create_if_missing=True,version=api_pb2.VOLUME_FS_VERSION_V2)
    try:
        existing=vol.listdir('/releases/'+release.name)
        source_exists=any(x.path.endswith('/source.tar') for x in existing)
    except modal.exception.NotFoundError:source_exists=False
    if source_exists:
        remote_sha=hashlib.sha256(b''.join(vol.read_file('/releases/'+release.name+'/source.tar'))).hexdigest()
        if remote_sha!=hashlib.sha256((release/'source.tar').read_bytes()).hexdigest():raise ValueError('Immutable remote release changed')
    with vol.batch_upload() as batch:
        if not source_exists:batch.put_file(release/'source.tar','/releases/'+release.name+'/source.tar')
        batch.put_file(target/'request.json','/jobs/'+label+'/request.json')
    app=modal.App.lookup(APP,environment_name='staging',create_if_missing=True)
    operation=budget().reserve_worker('pokemon/rl/'+label,seconds,gpu='L4',cpu_cores=4,memory_gib=8,sandbox=True,extra_usd=.02)
    write(target/'reservation.json',{'operation':operation,'max_allocated_seconds':seconds,'source_release':str(release)})
    bootstrap='import os,tarfile;from pathlib import Path;r=Path("/runs/releases/'+release.name+'");tarfile.open(r/"source.tar").extractall("/tmp/pk",filter="data");os.symlink("/opt/pk/candidates","/tmp/pk/source/candidates");os.environ["PYTHONPATH"]="/tmp/pk/source";os.execv("/usr/local/bin/python",["python","-m","integrations.pokemon.reference_training.remote_worker","/runs/jobs/'+label+'/request.json"])'
    sb=modal.Sandbox.create('python','-c',bootstrap,app=app,image=modal.Image.from_id(IMAGE),gpu='L4',cpu=(4,4),memory=(8192,8192),timeout=seconds-25,volumes={'/runs':vol},env={'PYTHONPATH':'/opt/pk','CUBLAS_WORKSPACE_CONFIG':':4096:8','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'},block_network=True,secrets=[],include_oidc_identity_token=False,environment_name='staging',tags={'pk_role':'reference-rl','pk_job':label})
    budget().attach(operation,sb.object_id);write(target/'provider.json',{'sandbox_id':sb.object_id,'app_id':app.app_id,'operation':operation,'volume':VOLUME})
    return {'label':label,'sandbox_id':sb.object_id,'app_id':app.app_id,'operation':operation,'release':str(release)}

def download_tree(volume,remote_path,output):
    """Atomic resumable collection; never mistake an interrupted file for an export."""
    output=Path(output);output.mkdir(parents=True,exist_ok=True);marker=output/'collection-complete.json'
    if marker.exists():
        old=json.loads(marker.read_text())
        if all((output/name).is_file() and hashlib.sha256((output/name).read_bytes()).hexdigest()==sha for name,sha in old['files'].items()):return old
    files={}
    for entry in volume.listdir(remote_path,recursive=True):
        if entry.type.name!='FILE':continue
        rel=Path(entry.path).relative_to(remote_path.lstrip('/'));local=output/rel;local.parent.mkdir(parents=True,exist_ok=True)
        temporary=local.with_name(local.name+'.download-'+str(time.time_ns()))
        with temporary.open('xb') as f:
            for chunk in volume.read_file(entry.path):f.write(chunk)
        sha=hashlib.sha256(temporary.read_bytes()).hexdigest()
        if local.exists() and hashlib.sha256(local.read_bytes()).hexdigest()==sha:temporary.unlink()
        else:
            if local.exists():local.rename(local.with_name(local.name+'.incomplete-'+str(time.time_ns())))
            temporary.replace(local)
        files[str(rel)]=sha
    result={'remote_path':remote_path,'files':files,'complete':True};write(marker,result);return result

def collect(label):
    import modal
    target=BASE/'jobs'/label;p=json.loads((target/'provider.json').read_text());info=provider_info(p['app_id'],p['sandbox_id'])
    if not info or not info.get('finished_at'):return {'status':'running-or-unresolved','provider':info}
    reservation=p['operation'];state=budget().snapshot();op=next(r for r in state['operations'] if r['id']==reservation)
    if op['status']!='settled':budget().settle_worker(reservation,info)
    write(target/'allocation.json',info)
    sb=modal.Sandbox.from_id(p['sandbox_id']);(target/'stdout.log').write_text(sb.stdout.read());(target/'stderr.log').write_text(sb.stderr.read())
    volume=modal.Volume.from_name(VOLUME,environment_name='staging')
    output=target/'collected';download_tree(volume,'/jobs/'+label,output)
    result=json.loads((output/'worker-result.json').read_text()) if (output/'worker-result.json').exists() else {'status':'failed-no-result'}
    if result.get('proof',{}).get('valid') and result.get('training',{}).get('source_admitted') and result.get('training',{}).get('trained_samples',0)>0:
        write(BASE/'proof-admitted.json',{'label':label,'provider':info,'result':result,'release':json.loads((target/'reservation.json').read_text())['source_release']})
    return {'provider':info,'result':result,'role_usage':role_usage()}

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['freeze','launch','collect']);p.add_argument('--label');p.add_argument('--mode',default='proof');p.add_argument('--seconds',type=int,default=600);p.add_argument('--recipe',default='selfplay');p.add_argument('--resume');p.add_argument('--seed',type=int,default=1729);p.add_argument('--release-path',type=Path);a=p.parse_args()
    r=str(freeze()) if a.command=='freeze' else launch(a.label,a.mode,a.seconds,a.recipe,a.resume,a.seed,a.release_path) if a.command=='launch' else collect(a.label)
    print(json.dumps(r,indent=2))
if __name__=='__main__':main()
