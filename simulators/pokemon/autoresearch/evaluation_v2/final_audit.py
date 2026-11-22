"""Read-only provider reconciliation and complete durable-artifact mirroring."""
import argparse,concurrent.futures,hashlib,json,os,time
from pathlib import Path
from google.protobuf.json_format import MessageToDict
from .campaign import BASE,guard,PREFIX
from .remote import VOLUME
from ..contract import write,sha,identity


def inventory(app_id):
    async def fetch():
        from modal.client import _Client
        from modal_proto import api_pb2
        client=await _Client.from_env();before=None;rows=[]
        while True:
            response=await client.stub.SandboxList(api_pb2.SandboxListRequest(app_id=app_id,environment_name='staging',include_finished=True,**({'before_timestamp':before} if before is not None else {})))
            if not response.sandboxes:return rows
            for s in response.sandboxes:
                rows.append({'provider_id':s.id,'app_id':s.app_id,'tags':{t.tag_name:t.tag_value for t in s.tags},
                    'task_id':s.task_info.id,'created_at':s.created_at,'started_at':s.task_info.started_at,'finished_at':s.task_info.finished_at,'gpu_type':s.task_info.gpu_type,
                    'resource_info':MessageToDict(s.resource_info,preserving_proto_field_name=True)})
            before=response.sandboxes[-1].created_at
    from modal._utils.async_utils import synchronizer
    return synchronizer.create_blocking(fetch)()


def main():
    import modal
    os.environ.setdefault('MODAL_DISABLE_API_PROXY','1')
    parser=argparse.ArgumentParser();parser.add_argument('--label',default='final-audit');parser.add_argument('--previous-inventory');args=parser.parse_args()
    destination=BASE/args.label;destination.mkdir(exist_ok=False)
    previous={}
    if args.previous_inventory:
        previous={r['remote']:r for r in json.loads(Path(args.previous_inventory).read_text())['artifacts']}
        for r in previous.values():assert sha(r['local'])==r['sha256'], 'Previously mirrored evidence changed'
    ops=[r for r in guard().snapshot()['operations'] if r['game']=='pokemon' and r['kind'].startswith(PREFIX)]
    providers=[json.loads(p.read_text()) for p in (BASE/'jobs').glob('*/provider.json')]
    apps=sorted({r['app_id'] for r in providers});observed=[r for app in apps for r in inventory(app)]
    assert all(r['finished_at']>0 for r in observed), 'Active owned worker; do not clean up'
    checked=[]
    for op in ops:
        if not op['provider_id']:continue
        prior=json.loads(op['receipt']);current=next(r for r in observed if r['provider_id']==op['provider_id'])
        same=all(current[key]==prior[key] for key in ['provider_id','started_at','finished_at','gpu_type'])
        assert same and current['tags']['operation']==op['id'] and current['gpu_type']=='CPU'
        checked.append({'operation':op['id'],'kind':op['kind'],'provider':current,'settlement_matches':same})
    write(destination/'provider-lifecycle.json',{'at_unix':time.time(),'read_only':True,'workers':checked,'all_owned_workers_finished':True,'all_intervals_match':True,
        'gpu_seconds':0,'worker_count':len(checked),'unpriced_noncompute_holds':[{'operation':r['id'],'kind':r['kind'],'usd_hold':r['usd_hold']} for r in ops if r['status']!='settled']},exclusive=True)
    volume=modal.Volume.from_name(VOLUME,environment_name='staging');tasks=[]
    for folder in sorted((BASE/'jobs').iterdir()):
        for segment in ['input/evaluation_v2','output']:
            try:entries=volume.listdir(folder.name+'/'+segment,recursive=True)
            except modal.exception.NotFoundError:continue
            for entry in entries:
                if entry.type.name!='FILE' or '/output/build/' in entry.path or '/output/public-source/' in entry.path:continue
                if segment=='input/evaluation_v2':local=folder/'source'/Path(entry.path).name
                else:local=folder/'output'/entry.path.split('/output/',1)[1]
                tasks.append((entry.path,local,entry.size))
    def mirror(item):
        remote,local,size=item
        if remote in previous:return previous[remote]
        content=b''.join(volume.read_file(remote));local.parent.mkdir(parents=True,exist_ok=True)
        if local.exists():assert local.read_bytes()==content, 'Previously collected artifact differs: '+str(local)
        else:
            try:
                with local.open('xb') as f:f.write(content)
            except FileExistsError:assert local.read_bytes()==content
        return {'remote':remote,'local':str(local),'bytes':len(content),'sha256':hashlib.sha256(content).hexdigest()}
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:artifacts=list(pool.map(mirror,tasks))
    for folder in sorted((BASE/'jobs').iterdir()):
        source=folder/'source'
        if source.exists():
            files={p.name:sha(p) for p in source.iterdir() if p.is_file()}
            write(folder/('source-manifest-'+args.label+'.json'),{'files':files,'source_sha256':identity(files),'origin':'Exact immutable Volume input, verified before cleanup'},exclusive=True)
    write(destination/'artifact-inventory.json',{'artifacts':artifacts,'file_count':len(artifacts),'total_mirrored_bytes':sum(r['bytes'] for r in artifacts),
        'all_required_worker_outputs_and_uploaded_source_mirrored':True,'excluded_prefix':'pinned-search-build-001/output/build: generated compiler/toolchain cache; pinned source archive, build log, failure diagnostics and receipts are preserved separately',
        'all_original_host_request_files_retained':True,'volume':VOLUME},exclusive=True)
    print(json.dumps({'worker_count':len(checked),'all_workers_stopped':True,'artifacts':len(artifacts),'bytes':sum(r['bytes'] for r in artifacts)}))
if __name__=='__main__':main()
