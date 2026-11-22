"""Hash settled owned evidence in a CPU worker, then compare retained local bytes.

No training is repeated and no remote/local evidence is changed or removed.
"""
import argparse,hashlib,json,time
from pathlib import Path
import modal
ROOT=Path(__file__).resolve().parents[2] if modal.is_local() else Path('/workspace')
VOLUMES={kind:modal.Volume.from_name(f'autoresearch-tavern-{kind}-evidence-20261108') for kind in ['device','original']}
app=modal.App('autoresearch-tavern-retention-20261108')

@app.function(image=modal.Image.debian_slim(python_version='3.12'),cpu=(2,2),memory=(2048,2048),
    timeout=180,max_containers=1,min_containers=0,buffer_containers=0,retries=0,
    single_use_containers=True,volumes={f'/data/{k}':v for k,v in VOLUMES.items()})
def remote_hashes(requests:list[dict]):
    start=time.perf_counter();result=[]
    for request in requests:
        kind,run=request['kind'],request['run']
        if kind not in ['device','original'] or '/' in run or run.startswith('.'):raise ValueError('owned run names only')
        root=Path('/data')/kind/run
        if not root.is_dir():raise FileNotFoundError(root)
        rows=[]
        for path in sorted(root.rglob('*')):
            if path.is_file():
                h=hashlib.sha256()
                with path.open('rb') as stream:
                    for chunk in iter(lambda:stream.read(4<<20),b''):h.update(chunk)
                rows.append({'path':str(path.relative_to(root)),'bytes':path.stat().st_size,'sha256':h.hexdigest()})
        result.append({'kind':kind,'run':run,'files':rows})
    return {'gpu':None,'cpu_cores':2,'ram_gib':2,'worker_seconds':time.perf_counter()-start,'runs':result}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--device-run',action='append',default=[])
    p.add_argument('--original-run',action='append',default=[]);a=p.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    requests=([{'kind':'original','run':run} for run in a.original_run]+
              [{'kind':'device','run':run} for run in a.device_run])
    if not requests:
        requests=[{'kind':'original','run':'matched-l4-v1'},*[{'kind':'device','run':run} for run in
            ['device-b200-arena-v11','device-b200-atomic-v12','device-b200-allocshim-v13']]]
    if len({(r['kind'],r['run']) for r in requests})!=len(requests):raise ValueError('duplicate audit request')
    with app.run(environment_name='staging'):
        print(json.dumps({'app_id':app.app_id,'function_id':remote_hashes.object_id}),flush=True)
        result=remote_hashes.remote(requests)
        result.update(app_id=app.app_id,function_id=remote_hashes.object_id)
    for run in result['runs']:
        base=ROOT/'evidence/tavern-implementation'/('device-port' if run['kind']=='device' else 'original-baselines-20261108')/run['run']/'retained'/run['run']
        for row in run['files']:
            path=base/row['path'];h=hashlib.sha256()
            if not path.is_file():raise FileNotFoundError(path)
            with path.open('rb') as stream:
                for chunk in iter(lambda:stream.read(4<<20),b''):h.update(chunk)
            if (path.stat().st_size,h.hexdigest())!=(row['bytes'],row['sha256']):raise AssertionError(path)
        run['local_files_match']=True
    result.update(all_retained_bytes_match=True,files=sum(len(r['files']) for r in result['runs']),
                  bytes=sum(f['bytes'] for r in result['runs'] for f in r['files']))
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='runs'}),flush=True)


if __name__=='__main__':main()
