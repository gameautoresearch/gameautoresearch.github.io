"""Read-only CPU-side hashes avoid repeated bulk network artifact downloads."""
import json
from pathlib import Path
import modal

app=modal.App('autoresearch-tavern-device-retention-20261108')
volume=modal.Volume.from_name('autoresearch-tavern-device-evidence-20261108')


@app.function(cpu=(4,4),memory=(4096,4096),timeout=300,retries=0,
    min_containers=0,buffer_containers=0,max_containers=1,scaledown_window=2,
    single_use_containers=True,volumes={'/results':volume})
def manifest(run_ids:list[str]):
    import hashlib,time
    started=time.perf_counter();rows=[]
    for run in run_ids:
        if not run or '/' in run or run in ['.','..']:raise ValueError('explicit run basename required')
        source=Path('/results')/run
        if not source.is_dir():raise FileNotFoundError(run)
        for p in sorted(source.rglob('*')):
            if not p.is_file():continue
            h=hashlib.sha256()
            with p.open('rb') as stream:
                for chunk in iter(lambda:stream.read(8<<20),b''):h.update(chunk)
            rows.append({'path':str(p.relative_to('/results')),'bytes':p.stat().st_size,'sha256':h.hexdigest()})
    return {'run_ids':run_ids,'files':rows,'source':'read-only mounted Modal volume CPU SHA256',
            'worker_seconds':time.perf_counter()-started,'gpu_executed':False}


@app.local_entrypoint()
def main(run_ids:str,manifest_path:str):
    dest=Path(manifest_path)
    if dest.exists():raise FileExistsError(dest)
    call=manifest.spawn(run_ids.split(','))
    print(json.dumps({'app_id':app.app_id,'function_call_id':call.object_id,'run_ids':run_ids}),flush=True)
    result=call.get()
    with dest.open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps({'manifest_path':str(dest),'files':len(result['files']),'worker_seconds':result['worker_seconds']}),flush=True)
