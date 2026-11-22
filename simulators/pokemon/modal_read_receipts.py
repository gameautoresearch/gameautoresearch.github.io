"""Read compact settled/in-progress owned receipts without model-blob transfer."""
import json,hashlib,time
from pathlib import Path
import modal
app=modal.App('pokemon-receipt-reader-20261108')
volumes={f'/{name}':modal.Volume.from_name(f'autoresearch-pokemon-{name}-20261108-receipts',environment_name='staging') for name in ('structural','massive')}
@app.function(cpu=1,memory=1024,timeout=120,volumes=volumes)
def read(label,kind):
    root=Path('/'+kind);out=root/label
    p=out/'results.json'
    if not p.exists():p=out/'partial.json'
    if not p.exists():return {'label':label,'ready':False}
    d=json.loads(p.read_text());summary={k:d[k] for k in ('environment','parity','stochastic_graph_exact','stochastic_graph_checks','cpus','median_decisions_per_second') if k in d}
    profiles={}
    for k,g in d.get('profiles',{}).items():
        rows=g['rows']
        if isinstance(rows,dict):rows=[dict(r,variant=v) for v,group in rows.items() for r in group]
        profiles[k]={'median_seconds':g['median_seconds'],'rows':[{key:x[key] for key in ('variant','seconds','legal_current_policy_decisions','timings','worker_metrics','completed_games','truncations','late_policy_samples_turn50','peak_cuda_allocated_bytes','rss_max_kib','ppo') if key in x} for x in rows]}
    summary['profiles']=profiles;summary['learning']=[]
    for r in d.get('learning',[]):
        row={k:r[k] for k in ('variant','seed','train_seconds','allocated_worker_seconds','policy_samples','trained_samples','inclusive_seconds','inclusive_allocated_seconds','updates','faults','truncations','natural_terminals','completed_games','timings','worker_metrics','peak_cuda_bytes','config','late_policy_samples_turn50','trained_samples_per_allocated_hour') if k in r}
        for stage in ('initial_evaluation','final_evaluation'):
            if stage in r:row[stage]={k:r[stage][k] for k in ('score','wins','losses','draws','truncations','valid_complete_schedule','seconds') if k in r[stage]}
        summary['learning'].append(row)
    if 'rows' in d and not profiles:
        summary['rows']=[{k:r[k] for k in ('repeat','workers','seconds','games','decisions','late_decisions','natural','truncations','faults','decisions_per_second') if k in r} for r in d['rows']]
    a=out/'allocation.json'
    if a.exists():
        raw=json.loads(a.read_text());summary['allocation']={k:raw[k] for k in ('error','traceback','resource_limits','cgroup_memory_peak_bytes','cgroup_cpu_max','cold_import_cuda_seconds','allocated_body_seconds_before_archive','estimated_body_cost_usd') if k in raw}
    archive=root/(label+'.tar.gz')
    if archive.exists():summary['archive']={'bytes':archive.stat().st_size,'sha256':hashlib.sha256(archive.read_bytes()).hexdigest()}
    return dict(label=label,kind=kind,terminal=p.name=='results.json',summary=summary)
@app.function(cpu=1,memory=512,timeout=300,max_containers=1,volumes=volumes)
def stream_archive(label,kind):
    with (Path('/'+kind)/(label+'.tar.gz')).open('rb') as file:
        while chunk:=file.read(512*1024):yield chunk

@app.local_entrypoint()
def main(kind:str='structural',label:str='transport-1',output:str='',download:bool=False):
    if download:
        import tarfile
        target=Path(output);target.mkdir(exist_ok=False)
        archive=target/'artifacts.tar.gz';count=0
        with archive.open('wb') as file:
            for chunk in stream_archive.remote_gen(label,kind):
                file.write(chunk);count+=len(chunk)
                if count%(4*1024*1024)==0:print(f'Downloaded {count} bytes',flush=True)
        result=read.remote(label,kind)
        digest=hashlib.sha256(archive.read_bytes()).hexdigest()
        assert digest==result['summary']['archive']['sha256']
        with tarfile.open(archive) as tar:tar.extractall(target,filter='data')
        (target/'recovery-receipt.json').write_text(json.dumps(dict(**result,verified_sha256=digest),indent=2)+'\n')
        print(json.dumps({'bytes':count,'verified_sha256':digest}),flush=True)
        return
    result=read.remote(label,kind)
    if output:Path(output).write_text(json.dumps(result,indent=2)+'\n')
    else:print(json.dumps(result,indent=2))
