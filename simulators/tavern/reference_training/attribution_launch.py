"""Fresh successor ledger and finite worker for the explicitly reopened gap audit."""
import argparse
from datetime import datetime,timedelta,timezone
import json
import os
from pathlib import Path
import re
import time
os.environ.setdefault('MODAL_DISABLE_API_PROXY','1')
from .strong.common import ROOT,write,sha
OLD=ROOT/'evidence/tavern-population-throughput-20261109'
ATTRIBUTION_ROOT=ROOT/'evidence/tavern-throughput-attribution-20261109'
VOLUME='tavern-throughput-attribution-20261109'


def budget():
    from tools.overnight_budget import CampaignBudget
    p=ATTRIBUTION_ROOT/'authorization.json'
    if not p.exists():
        old=json.loads((OLD/'authorization.json').read_text());end=(datetime.now(timezone.utc)+timedelta(hours=3)).isoformat()
        write(p,dict(campaign_id='tavern-throughput-attribution-20261109',
            authorized_by='Explicit delegated user follow-up: Resolve Battlegrounds measured performance gap',
            scope='Same-policy same-mixture controlled attribution and inexpensive measured fixes only; no strength campaign/private grading/model calls',
            deadline_utc=end,worker_stop_utc=end,
            global_limits=dict(gpu_allocated_seconds=1200,resource_charged_or_reserved_usd=3,model_charged_or_reserved_usd=0,max_concurrent_gpu_workers=1),
            initial_grants=dict(tavern=dict(gpu_seconds=1200,resource_usd=3,model_usd=0)),pricing=old['pricing'],
            predecessor_closed_sha256=sha(OLD/'CLOSED.json'),predecessor_ledger_sha256=sha(OLD/'campaign.sqlite3')))
    return CampaignBudget('tavern',ATTRIBUTION_ROOT/'campaign.sqlite3',p)


def launch(name,cap,plan='crossover',cache=4,threads=4):
    if (ATTRIBUTION_ROOT/'CLOSED.json').exists():raise ValueError('Successor task closed')
    if cap not in range(120,701):raise ValueError('Finite120..700 second verification required')
    import modal
    from simulators.tavern.autoresearch.provider import configure_sandbox_waits
    configure_sandbox_waits();b=budget();out=ATTRIBUTION_ROOT/name;out.mkdir(exist_ok=False)
    parent=(ATTRIBUTION_ROOT/'runtime-final' if plan=='single' and (ATTRIBUTION_ROOT/'runtime-final/image.json').exists() else OLD/'l4-bounded-v4');manifest=json.loads((parent/'source-files.json').read_text())
    if any(sha(parent/'source'/f)!=h for f,h in manifest.items()):raise ValueError('Frozen runtime mutated')
    image_id=(json.loads((parent/'image.json').read_text())['image_id'] if (parent/'image.json').exists() else re.findall(r'Built image (im-\w+)',(OLD/'l4-bounded-v4-launch.log').read_text())[-1])
    source=(Path(__file__).parent/'attribution.py').read_text();(out/'runner.py').write_text(source)
    config=json.loads((ATTRIBUTION_ROOT/'config.json').read_text())
    worker_env=None
    if plan=='single':
        worker_env={k:str(threads) for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']}
        config.update(actor_graph_max_shapes=cache,host_thread_environment=worker_env,profile_synchronized=False)
    write(out/'config.json',config)
    argv=['--one','single','--sync','0','--cache',str(cache),'--threads',str(threads)] if plan=='single' else ['--plan',plan]
    command=("import sys,runpy\nfrom pathlib import Path\n"
        f"root=Path('/retained/{name}');root.mkdir(parents=True,exist_ok=False)\n"
        f"(root/'runner.py').write_text({source!r})\n"
        f"(root/'config.json').write_text({json.dumps(config)!r})\n"
        f"sys.argv=[str(root/'runner.py'),'--root',str(root),'--config',str(root/'config.json'),'--cap',{str(cap-45)!r}]+{argv!r}\n"
        "runpy.run_path(str(root/'runner.py'),run_name='__main__')\n")
    (out/'worker-command.py').write_text(command)
    operation=b.reserve_worker('attribution:'+name,cap,'L4',4,16);write(out/'reservation.json',dict(operation=operation,cap=cap))
    app=modal.App('tavern-attribution-'+name);volume=modal.Volume.from_name(VOLUME,environment_name='staging',create_if_missing=True,version=2)
    with modal.enable_output(),app.run(detach=True,environment_name='staging'):
        sb=modal.Sandbox.create('python','-u','-c',command,app=app,image=modal.Image.from_id(image_id),
            gpu='L4',cpu=(4,4),memory=(16384,16384),timeout=cap-25,block_network=True,secrets=[],
            volumes={'/retained':volume},env=worker_env,include_oidc_identity_token=False,name='tavern-attribution-'+name,
            tags={'campaign':'tavern-throughput-attribution-20261109','purpose':'controlled-attribution'})
        b.attach(operation,sb.object_id)
        write(out/'provider.json',dict(app_id=app.app_id,sandbox_id=sb.object_id,volume_name=VOLUME,
            name=name,gpu_operation=operation,cap_seconds=cap,image_id=image_id,automatic_retries=0,
            runner_sha256=sha(out/'runner.py'),command_sha256=sha(out/'worker-command.py'),
            frozen_training_source_sha256=sha(parent/'source-files.json'),plan=plan,launched_unix=time.time()))
        sb.detach()
    print((out/'provider.json').read_text(),flush=True)


def status(name,collect):
    import modal
    from simulators.tavern.autoresearch.provider import configure_sandbox_waits,allocation_info
    configure_sandbox_waits();b=budget();out=ATTRIBUTION_ROOT/name;p=json.loads((out/'provider.json').read_text());info=allocation_info(p['app_id'],p['sandbox_id'])
    if info and info['started_at'] and info['finished_at']:
        ops={o['id']:o for o in b.snapshot()['operations']}
        if ops[p['gpu_operation']]['status']!='settled':b.settle_worker(p['gpu_operation'],info)
    write(out/'provider-status.json',dict(provider=info,checked_unix=time.time()))
    if collect:
        if not info or not info['finished_at']:raise ValueError('Worker still active')
        v=modal.Volume.from_name(VOLUME,environment_name='staging');dest=out/'collected'
        for entry in v.iterdir('/'+name,recursive=True):
            if entry.type!=1:continue
            f=entry.path.lstrip('/');assert f.startswith(name+'/')
            if Path(f).suffix not in ('.json','.jsonl','.csv','.log','.py'):continue
            target=dest/f[len(name)+1:]
            if target.exists():continue
            target.parent.mkdir(parents=True,exist_ok=True);tmp=target.with_suffix('.download')
            with tmp.open('wb') as handle:
                for chunk in v.read_file('/'+f):handle.write(chunk)
            tmp.replace(target)
        write(out/'metadata-collection.json',dict(files={str(f.relative_to(dest)):sha(f) for f in dest.rglob('*') if f.is_file()}))
    write(ATTRIBUTION_ROOT/'budget-snapshot.json',b.snapshot());print(json.dumps(info,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['launch','status']);p.add_argument('--name',required=True);p.add_argument('--cap',type=int,default=600);p.add_argument('--plan',choices=['crossover','cache','threads','single'],default='crossover');p.add_argument('--cache',type=int,choices=[4,16],default=4);p.add_argument('--threads',type=int,choices=[1,4],default=4);p.add_argument('--collect',action='store_true');a=p.parse_args()
    if a.command=='launch':launch(a.name,a.cap,a.plan,a.cache,a.threads)
    else:status(a.name,a.collect)
