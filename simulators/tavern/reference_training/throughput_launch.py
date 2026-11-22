"""Launch one bounded L4 population-throughput verification, in a fresh ledger.

No historical campaign is resumed. The user authorized trainer performance work;
the operator's conservative engineering limit is1,200 cumulative allocated L4
seconds, including failures/setup/idle, four physical CPU cores and16GiB RAM.
"""
import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
os.environ.setdefault('MODAL_DISABLE_API_PROXY', '1')
from .strong.common import ROOT, GAME, write, sha

ROOT_OUT = ROOT/'evidence/tavern-population-throughput-20261109'
VOLUME = 'tavern-population-throughput-20261109'


def budget():
    from tools.overnight_budget import CampaignBudget
    auth_path = ROOT_OUT/'authorization.json'
    if not auth_path.exists():
        pricing = json.loads((ROOT/'evidence/overnight-20261108-1533Z/authorization.json').read_text())['pricing']
        pricing.update(retrieved_at_utc=datetime.now(timezone.utc).isoformat(),
                       verified_source='https://modal.com/pricing', reservation_margin=1.1)
        end = (datetime.now(timezone.utc)+timedelta(hours=3)).isoformat()
        write(auth_path, dict(campaign_id='tavern-population-throughput-20261109',
              authorized_by='User: do so for battlegrounds; repair/profile the actual population trainer around Rust CPU simulation',
              scope='Bounded performance verification only; operator-selected conservative ceiling; no old campaign, long strength training, private grading or GPU simulator rewrite',
              deadline_utc=end, worker_stop_utc=end,
              global_limits=dict(gpu_allocated_seconds=1200, resource_charged_or_reserved_usd=3,
                                 model_charged_or_reserved_usd=0, max_concurrent_gpu_workers=1),
              initial_grants=dict(tavern=dict(gpu_seconds=1200, resource_usd=3, model_usd=0)), pricing=pricing))
    return CampaignBudget('tavern', ROOT_OUT/'campaign.sqlite3', auth_path)


def launch(name, seconds, graphs=False):
    if (ROOT_OUT/'CLOSED.json').exists():
        raise ValueError('This bounded performance task is closed; unused allowance does not reopen it')
    if seconds not in range(180,1081):
        raise ValueError('Finite verification180..1080 seconds required')
    import modal
    sys.path.insert(0, str(GAME/'autoresearch'))
    from provider import configure_sandbox_waits, image_for
    configure_sandbox_waits()
    b = budget()
    out = ROOT_OUT/name
    out.mkdir(parents=True, exist_ok=False)
    source = out/'source'
    game = source/'simulators/tavern'
    game.mkdir(parents=True)
    for f in GAME.glob('*.py'):
        shutil.copy2(f, game/f.name)
    shutil.copytree(GAME/'reference_training', game/'reference_training',
                    ignore=shutil.ignore_patterns('__pycache__', '.pytest_cache'))
    helpers = game/'autoresearch'
    helpers.mkdir()
    for name_ in ('__init__.py','provider.py','contract.py','benchmark.json','policy_worker.py'):
        shutil.copy2(GAME/'autoresearch'/name_, helpers/name_)
    shutil.copytree(GAME/'autoresearch/recipes', helpers/'recipes')
    (source/'tools').mkdir()
    for name_ in ('open_credit_campaign.py','overnight_budget.py'):
        shutil.copy2(ROOT/'tools'/name_, source/'tools'/name_)
    # The CLI's local-lock path must also exist in this isolated source image.
    (source/'evidence').mkdir()
    (source/'evidence/local-performance.lock').touch()
    write(out/'source-files.json', {str(p.relative_to(source)):sha(p) for p in source.rglob('*') if p.is_file()})
    inputs = ROOT_OUT/'inputs-quarter157'
    expected = json.loads((inputs/'manifest.json').read_text())
    if any(sha(inputs/f) != h for f,h in expected.items()):
        raise ValueError('Frozen trained-parent inputs changed')
    base = ROOT/'evidence/overnight-20261108-1533Z/tavern/release-admission-v2'
    app = modal.App('tavern-population-throughput-'+name)
    image = (image_for(base).add_local_dir(source,'/opt/task',copy=True)
             .add_local_dir(inputs,'/inputs',copy=True)
             .env(dict(PYTHONPATH='/opt/task', PYTHONDONTWRITEBYTECODE='1',
                       TAVERNGYM_WORKERS='pooled', TAVERNGYM_SCHEDULER='dynamic'))
             .workdir('/opt/task'))
    volume = modal.Volume.from_name(VOLUME,environment_name='staging',create_if_missing=True,version=2)
    build = b.reserve_worker('image:'+name,300,None,4,16,sandbox=False,extra_usd=.1)
    write(out/'build-reservation.json',dict(operation=build,unpriced_hold_retained=True))
    with modal.enable_output(), app.run(detach=True,environment_name='staging'):
        b.attach(build,app.app_id)
        image.build(app)
        gpu = b.reserve_worker('verify:'+name,seconds,'L4',4,16)
        # Persist the hold before dispatch. Failed launches never silently refund it.
        write(out/'gpu-reservation.json',dict(operation=gpu,cap_seconds=seconds))
        command = ['python','-u','-m','simulators.tavern.reference_training.throughput_guest',
                   '--root','/retained/'+name,'--seconds',str(seconds-30)]
        if graphs:
            command += ['--graphs']
        sb = modal.Sandbox.create(*command,app=app,image=image,
             gpu='L4',cpu=(4,4),memory=(16384,16384),timeout=seconds-30,
             block_network=True,secrets=[],volumes={'/retained':volume},environment_name='staging',
             include_oidc_identity_token=False,name='tavern-throughput-'+name,
             tags={'campaign':'tavern-population-throughput-20261109','purpose':'bounded-verification'})
        b.attach(gpu,sb.object_id)
        write(out/'provider.json',dict(app_id=app.app_id,sandbox_id=sb.object_id,volume_name=VOLUME,
              name=name,launched_unix=time.time(),cap_seconds=seconds,gpu_operation=gpu,
              build_operation=build,source_manifest_sha256=sha(out/'source-files.json'),
              input_manifest_sha256=sha(inputs/'manifest.json'),graphs=graphs,automatic_retries=0))
        sb.detach()
    print((out/'provider.json').read_text(),flush=True)


def status(name, collect=False, metadata_only=False):
    import modal
    sys.path.insert(0,str(GAME/'autoresearch'))
    from provider import configure_sandbox_waits, allocation_info
    configure_sandbox_waits()
    out = ROOT_OUT/name
    launch = json.loads((out/'provider.json').read_text())
    info = allocation_info(launch['app_id'],launch['sandbox_id'])
    b = budget()
    if info and info.get('started_at') and info.get('finished_at'):
        operations = {o['id']:o for o in b.snapshot()['operations']}
        if operations[launch['gpu_operation']]['status'] != 'settled':
            b.settle_worker(launch['gpu_operation'],info)
    write(out/'provider-status.json',dict(provider=info,checked_unix=time.time()))
    if collect:
        if not info or not info.get('finished_at'):
            raise ValueError('Cannot finalize an active allocation')
        volume = modal.Volume.from_name(VOLUME,environment_name='staging')
        dest = out/'collected'
        prefix = name+'/'
        for entry in volume.iterdir('/'+name,recursive=True):
            if entry.type != 1:
                continue
            path = entry.path.lstrip('/')
            if not path.startswith(prefix):
                raise ValueError('Unexpected volume prefix')
            if metadata_only and Path(path).suffix not in ('.json','.jsonl','.log','.txt'):
                continue
            target = dest/path[len(prefix):]
            if target.exists():
                continue
            target.parent.mkdir(parents=True,exist_ok=True)
            temp = target.with_suffix(target.suffix+'.download')
            with temp.open('wb') as f:
                for chunk in volume.read_file('/'+path):
                    f.write(chunk)
            temp.replace(target)
        write(out/('metadata-collection.json' if metadata_only else 'collection.json'),
              dict(metadata_only=metadata_only, private_archive_volume=VOLUME,
                   files={str(p.relative_to(dest)):sha(p) for p in dest.rglob('*') if p.is_file() and p.suffix != '.download'}))
    write(ROOT_OUT/'budget-snapshot.json',b.snapshot())
    print(json.dumps(info,indent=2),flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['launch','status'])
    p.add_argument('--name',required=True)
    p.add_argument('--seconds',type=int,default=1080)
    p.add_argument('--collect',action='store_true')
    p.add_argument('--graphs',action='store_true')
    p.add_argument('--metadata-only',action='store_true')
    a=p.parse_args()
    if a.command == 'launch':
        launch(a.name,a.seconds,a.graphs)
    else:
        status(a.name,a.collect,a.metadata_only)
