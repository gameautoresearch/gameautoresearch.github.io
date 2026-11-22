"""Verify one qualified recipe against complete fresh-worker allocation time.

Reuses the exact qualified image without another build, under the same ledger.
It launches no successor and refuses missing/mismatched qualification artifacts.
"""
import argparse
import json
import os
import re
import time
from pathlib import Path
os.environ.setdefault('MODAL_DISABLE_API_PROXY', '1')
from .strong.common import ROOT, write, sha
from .throughput_launch import ROOT_OUT, VOLUME, budget


def launch(name, parent_name, seconds, games=256, updates=4):
    if (ROOT_OUT/'CLOSED.json').exists():
        raise ValueError('This bounded performance task is closed')
    if seconds not in range(120, 301) or games not in (192,256) or updates not in range(2,5):
        raise ValueError('Bounded selected-recipe verification only')
    parent = ROOT_OUT/parent_name
    suite = parent/'collected/suite'
    completed = json.loads((suite/'completed.json').read_text())
    if not completed['all_complete']:
        raise ValueError('Completed qualification required')
    for filename in ('resume-parity.json','parity-trial0.json','parity-trial1.json'):
        if not all(json.loads((suite/filename).read_text()).values()):
            raise ValueError('Exact CUDA qualification missing')
    config = json.loads((suite/f'optimized-{games}-config.json').read_text())
    config.update(updates=updates, profile_synchronized=False)
    # Source in the existing image must remain the tested immutable snapshot.
    manifest = json.loads((parent/'source-files.json').read_text())
    if any(sha(parent/'source'/f) != h for f,h in manifest.items()):
        raise ValueError('Qualified source snapshot mutated')
    image_ids = re.findall(r'Built image (im-\w+)', (ROOT_OUT/(parent_name+'-launch.log')).read_text())
    if not image_ids:
        raise ValueError('No image-build receipt')
    image_id = image_ids[-1]
    out = ROOT_OUT/name
    out.mkdir(parents=True, exist_ok=False)
    write(out/'config.json',config)
    # The wrapper writes its terminal status before exiting; provider timestamps
    # remain authoritative for startup/import/setup/checkpoint/idle/teardown time.
    command = (
        "import json,time,subprocess,traceback\n"
        "from pathlib import Path\n"
        f"root=Path('/retained/{name}');root.mkdir(parents=True,exist_ok=False)\n"
        "entered=time.time();status={'status':'started','entry_unix':entered}\n"
        "try:\n"
        " from simulators.tavern.reference_training.strong.train import run\n"
        f" c=json.loads({json.dumps(config)!r})\n"
        f" run(c,root/'training','/inputs',{seconds-65},limit_updates={updates})\n"
        " status.update(status='completed',finished_unix=time.time())\n"
        "except BaseException:\n"
        " status.update(status='failed',finished_unix=time.time(),traceback=traceback.format_exc());raise\n"
        "finally:\n"
        " (root/'lifecycle.json').write_text(json.dumps(status,indent=2)+'\\n')\n"
        " subprocess.run(['sync',str(root)],check=True,timeout=10)\n"
    )
    (out/'worker-command.py').write_text(command)
    import modal
    from simulators.tavern.autoresearch.provider import configure_sandbox_waits
    configure_sandbox_waits()
    b = budget()
    operation = b.reserve_worker('selected-session:'+name,seconds,'L4',4,16)
    write(out/'gpu-reservation.json',dict(operation=operation,cap_seconds=seconds))
    app = modal.App('tavern-population-throughput-'+name)
    image = modal.Image.from_id(image_id)
    volume = modal.Volume.from_name(VOLUME,environment_name='staging')
    with modal.enable_output(),app.run(detach=True,environment_name='staging'):
        sb = modal.Sandbox.create('python','-u','-c',command,app=app,image=image,gpu='L4',
             cpu=(4,4),memory=(16384,16384),timeout=seconds-30,block_network=True,secrets=[],
             volumes={'/retained':volume},include_oidc_identity_token=False,
             name='tavern-throughput-'+name,
             tags={'campaign':'tavern-population-throughput-20261109','purpose':'selected-session-verification'})
        b.attach(operation,sb.object_id)
        write(out/'provider.json',dict(app_id=app.app_id,sandbox_id=sb.object_id,volume_name=VOLUME,
              name=name,launched_unix=time.time(),cap_seconds=seconds,gpu_operation=operation,
              reused_image_id=image_id,qualified_parent=parent_name,automatic_retries=0,
              source_manifest_sha256=sha(parent/'source-files.json'),
              worker_command_sha256=sha(out/'worker-command.py'),image_build_operations=0))
        sb.detach()
    print((out/'provider.json').read_text(),flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--name',required=True)
    p.add_argument('--parent',required=True)
    p.add_argument('--seconds',type=int,default=240)
    p.add_argument('--games',type=int,default=256)
    p.add_argument('--updates',type=int,default=4)
    a=p.parse_args()
    launch(a.name,a.parent,a.seconds,a.games,a.updates)
