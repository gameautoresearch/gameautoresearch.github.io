"""Read detached progress or collect durable results; never starts a GPU."""
from pathlib import Path
import argparse
import json
import modal
from .bootstrap import ROOT


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=ROOT/'evidence/generals-averagejoe-continuation-20261109')
    p.add_argument('--collect', action='store_true')
    a = p.parse_args()
    launch = json.loads((a.root/'provider-call.json').read_text())
    from simulators.generals.autoresearch.provider import configure_sandbox_waits, allocation_info, read_rpc
    configure_sandbox_waits()
    provider = allocation_info(launch['app_id'], launch['sandbox_id'])
    volume = modal.Volume.from_name(launch['volume_name'], environment_name='staging')
    def read(name):
        return b''.join(volume.read_file(name)).decode()
    result = {'provider': provider, 'volume_name': launch['volume_name']}
    for name in ('lifecycle.json', 'output/status.json', 'output/latest.json', 'output/result.json',
                 'control/observer-started.json', 'control/allocation.json'):
        try:
            text = read('/'+name); result[name] = json.loads(text)
        except Exception as exc:
            result[name] = {'unavailable': type(exc).__name__}
    # Reading the active sandbox also exposes the latest uncommitted log lines.
    if provider and not provider.get('finished_at'):
        try:
            sb = modal.Sandbox.from_id(launch['sandbox_id'])
            result['trainer_tail'] = read_rpc(lambda: sb.filesystem.read_text('/retained/trainer.log'))[-9000:]
            sb.detach()
        except Exception as exc:
            result['trainer_tail_error'] = str(exc)
    if a.collect:
        if not provider or not provider.get('finished_at'):
            raise RuntimeError('Wait for worker completion before collecting final artifacts')
        destination = a.root/'collected'; destination.mkdir(exist_ok=False)
        for entry in volume.iterdir('/', recursive=True):
            if entry.type == 1:  # modal.volume.FileEntryType.FILE
                target = destination/entry.path.lstrip('/'); target.parent.mkdir(parents=True, exist_ok=True)
                with target.open('wb') as f:
                    for chunk in volume.read_file(entry.path):
                        f.write(chunk)
        result['collected'] = str(destination)
    (a.root/'last-status.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
