"""Retain owned large results in a private volume and fetch compact readouts.

Large source/state archives need not block laptop receipt retrieval. This CPU-only
reader accepts explicit owned call IDs from local launch records, stores the full
immutable archive with SHA-256, and returns logs/JSON plus member hashes.
"""
import json
from pathlib import Path
import modal

app = modal.App('autoresearch-generals-receipt-reader-20261108')
volume = modal.Volume.from_name('generals-frontier-evidence-20261108', create_if_missing=True)


@app.function(image=modal.Image.debian_slim(python_version='3.12'), cpu=(2, 2), memory=(8192, 8192),
              volumes={'/retained': volume}, retries=0, timeout=600,
              max_containers=1, min_containers=0, buffer_containers=0, scaledown_window=2)
def read_owned(call_id: str, tag: str):
    import hashlib
    import io
    import tarfile
    import time
    start = time.monotonic()
    binary, receipt = modal.FunctionCall.from_id(call_id).get(timeout=30)
    sha = hashlib.sha256(binary).hexdigest()
    folder = Path('/retained')/tag
    folder.mkdir(exist_ok=True)
    target = folder/'artifacts.tar.gz'
    if target.exists():
        if hashlib.sha256(target.read_bytes()).hexdigest() != sha:
            raise RuntimeError('Existing retained archive differs; refuse overwrite')
    else:
        target.write_bytes(binary)
    volume.commit()
    members, selected = [], {}
    with tarfile.open(fileobj=io.BytesIO(binary), mode='r:gz') as tar:
        for member in tar:
            if not member.isfile():
                continue
            if ((member.name.endswith(('.json', '.log', '.jsonl')) and member.size < 20_000_000)
                    or (member.name.endswith('source.tgz') and member.size < 2_000_000)):
                data = tar.extractfile(member).read()
                selected[member.name] = data
                members.append(dict(name=member.name, bytes=member.size, sha256=hashlib.sha256(data).hexdigest()))
            else:
                members.append(dict(name=member.name, bytes=member.size))
    info = dict(call_id=call_id, tag=tag, full_archive_sha256=sha, full_archive_bytes=len(binary),
                volume='generals-frontier-evidence-20261108', volume_path=f'{tag}/artifacts.tar.gz',
                source_receipt=receipt, members=members, reader_seconds=time.monotonic()-start,
                reader_allocation=dict(gpu=None, cpu_physical_cores=2, ram_gib=8),
                privacy='Account-private staging volume; no public publishing')
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode='w:gz') as tar:
        for name, data in selected.items():
            member = tarfile.TarInfo(name)
            member.size = len(data)
            tar.addfile(member, io.BytesIO(data))
        data = (json.dumps(info, indent=2)+'\n').encode()
        member = tarfile.TarInfo('retained-archive.json')
        member.size = len(data)
        tar.addfile(member, io.BytesIO(data))
    return output.getvalue(), info


@app.local_entrypoint()
def main(tags: str, suffix: str = ''):
    import io
    import tarfile
    from modal.config import config
    if config.get('environment') != 'staging':
        raise ValueError('staging only')
    if suffix and not suffix.replace('-', '').isalnum():
        raise ValueError('Unsafe suffix')
    root = Path(__file__).resolve().parents[2]
    base = root/'evidence/generals-implementation/structural-20261108'
    for tag in tags.split(','):
        if not tag.replace('-', '').isalnum():
            raise ValueError('Unsafe tag')
        source = base/'scaling'/tag/'launch.json'
        row = json.loads(source.read_text())
        destination = base/'receipt-reader'/(tag+suffix)
        destination.mkdir(parents=True, exist_ok=False)
        binary, receipt = read_owned.remote(row['call_id'], tag)
        (destination/'readout.tar.gz').write_bytes(binary)
        with tarfile.open(fileobj=io.BytesIO(binary), mode='r:gz') as tar:
            tar.extractall(destination, filter='data')
        (destination/'reader.json').write_text(json.dumps(dict(reader_app_id=app.app_id, **receipt), indent=2)+'\n')
        print(json.dumps(dict(tag=tag, archive_bytes=receipt['full_archive_bytes'], sha256=receipt['full_archive_sha256'],
                              receipt=receipt['source_receipt'])), flush=True)
