#!/usr/bin/env python3
"""Package a staged run snapshot into campaign archives, validating every input hash."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile


class HashingReader:
    def __init__(self, stream):
        self.stream = stream
        self.sha = hashlib.sha256()

    def read(self, size=-1):
        data = self.stream.read(size)
        self.sha.update(data)
        return data


def group_for(path: str) -> str:
    parts = Path(path).parts
    if parts[0] == 'evidence':
        return parts[1]
    if parts[0] == 'collected':
        return 'polytopia-historical-modal'
    if parts[0] == 'simulators':
        return 'polytopia-historical-local'
    return parts[0]


def package(snapshot: Path, output: Path) -> dict:
    if not (snapshot/'summary.json').exists():
        raise ValueError('Snapshot collection has not completed.')
    output.mkdir(parents=True, exist_ok=True)
    groups = defaultdict(list)
    for line in (snapshot/'files.jsonl').read_text().splitlines():
        r = json.loads(line)
        groups[group_for(r['path'])].append(r)
    archives = []
    for group, records in sorted(groups.items()):
        target = output/(group+'.tar.zst')
        receipt = output/(group+'.archive.json')
        if target.exists() and receipt.exists():
            archives.append(json.loads(receipt.read_text()))
            continue
        tmp = target.with_name(target.name+'.partial')
        proc = subprocess.Popen(['zstd', '-q', '-T2', '-3', '-f', '-o', str(tmp)], stdin=subprocess.PIPE)
        try:
            with tarfile.open(fileobj=proc.stdin, mode='w|') as tar:
                for r in records:
                    path = snapshot/'files'/r['path']
                    info = tar.gettarinfo(str(path), arcname=r['path'])
                    if info.size != r['bytes']:
                        raise RuntimeError('Snapshot size changed: '+r['path'])
                    with path.open('rb') as source:
                        reader = HashingReader(source)
                        tar.addfile(info, reader)
                    if reader.sha.hexdigest() != r['sha256']:
                        raise RuntimeError('Snapshot checksum changed: '+r['path'])
            proc.stdin.close()
            if proc.wait() != 0:
                raise RuntimeError('Compression failed for '+group)
        except BaseException:
            proc.terminate(); proc.wait()
            raise
        tmp.replace(target)
        with target.open('rb') as source:
            sha = hashlib.file_digest(source, 'sha256').hexdigest()
        metadata = {'path': target.name, 'sha256': sha, 'bytes': target.stat().st_size,
                    'files': len(records), 'uncompressed_bytes': sum(r['bytes'] for r in records)}
        receipt.write_text(json.dumps(metadata, indent=2)+'\n')
        (output/(group+'.files.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in records))
        archives.append(metadata)
        print('PACKAGED', group, metadata['files'], metadata['uncompressed_bytes'], '->', metadata['bytes'], flush=True)
    shutil.copyfile(snapshot/'exclusions.jsonl', output/'exclusions.jsonl')
    shutil.copyfile(snapshot/'summary.json', output/'snapshot-summary.json')
    summary = {'archives': archives, 'files': sum(r['files'] for r in archives),
               'uncompressed_bytes': sum(r['uncompressed_bytes'] for r in archives),
               'archive_bytes': sum(r['bytes'] for r in archives),
               'extraction': 'zstd -dc CAMPAIGN.tar.zst | tar -xf -',
               'originals_unchanged': True, 'every_staged_input_sha256_verified_during_packaging': True}
    (output/'MANIFEST.json').write_text(json.dumps(summary, indent=2)+'\n')
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--snapshot', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    print(json.dumps(package(args.snapshot, args.output), indent=2))


if __name__ == '__main__':
    main()
