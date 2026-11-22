#!/usr/bin/env python3
"""Read-only remote/local artifact SHA audit before any owned volume cleanup.

Run with the Modal CLI's Python environment. This script does not delete resources.
Every run ID and output path is explicit; historical receipts are never overwritten.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import modal
from modal.types import FileEntryType
from concurrent.futures import ThreadPoolExecutor

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'evidence/tavern-implementation/structural-20261108'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-id',action='append',required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    vol=modal.Volume.from_name('autoresearch-tavern-structural-evidence-20261108',environment_name='staging')
    rows=[];started=time.perf_counter()
    def verify(entry, destination):
        local=destination/entry.path
        if not local.is_file():raise FileNotFoundError(local)
        if local.stat().st_size!=entry.size:raise RuntimeError(f'size mismatch: {entry.path}')
        h=hashlib.sha256()
        for chunk in vol.read_file(entry.path):h.update(chunk)
        local_hash=hashlib.sha256(local.read_bytes()).hexdigest()
        if local_hash!=h.hexdigest():raise RuntimeError(f'SHA mismatch: {entry.path}')
        return {'path':entry.path,'bytes':entry.size,'sha256':local_hash}
    for run in a.run_id:
        destination=BASE/run/'retained'
        entries=[e for e in vol.iterdir(run,recursive=True) if e.type==FileEntryType.FILE]
        with ThreadPoolExecutor(max_workers=8) as pool:
            rows.extend(pool.map(lambda entry:verify(entry,destination),entries))
        print(json.dumps({'verified':run,'files_so_far':len(rows)}),flush=True)
    a.out.write_text(json.dumps({'verified':True,'files':rows,'file_count':len(rows),
        'bytes':sum(r['bytes'] for r in rows),'audit_wall_seconds':time.perf_counter()-started,
        'source':'Modal Volume read_file SHA256 matched retained local bytes; no remote mutations'},indent=2)+'\n')


if __name__=='__main__':main()
