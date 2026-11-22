"""Resume atomic artifact retention and verify remote bytes before cleanup.

Partial CLI downloads never count as retained. Existing matching files are kept;
replacement is atomic after the complete remote stream and expected length arrive.
Every invocation writes a new receipt and performs no remote mutations.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
import time
import modal
from modal.types import FileEntryType

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'evidence/tavern-implementation/device-port'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-id',action='append',required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--manifest',type=Path,help='authoritative read-only remote SHA manifest')
    a=p.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    volume=modal.Volume.from_name('autoresearch-tavern-device-evidence-20261108',environment_name='staging')
    started=time.perf_counter();rows=[]
    manifest=json.loads(a.manifest.read_text()) if a.manifest else None
    if manifest is not None and manifest['run_ids']!=a.run_id:raise ValueError('remote manifest run list differs')
    expected={r['path']:r for r in manifest['files']} if manifest else {}
    candidates={}
    def retain(entry,destination):
        local=destination/entry.path;local.parent.mkdir(parents=True,exist_ok=True)
        remote=expected.get(entry.path)
        if remote and local.is_file() and local.stat().st_size==remote['bytes'] and hashlib.sha256(local.read_bytes()).hexdigest()==remote['sha256']:
            return dict(remote,existing_bytes_matched=True)
        temp=local.with_name(local.name+'.retaining')
        if remote:
            # Frozen local compiler outputs and earlier verified binary copies
            # often already contain these exact bytes. A remote SHA match is
            # mandatory; filenames or source receipts alone never suffice.
            name=Path(entry.path).name
            if name not in candidates:candidates[name]=list(BASE.rglob(name))
            for candidate in candidates[name]:
                if candidate==local or not candidate.is_file() or candidate.stat().st_size!=remote['bytes']:continue
                if hashlib.sha256(candidate.read_bytes()).hexdigest()!=remote['sha256']:continue
                shutil.copyfile(candidate,temp);temp.replace(local)
                return dict(remote,existing_bytes_matched=False,recovered_from_verified_copy=str(candidate))
        for attempt in range(3):
            h=hashlib.sha256();size=0
            try:
                with temp.open('wb') as target:
                    for chunk in volume.read_file(entry.path):
                        target.write(chunk);h.update(chunk);size+=len(chunk)
                break
            except Exception:
                if attempt==2:raise
                time.sleep(1+attempt)
        if size!=entry.size:raise RuntimeError(('incomplete remote stream',entry.path,size,entry.size))
        if remote and h.hexdigest()!=remote['sha256']:raise RuntimeError(('remote bytes changed since manifest',entry.path))
        same=local.exists() and local.stat().st_size==size and hashlib.sha256(local.read_bytes()).hexdigest()==h.hexdigest()
        if same:temp.unlink()
        else:temp.replace(local)
        return {'path':entry.path,'bytes':size,'sha256':h.hexdigest(),'existing_bytes_matched':same}
    for run in a.run_id:
        destination=BASE/run/'retained'
        entries=([SimpleNamespace(path=r['path'],size=r['bytes']) for r in manifest['files'] if r['path'].startswith(run+'/')]
            if manifest else [e for e in volume.iterdir(run,recursive=True) if e.type==FileEntryType.FILE])
        with ThreadPoolExecutor(max_workers=4) as pool:
            for row in pool.map(lambda entry:retain(entry,destination),entries):rows.append(row)
        print(json.dumps({'retained_verified':run,'files_so_far':len(rows)}),flush=True)
        a.out.with_suffix('.progress.json').write_text(json.dumps({'completed_through':run,'files':rows},indent=2)+'\n')
    receipt={'verified':True,'file_count':len(rows),'bytes':sum(row['bytes'] for row in rows),
        'files':rows,'wall_seconds':time.perf_counter()-started,
        'source':'complete Modal Volume.read_file streams; atomic local retention, no remote mutation'}
    a.out.write_text(json.dumps(receipt,indent=2)+'\n')


if __name__=='__main__':main()
