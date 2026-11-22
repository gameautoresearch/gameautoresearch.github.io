"""Root-side export after stopping the unprivileged trainer. No symlink following."""
from pathlib import Path
import argparse
import json
import os
import stat
from .bundle import seal,archive_bundle,ARCHIVE

def safe_read(path,limit):
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)  # a FIFO the trainer left must not block the export
    try:
        s=os.fstat(fd)
        if not stat.S_ISREG(s.st_mode) or s.st_size>limit:raise ValueError('Invalid trainer output file')
        with os.fdopen(fd,'rb',closefd=False) as f:return f.read(limit+1)
    finally:os.close(fd)

def main():
    target=Path('/job/trusted-export');target.mkdir(mode=0o700)
    for name,origin,cap in [(n,'/job/input/'+n,524288) for n in ('policy.py','learner.py','train.py')]+[('model.json','/job/output/model.json',65536),('weights.npz','/job/output/weights.npz',134217728)]:
        (target/name).write_bytes(safe_read(origin,cap))
    identity=seal(target,'/job/export',{'kind':'allocated_worker_output'})
    # One file for the host: six separate copies did not fit the end of a short run.
    archive_bundle('/job/export','/job/'+ARCHIVE)
    print(json.dumps({'bundle_sha256':identity}),flush=True)

if __name__=='__main__':main()
