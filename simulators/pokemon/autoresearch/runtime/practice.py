"""Explicit public practice in a finite training lease; no idle warm workers."""
import json
import math
import time
from pathlib import Path


def public_practice(seconds, total_seconds, seed_start, clusters):
    if type(seconds)is not int or seconds<60 or seconds>total_seconds-180:
        raise ValueError('Combined practice needs >=60s and must leave >=180s for training/setup')
    if type(clusters)is not int or not 1<=clusters<=16 or type(seed_start)is not int or not 900_000_000<=seed_start<seed_start+clusters<1_000_000_000:
        raise ValueError('Public practice seeds and 1..16 clusters required')
    return dict(seconds=seconds,seeds=list(range(seed_start,seed_start+clusters)))


def startup_summary(jobs):
    waits=[]
    for job in jobs:
        receipt=job.get('receipt',{})
        if isinstance(receipt,str):receipt=json.loads(receipt)
        created,start=receipt.get('created_at'),receipt.get('started_at')
        if type(created)in(int,float) and type(start)in(int,float) and math.isfinite(created) and math.isfinite(start) and 0<created<=start:
            waits.append(start-created)
    waits.sort()
    return dict(measured_intervals=len(waits),provisioning_wall_seconds=sum(waits),
        median_startup_seconds=waits[len(waits)//2]if waits else None,
        p95_startup_seconds=waits[min(len(waits)-1,math.ceil(.95*len(waits))-1)]if waits else None,
        idle_gpu_retention_enabled=False,recommend_combined_practice=bool(waits and waits[-1]>120))


def run_practice(spec, out, inputs, deadline):
    """The caller supplies a trusted wall timer; inference sees only observations."""
    import signal
    from .bundle import verify
    from .arena import run
    from .worker import _deadline,ASSESSMENT_PUBLICATION_RESERVE_SECONDS
    request=dict(spec['public_practice']['assessment'])
    bundle=out/'stage/bundle';manifest=verify(bundle)
    request['candidate']=dict(kind='bundle-native',name='same-lease-trained-stage',
        path=str(bundle),bundle_sha256=manifest['bundle_sha256'])
    request['public_development']=True
    if not request.get('seeds') or any(type(s)is not int or not 900_000_000<=s<1_000_000_000 for s in request['seeds']):
        raise ValueError('Combined assessment must use public practice seeds')
    path=out/'public-practice-request.json';path.write_text(json.dumps(request));path.chmod(0o600)
    target=out/'practice';target.mkdir(mode=0o700)
    remaining=min(spec['public_practice']['seconds'],deadline-time.time()-ASSESSMENT_PUBLICATION_RESERVE_SECONDS)
    if remaining<=0:raise TimeoutError('No combined practice time remains')
    signal.setitimer(signal.ITIMER_REAL,remaining)
    try:run(request,target,str(path))
    finally:signal.setitimer(signal.ITIMER_REAL,max(.001,deadline-time.time()-30))
