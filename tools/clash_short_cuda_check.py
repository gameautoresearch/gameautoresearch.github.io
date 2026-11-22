#!/usr/bin/env python3
"""One-L40S check of the Clash short form on CUDA. Spends worker time (Modal, game-autoresearch-project).

In the harness's own Clash worker image (which prebuilds the short-form CPU and CUDA engines and the graph
bridges), one sandbox:
  1. loads the short-form CUDA engine and checks the cadence/timeline it reports;
  2. plays the same seeds and random legal actions on native CPU and native CUDA, comparing every public
     output after every decision until all worlds end naturally, for the short form and the pinned form
     (the pinned CUDA engine is compiled in the sandbox);
  3. trains the unmodified starter recipe (1,024 worlds, learner graph, device buckets, Triton GAE) for a
     few updates through the trusted fast entry, so graph capture and conditional bodies run.

    MODAL_PROFILE=duper .venv-clash-ar/bin/python tools/clash_short_cuda_check.py --out NEW_DIR

NEW_DIR gets check.json (everything the sandbox reported), the starter run's small outputs, the provider
receipt and the app record. The sandbox has no network and a hard timeout.
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REMOTE = r'''
import json,os,subprocess,sys,time,traceback
import numpy as np
sys.path[:0]=['/opt/clash','/opt/clash/simulators/clash/native','/opt/clash/simulators/clash']
out={'started_unix':time.time()}
def record(k,v):
    out[k]=v;json.dump(out,open('/retained/check.tmp','w'),indent=1,default=str);os.replace('/retained/check.tmp','/retained/check.json')
record('gpu',subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'],text=True).strip())
import torch,match_forms
from runtime import NativeCUDA,build_cuda
from cpu_batch import NativeCPUBatch
record('torch',[torch.__version__,torch.version.cuda,torch.cuda.get_device_name()])
def parity(form,count=64,seed=20261114,render_mode=4):
    t=time.time();lib=build_cuda(form);build=time.time()-t
    cpu=NativeCPUBatch(list(range(9000,9000+count)),threads=4,form=form)
    gpu=NativeCUDA(list(range(9000,9000+count)),launch_threads=1,render_mode=render_mode,form=form)
    rng=np.random.default_rng(seed);decisions=0;compared=0;done=np.zeros(count,bool);lengths=[0]*count;winners=[None]*count
    def same():
        for name in ('observations','masks','infos','rewards'):
            a=getattr(cpu,name);b=getattr(gpu,name).cpu().numpy()
            if not np.array_equal(a,b):raise AssertionError(f'{form}: CPU/CUDA {name} differ at decision {decisions}: {int((a!=b).sum())} entries')
    try:
        same()
        while not done.all():
            if decisions>3000:raise AssertionError('worlds did not end')
            actions=np.zeros((count,2),np.int32)
            for i in range(count):
                if done[i]:continue
                for s in (0,1):
                    legal=np.flatnonzero(cpu.masks[i,s]);actions[i,s]=0 if rng.random()<.7 else int(rng.choice(legal))
            cpu.step(actions);gpu.step(actions);decisions+=1;same();compared+=1
            for i in range(count):
                if not done[i]:
                    lengths[i]+=1
                    if cpu.infos[i,2]>=0:done[i]=True;winners[i]=int(cpu.infos[i,2])
                    assert not cpu.infos[i,5] and not cpu.infos[i,11] and not cpu.infos[i,7] and not cpu.infos[i,8] and not cpu.infos[i,9]
        return {'form':form,'reported':gpu.match_form,'worlds':count,'compared_decisions':compared,'compared_public_ints_per_decision':count*(2*7024+2*2307+12+2),
                'natural_ends':int(done.sum()),'decisions_per_match':{'mean':float(np.mean(lengths)),'min':min(lengths),'max':max(lengths)},
                'winners':{str(k):winners.count(k) for k in (0,1,2)},'render_mode':render_mode,'library':lib.name,'build_or_reuse_seconds':build,'seconds':time.time()-t,'passed':True}
    finally:
        cpu.close();gpu.close()
try:
    record('parity_short_v1',parity('short-v1'))
    record('parity_short_v1_render0',parity('short-v1',count=16,render_mode=0))
    record('parity_full',parity('full',count=32))
    import shutil,pathlib
    from simulators.clash.autoresearch.common import HERE,dump
    source=pathlib.Path('/tmp/source');source.mkdir()
    for n in ('policy.py','learner.py','train.py'):shutil.copyfile(HERE/'starter'/n,source/n)
    shutil.copyfile(HERE/'recipes/starter.json',source/'config.json')
    from simulators.clash.autoresearch.train_entry import run
    t=time.time();r=run(source,'/retained/starter-run',900,updates=int(os.environ.get('CHECK_UPDATES','12')))
    rows=[json.loads(l) for l in open('/retained/starter-run/progress.jsonl')]
    record('starter',{'result':{k:v for k,v in r.items() if k not in ('loss_summary',)},'wall_seconds':time.time()-t,'progress':rows,
                      'new_decisions_per_process_second':r['new_learner_decisions']/r['seconds']})
    record('status','passed')
except BaseException as exc:
    record('status','failed');record('error',traceback.format_exc())
finally:
    subprocess.run(['sync'],timeout=30)
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--timeout', type=int, default=2700, help='hard sandbox seconds (default 45 min)')
    parser.add_argument('--updates', type=int, default=12)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(ROOT))
    os.environ.setdefault('MODAL_DISABLE_API_PROXY', '1')
    import modal
    from harness import hardware
    from harness.policy import ENVIRONMENT
    from harness.games.clash.provider import image_for, snapshot
    from simulators.clash.autoresearch.common import dump, sha
    from simulators.generals.autoresearch.provider import allocation_info, configure_sandbox_waits, read_rpc
    configure_sandbox_waits()
    archive = snapshot(out / 'uploads')
    app = modal.App('game-autoresearch-clash-short-check')
    record = {'environment': ENVIRONMENT, 'gpu': hardware.GPU, 'cpu': 4, 'memory_gib': 16, 'timeout_seconds': args.timeout,
              'source_archive_sha256': sha(archive), 'started_unix': time.time()}
    with app.run(environment_name=ENVIRONMENT):
        record['app_id'] = app.app_id
        image = image_for(archive)
        mark = time.time()
        read_rpc(lambda: image.build(app))
        record.update(image_id=image.object_id, image_build_wall_seconds=time.time() - mark)
        dump(out / 'app.json', record)
        # Outputs live on a Volume so a dropped exec stream or a terminated sandbox cannot lose them.
        import uuid
        vname = 'clash-short-check-' + uuid.uuid4().hex
        volume = modal.Volume.from_name(vname, environment_name=ENVIRONMENT, create_if_missing=True, version=2)
        record['volume'] = vname
        sb = modal.Sandbox.create('/bin/sleep', str(args.timeout), app=app, image=image, gpu=hardware.GPU, cpu=(4, 4),
                                  memory=(16 * 1024, 16 * 1024), timeout=args.timeout, block_network=True, secrets=[],
                                  volumes={'/retained': volume})
        record['sandbox_id'] = sb.object_id
        dump(out / 'app.json', record)
        try:
            read_rpc(lambda: sb.filesystem.write_text(REMOTE, '/retained/check.py'))
            command = (f'CHECK_UPDATES={args.updates} CLASH_CUDA_LAUNCH_THREADS=1 CLASH_CUDA_RENDER_MODE=4 '
                       'python -X faulthandler -u /retained/check.py > /retained/stdout.log 2>&1; echo $? > /retained/exit')
            proc = sb.exec('sh', '-c', command, timeout=args.timeout - 60)
            deadline = time.time() + args.timeout - 60
            while time.time() < deadline:
                try:
                    if read_rpc(proc.poll) is not None:
                        break
                except Exception as exc:  # a dropped stream: fall back to the exit marker on the sandbox
                    record.setdefault('poll_errors', []).append(type(exc).__name__)
                    try:
                        read_rpc(lambda: sb.filesystem.read_text('/retained/exit'))
                        break
                    except Exception:
                        pass
                time.sleep(5)
        finally:
            sb.terminate()
            receipt = None
            for _ in range(60):
                receipt = allocation_info(app.app_id, sb.object_id)
                if receipt and receipt.get('finished_at'):
                    break
                time.sleep(.5)
            record['provider_receipt'] = receipt
            if receipt and receipt.get('started_at') and receipt.get('finished_at'):
                record['gpu_worker_seconds'] = receipt['finished_at'] - receipt['started_at']
            record['finished_unix'] = time.time()
            dump(out / 'app.json', record)
        # Bulk download after the GPU has stopped.
        retained = out / 'retained'
        retained.mkdir()
        for entry in volume.iterdir('/', recursive=True):
            if entry.type.name != 'FILE':
                continue
            target = retained / entry.path.lstrip('/')
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('wb') as stream:
                for chunk in volume.read_file(entry.path):
                    stream.write(chunk)
        dump(out / 'app.json', record)
    (out / 'check.json').write_bytes((out / 'retained/check.json').read_bytes())
    check = json.loads((out / 'check.json').read_text())
    print(json.dumps({'status': check.get('status'), 'gpu': check.get('gpu'), 'gpu_worker_seconds': record.get('gpu_worker_seconds'),
                      'parity': {k: (v.get('passed'), v.get('compared_decisions'), v.get('decisions_per_match')) for k, v in check.items() if k.startswith('parity')},
                      'starter': (check.get('starter') or {}).get('result'), 'error': (check.get('error') or '')[-1500:]}, indent=1, default=str))


if __name__ == '__main__':
    main()
