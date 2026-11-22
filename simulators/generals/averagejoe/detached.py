"""One detached bounded GPU Sandbox and an independent CPU receipt collector.

The trainer is the Sandbox entrypoint (not an exec tied to a host controller).
Provider timeout and an in-container watchdog each bound lifetime. A remote CPU
observer settles allocation after exit. It cannot launch or retry GPU workers.
"""
from pathlib import Path
import argparse
import json
import shutil
import time
import modal
from .bootstrap import ROOT, HERE, ENGINE, UPSTREAM, binding, sha


def write(path, value):
    path = Path(path); tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n'); tmp.replace(path)


def observe(app_id, sandbox_id, volume_name, launched_unix):
    """Remote CPU only. Recoverably reconcile this exact sandbox, never create one."""
    from pathlib import Path
    import json
    import time
    import modal
    from simulators.generals.autoresearch.provider import configure_sandbox_waits, allocation_info, read_rpc
    configure_sandbox_waits()
    root = Path('/retained'); control = root/'control'; control.mkdir(exist_ok=True)
    volume = modal.Volume.from_name(volume_name, environment_name='staging')
    sandbox = modal.Sandbox.from_id(sandbox_id)
    def save(name, value):
        (control/name).write_text(json.dumps(value, indent=2)+'\n'); volume.commit()
    save('observer-started.json', {'unix': time.time(), 'app_id': app_id,
         'sandbox_id': sandbox_id, 'can_create_gpu_workers': False, 'laptop_dependency': False})
    info = None
    while time.time() < launched_unix+3540:
        info = allocation_info(app_id, sandbox_id)
        save('observer-status.json', {'checked_unix': time.time(), 'provider': info})
        if info and info.get('finished_at'):
            break
        if time.time() >= launched_unix+3490:
            sandbox.terminate(wait=True)
        time.sleep(10)
    if not info or not info.get('finished_at'):
        sandbox.terminate(wait=True)
        for _ in range(20):
            info = allocation_info(app_id, sandbox_id)
            if info and info.get('finished_at'):
                break
            time.sleep(.5)
    if not info or not info.get('started_at') or not info.get('finished_at'):
        save('allocation.json', {'status': 'unresolved_hold_retained', 'provider': info,
             'reserved_gpu_seconds': 3600, 'sandbox_id': sandbox_id})
        raise RuntimeError('Provider allocation unsettled; original hold retained')
    elapsed = info['finished_at']-info['started_at']
    volume.reload()
    lifecycle = json.loads((root/'lifecycle.json').read_text()) if (root/'lifecycle.json').exists() else None
    result = json.loads((root/'output/result.json').read_text()) if (root/'output/result.json').exists() else None
    # Check every final checkpoint file from durable storage after the worker exits.
    import hashlib
    checkpoint_valid = False
    if result:
        ck = root/'output/checkpoints/final'; manifest = json.loads((ck/'state.json').read_text())
        checkpoint_valid = all(hashlib.sha256((ck/n).read_bytes()).hexdigest() == h for n, h in manifest['files'].items())
    receipt = {'status': 'settled_complete' if result and checkpoint_valid and lifecycle and lifecycle.get('trainer_returncode') == 0 else 'failed_settled',
               'provider': info, 'allocated_gpu_seconds': elapsed, 'cap_gpu_seconds': 3600,
               'within_cap': elapsed <= 3600, 'owned_active_gpu_workers': 0,
               'final_checkpoint_hashes_verified_from_volume': checkpoint_valid,
               'estimated_compute_usd': elapsed*(.001097+4*.00003942+16*.00000667),
               'invoice_verified': False, 'automatic_gpu_retries': 0, 'lifecycle': lifecycle,
               'remote_volume': volume_name, 'laptop_dependency': False}
    save('allocation.json', receipt)
    save('gpu-budget.json', {'cap_gpu_seconds': 3600, 'attempts': [receipt],
         'status': 'closed', 'unsettled_gpu_reservations': 0})
    lines = ['# AverageJoe autonomous continuation', '',
             '**'+receipt['status']+'**. Allocated H100 seconds: '+str(round(elapsed,3)), '']
    if result:
        lines += [f"Updates: {result['initial_iteration']} → {result['iteration']}; final curriculum stage {result['stage']}.", '',
                  '| Assessment | Wins | Losses | Caps |', '|---|---:|---:|---:|']
        for label in ('baseline', 'final'):
            row = result[label]; lines.append(f"| {label} | {row['wins']} | {row['losses']} | {row['caps']} |")
        lines += ['', 'Assessment opponent is upstream random. These are curriculum results, not expert-strength calibration.',
                  '', 'Full state, source, map pools, assessments and viewing games are retained in this private Volume.']
    (control/'readout.md').write_text('\n'.join(lines)+'\n'); volume.commit()
    return receipt


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=ROOT/'evidence/generals-averagejoe-continuation-20261109')
    p.add_argument('--prepare-only', action='store_true')
    a = p.parse_args(); a.root = a.root.resolve()
    authorization = json.loads((a.root/'authorization.json').read_text())
    if authorization['new_gpu_seconds_cap'] != 3600:
        raise ValueError('Separate one-hour authorization required')
    if (a.root/'launch-claim.json').exists():
        raise ValueError('Attempt already reserved; reconcile it instead of retrying')
    from .lineage import verify_parent
    parent = ROOT/'evidence/generals-averagejoe-20261109/smoke-02/artifacts'
    parent_source = parent.parent/'source'
    lineage = verify_parent(parent, parent_source)
    write(a.root/'lineage.json', lineage)
    write(a.root/'source-binding.json', binding())
    snapshot = a.root/'source'
    if snapshot.exists():
        raise FileExistsError('Use preserved snapshot; no implicit overwrite/retry')
    (snapshot/'simulators/generals').mkdir(parents=True)
    shutil.copytree(HERE, snapshot/'simulators/generals/averagejoe', ignore=shutil.ignore_patterns('__pycache__', '.pytest_cache'))
    # Only transport/receipt helpers are needed by the remote CPU observer.
    helpers = snapshot/'simulators/generals/autoresearch'; helpers.mkdir()
    for name in ('__init__.py', 'provider.py', 'transport.py', 'contract.py', 'bundle.py'):
        shutil.copy2(ROOT/'simulators/generals/autoresearch'/name, helpers/name)
    (snapshot/'vendor/engines').mkdir(parents=True)
    for source, name in ((UPSTREAM, 'AverageJoe'), (ENGINE, 'generals-bots')):
        shutil.copytree(source, snapshot/'vendor/engines'/name, ignore=shutil.ignore_patterns('.git', '__pycache__', '.pytest_cache'))
    write(a.root/'source-files.json', {str(f.relative_to(snapshot)):sha(f) for f in snapshot.rglob('*') if f.is_file()})
    if a.prepare_only:
        return
    from simulators.generals.autoresearch.provider import configure_sandbox_waits, allocation_info
    configure_sandbox_waits()
    volume_name = 'generals-averagejoe-continuation-20261109'
    volume = modal.Volume.from_name(volume_name, environment_name='staging', create_if_missing=True, version=2)
    app = modal.App('generals-averagejoe-continuation-20261109')
    image = (modal.Image.debian_slim(python_version='3.12').apt_install('git')
             .pip_install('jax[cuda12]==0.12.0.1','flax==0.12.12','optax==0.2.8','numpy==2.5.4',
                          'pygame==2.6.1','python-socketio[client]==5.18.1','pytest==9.2.0','psutil==7.2.2')
             .pip_install('equinox==0.13.8','ruamel.yaml==0.19.1')
             .add_local_dir(snapshot, '/workspace', copy=True)
             .add_local_dir(ROOT/'evidence/generals-averagejoe-20261109/inputs-v2', '/inputs', copy=True)
             .add_local_dir(parent, '/parent', copy=True)
             .add_local_dir(parent_source, '/parent-source', copy=True)
             .add_local_file(a.root/'authorization.json','/job-metadata/authorization.json',copy=True)
             .add_local_file(a.root/'source-binding.json','/job-metadata/source-binding.json',copy=True)
             .add_local_file(a.root/'lineage.json','/job-metadata/lineage.json',copy=True)
             .env({'PYTHONPATH':'/workspace','OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4',
                   'XLA_PYTHON_CLIENT_PREALLOCATE':'false','JAX_DEFAULT_MATMUL_PRECISION':'tensorfloat32',
                   'AVERAGEJOE_VOLUME_V2':'1'}).workdir('/workspace'))
    observer = app.function(image=image, cpu=(.25,.25), memory=(1024,1024), volumes={'/retained':volume},
                           timeout=3750, startup_timeout=120, retries=0, max_containers=1,
                           min_containers=0, buffer_containers=0, single_use_containers=True,
                           serialized=True, include_source=False, name='settle_allocation')(observe)
    with modal.enable_output(), app.run(detach=True, environment_name='staging'):
        built = time.monotonic(); image.build(app)
        write(a.root/'image-build.json', {'elapsed_seconds': time.monotonic()-built, 'gpu_allocations': 0, 'unpriced_reserve_usd': .5})
        launched = time.time()
        write(a.root/'launch-claim.json', {'reserved_gpu_seconds':3600, 'automatic_gpu_retries':0,
              'launch_requested_unix':launched, 'app_id':app.app_id, 'remote_volume':volume_name})
        sandbox = modal.Sandbox.create('python','-u','-m','simulators.generals.averagejoe.autonomous',
                     '--root','/retained','--deadline',str(launched+3400),
                     app=app, image=image, gpu='H100!', cpu=(4,4), memory=(16384,16384),
                     timeout=3500, block_network=True, secrets=[], volumes={'/retained':volume},
                     env={'JAX_PLATFORMS':'cuda'}, environment_name='staging', include_oidc_identity_token=False,
                     name='averagejoe-continuation-20261109', tags={'campaign':'averagejoe-continuation-20261109','attempt':'01'})
        receipt = {'app_id':app.app_id,'sandbox_id':sandbox.object_id,
                   'volume_name':volume_name,'launch_requested_unix':launched,
                   'watchdog_deadline_unix':launched+3400,'provider_timeout_seconds':3500,
                   'cap_gpu_seconds':3600,'laptop_dependency':False,'automatic_gpu_retries':0}
        write(a.root/'provider-call.json',receipt)
        call = observer.spawn(app.app_id, sandbox.object_id, volume_name, launched)
        receipt['observer_call_id'] = call.object_id
        write(a.root/'provider-call.json',receipt)
        for _ in range(120):
            info = allocation_info(app.app_id,sandbox.object_id)
            if info and info.get('started_at'):
                write(a.root/'provider-start.json',info);break
            time.sleep(.5)
        sandbox.detach()
        print(json.dumps(receipt,indent=2),flush=True)


if __name__ == '__main__':
    main()
