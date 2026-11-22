"""New user-authorized structural campaign; historical budget records are immutable.

Each call is a finite, single-use L4/4-core/16GiB allocation. No warm containers,
automatic retries or historical campaign budget resets. Actual full worker time
and local cold-launch time are recorded independently of throughput diagnostics.
"""
import json
import os
from pathlib import Path
import time
import modal
from integrations.generals.modal_efficiency import image, ROOT, RATE_SECOND

if modal.is_local():
    image = image.add_local_file(ROOT/'evidence/generals-implementation/structural-20261108/kernels-1/source.tgz', '/inputs/baseline.tgz')

app = modal.App('autoresearch-generals-structural-20261108')


@app.function(image=image, gpu='L4', cpu=(4, 4), memory=(16384, 16384),
              max_containers=1, min_containers=0, buffer_containers=0, retries=0,
              timeout=1800, startup_timeout=180, scaledown_window=2,
              single_use_containers=True)
def worker(batches: str, variants: str, modes: str):
    import io
    import subprocess
    import sys
    import tarfile
    start = time.monotonic()
    gpu_name = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], text=True).strip()
    rate = (.001097 if 'H100' in gpu_name else .000222)+4*.0000131+16*.00000222
    folder = Path('/artifacts')
    folder.mkdir(exist_ok=True)
    commands = []
    flags = {'default': '', 'while': '--xla_gpu_enable_command_buffer=+WHILE',
             'unroll': '--xla_gpu_enable_command_buffer=+WHILE --xla_gpu_command_buffer_unroll_loops=true',
             'off': '--xla_gpu_enable_command_buffer=', 'learning': '', 'validation': '', 'implicit-validation': ''}
    for mode in modes.split(','):
        env = dict(os.environ, XLA_FLAGS=flags[mode],
                   GENERALS_ALLOCATION_CONFIRMED='1',
                   GENERALS_ALLOCATION_START_UNIX=str(time.time()-(time.monotonic()-start)))
        command = [sys.executable, '-m', 'integrations.generals.structural_experiment',
                   '--output', str(folder/mode), '--snapshot', '/inputs/smoke.msgpack',
                   '--batches', batches, '--variants', variants, '--repeats', '7']
        if mode == 'learning':
            command = [sys.executable, '-m', 'integrations.generals.full_loop_experiment',
                       '--output', str(folder/'learning'), '--baseline-archive', '/inputs/baseline.tgz',
                       '--device', 'gpu', '--candidate', 'integrations/generals/recipes/l4-fast.json',
                       '--steps', '512', '--cli-check']
        if mode in ('validation', 'implicit-validation'):
            command = [sys.executable, '-m', 'integrations.generals.validate_frontier',
                       '--output', str(folder/'validation')]
            if mode == 'implicit-validation':
                command += ['--recipe', 'integrations/generals/recipes/h100-implicit.json']
        t0 = time.monotonic()
        with (folder/f'{mode}.log').open('w') as log:
            try:
                result = subprocess.run(command, cwd='/workspace', env=env, stdout=log,
                                        stderr=subprocess.STDOUT, timeout=550)
                code = result.returncode
            except subprocess.TimeoutExpired:
                code = 124
        commands.append(dict(mode=mode, command=command, xla_flags=flags[mode],
                             elapsed_seconds=time.monotonic()-t0, exit_code=code))
        (folder/'commands.json').write_text(json.dumps(commands, indent=2)+'\n')
        print(json.dumps(commands[-1]), flush=True)
    receipt = dict(function_seconds=time.monotonic()-start, commands=commands,
                   allocation=dict(gpu=gpu_name, gpu_count=1, cpu_physical_cores=4, ram_gib=16),
                   estimated_resource_usd=(time.monotonic()-start)*rate,
                   rate_usd_per_second=rate,
                   accounting='All allocated worker time including imports, compile, diagnostics, evaluation, idle and artifact preparation; provider invoice separate.')
    (folder/'worker-receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode='w:gz') as tar:
        for file in folder.rglob('*'):
            if file.is_file() and not any(p.startswith('cache-') for p in file.relative_to(folder).parts):
                tar.add(file, arcname=str(file.relative_to(folder)))
    receipt['function_seconds'] = time.monotonic()-start
    receipt['estimated_resource_usd'] = receipt['function_seconds']*rate
    receipt['archive_excludes'] = 'Derived JAX executable caches; all sources, inputs, logs, checkpoints, replay and metrics retained'
    return archive.getvalue(), receipt


@app.local_entrypoint()
def main(tag: str, batches: str = '16,128,512',
         variants: str = 'direct-baseline,direct-optimized,direct-guarded,direct-compact,direct-unroll2',
         modes: str = 'default,while,unroll'):
    import fcntl
    import hashlib
    import io
    import tarfile
    from modal.config import config
    if config.get('environment') != 'staging':
        raise RuntimeError('Pass --env staging')
    if not tag.replace('-', '').isalnum() or any(m not in ('default', 'while', 'unroll', 'off', 'learning', 'validation') for m in modes.split(',')):
        raise ValueError('Invalid tag/modes')
    folder = ROOT/'evidence/generals-implementation/structural-20261108/cuda'
    folder.mkdir(parents=True, exist_ok=True)
    with (folder/'launch.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = folder/'campaign.json'
        ledger = json.loads(path.read_text()) if path.exists() else dict(
            authorization='2026-11-08 user superseded prior dollar/hour/attempt ceilings; new campaign; historical receipts untouched', runs=[])
        if any(r['tag'] == tag for r in ledger['runs']):
            raise ValueError('Existing tag; inspect/reattach instead of duplicate launch')
        row = dict(tag=tag, app_id=app.app_id, status='starting', start_unix=time.time(),
                   batches=batches, variants=variants, modes=modes)
        ledger['runs'].append(row)
        path.write_text(json.dumps(ledger, indent=2)+'\n')
        start = time.monotonic()
        try:
            call = worker.spawn(batches, variants, modes)
            row.update(status='submitted', call_id=call.object_id)
            path.write_text(json.dumps(ledger, indent=2)+'\n')
            binary, receipt = call.get()
            dest = folder/tag
            dest.mkdir(exist_ok=False)
            (dest/'artifacts.tar.gz').write_bytes(binary)
            with tarfile.open(fileobj=io.BytesIO(binary), mode='r:gz') as tar:
                tar.extractall(dest, filter='data')
            row.update(status='succeeded' if all(c['exit_code'] == 0 for c in receipt['commands']) else 'failed',
                       receipt=receipt, archive_sha256=hashlib.sha256(binary).hexdigest(), archive_bytes=len(binary))
        except BaseException as exc:
            row.update(status='client_error_inspect_call_before_retry', error_type=type(exc).__name__)
            raise
        finally:
            row['cold_launch_seconds'] = time.monotonic()-start
            path.write_text(json.dumps(ledger, indent=2)+'\n')
        print(json.dumps(row, indent=2), flush=True)
