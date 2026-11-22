"""Massively parallel device scaling on fixed L4 or H100 allocations.

Each batch runs in a fresh child process on the same allocated GPU. The complete
allocation includes compile/autotune, all probes, serialization and idle time.
Large-batch PPO changes are logged separately from fixed-batch kernel changes.
"""
import json
import time
import modal
from integrations.generals.modal_structural import image, ROOT

app = modal.App('autoresearch-generals-scale-20261108')


def execute(batches, variants, minibatch_size):
    import io
    import os
    from pathlib import Path
    import subprocess
    import sys
    import tarfile
    start = time.monotonic()
    output = Path('/artifacts')
    output.mkdir(exist_ok=True)
    commands = []
    for batch in batches.split(','):
        command = [sys.executable, '-m', 'integrations.generals.structural_experiment',
                   '--output', str(output/f'batch-{batch}'), '--snapshot', '/inputs/smoke.msgpack',
                   '--batches', batch, '--variants', variants, '--massive', '--repeats', '7',
                   '--minibatch-size', str(minibatch_size)]
        env = dict(os.environ)
        if variants in ('split', 'split-bf16'):
            env['JAX_PLATFORMS'] = 'cuda,cpu'
            command = [sys.executable, '-m', 'integrations.generals.split_experiment',
                       '--output', str(output/f'batch-{batch}'), '--snapshot', '/inputs/smoke.msgpack', '--batches', batch,
                       '--dtype', 'bfloat16' if variants == 'split-bf16' else 'float32']
        t0 = time.monotonic()
        with (output/f'batch-{batch}.log').open('w') as log:
            try:
                result = subprocess.run(command, cwd='/workspace', env=env, stdout=log, stderr=subprocess.STDOUT, timeout=600)
                code = result.returncode
            except subprocess.TimeoutExpired:
                code = 124
        row = dict(batch=int(batch), command=command, seconds=time.monotonic()-t0, exit_code=code)
        commands.append(row)
        (output/'commands.json').write_text(json.dumps(commands, indent=2)+'\n')
        print(json.dumps(row), flush=True)
    receipt = dict(commands=commands, function_seconds=time.monotonic()-start,
                   cpu_physical_cores=4, ram_gib=16, single_gpu=True,
                   note='Identical attached CPU/RAM for both GPU models. Actual GPU model/memory is in child manifest. No utilization or setup discounts.')
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode='w:gz') as tar:
        for file in output.rglob('*'):
            if file.is_file():
                tar.add(file, arcname=str(file.relative_to(output)))
    receipt['function_seconds'] = time.monotonic()-start
    return archive.getvalue(), receipt


@app.function(image=image, gpu='L4', cpu=(4, 4), memory=(16384, 16384),
              max_containers=1, min_containers=0, buffer_containers=0, retries=0,
              timeout=2400, startup_timeout=180, scaledown_window=2, single_use_containers=True)
def l4(batches: str, variants: str, minibatch_size: int):
    return execute(batches, variants, minibatch_size)


@app.function(image=image, gpu='H100', cpu=(4, 4), memory=(16384, 16384),
              max_containers=1, min_containers=0, buffer_containers=0, retries=0,
              timeout=2400, startup_timeout=180, scaledown_window=2, single_use_containers=True)
def h100(batches: str, variants: str, minibatch_size: int):
    return execute(batches, variants, minibatch_size)


@app.local_entrypoint()
def main(tag: str, gpu: str = 'L4', batches: str = '128,512,2048,8192',
         variants: str = 'direct-scaled,bf16-scaled', minibatch_size: int = 0):
    import hashlib
    import io
    import tarfile
    from modal.config import config
    if config.get('environment') != 'staging' or gpu not in ('L4', 'H100') or not tag.replace('-', '').isalnum():
        raise ValueError('Use staging, L4/H100 and a unique safe tag')
    destination = ROOT/'evidence/generals-implementation/structural-20261108/scaling'/tag
    destination.mkdir(parents=True, exist_ok=False)
    path = destination/'launch.json'
    row = dict(tag=tag, gpu=gpu, status='starting', app_id=app.app_id, start_unix=time.time(),
               batches=batches, variants=variants, minibatch_size=minibatch_size,
               authorization='2026-11-08 user requested massive parallelism and superseded old experiment caps')
    path.write_text(json.dumps(row, indent=2)+'\n')
    start = time.monotonic()
    try:
        call = (l4 if gpu == 'L4' else h100).spawn(batches, variants, minibatch_size)
        row.update(status='submitted', call_id=call.object_id)
        path.write_text(json.dumps(row, indent=2)+'\n')
        binary, receipt = call.get()
        (destination/'artifacts.tar.gz').write_bytes(binary)
        with tarfile.open(fileobj=io.BytesIO(binary), mode='r:gz') as tar:
            tar.extractall(destination, filter='data')
        row.update(status='succeeded' if all(c['exit_code'] == 0 for c in receipt['commands']) else 'partial_failed',
                   receipt=receipt, archive_bytes=len(binary), archive_sha256=hashlib.sha256(binary).hexdigest())
    except BaseException as exc:
        row.update(status='client_error_inspect_call_before_retry', error_type=type(exc).__name__)
        raise
    finally:
        row['cold_launch_seconds'] = time.monotonic()-start
        path.write_text(json.dumps(row, indent=2)+'\n')
    print(json.dumps(row, indent=2), flush=True)
