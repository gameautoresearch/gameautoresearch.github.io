"""Matched elapsed training-window learning tests; no campaign spending ceiling.

The finite window is the experimental treatment, not a resurrected authorization
limit. Full GPU allocation (startup, cold compile, evaluation, save/read/replay,
artifact preparation and idle) is separately retained for the actual denominator.
"""
import json
import time
import modal
from integrations.generals.modal_structural import image, ROOT

app = modal.App('autoresearch-generals-mass-learning-20261108')


def execute(variant, seed, seconds, envs, minibatch_size):
    import io
    import os
    from pathlib import Path
    import subprocess
    import sys
    import tarfile
    start, start_unix = time.monotonic(), time.time()
    output = Path('/artifacts')
    output.mkdir()
    with tarfile.open(output/'executed-source.tgz', 'w:gz') as tar:
        for file in Path('/workspace/integrations/generals').rglob('*'):
            if "autoresearch" in file.relative_to(Path('/workspace/integrations/generals')).parts:
                continue
            if file.is_file() and file.suffix in ('.py', '.json', '.lock'):
                tar.add(file, arcname='integrations/generals/'+str(file.relative_to('/workspace/integrations/generals')))
    cwd = Path('/workspace')
    if variant == 'baseline':
        cwd = output/'baseline-source'
        cwd.mkdir()
        with tarfile.open('/inputs/baseline.tgz') as tar:
            tar.extractall(cwd, filter='data')
        (cwd/'candidates').mkdir()
        (cwd/'candidates/generals-bots').symlink_to('/workspace/candidates/generals-bots', target_is_directory=True)
    gpu_name = subprocess.check_output(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], text=True).strip()
    cfg = json.loads(Path('/workspace/integrations/generals/recipes/l4-small.json').read_text())
    cfg.update(seed=seed, max_updates=1000000, allocated_worker_seconds=seconds,
               checkpoint_interval=1000000, checkpoint_reserve_seconds=10,
               evaluation_reserve_seconds=10, development_eval_seeds=16,
               execution='optimized', hardware=dict(gpu_model=gpu_name, gpu_count=1,
                   attached_cpu_cores=4, attached_ram_gib=16,
                   isolation='Modal fixed physical CPU/RAM and one GPU; all allocation time counts'))
    if variant in ('frontier', 'implicit'):
        if envs*cfg['rollout_steps'] % minibatch_size:
            raise ValueError('Minibatch size must divide rollout samples')
        cfg.update(num_envs=envs, minibatches=envs*cfg['rollout_steps']//minibatch_size, packed_rollout=True,
                   historical_convolution='implicit' if variant == 'implicit' else 'slice', compact_opponent_inference=True,
                   sparse_reset_capacity=128, compute_dtype='bfloat16', compact_checkpoint=True)
    recipe = output/'recipe.json'
    recipe.write_text(json.dumps(cfg, indent=2)+'\n')
    env = dict(os.environ, PYTHONPATH=str(cwd)+':/workspace/candidates/generals-bots',
               GENERALS_ALLOCATION_CONFIRMED='1',
               JAX_COMPILATION_CACHE_DIR='/tmp/training-cache')
    command = [sys.executable, '-m', 'integrations.generals.lease', '--recipe', str(recipe),
               '--ledger', str(output/'allocation.json'), '--', sys.executable, '-m',
               'integrations.generals', 'train', '--device', 'gpu', '--recipe', str(recipe),
               '--output', str(output/'train')]
    with (output/'train.log').open('w') as log:
        try:
            result = subprocess.run(command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=seconds+30)
            code = result.returncode
        except subprocess.TimeoutExpired:
            code = 124
    checks = []
    if code == 0:
        # Public practice only. This is outside the training treatment window but
        # inside the charged full worker allocation and reported as such.
        env.update(GENERALS_ALLOCATION_START_UNIX=str(start_unix))
        commands = [
            [sys.executable, '-m', 'integrations.generals', 'evaluate', '--device', 'gpu',
             '--recipe', str(recipe), '--checkpoint', str(output/'train/latest.msgpack'),
             '--split', 'practice', '--opponents', 'random,expansion', '--output', str(output/'evaluation')],
            [sys.executable, '-m', 'integrations.generals.replay', str(output/'evaluation/replay.npz'),
             '--output', str(output/'replay-verification.json')],
        ]
        for i, command in enumerate(commands):
            t0 = time.monotonic()
            with (output/f'check-{i}.log').open('w') as log:
                check = subprocess.run(command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=120)
            checks.append(dict(command=command, seconds=time.monotonic()-t0, exit_code=check.returncode))
            if check.returncode:
                break
    receipt = dict(variant=variant, seed=seed, training_window_seconds=seconds, gpu_name=gpu_name,
                   cpu_physical_cores=4, ram_gib=16, exit_code=code, checks=checks,
                   function_seconds_before_archive=time.monotonic()-start,
                   rate_usd_per_second=(.000222 if 'L4' in gpu_name else .001097)+4*.0000131+16*.00000222,
                   config=cfg, scope='All worker phases included; training window is only the treatment, not the full denominator')
    (output/'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    binary = io.BytesIO()
    with tarfile.open(fileobj=binary, mode='w:gz') as tar:
        for file in output.rglob('*'):
            if file.is_file():
                tar.add(file, arcname=str(file.relative_to(output)))
    receipt['function_seconds'] = time.monotonic()-start
    receipt['estimated_resource_usd'] = receipt['function_seconds']*receipt['rate_usd_per_second']
    return binary.getvalue(), receipt


@app.function(image=image, gpu='L4', cpu=(4, 4), memory=(16384, 16384),
              max_containers=3, min_containers=0, buffer_containers=0, retries=0,
              timeout=600, startup_timeout=180, scaledown_window=2, single_use_containers=True)
def l4(variant: str, seed: int, seconds: int, envs: int, minibatch_size: int):
    return execute(variant, seed, seconds, envs, minibatch_size)


@app.function(image=image, gpu='H100', cpu=(4, 4), memory=(16384, 16384),
              max_containers=3, min_containers=0, buffer_containers=0, retries=0,
              timeout=600, startup_timeout=180, scaledown_window=2, single_use_containers=True)
def h100(variant: str, seed: int, seconds: int, envs: int, minibatch_size: int):
    return execute(variant, seed, seconds, envs, minibatch_size)


@app.local_entrypoint()
def main(tag: str, variant: str = 'frontier', gpu: str = 'L4', seeds: str = '42,43,44', seconds: int = 180,
         envs: int = 8192, minibatch_size: int = 8192):
    import io
    import hashlib
    import tarfile
    from modal.config import config
    if (config.get('environment') != 'staging' or variant not in ('baseline', 'frontier', 'implicit')
            or gpu not in ('L4', 'H100') or not tag.replace('-', '').isalnum() or not 60 <= seconds <= 300):
        raise ValueError('Use staging and a finite 60–300 second learning comparison')
    destination = ROOT/'evidence/generals-implementation/structural-20261108/learning-scale'/tag
    destination.mkdir(parents=True, exist_ok=False)
    path = destination/'launch.json'
    record = dict(tag=tag, variant=variant, gpu=gpu, seconds=seconds, app_id=app.app_id,
                  start_unix=time.time(), status='starting', calls=[],
                  authorization='New user-authorized frontier campaign; no old spending/hour/attempt limits')
    path.write_text(json.dumps(record, indent=2)+'\n')
    start = time.monotonic()
    calls = []
    for seed in map(int, seeds.split(',')):
        call = (l4 if gpu == 'L4' else h100).spawn(variant, seed, seconds, envs, minibatch_size)
        row = dict(seed=seed, call_id=call.object_id, status='submitted')
        calls.append((call, row))
        record['calls'].append(row)
        path.write_text(json.dumps(record, indent=2)+'\n')
    try:
        for call, row in calls:
            binary, receipt = call.get()
            folder = destination/f'seed-{row["seed"]}'
            folder.mkdir()
            (folder/'artifacts.tar.gz').write_bytes(binary)
            with tarfile.open(fileobj=io.BytesIO(binary), mode='r:gz') as tar:
                tar.extractall(folder, filter='data')
            row.update(status='succeeded' if receipt['exit_code'] == 0 and all(c['exit_code'] == 0 for c in receipt['checks']) else 'failed',
                       receipt=receipt, archive_bytes=len(binary), archive_sha256=hashlib.sha256(binary).hexdigest())
            path.write_text(json.dumps(record, indent=2)+'\n')
            print(json.dumps(dict(seed=row['seed'], status=row['status'], receipt=receipt)), flush=True)
        record['status'] = 'succeeded' if all(r['status'] == 'succeeded' for r in record['calls']) else 'partial_failed'
    except BaseException as exc:
        record.update(status='client_error_inspect_calls_before_retry', error_type=type(exc).__name__)
        raise
    finally:
        record['cold_launch_seconds'] = time.monotonic()-start
        path.write_text(json.dumps(record, indent=2)+'\n')
