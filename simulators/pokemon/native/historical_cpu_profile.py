"""Bounded reproduction of resident-4 CPU simulation with L4 neural learning.

The archived native package and Linux binaries are uploaded unchanged. No CUDA
simulator is built or stepped. This is a constructed workload, not full admission.
One finite worker, no retries, and a new conservative allocation reservation.
"""
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import tarfile
import tempfile
import time

import modal

ROOT = Path(__file__).resolve().parents[3] if modal.is_local() else Path('/workspace')
OUT = ROOT / 'evidence/pokemon-implementation/native-cpu-completion-20261109'
ARCHIVE = ROOT / 'evidence/pokemon-implementation/native-rewrite-20261108/resident-4/receipts'
if modal.is_local():
    staging = tempfile.TemporaryDirectory(prefix='pokemon-historical-cpu-')
    UPLOAD = Path(staging.name)
    shutil.copytree(ARCHIVE / 'cpu-build', UPLOAD / 'cpu-build')
    shutil.copyfile(ARCHIVE / 'libpokemon_cuda.so', UPLOAD / 'libpokemon_cuda.so')
    for name in ('network.py', 'efficient.py'):
        shutil.copyfile(ROOT / 'simulators/pokemon' / name, UPLOAD / name)
    expected_integration = '77cbbb16351c1856077c4cdb5742db0330a8e8628d22f42ac44fb8abdc98badb'
    if hashlib.sha256(b''.join((UPLOAD / n).read_bytes() for n in ('network.py', 'efficient.py'))).hexdigest() != expected_integration:
        raise ValueError('Historical neural integration source changed')
    raw = (ROOT / 'evidence/pokemon-implementation/native-rewrite-20261108/gpu-corpus-3/oracle-fixtures.jsonl').read_bytes()
    if hashlib.sha256(raw).hexdigest() != '33fa7bb97955e5178a26763a426e1ea2fd9424c7d7ca4925ea0438826537bad9':
        raise ValueError('Historical fixture source changed')
    fixtures = [json.loads(line) for line in raw.splitlines()]
    (UPLOAD / 'fixtures.json.gz').write_bytes(gzip.compress(json.dumps(fixtures).encode(), mtime=0))
    historical = json.loads((ARCHIVE / 'results.json').read_bytes())
    expected_profile = next(r for r in historical['profiles'] if r['backend'] == 'native_cpu' and r['n'] == 2048)
    (UPLOAD / 'expected-profile.json').write_text(json.dumps(expected_profile))

app = modal.App('pokemon-native-cpu-history-20261109')
volume = modal.Volume.from_name('pokemon-native-cpu-history-20261109', environment_name='staging', create_if_missing=True)
image = (modal.Image.from_registry('nvidia/cuda:13.0.2-devel-ubuntu24.04', add_python='3.12')
         .pip_install('numpy==2.5.4', 'torch==2.15.0')
         .env({'PYTHONPATH': '/workspace', 'CUBLAS_WORKSPACE_CONFIG': ':4096:8',
               'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'})
         .add_local_dir(UPLOAD / 'cpu-build/source', '/workspace/integrations/pokemon/native')
         .add_local_file(UPLOAD / 'cpu-build/libpokemon_native.so', '/workspace/cpu-build/libpokemon_native.so')
         .add_local_file(UPLOAD / 'cpu-build/build.json', '/workspace/cpu-build/build.json')
         .add_local_file(UPLOAD / 'libpokemon_cuda.so', '/workspace/libpokemon_cuda.so')
         .add_local_file(UPLOAD / 'network.py', '/workspace/integrations/pokemon/network.py')
         .add_local_file(UPLOAD / 'efficient.py', '/workspace/integrations/pokemon/efficient.py')
         .add_local_file(UPLOAD / 'fixtures.json.gz', '/workspace/fixtures.json.gz')
         .add_local_file(UPLOAD / 'expected-profile.json', '/workspace/expected-profile.json')) if modal.is_local() else modal.Image.debian_slim()


@app.function(image=image, gpu='L4', cpu=(4, 4), memory=(8192, 8192),
              timeout=180, startup_timeout=300, max_containers=1,
              min_containers=0, buffer_containers=0, retries=0,
              single_use_containers=True, scaledown_window=2, volumes={'/receipts': volume})
def profile(label):
    started = time.perf_counter()
    result = dict(full_gen9_admission=False, simulation_backend='archived native CPU',
                  resources=dict(gpu='L4', cpus=4, RAM_GiB=8), rows=[])
    try:
        import statistics
        import torch
        from integrations.pokemon.native.resident import ResidentBattles
        from integrations.pokemon.native.resident_ppo import ResidentPPO
        from integrations.pokemon.network import Policy
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        fixtures = json.loads(gzip.decompress(Path('/workspace/fixtures.json.gz').read_bytes()))
        result.update(torch=torch.__version__, cuda=torch.version.cuda,
                      gpu=torch.cuda.get_device_name(), profile_templates=len(fixtures))
        torch.manual_seed(1729)
        torch.cuda.manual_seed_all(1729)
        tick = time.perf_counter()
        env = ResidentBattles('/workspace/cpu-build/libpokemon_native.so', '/workspace/libpokemon_cuda.so',
                              fixtures, 2048, backend='native_cpu', cpu_threads=3, phase_limit=128,
                              developer_scope='Exact archived component CPU reproduction; not full Gen9', diversify=True)
        model = Policy(hidden=64, embedding=8).cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=.001, eps=1e-5)
        trainer = ResidentPPO(env, model, optimizer, steps=32, minibatch=4096, epochs=2)
        result['setup_seconds'] = time.perf_counter() - tick
        result['warmup'] = trainer.collect_update()
        for repeat in range(5):
            torch.cuda.synchronize()
            tick = time.perf_counter()
            row = trainer.collect_update()
            torch.cuda.synchronize()
            row.update(repeat=repeat, seconds=time.perf_counter() - tick)
            row['samples_per_second'] = row['samples'] / row['seconds']
            result['rows'].append(row)
        result['median_samples_per_second'] = statistics.median(r['samples_per_second'] for r in result['rows'])
        expected = json.loads(Path('/workspace/expected-profile.json').read_text())
        keys = ('samples', 'natural', 'truncations', 'faults')
        result['exact_archived_workload_counts'] = all(
            all(a[k] == b[k] for k in keys)
            for a, b in zip([result['warmup'], *result['rows']], [expected['warmup'], *expected['rows']]))
        if not result['exact_archived_workload_counts']:
            raise AssertionError('Archived workload counts were not reproduced')
        result['source_signature'] = env.source_signature
        result['peak_cuda_bytes'] = torch.cuda.max_memory_allocated()
        env.close()
    except BaseException:
        import traceback
        result['error'] = traceback.format_exc()
    result['body_before_receipt_seconds'] = time.perf_counter() - started
    target = Path('/receipts') / label
    target.mkdir(exist_ok=False)
    (target / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
    volume.commit()
    result['body_seconds'] = time.perf_counter() - started
    return result


@app.local_entrypoint()
def main(label: str = 'historical-1'):
    from modal.config import config
    if config.get('environment') != 'staging':
        raise RuntimeError('Staging only')
    reserved = sum(json.loads(p.read_text()).get('reserved_L4_seconds', 0)
                   for p in OUT.glob('*/allocation.json'))
    if reserved + 480 > 1200:
        raise RuntimeError('Separate 1200-second L4 validation reservation exhausted')
    target = OUT / label
    target.mkdir(parents=True, exist_ok=False)
    # 180-second body + up to 300-second worker startup, conservatively reserved.
    # This task does not reuse or mutate any historical ledger.
    ledger = dict(status='reserved', aggregate_cap_L4_seconds=1200,
                  reserved_L4_seconds=480, timeout_seconds=180, startup_timeout_seconds=300,
                  retries=0, resources=dict(gpu='L4', cpus=4, RAM_GiB=8), label=label)
    (target / 'allocation.json').write_text(json.dumps(ledger, indent=2) + '\n')
    started = time.perf_counter()
    call = profile.spawn(label)
    ledger.update(status='running', function_call_id=call.object_id)
    (target / 'allocation.json').write_text(json.dumps(ledger, indent=2) + '\n')
    try:
        result = call.get(timeout=500)
        (target / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
        ledger.update(status='failed_with_receipt' if 'error' in result else 'completed',
                      client_seconds=time.perf_counter() - started, body_seconds=result['body_seconds'],
                      financial_settlement_verified=False)
    finally:
        (target / 'allocation.json').write_text(json.dumps(ledger, indent=2) + '\n')
    print(json.dumps(result, indent=2))
