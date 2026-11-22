"""Dedicated full-loop validation, independent of the serialized tuning queue."""
import json
from pathlib import Path
import time
import modal
from integrations.generals.modal_structural import image, worker as structural_worker, ROOT, RATE_SECOND

app = modal.App('autoresearch-generals-full-loop-20261108')


@app.function(image=image, gpu='L4', cpu=(4, 4), memory=(16384, 16384),
              max_containers=1, min_containers=0, buffer_containers=0, retries=0,
              timeout=1200, startup_timeout=180, scaledown_window=2,
              single_use_containers=True)
def worker(mode: str):
    return structural_worker.local('128', '', mode)


@app.function(image=image, gpu='H100', cpu=(4, 4), memory=(16384, 16384),
              max_containers=1, min_containers=0, buffer_containers=0, retries=0,
              timeout=1200, startup_timeout=180, scaledown_window=2,
              single_use_containers=True)
def h100_worker():
    return structural_worker.local('128', '', 'implicit-validation')


@app.local_entrypoint()
def main(tag: str, mode: str = 'learning'):
    import hashlib
    import io
    import tarfile
    from modal.config import config
    if config.get('environment') != 'staging' or not tag.replace('-', '').isalnum() or mode not in ('learning', 'validation', 'implicit-validation'):
        raise ValueError('staging and a safe unique tag are required')
    folder = ROOT/'evidence/generals-implementation/structural-20261108/cuda-learning'
    folder.mkdir(parents=True, exist_ok=True)
    destination = folder/tag
    destination.mkdir(exist_ok=False)
    path = destination/'launch.json'
    row = dict(tag=tag, app_id=app.app_id, status='starting', start_unix=time.time(),
               authorization='New 2026-11-08 campaign; old caps/attempt limits explicitly superseded; historical ledgers unchanged')
    path.write_text(json.dumps(row, indent=2)+'\n')
    start = time.monotonic()
    try:
        call = h100_worker.spawn() if mode == 'implicit-validation' else worker.spawn(mode)
        row.update(status='submitted', call_id=call.object_id)
        path.write_text(json.dumps(row, indent=2)+'\n')
        binary, receipt = call.get()
        (destination/'artifacts.tar.gz').write_bytes(binary)
        with tarfile.open(fileobj=io.BytesIO(binary), mode='r:gz') as tar:
            tar.extractall(destination, filter='data')
        row.update(status='succeeded' if all(c['exit_code'] == 0 for c in receipt['commands']) else 'failed',
                   receipt=receipt, archive_sha256=hashlib.sha256(binary).hexdigest(), archive_bytes=len(binary))
    except BaseException as exc:
        row.update(status='client_error_inspect_call_before_retry', error_type=type(exc).__name__)
        raise
    finally:
        row['cold_launch_seconds'] = time.monotonic()-start
        path.write_text(json.dumps(row, indent=2)+'\n')
    print(json.dumps(row, indent=2), flush=True)
