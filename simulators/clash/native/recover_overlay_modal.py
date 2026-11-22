"""CPU-only recovery of exact old overlay from two explicitly owned Modal apps."""
from pathlib import Path
import hashlib,json,time
import modal
SOURCE_IMAGE='im-eZQ5seKRMmQSU4Tlmqd7Go'
EXPECTED='13e16dc5dcad00c003fde86b470c12d5b99eaeea85cc9cb6cce921c0d03c7379'
OWN_APPS={'native-optimized-1':'ap-07NzJheIOEGNM5bIrO9PoT','native-baselines-1':'ap-fIlypBsRL74c31zwDW6CZi'}
app=modal.App('clash-overlay-recovery-20261108')
volume=modal.Volume.from_name('clash-structural-artifacts-20261108',create_if_missing=False)
image=modal.Image.from_id(SOURCE_IMAGE)

@app.function(image=image,cpu=(1,1),memory=(4096,4096),volumes={'/artifacts':volume},max_containers=1,min_containers=0,scaledown_window=2,retries=0,timeout=180)
def recover(recovery_id:str):
    import os,shutil
    started=time.perf_counter();cpu=time.process_time();source=Path('/opt/upload.tar.gz')
    receipt={'format':1,'source_image_id':SOURCE_IMAGE,'owned_apps':OWN_APPS,'expected_sha256':EXPECTED,'gpu_count':0,'cpu_physical_cores':1,'ram_gib':4,'scope':'read-only exact archived overlay recovery from immutable owned image; no simulation/training/GPU'}
    if not source.is_file():receipt.update(status='not-recoverable-from-image-alone',reason='/opt/upload.tar.gz is absent; original add_local_file was a function mount, not a baked image layer')
    else:
        digest=hashlib.sha256(source.read_bytes()).hexdigest();receipt['observed_sha256']=digest;receipt['source_bytes']=source.stat().st_size
        if digest!=EXPECTED:receipt.update(status='hash-mismatch-no-copy',reason='retained image file differs from exact expected overlay; no source substituted')
        else:receipt['status']='recovered-exact-overlay'
    for run_id in OWN_APPS:
        target=Path('/artifacts')/run_id/recovery_id;target.mkdir(parents=False,exist_ok=False)
        if receipt['status']=='recovered-exact-overlay':
            destination=target/'source-overlay.tar.gz'
            with destination.open('xb') as handle:
                with source.open('rb') as raw:shutil.copyfileobj(raw,handle)
            assert hashlib.sha256(destination.read_bytes()).hexdigest()==EXPECTED
        receipt.update(function_body_seconds=time.perf_counter()-started,cpu_process_seconds=time.process_time()-cpu,conservative_load_headroom_seconds=60)
        (target/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    volume.commit();receipt['function_seconds_including_commit']=time.perf_counter()-started
    return receipt

@app.local_entrypoint()
def main(recovery_id:str='recovered-overlay-1'):
    from modal.config import config
    if config.get('environment')!='staging':raise RuntimeError('staging only')
    root=Path(__file__).resolve().parents[3];out=root/'evidence/clash-implementation/native-training-20261108'/recovery_id;out.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter();call=recover.spawn(recovery_id)
    (out/'submitted.json').write_text(json.dumps({'function_call_id':call.object_id,'source_image_id':SOURCE_IMAGE,'owned_apps':OWN_APPS,'gpu_count':0},indent=2)+'\n')
    result=call.get();result.update(controller_seconds=time.perf_counter()-started,function_call_id=call.object_id)
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
