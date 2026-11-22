"""Host readers of a stopped training worker's retained output. Nothing here launches work.

A failed stage stays failed. Checkpoints the supervisor mirrored to durable
output are collected as opaque bytes; no candidate pickle is evaluated on the host.
"""
import hashlib
import json
import os
from pathlib import Path

from . import sdk


def collect_durable_checkpoint(volume, operation, output, request):
    """Collect a stopped worker's retained state even without results.tar.

    No pickle evaluation, worker creation or success promotion. All remote paths
    are fixed except for two validated SHA256 generation names.
    """
    import shutil
    import tempfile
    prefix=operation+'/output/stage/durable-checkpoints/'
    def small(path,limit):
        data=bytearray()
        for chunk in volume.read_file(path):
            data.extend(chunk)
            if len(data)>limit:raise ValueError('Oversized checkpoint metadata')
        return bytes(data)
    index=json.loads(small(prefix+'latest.json',16384))
    keys=index.get('generations',[])
    if index.get('schema')!='pokemon-durable-checkpoint-index-v1' or not isinstance(keys,list)or not 1<=len(keys)<=2:
        raise ValueError('Invalid durable checkpoint index')
    stage=Path(output)/'stage';stage.mkdir(parents=True,exist_ok=True)
    errors=[]
    for key in keys:
        if not isinstance(key,str)or len(key)!=64 or any(c not in '0123456789abcdef'for c in key):
            errors.append('invalid_generation');continue
        try:
            manifest=json.loads(small(prefix+key+'/manifest.json',1024*1024))
            meta=manifest['files']['checkpoint.pt'];size=meta['size_bytes']
            if (manifest.get('schema')!='pokemon-failed-training-artifacts-v1'
                    or manifest.get('status')!='retained_unvalidated'
                    or manifest.get('eligible_for_submission')is not False
                    or manifest.get('original_stage_successful')is not False
                    or manifest.get('checkpoint_retained')is not True
                    or manifest.get('source_sha256')!=request['source_sha256']
                    or manifest.get('config_sha256')!=sdk.identity(request['config'])
                    or manifest.get('runtime_sha256')!=sdk.runtime_signature()
                    or meta['sha256']!=key or type(size)is not int or not 0<size<=512*1024*1024):
                raise ValueError('Durable checkpoint provenance differs')
            with tempfile.TemporaryDirectory(prefix='.recover-',dir=stage)as temporary:
                candidate=Path(temporary);source=candidate/'source';source.mkdir()
                for name in (*sdk.EDITABLE,'config.json'):
                    (source/name).write_bytes(small(prefix+key+'/source/'+name,1024*1024))
                if sdk.source_signature(source)!=manifest['source_sha256']or sdk.identity(json.loads((source/'config.json').read_bytes()))!=manifest['config_sha256']:
                    raise ValueError('Durable source/config differs')
                count=0;digest=hashlib.sha256()
                with(candidate/'checkpoint.pt').open('xb')as stream:
                    for chunk in volume.read_file(prefix+key+'/checkpoint.pt'):
                        count+=len(chunk)
                        if count>size:raise ValueError('Oversized durable checkpoint')
                        stream.write(chunk);digest.update(chunk)
                    stream.flush();os.fsync(stream.fileno())
                if count!=size or digest.hexdigest()!=key:raise ValueError('Durable checkpoint content differs')
                manifest.update(collection_kind='direct_durable_volume',collection_errors=errors)
                (candidate/'manifest.json').write_text(json.dumps(manifest,indent=2))
                destination=stage/'recovery'
                if destination.exists():
                    if destination.is_symlink()or sdk.sha(destination/'checkpoint.pt')!=key:
                        raise ValueError('Existing collected recovery differs')
                else:os.replace(candidate,destination)
            return dict(available=True,checkpoint_sha256=key,submission_allowed=False,
                original_stage_successful=False,collection_kind='direct_durable_volume')
        except (OSError,ValueError,KeyError,TypeError)as exc:errors.append(type(exc).__name__)
    raise ValueError('No intact bound durable checkpoint: '+','.join(errors))


def training_outcome(output, successful=False):
    """Safe own-stage diagnostics; retains failure and separates recoverability."""
    output=Path(output);receipt_path=output/'stage/supervisor.json';receipt={}
    if receipt_path.is_file()and receipt_path.stat().st_size<=1024*1024:receipt=json.loads(receipt_path.read_bytes())
    recovery_path=output/'stage/recovery/manifest.json';recovery={}
    if recovery_path.is_file()and recovery_path.stat().st_size<=1024*1024:recovery=json.loads(recovery_path.read_bytes())
    retained=bool((successful and(output/'stage/checkpoint.pt').is_file())or recovery.get('checkpoint_retained'))
    return dict(stage_successful=bool(successful),termination_reason=receipt.get('termination_reason','worker_output_missing'),
        phase=receipt.get('phase'),checkpoint_retained=retained,
        recovery_validation='validated_successful_stage'if successful and retained else'unvalidated_until_explicit_resume'if retained else'no_retained_checkpoint',
        collection_kind=recovery.get('collection_kind','worker_archive'),
        explicit_recovery_required=retained and not successful,failed_allocation_still_charged=not successful,
        publication_error_type=receipt.get('periodic_checkpoint_error_type'),
        # A sealed stage whose trainer the supervisor ended (stop request, stage time limit) says so in its receipt.
        **{flag:True for flag in('stopped_early','cut_short')if successful and receipt.get(flag)is True})
