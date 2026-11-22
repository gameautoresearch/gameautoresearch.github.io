"""Reject stale/generated native constants without invoking a JVM at runtime."""
from pathlib import Path
import hashlib,json,sys
HERE=Path(__file__).resolve().parent

def validate_data():
    from data_codegen import emit
    folder=HERE/'generated-v2';source=folder/'task-data.json';manifest=folder/'manifest.json';header=folder/'config.hpp'
    data=json.loads(source.read_text());receipt=json.loads(manifest.read_text());digest=hashlib.sha256(source.read_bytes()).hexdigest()
    if digest!=receipt['task_data_sha256']:raise ValueError('native task data differs from immutable export receipt')
    lock=HERE.parent/'rules.lock.json';locked=json.loads(lock.read_text())
    want=hashlib.sha256(lock.read_bytes()).hexdigest()
    if data['provenance']['rules_lock_sha256']!=want:raise ValueError('native exported data rules/source pin differs')
    if data['provenance']['upstream_commit']!='f0659dafabfb9d2a4a8d3b22a9d14de306f5e4de':raise ValueError('unsupported native upstream pin')
    expected=emit(data)
    if not header.exists() or header.read_text()!=expected:raise ValueError('native generated configuration is stale; recompile pinned constants explicitly')
    return {'task_data_sha256':digest,'config_sha256':hashlib.sha256(header.read_bytes()).hexdigest(),'rules_lock_sha256':want,'upstream_commit':data['provenance']['upstream_commit']}
