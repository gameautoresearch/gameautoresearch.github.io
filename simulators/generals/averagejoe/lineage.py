"""Explicit, source-bound admission of an older checkpoint into added orchestration.

Existing model/optimizer/engine/framework files must be byte-identical. Only new
modules may be added. The archived old source is verified independently, rather
than weakening checkpoint.verify's exact-current-source rule.
"""
from pathlib import Path
import hashlib
import json
from .bootstrap import binding, sha


def verify_parent(parent, archived_source):
    parent, archived_source = Path(parent), Path(archived_source)
    old = json.loads((parent / 'source.json').read_text())
    sources = old['sources']
    digest = hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()
    if digest != old['sha256']:
        raise ValueError('Parent source manifest digest mismatch')
    layouts = {
        'averagejoe_files': 'vendor/engines/AverageJoe',
        'engine_files': 'vendor/engines/generals-bots',
        'framework_files': 'simulators/generals/averagejoe',
    }
    for field, base in layouts.items():
        for name, expected in sources[field].items():
            if sha(archived_source / base / name) != expected:
                raise ValueError('Archived parent source changed: ' + name)
    current = binding()
    for field in ('averagejoe_commit', 'averagejoe_files', 'engine_files', 'recipe_sha256'):
        if current['sources'][field] != sources[field]:
            raise ValueError('Incompatible continuation source: ' + field)
    for name, expected in sources['framework_files'].items():
        if current['sources']['framework_files'].get(name) != expected:
            raise ValueError('Existing framework source changed: ' + name)
    meta = json.loads((parent / 'final-state.json').read_text())
    if meta['binding_sha256'] != old['sha256'] or sha(parent / 'final-state.eqx') != meta['sha256']:
        raise ValueError('Parent checkpoint/source checksum mismatch')
    from .framework import recipe
    if json.loads((parent / 'config.json').read_text()) != recipe().to_dict():
        raise ValueError('Parent recipe drift')
    if meta['stage'] != 0 or not meta['exact_tensor_readback']:
        raise ValueError('This adapter requires the verified stage-0 parent')
    return {'parent_binding_sha256': old['sha256'], 'new_binding_sha256': current['sha256'],
            'parent_checkpoint_sha256': meta['sha256'], 'parent_iteration': meta['iteration'],
            'admission': 'Archived parent verified; all existing source/config identical; added orchestration only',
            'added_modules': sorted(set(current['sources']['framework_files']) - set(sources['framework_files']))}
