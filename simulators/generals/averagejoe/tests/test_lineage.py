"""A continuation must not silently accept changed parent source or weights."""
import hashlib
import json
import pytest
from simulators.generals.averagejoe import lineage


def fixture(tmp_path, monkeypatch):
    archive = tmp_path/'archive'; parent = tmp_path/'parent'; parent.mkdir()
    code = archive/'simulators/generals/averagejoe/framework.py'; code.parent.mkdir(parents=True)
    code.write_text('unchanged model code\n')
    source = {'averagejoe_commit': 'pin', 'averagejoe_files': {}, 'engine_files': {},
              'recipe_sha256': 'recipe', 'framework_files': {'framework.py': lineage.sha(code)}}
    digest = hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest()
    (parent/'source.json').write_text(json.dumps({'sha256': digest, 'sources': source}))
    state = parent/'final-state.eqx'; state.write_bytes(b'fixture state')
    (parent/'final-state.json').write_text(json.dumps({'binding_sha256': digest, 'sha256': lineage.sha(state),
           'stage': 0, 'iteration': 20, 'exact_tensor_readback': True}))
    from simulators.generals.averagejoe.framework import recipe
    (parent/'config.json').write_text(json.dumps(recipe().to_dict()))
    current = {**source, 'framework_files': {**source['framework_files'], 'continuation.py': 'added'}}
    monkeypatch.setattr(lineage, 'binding', lambda: {'sha256': 'new', 'sources': current})
    return parent, archive, current


def test_added_orchestration_admitted_but_changed_model_rejected(tmp_path, monkeypatch):
    parent, archive, current = fixture(tmp_path, monkeypatch)
    assert lineage.verify_parent(parent, archive)['added_modules'] == ['continuation.py']
    current['framework_files']['framework.py'] = 'different'
    with pytest.raises(ValueError, match='Existing framework source changed'):
        lineage.verify_parent(parent, archive)


def test_parent_bytes_and_archived_source_tampering_rejected(tmp_path, monkeypatch):
    parent, archive, _ = fixture(tmp_path, monkeypatch)
    (parent/'final-state.eqx').write_bytes(b'changed checkpoint')
    with pytest.raises(ValueError, match='checksum mismatch'):
        lineage.verify_parent(parent, archive)
    (archive/'simulators/generals/averagejoe/framework.py').write_text('changed archive')
    with pytest.raises(ValueError, match='Archived parent source changed'):
        lineage.verify_parent(parent, archive)
