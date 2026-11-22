"""Offline release construction: frozen worker source, the researcher workspace and private grading seeds.

No training, worker allocation or network access happens here.
"""
from pathlib import Path
import json
import secrets
import shutil
import time

from .contract import DEFAULT, GAME, HERE, PANEL, ROOT, RUNTIME, frozen_manifest, sha, write_json

CONTROL = ('__init__.py', 'contract.py', 'bundle.py', 'wire.py', 'policy_worker.py', 'arena.py', 'grading.py',
           'train_entry.py', 'evaluate_entry.py', 'export_entry.py')
BENZENE = 'vendor/engines/benzene'


def prepare(out):
    """Build a new release in `out`. The MoHex source is staged for the worker image to compile."""
    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError('Choose a fresh release directory; releases and seed manifests are immutable')
    lock = json.loads((GAME / 'source.lock.json').read_text())
    out.mkdir(parents=True)
    frozen, public, private = out / 'frozen', out / 'public', out / 'private'
    frozen.mkdir()
    public.mkdir()
    private.mkdir(mode=0o700)
    (frozen / 'simulators').mkdir()
    (frozen / 'simulators/__init__.py').write_text('')
    game = frozen / 'simulators/hex'
    for name in RUNTIME:
        target = game / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(GAME / name, target)
    for name in CONTROL:
        target = game / 'autoresearch' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(HERE / name, target)
    # MoHex source at the same relative path as in the repository, verified against the pin.
    upstream = json.loads((ROOT / 'vendor/upstream-git/manifest.json').read_text())['benzene']
    if upstream['commit'] != lock['commit']:
        raise ValueError('MoHex source lock and upstream metadata disagree')
    for name in lock['files']:
        source, target = ROOT / BENZENE / name, frozen / BENZENE / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    changed = sorted(name for name, value in lock['files'].items() if sha(frozen / BENZENE / name) != value)
    if changed != sorted(lock.get('local_changes', {})):
        raise ValueError('Staged MoHex source differs from its pin beyond the recorded local changes: ' + ', '.join(changed))
    for name, value in lock.get('local_changes', {}).items():
        if sha(frozen / BENZENE / name) != value:
            raise ValueError('Recorded MoHex local change drifted: ' + name)
    shutil.copytree(HERE / 'starter', public / 'agent', ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(HERE / 'recipes', public / 'recipes')
    for name in ('benchmark.json', 'CONTRACT.md'):
        shutil.copy2(HERE / name, public / name)
    # The researcher sees only the canonical workspace built from this public tree.
    from harness.workspace import stage as stage_workspace
    stage_workspace('hex', public)
    write_json(public / 'session.json', DEFAULT)
    write_json(public / 'engine-manifest.json', frozen_manifest())
    write_json(public / 'practice-opponents.json', list(PANEL))
    write_json(private / 'grade-seeds.json', {'split': 'heldout', 'opening_seed': secrets.randbits(128),
                                              'policy_seed': secrets.randbits(63),
                                              'openings': DEFAULT['grade_openings']}, exclusive=True)
    write_json(private / 'panel.json', {'practice': list(PANEL), 'official': list(PANEL)}, exclusive=True)
    manifest = {'version': DEFAULT['version'], 'mode': DEFAULT['mode'], 'created_unix': time.time(),
                'engine': frozen_manifest(), 'mohex': {'commit': lock['commit'], 'files': len(lock['files']),
                                                       'local_changes': sorted(lock.get('local_changes', {}))},
                'public_files': {str(p.relative_to(public)): sha(p) for p in public.rglob('*') if p.is_file()},
                'private_seed_commitments': {'grade-seeds.json': sha(private / 'grade-seeds.json')},
                'gpu_jobs_started': 0, 'model_calls': 0, 'official_grading_ready': True,
                'readiness': 'prepared offline; the raw panel score needs no calibration run'}
    manifest['frozen_files'] = {str(p.relative_to(frozen)): sha(p) for p in frozen.rglob('*') if p.is_file()}
    write_json(out / 'release.json', manifest, exclusive=True)
    audit_release(out)
    return {k: v for k, v in manifest.items() if not k.endswith('_files')}


def audit_release(out):
    out = Path(out)
    m = json.loads((out / 'release.json').read_text())
    for tree, key in (('public', 'public_files'), ('frozen', 'frozen_files')):
        actual = {str(p.relative_to(out / tree)) for p in (out / tree).rglob('*') if p.is_file()}
        if actual != set(m[key]):
            raise ValueError('Unmanifested or missing release files: ' + tree)
    for tree, key in (('public', 'public_files'), ('frozen', 'frozen_files')):
        for name, value in m[key].items():
            if sha(out / tree / name) != value:
                raise ValueError(f'{tree.capitalize()} context drift: {name}')
    for name, value in m['private_seed_commitments'].items():
        if sha(out / 'private' / name) != value:
            raise ValueError('Private seed commitment drift')
    for root in (out / 'public', out / 'frozen'):
        for p in root.rglob('*'):
            if p.is_symlink():
                raise ValueError('Symlink in staged release')
            if any(part in ('private', 'hidden') for part in p.relative_to(root).parts) or \
                    p.name == 'grade-seeds.json' or p.suffix in ('.key', '.pt'):
                raise ValueError('Private asset staged: ' + str(p))
    return {'public_files': len(m['public_files']), 'frozen_files': len(m['frozen_files']),
            'official_grading_ready': m['official_grading_ready']}
