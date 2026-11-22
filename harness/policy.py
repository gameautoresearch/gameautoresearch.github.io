"""Host policy and identities: one Modal environment, source-bound releases and runs."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

from harness import workspace

ROOT = Path(__file__).resolve().parents[1]
ENVIRONMENT = 'game-autoresearch-project'
MODAL_WORKSPACE = 'duper'
HARNESS = 'terminus-harness-1'
HARBOR = 'harbor[modal]==0.25.0'
DEFAULT_GPU = 'L40S'
ARCHIVE_DATASET = 'game-autoresearch/game-autoresearch'
SCORED_CHECKS = ('exact_source_real_native_qualification', 'isolation_qualification', 'accounting_qualification',
                 'multiseed_calibration', 'explicit_new_campaign_authority')
COMPLETION = ('research_done', 'workers_settled', 'evaluation_valid', 'archive_verified')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def file_hashes(root):
    root = Path(root)
    if any(p.is_symlink() for p in root.rglob('*')):
        raise ValueError('Symlinks in sealed release')
    return {p.relative_to(root).as_posix(): workspace.sha(p)
            for p in sorted(root.rglob('*')) if p.is_file() and '__pycache__' not in p.parts}


def repository_files(directory):
    """Source files git knows or would add: never ignored caches, builds or local run state.

    A staged copy of the repository (the cloud coordinator's image) has no git
    metadata and contains exactly these files, so there every file is listed.
    """
    if not (ROOT / '.git').exists():
        return [p for p in sorted(Path(directory).rglob('*'))
                if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc']
    listed = subprocess.run(['git', 'ls-files', '-z', '--cached', '--others', '--exclude-standard', '--', str(directory)],
                            cwd=ROOT, capture_output=True, check=True).stdout.decode()
    return [ROOT / name for name in listed.split('\0') if name and (ROOT / name).is_file()]


def in_modal():
    """True inside a Modal container, where the coordinator runs for a cloud session."""
    return bool(os.environ.get('MODAL_TASK_ID'))


def source_identity(game, code_only=False):
    """Everything that decides how this game's sessions run: the harness and the game's research substrate.

    `code_only` keeps only Python source: what fresh final training and grading execute. Documentation and
    the researcher-facing templates still bind a session's start, but editing them must not invalidate the
    finals of sessions already running.
    """
    paths = []
    for path in repository_files(ROOT / 'harness'):
        parts = path.relative_to(ROOT / 'harness').parts
        # The shared core and this game's backend; another game's backend cannot affect this one.
        # The cloud launcher (cloud.py, its requirements) only decides where the coordinator process runs, as
        # tools/ does; it never changes what a session does, so editing it does not invalidate releases.
        if (parts[0] in ('tests', 'cloud_requirements') or parts == ('cloud.py',)
                or (parts[0] == 'games' and len(parts) > 2 and parts[1] != game)):
            continue
        if path.suffix in (('.py',) if code_only else ('.py', '.md', '.toml', '')):
            paths.append(path)
    paths += [p for p in repository_files(ROOT / workspace.catalog()[game]['autoresearch']) if p.suffix == '.py']
    return digest({str(p.relative_to(ROOT)): workspace.sha(p) for p in sorted(set(paths))})


def sidecar(release):
    release = Path(release).resolve()
    return release.with_name(release.name + '.harness.json')


def bind_release(game, release):
    """Seal a new constructor result. The release's own files are never changed."""
    release = Path(release).resolve()
    path = sidecar(release)
    if path.exists():
        raise FileExistsError('Release binding already exists')
    meta = workspace.verify(release)
    if meta['game'] != game:
        raise ValueError('Wrong game workspace')
    record = dict(schema=1, harness=HARNESS, game=game, release=str(release), environment=ENVIRONMENT,
                  modal_workspace=MODAL_WORKSPACE, source_identity=source_identity(game),
                  code_identity=source_identity(game, code_only=True),
                  workspace_sha256=meta['sha256'], release_files=file_hashes(release))
    record['sha256'] = digest(record)
    path.write_text(json.dumps(record, indent=2) + '\n')
    path.chmod(0o400)
    return record


def verify_release(game, release, code_only=False):
    """`code_only` (final training, grading) checks the Python source recorded at release time; releases bound
    before code identities existed fall back to the full check."""
    path = sidecar(release)
    if not path.is_file() or path.is_symlink():
        raise ValueError('No harness binding for this release. Releases from earlier harnesses are not adopted: '
                         'prepare a new one with `python -m harness prepare`')
    record = json.loads(path.read_text())
    body = {k: v for k, v in record.items() if k != 'sha256'}
    # The recorded path is informational: a release is identified by its bytes, so the
    # same release verifies wherever it is mounted (this machine or a cloud coordinator).
    if digest(body) != record.get('sha256') or record.get('harness') != HARNESS or record.get('game') != game:
        raise ValueError('Release binding differs')
    if record['environment'] != ENVIRONMENT or record['modal_workspace'] != MODAL_WORKSPACE:
        raise ValueError('Project environment differs')
    if code_only and 'code_identity' in record:
        if record['code_identity'] != source_identity(game, code_only=True):
            raise ValueError('Harness or game research code changed since this release; prepare a new release')
    elif record['source_identity'] != source_identity(game):
        raise ValueError('Harness or game research source changed since this release; prepare a new release')
    if record['release_files'] != file_hashes(release):
        raise ValueError('Release bytes changed')
    return record


def bind_run(game, release, out, settings):
    """A session directory belongs to exactly one release, variant, model and budget."""
    record = verify_release(game, release)
    out = Path(out).resolve()
    path = out.with_name(out.name + '.harness-run.json')
    value = json.loads(json.dumps(dict(harness=HARNESS, game=game, release_sha256=record['sha256'], session=str(out),
                                       settings=settings), sort_keys=True, default=str))
    if path.exists() or out.exists():
        raise FileExistsError('Choose a fresh session directory; existing sessions and budgets are never reset')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')
    path.chmod(0o400)
    return value


def run_binding(out):
    out = Path(out).resolve()
    path = out.with_name(out.name + '.harness-run.json')
    if not path.is_file():
        raise ValueError('Not a harness session directory')
    return json.loads(path.read_text())


def require_modal():
    """Paid work uses only the project environment of the duper workspace."""
    if in_modal():
        # The container's identity is the workspace that launched it; pin the environment.
        if os.environ.get('MODAL_ENVIRONMENT') != ENVIRONMENT:
            raise ValueError('Cloud coordinator must run in the project environment')
        return
    direct_modal(os.environ)
    from modal.config import _profile
    if _profile != MODAL_WORKSPACE:
        raise ValueError('Select MODAL_PROFILE=duper before starting the process; no fallback profile is allowed')
    os.environ['MODAL_ENVIRONMENT'] = ENVIRONMENT


def direct_modal(env):
    """Modal traffic goes direct, not through a local HTTP/SOCKS proxy.

    A session holds a long-lived channel to the researcher's box for every
    keystroke; through a desktop proxy that channel times out. Model calls
    still use the proxy. Set MODAL_DISABLE_API_PROXY=0 to override.
    """
    env.setdefault('MODAL_DISABLE_API_PROXY', '1')


def require_scored(game, release, receipt):
    """A scored session needs retained evidence for every gate, bound to this exact release."""
    binding = verify_release(game, release)
    if receipt is None:
        raise ValueError('Scored launch blocked: no exact-source qualification and new campaign authority')
    data = json.loads(Path(receipt).read_text())
    if data.get('release_sha256') != binding['sha256'] or not all(data.get(k) is True for k in SCORED_CHECKS):
        raise ValueError('Scored launch blocked: missing or mismatched qualification')
    evidence = data.get('evidence', {})
    if set(evidence) != set(SCORED_CHECKS):
        raise ValueError('Retained qualification evidence required for every check')
    for item in evidence.values():
        path = Path(item['path'])
        if not path.is_file() or workspace.sha(path) != item['sha256']:
            raise ValueError('Qualification evidence missing or changed')
    return data


def completion(out):
    """A run is complete only when research, settlement, evaluation and the private archive are each evidenced."""
    out = Path(out)
    receipt = out / 'completion-evidence.json'
    data = json.loads(receipt.read_text()) if receipt.is_file() else {}
    binding = out.with_name(out.name + '.harness-run.json')
    bound = binding.is_file() and data.get('run_binding_sha256') == workspace.sha(binding)
    values = {}
    for name in COMPLETION:
        item = data.get(name, {})
        path = Path(item.get('path', ''))
        values[name] = bool(bound and item.get('passed') is True and path.is_file()
                            and workspace.sha(path) == item.get('sha256'))
    archive = data.get('archive_verified', {})
    retained = {str(p.relative_to(out)): workspace.sha(p) for p in sorted(out.rglob('*'))
                if p.is_file() and p.name not in ('completion.json', 'completion-evidence.json')}
    # A transcript-only upload is never a complete run archive.
    values['archive_verified'] &= (all(archive.get(k) is True for k in ('full_retained_context',
                                                                         'remote_sizes_and_checksums_verified',
                                                                         'private_dataset'))
                                   and bool(archive.get('dataset_commit')) and archive.get('retained_files') == retained
                                   and archive.get('dataset') == ARCHIVE_DATASET)
    return dict(**values, complete=all(values.values()), normalized_reference_score=None)
