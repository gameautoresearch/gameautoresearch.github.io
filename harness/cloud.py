"""The cloud coordinator: run one harness command in a Modal container instead of on this machine.

    .venv-harness/bin/python -m harness.cloud launch --campaign NAME [--secret NAME ...] -- \
        run pokemon recipe --release evidence/releases-.../pokemon-2 --out pokemon/opus/main --model ... --execute
    .venv-harness/bin/python -m harness.cloud status --campaign NAME --out pokemon/opus/main
    .venv-harness/bin/python -m harness.cloud pull --campaign NAME [--out pokemon/opus/main] --to runs/cloud

Everything a session does still happens exactly as `python -m harness ...` does it: the same broker,
backends, Harbor, budgets and records. Only the process that runs it moves. On this machine the
coordinator has to stay awake and connected for the whole session (about 8 h for 3 h of research and a
1 h final); a sleeping laptop or a network drop stops its workers. In a Modal container it does not.

How it is put together:
- Image (one per game): the repository's source as git lists it (exactly what source identities are
  computed from, so releases verify unchanged), the game's host requirements in `/opt/game`, and Harbor
  with its pinned lock in `/opt/harbor`. The two keep their own Modal client versions, as on this machine.
- Releases are uploaded once to the `ar-coordinator-releases` Volume, keyed by their binding hash, with
  their `.harness.json` beside them, and verified there by the same byte check as here.
- The session runs in the container's local disk under `/work/<campaign>/` (POSIX locks and fsync behave
  as on a laptop). A mirror thread copies new and changed files to the `ar-coordinator-records` Volume and
  commits it every `MIRROR_SECONDS` and at the end; `pull` brings them back here.
- Model keys reach the container only as Modal Secrets (`--secret`), as environment variables for Harbor
  on the coordinator. They never enter the researcher's box (Terminus-2) and never enter a record.
- Each session is its own deployed app (`ar-coord-<campaign>-<out>`), so launching or stopping one never
  touches another. A finished app is idle and costs nothing; `stop` ends a running one.

Not supported in the cloud: `--key-file` and a local `--api-base` (the Claude OAuth proxy reads this
Mac's keychain). Use a Modal Secret with the provider's API key instead.
"""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tarfile
import tempfile
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
ENVIRONMENT = 'game-autoresearch-project'
MODAL_WORKSPACE = 'duper'
RECORDS_VOLUME = 'ar-coordinator-records'
RELEASES_VOLUME = 'ar-coordinator-releases'
RECORDS, RELEASES, REPO, WORK = '/records', '/releases', '/repo', '/work'
GAME_VENV, HARBOR_VENV = '/opt/game', '/opt/harbor'
PYTHON = '3.12.9'
MIRROR_SECONDS = 60
# How often a running coordinator looks for the operator's pause request in the records Volume.
REQUEST_SECONDS = 15
PAUSE_REQUEST = 'pause.request'
# A Modal function lives at most 24 hours; the coordinator gets all of it.
TIMEOUT_SECONDS = 86400
# (reservation, cap): Modal bills max(reservation, use); a coordinator mostly waits on Harbor and workers.
CPU, MEMORY_MIB = (0.25, 2.0), (1024, 8192)
TORCH_CPU_INDEX = 'https://download.pytorch.org/whl/cpu'
# The game's host requirements: what its research interpreter was built from.
HOST_REQUIREMENTS = {
    'clash': 'simulators/clash/autoresearch/requirements-host.txt',
    'doudizhu': 'simulators/doudizhu/autoresearch/requirements-host.txt',
    'generals': 'simulators/generals/autoresearch/requirements-host.txt',
    'pokemon': 'simulators/pokemon/autoresearch/requirements-host.txt',
    'polytopia': 'simulators/polytopia/autoresearch/requirements-host.txt',
    'tavern': 'simulators/tavern/autoresearch/requirements-host.txt',
    'hex': 'simulators/hex/requirements.txt',
}
# Every game's host requirements pin the workers' Modal client (1.5.2) except Hex's, which lists only its
# engine's packages; its research interpreter here carries the same client.
HOST_PINS = {'hex': ('modal[api-proxy-support]==1.5.2',)}
HARBOR_LOCK = 'harness/cloud_requirements/harbor.lock.txt'
# Harbor 0.25.0's add_anthropic_caching puts cache_control on an empty text block, which Anthropic rejects (a Claude
# researcher whose turn came back empty fails its trial); the image applies this fix after installing the lock.
HARBOR_PATCH = 'harness/cloud_requirements/harbor-0.25.0-empty-cache-block.patch'
# What the image holds from the repository (as git lists it). Large research archives that no host
# code reads are left out.
SOURCE = ('harness', 'simulators', 'integrations', 'tools', 'AGENTS.md')
SOURCE_EXCLUDE = ('simulators/polytopia/benchmark/', 'simulators/polytopia/third_party/',
                  'simulators/polytopia/docs/', 'simulators/polytopia/web/', 'simulators/polytopia/research_notes/')
# Git-ignored files a game's host code still reads, carried into the image anyway. Clash's release verification
# hashes the pinned native task data (`simulators/clash/native/data_binding.py`), which is generated and ignored.
IGNORED_REQUIRED = {'clash': ('simulators/clash/native/generated-v2',)}
# Upstream engines a game's host imports by path (the local research venvs have them as editable installs) and
# whose pinned git metadata the host checks (Generals' frozen_manifest): the image puts them on PYTHONPATH and
# restores their .git from the tracked provenance archive in vendor/upstream-git/.
ENGINE_PATHS = {'generals': ('vendor/engines/generals-bots',)}
UPSTREAM_GIT = {'generals': ('generals-bots',)}
# Compatibility links a game's host resolves paths through (Generals reads its engine as candidates/generals-bots).
LINKS = {'generals': ('candidates',)}
PATH_FLAGS = {'--out', '--session'}
# Read-only inputs a command takes from the campaign's records (a continuation seed): `launch` uploads the local
# folder to <campaign>/inputs/, and the coordinator copies it into /work before the command starts.
INPUT_FLAGS = {'--continue-from'}
INPUTS = 'inputs'
LOCAL_ONLY = ('--key-file',)


# ---------------------------------------------------------------- staging (this machine)

def source_files(game):
    """The repository files the coordinator image carries, as git lists them, plus the game's upstream engines."""
    from harness import workspace
    upstream = tuple(workspace.catalog()[game].get('upstream', ()))
    archives = tuple(f'vendor/upstream-git/{name}.tar.gz' for name in UPSTREAM_GIT.get(game, ()))
    listed = subprocess.run(['git', 'ls-files', '-z', '--cached', '--others', '--exclude-standard', '--',
                             *SOURCE, *LINKS.get(game, ()), *upstream, *archives], cwd=ROOT, capture_output=True, check=True).stdout.decode()
    required = IGNORED_REQUIRED.get(game, ())
    if required:
        listed += subprocess.run(['git', 'ls-files', '-z', '--others', '--ignored', '--exclude-standard', '--',
                                  *required], cwd=ROOT, capture_output=True, check=True).stdout.decode()
    names = sorted({name for name in listed.split('\0') if name and not name.startswith(SOURCE_EXCLUDE)})
    return [name for name in names if (ROOT / name).is_symlink() or (ROOT / name).is_file()]


def source_archive(game, directory):
    """A deterministic tar of `source_files(game)`; returns (path, sha256). Symlinks (compatibility links) stay links."""
    path = Path(directory) / f'{game}-source.tar'
    with tarfile.open(path, 'w', format=tarfile.PAX_FORMAT) as tar:
        for name in source_files(game):
            info = tar.gettarinfo(str(ROOT / name), arcname=name)
            info.uid = info.gid = 0
            info.uname = info.gname = 'root'
            info.mtime = 0
            if info.isreg():
                with open(ROOT / name, 'rb') as stream:
                    tar.addfile(info, stream)
            else:
                tar.addfile(info)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return path, digest


def requirement_files(game, directory):
    """Copy the game's host requirements (and the files they include) into one folder for the image."""
    source = ROOT / HOST_REQUIREMENTS[game]
    folder = Path(directory) / f'{game}-requirements'
    folder.mkdir(parents=True, exist_ok=True)
    seen = []

    def copy(path, name):
        shutil.copyfile(path, folder / name)
        seen.append(name)
        for line in path.read_text().splitlines():
            line = line.strip()
            if line.startswith('-r '):
                included = (path.parent / line[3:].strip()).resolve()
                copy(included, included.name)

    copy(source, 'host.txt')
    # Includes are rewritten to the flat folder.
    for name in seen:
        target = folder / name
        lines = []
        for line in target.read_text().splitlines():
            if line.strip().startswith('-r '):
                line = '-r ' + Path(line.strip()[3:].strip()).name
            lines.append(line)
        target.write_text('\n'.join(lines) + '\n')
    (folder / 'pins.txt').write_text(''.join(line + '\n' for line in HOST_PINS.get(game, ())))
    return folder


def image(game, staging):
    """The coordinator image for one game."""
    import modal
    archive, digest = source_archive(game, staging)
    requirements = requirement_files(game, staging)
    pythonpath = ':'.join([REPO, *(f'{REPO}/{path}' for path in ENGINE_PATHS.get(game, ()))])
    install = (f'uv pip install --python {GAME_VENV}/bin/python --index-strategy unsafe-best-match '
               f'--extra-index-url {TORCH_CPU_INDEX} -r /tmp/requirements/host.txt -r /tmp/requirements/pins.txt')
    return (modal.Image.debian_slim(python_version='3.12')
            .apt_install('git', 'ca-certificates', 'rsync', 'patch')
            .pip_install('uv==0.9.25')
            .run_commands(f'uv python install {PYTHON}')
            .add_local_file(ROOT / HARBOR_LOCK, '/tmp/harbor.lock.txt', copy=True)
            .run_commands(f'uv venv --python {PYTHON} {HARBOR_VENV}',
                          f'uv pip install --python {HARBOR_VENV}/bin/python -r /tmp/harbor.lock.txt')
            .add_local_file(ROOT / HARBOR_PATCH, '/tmp/harbor.patch', copy=True)
            .run_commands(f'patch --forward {HARBOR_VENV}/lib/python3.12/site-packages/harbor/llms/utils.py /tmp/harbor.patch',
                          f'grep -q "and not msg" {HARBOR_VENV}/lib/python3.12/site-packages/harbor/llms/utils.py')
            .add_local_dir(requirements, '/tmp/requirements', copy=True)
            .run_commands(f'uv venv --python {PYTHON} {GAME_VENV}', install)
            .add_local_file(archive, '/tmp/source.tar', copy=True)
            .run_commands(f'mkdir -p {REPO} {WORK} && tar -xf /tmp/source.tar -C {REPO} && rm /tmp/source.tar',
                          f'echo {digest} > /etc/ar-source-sha256',
                          'git config --system --add safe.directory "*"',
                          f'echo {pythonpath} > /etc/ar-pythonpath',
                          *[f'tar -xzf {REPO}/vendor/upstream-git/{name}.tar.gz -C {REPO}/vendor/engines/{name} .git'
                            for name in UPSTREAM_GIT.get(game, ())])
            .env({'PYTHONPATH': pythonpath,
                  'PYTHONUNBUFFERED': '1', 'MODAL_ENVIRONMENT': ENVIRONMENT,
                  'HARNESS_HARBOR': f'{HARBOR_VENV}/bin/harbor'})), digest


def release_key(game, release):
    """Where a release lives in the releases Volume: by game and binding hash, under its own folder name."""
    from harness import policy
    record = policy.verify_release(game, release)
    return f"{game}/{record['sha256'][:16]}", record


def release_archive(release, path):
    """One deterministic tar of a release folder and its binding (`NAME/...`, `NAME.harness.json`)."""
    from harness import policy
    release = Path(release).resolve()
    with tarfile.open(path, 'w', format=tarfile.PAX_FORMAT) as tar:
        items = [(p, f'{release.name}/{p.relative_to(release).as_posix()}') for p in sorted(release.rglob('*'))
                 if p.is_file()]
        items.append((policy.sidecar(release), release.name + '.harness.json'))
        for source, name in items:
            info = tar.gettarinfo(str(source), arcname=name)
            info.uid = info.gid = 0
            info.uname = info.gname = 'root'
            info.mtime = 0
            with open(source, 'rb') as stream:
                tar.addfile(info, stream)
    return Path(path)


def stage_release(game, release, volume=None):
    """Upload a verified release and its binding once, as one tar (thousands of small files hang a batch upload).

    Returns (path of the release inside the container, path of its tar in the releases Volume). The container
    unpacks the tar into /work/releases/ before the harness starts; the release is then verified there as here.
    """
    import modal
    release = Path(release).resolve()
    key, record = release_key(game, release)
    volume = volume or modal.Volume.from_name(RELEASES_VOLUME, environment_name=ENVIRONMENT,
                                              create_if_missing=True, version=2)
    remote = f'{key}/{release.name}.tar'
    local_path = f'{WORK}/releases/{key}/{release.name}'
    try:
        next(iter(volume.read_file(remote + '.staged.json')))
        return local_path, f'{RELEASES}/{remote}'
    except Exception:
        pass
    with tempfile.TemporaryDirectory() as temp:
        archive = release_archive(release, Path(temp) / 'release.tar')
        marker = Path(temp) / 'staged.json'
        marker.write_text(json.dumps(dict(game=game, binding_sha256=record['sha256'], name=release.name,
                                          tar_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                                          staged_at=time.time())))
        with volume.batch_upload(force=True) as upload:
            upload.put_file(str(archive), remote)
        with volume.batch_upload(force=True) as upload:  # the marker only once the tar is in
            upload.put_file(str(marker), remote + '.staged.json')
    return local_path, f'{RELEASES}/{remote}'


def release_tar(path):
    """The staged tar of a release unpacked at `path` (/work/releases/<game>/<key>/<name>)."""
    relative = PurePosixPath(path).relative_to(f'{WORK}/releases')
    return f'{RELEASES}/{relative}.tar'


def unpack_releases(paths):
    """In the container: unpack each needed release from its staged tar into /work/releases/ (once)."""
    for path in paths:
        target = Path(path)
        if target.is_dir():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(release_tar(path)) as tar:
            tar.extractall(target.parent, filter='tar')


def restore(campaign, session, store=RECORDS, work=WORK):
    """In the container: bring a recorded session (its folder and binding) back into /work for final/grade/status.

    Returns the release path its binding names."""
    source, target = Path(store) / campaign / session, Path(work) / campaign / session
    binding = source.with_name(source.name + '.harness-run.json')
    if not binding.is_file():
        raise FileNotFoundError(f'No recorded session {campaign}/{session}')
    if not target.exists():
        shutil.copytree(source, target, symlinks=True)
        shutil.copy2(binding, target.with_name(binding.name))
        # Records mirrored before the sidecar fix can hold a WAL older than its database: drop it, or the restored
        # ledger reads as it was before its last checkpoint (polytopia-empty15-20261118: finals read as unfinalized).
        for side in stale_sidecars(target):
            side.unlink()
    return json.loads(binding.read_text())['settings']['release']


def rewrite(game, campaign, argv, stage=stage_release):
    """The harness argv as the container runs it: releases staged, session paths under /work/<campaign>/."""
    if not argv or argv[0] not in ('run', 'retrain', 'resume', 'final', 'grade', 'status', 'verify'):
        raise SystemExit('Cloud commands: run, retrain, resume, final, grade, status, verify')
    out, sessions, releases = list(argv), [], []
    for index, item in enumerate(out):
        if item in LOCAL_ONLY:
            raise SystemExit(f'{item} is a local file; give the key as a Modal Secret with --secret')
        if item == '--api-base' and index + 1 < len(out) and any(
                host in out[index + 1] for host in ('127.0.0.1', 'localhost', '[::1]')):
            raise SystemExit('A local --api-base is not reachable from the cloud; use a provider endpoint')
        if index and out[index - 1] == '--release':
            path, _ = stage(game, item)
            releases.append(path)
            out[index] = path
        elif index and out[index - 1] in PATH_FLAGS | INPUT_FLAGS:
            path = PurePosixPath(item)
            if path.is_absolute() or '..' in path.parts or not path.parts:
                raise SystemExit(f'{out[index - 1]} is a path inside the campaign, e.g. pokemon/opus/main')
            if out[index - 1] in PATH_FLAGS:
                sessions.append(str(path))
            out[index] = f'{WORK}/{campaign}/{path}'
    if argv[0] in ('run', 'retrain', 'resume', 'final', 'grade', 'status') and not sessions:
        raise SystemExit('Give --out (run, retrain) or --session (resume, final, grade, status)')
    return out, sessions, releases


def app_name(campaign, session):
    text = f'ar-coord-{campaign}-{session}'.lower()
    clean = ''.join(c if c.isalnum() else '-' for c in text).strip('-')
    while '--' in clean:
        clean = clean.replace('--', '-')
    if len(clean) > 56:
        clean = clean[:47] + '-' + hashlib.sha256(text.encode()).hexdigest()[:8]
    return clean


def check_name(value, what):
    if not value or not all(c.isalnum() or c in '-_.' for c in value) or value.startswith('.'):
        raise SystemExit(f'{what} must be a plain name (letters, digits, - _ .)')
    return value


# ---------------------------------------------------------------- in the container

SQLITE_SIDECARS = ('-wal', '-shm')


def stale_sidecars(root):
    """SQLite `-wal` / `-shm` files under `root` whose database no longer has them as live state: the sidecar is
    older than its database (SQLite checkpointed and removed it at close, but a copy outlived it). Reading the database
    with such a WAL replays old frames over the newer pages: a finalized ledger reads as unfinalized."""
    stale = []
    for db in Path(root).rglob('*.sqlite3'):
        for suffix in SQLITE_SIDECARS:
            side = Path(str(db) + suffix)
            if side.is_file() and side.stat().st_mtime < db.stat().st_mtime:
                stale.append(side)
    return stale


def mirror(source, target):
    """Copy new and changed files from `source` to `target`. Never deletes, except a database's `-wal` / `-shm` that
    no longer exists in `source` (SQLite removed it at close): a copy left behind would roll the database back when
    the records are read or restored. Returns how many were copied."""
    source, target = Path(source), Path(target)
    for db in target.rglob('*.sqlite3'):
        for suffix in SQLITE_SIDECARS:
            side = Path(str(db) + suffix)
            if side.is_file() and not (source / side.relative_to(target)).exists():
                side.unlink()
    copied = 0
    for path in sorted(source.rglob('*')):
        if path.is_symlink() or not path.is_file():
            continue
        dest = target / path.relative_to(source)
        try:
            stat = path.stat()
            if dest.is_file():
                theirs = dest.stat()
                if theirs.st_size == stat.st_size and int(theirs.st_mtime) == int(stat.st_mtime):
                    continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            temporary = dest.with_name('.' + dest.name + '.mirror-partial')
            shutil.copy2(path, temporary)
            os.replace(temporary, dest)
            copied += 1
        except FileNotFoundError:
            continue  # removed while copying (a temporary file); the next pass sees the final state
    return copied


BOX_APP = 'game-autoresearch-researchers'  # harness.session.APP: the app every researcher box runs in


def terminate_boxes(read_plan, session, sandboxes=None):
    """Terminate the researcher box of `session` if it is still running; returns the terminated sandbox ids.

    A session that closes normally terminates its own box. One whose coordinator ended without closing it (Modal
    preempted the container, or `stop` cancelled the call) leaves the Harbor box running until its own timeout, and
    nothing can use it again: the broker that served it is gone. `read_plan()` returns the session's `plan.json`
    (from the records Volume) or None; the box is found by the unique `box_tag` written there."""
    plan = read_plan()
    tag = (plan or {}).get('box_tag')
    if not tag:
        return []
    if sandboxes is None:
        import modal
        app = modal.App.lookup(BOX_APP, environment_name=ENVIRONMENT)
        sandboxes = list(modal.Sandbox.list(app_id=app.app_id, tags={'ar_session': tag}))
    ended = []
    for sandbox in sandboxes:
        sandbox.terminate()
        ended.append(sandbox.object_id)
    return ended


def coordinate(spec):
    """Run `python -m harness <argv>` with the game's interpreter, mirroring /work to the records Volume."""
    import modal
    records = modal.Volume.from_name(RECORDS_VOLUME, environment_name=ENVIRONMENT)
    campaign = spec['campaign']
    work, store = Path(WORK) / campaign, Path(RECORDS) / campaign
    work.mkdir(parents=True, exist_ok=True)
    log_name = spec['log']
    # Modal restarts a preempted container with the same input, in an empty /work. Re-running `run` or `retrain`
    # there would start a second session under the same path: its mirror would overwrite the first one's records,
    # its workers would collide with the first one's private Volume, and the first one's box and workers would be
    # orphaned. So a restart is refused and recorded; the operator decides how to recover from the records.
    prior = store / (log_name + '.coordinator.json')
    if spec['argv'][0] in ('run', 'retrain') and prior.is_file():
        before = json.loads(prior.read_text())
        if before.get('container') != os.environ.get('MODAL_TASK_ID'):
            note = dict(spec, refused='restart of a run/retrain coordinator after its container ended '
                        '(preemption): the session is not re-run', restarted_at=time.time(),
                        container=os.environ.get('MODAL_TASK_ID'), prior=before)
            plan = store / spec['session'] / 'plan.json'
            try:
                note['boxes_terminated'] = terminate_boxes(
                    lambda: json.loads(plan.read_text()) if plan.is_file() else None, spec['session'])
            except Exception as exc:
                note['box_cleanup_error'] = type(exc).__name__ + ': ' + str(exc)[:300]
            (store / f'{log_name}.coordinator-restart-{int(time.time())}.json').write_text(
                json.dumps(note, indent=2) + '\n')
            records.commit()
            return note
    # The game's own path (the repository plus any engines it imports by path), written into the image; never the
    # container's PYTHONPATH, which Modal's runtime extends with its own client.
    pythonpath = Path('/etc/ar-pythonpath')
    env = dict(os.environ, MODAL_ENVIRONMENT=ENVIRONMENT,
               PYTHONPATH=pythonpath.read_text().strip() if pythonpath.is_file() else REPO,
               PYTHONUNBUFFERED='1',
               HARNESS_HARBOR=f'{HARBOR_VENV}/bin/harbor')
    started = time.time()
    meta = dict(spec, started_at=started, source_sha256=Path('/etc/ar-source-sha256').read_text().strip(),
                container=os.environ.get('MODAL_TASK_ID'))
    (work / (log_name + '.coordinator.json')).write_text(json.dumps(meta, indent=2) + '\n')
    stop = threading.Event()

    def sync():
        mirror(work, store)
        records.commit()

    def requests():
        # `harness.cloud pause` writes the operator's request into the records Volume; hand it to the session.
        source = store / spec['session'] / PAUSE_REQUEST
        target = work / spec['session'] / PAUSE_REQUEST
        records.reload()
        if source.is_file() and target.parent.is_dir() and (
                not target.is_file() or target.read_bytes() != source.read_bytes()):
            temporary = target.with_name(target.name + '.tmp')
            temporary.write_bytes(source.read_bytes())
            temporary.replace(target)

    def loop():
        passes = 0
        while not stop.wait(REQUEST_SECONDS):
            passes += 1
            if spec['argv'][0] in ('run', 'resume'):
                try:
                    requests()
                except Exception as exc:
                    print('request check failed:', type(exc).__name__, str(exc)[:300], flush=True)
            if passes % max(1, MIRROR_SECONDS // REQUEST_SECONDS):
                continue
            try:
                sync()
            except Exception as exc:  # a failed pass is retried on the next one
                print('mirror failed:', type(exc).__name__, str(exc)[:300], flush=True)

    # Harbor's Modal preflight looks for ~/.modal.toml or token variables. In a container the Modal client
    # authenticates as the container and ignores both, so an empty placeholder (no credentials) satisfies it.
    placeholder = Path.home() / '.modal.toml'
    if not placeholder.exists():
        placeholder.write_text('# Modal authenticates this container itself; this file holds no credentials.\n')
    releases = list(spec.get('releases', ()))
    for item in spec.get('inputs', ()):
        if not (work / item).exists():
            shutil.copytree(store / item, work / item)
    for session in spec.get('restore', ()):
        release = restore(campaign, session)
        # A retrain reads only the recipe files of its source session and trains on its own --release; the
        # source's release (possibly a path of another machine, never staged here) is not unpacked.
        if spec['argv'][0] != 'retrain':
            releases.append(release)
    unpack_releases(releases)
    thread = threading.Thread(target=loop, daemon=True)
    thread.start()
    with open(work / (log_name + '.coordinator.log'), 'ab') as log:
        process = subprocess.Popen([f'{GAME_VENV}/bin/python', '-m', 'harness', *spec['argv']], cwd=REPO, env=env,
                                   stdout=log, stderr=subprocess.STDOUT)
        code = process.wait()
    stop.set()
    thread.join(timeout=30)
    meta.update(finished_at=time.time(), returncode=code)
    (work / (log_name + '.coordinator.json')).write_text(json.dumps(meta, indent=2) + '\n')
    for attempt in range(5):
        try:
            sync()
            break
        except Exception:
            time.sleep(10 * (attempt + 1))
    return meta


# ---------------------------------------------------------------- commands (this machine)

def input_name(folder):
    """Where a local input folder lives in the campaign: named by its contents, so a relaunch reuses it."""
    folder = Path(folder)
    if not folder.is_dir():
        raise SystemExit(f'{folder} is not a folder')
    files = sorted(p for p in folder.rglob('*') if p.is_file())
    if any(p.is_symlink() for p in folder.rglob('*')):
        raise SystemExit(f'{folder} holds a symbolic link')
    content = hashlib.sha256()
    for path in files:
        content.update(path.relative_to(folder).as_posix().encode() + b'\0' + hashlib.sha256(path.read_bytes()).digest())
    return f'{INPUTS}/{folder.name}-{content.hexdigest()[:12]}', files


def stage_inputs(campaign, argv, upload=None):
    """Replace each local input folder in argv by its campaign path, uploading it unless it is there already."""
    argv, inputs = list(argv), []
    for index, item in enumerate(argv[:-1]):
        if item in INPUT_FLAGS:
            name, files = input_name(argv[index + 1])
            if upload is not None:
                upload(Path(argv[index + 1]), f'{campaign}/{name}', files)
            argv[index + 1] = name
            inputs.append(name)
    return argv, inputs


def upload_input(records):
    def upload(folder, remote, files):
        try:
            present = any(True for _ in records.listdir(remote))
        except Exception:
            present = False
        if present:
            return
        with records.batch_upload() as batch:
            for path in files:
                batch.put_file(str(path), f'{remote}/{path.relative_to(folder).as_posix()}')
    return upload


def launch(campaign, argv, game=None, secrets=(), dry=False, staging=None):
    import modal
    check_name(campaign, '--campaign')
    from harness.workspace import GAMES
    game = game or next((item for item in argv[1:3] if item in GAMES), None)
    if argv and argv[0] in ('resume', 'final', 'grade', 'status') and game is None:
        raise SystemExit('Give --game for resume, final, grade and status')
    if game not in GAMES:
        raise SystemExit('Unknown game')
    if dry:
        argv, inputs = stage_inputs(campaign, argv)
        argv2, sessions, _ = rewrite(game, campaign, argv, stage=lambda g, r: (
            f'{WORK}/releases/{release_key(g, r)[0]}/{Path(r).resolve().name}', None))
        return dict(game=game, campaign=campaign, argv=argv2, app=app_name(campaign, sessions[0]) if sessions else None,
                    sources=len(source_files(game)), secrets=list(secrets), inputs=inputs, executed=False)
    os.environ['MODAL_ENVIRONMENT'] = ENVIRONMENT
    local, (argv, inputs) = argv, stage_inputs(campaign, argv)
    _, sessions, _ = rewrite(game, campaign, argv, stage=lambda g, r: ('', None))
    session = sessions[0] if sessions else argv[0]
    # One app per launch: a later `grade` of a session never redeploys the app still running its `run`.
    name = app_name(campaign, session if argv[0] in ('run', 'retrain') else f'{argv[0]}-{session}')
    if argv[0] in ('run', 'retrain'):
        # Sessions and budgets are never reset: a container starts with an empty /work, so refuse here what
        # `bind_run` would refuse on a laptop.
        records = modal.Volume.from_name(RECORDS_VOLUME, environment_name=ENVIRONMENT, create_if_missing=True, version=2)
        try:
            taken = any(True for _ in records.listdir(f'{campaign}/{session}'))
        except Exception:
            taken = False
        if taken or launches(campaign, session):
            raise SystemExit(f'{campaign}/{session} already exists; choose a fresh --out')
    stage_inputs(campaign, local, upload_input(
        modal.Volume.from_name(RECORDS_VOLUME, environment_name=ENVIRONMENT, create_if_missing=True, version=2)))
    argv2, sessions, release_paths = rewrite(game, campaign, argv)
    staging = Path(staging or tempfile.mkdtemp(prefix='ar-coordinator-'))
    built, digest = image(game, staging)
    records = modal.Volume.from_name(RECORDS_VOLUME, environment_name=ENVIRONMENT, create_if_missing=True, version=2)
    releases = modal.Volume.from_name(RELEASES_VOLUME, environment_name=ENVIRONMENT, create_if_missing=True, version=2)
    app = modal.App(name)
    function = app.function(image=built, volumes={RECORDS: records, RELEASES: releases},
                            secrets=[modal.Secret.from_name(item, environment_name=ENVIRONMENT) for item in secrets],
                            timeout=TIMEOUT_SECONDS, cpu=CPU, memory=MEMORY_MIB, retries=0, serialized=True,
                            max_containers=1,
                            # A coordinator cannot survive its container: never let Modal preempt it.
                            nonpreemptible=True)(coordinate)
    with modal.enable_output():
        app.deploy(name=name, environment_name=ENVIRONMENT)
    spec = dict(campaign=campaign, game=game, argv=argv2, session=session, app=name, log=log_name(session),
                secrets=list(secrets), source_sha256=digest, releases=release_paths, inputs=inputs,
                restore=sessions if argv[0] in ('resume', 'final', 'grade', 'status')
                else [argv[i + 1] for i, item in enumerate(argv[:-1]) if argv[0] == 'retrain' and item == '--session'])
    call = function.spawn(spec)
    record = dict(spec, call_id=call.object_id, launched_at=time.time(), environment=ENVIRONMENT)
    local = ROOT / 'runs' / 'cloud' / campaign / 'launches'
    local.mkdir(parents=True, exist_ok=True)
    (local / (name + '.json')).write_text(json.dumps(record, indent=2) + '\n')
    return record


def launches(campaign, out=None):
    folder = ROOT / 'runs' / 'cloud' / campaign / 'launches'
    items = [json.loads(p.read_text()) for p in sorted(folder.glob('*.json'))] if folder.is_dir() else []
    return [item for item in items if out is None or item['session'] == out]


def status(campaign, out=None):
    import modal
    result = []
    for item in launches(campaign, out):
        call = modal.FunctionCall.from_id(item['call_id'])
        try:
            value = call.get(timeout=0)
            state = dict(state='finished', returncode=value.get('returncode'))
        except TimeoutError:
            state = dict(state='running')
        except Exception as exc:
            state = dict(state='failed', error=type(exc).__name__ + ': ' + str(exc)[:300])
        result.append(dict(session=item['session'], app=item['app'], call_id=item['call_id'], **state))
    return result


def stop(campaign, out):
    import modal
    records = modal.Volume.from_name(RECORDS_VOLUME, environment_name=ENVIRONMENT)

    def plan_of(session):
        try:
            return json.loads(b''.join(records.read_file(f'{campaign}/{session}/plan.json')))
        except Exception:
            return None

    stopped, boxes = [], {}
    for item in launches(campaign, out):
        modal.FunctionCall.from_id(item['call_id']).cancel()
        stopped.append(item['app'])
        try:
            boxes[item['session']] = terminate_boxes(lambda: plan_of(item['session']), item['session'])
        except Exception as exc:
            boxes[item['session']] = 'cleanup failed: ' + type(exc).__name__ + ': ' + str(exc)[:200]
    return dict(cancelled=stopped, boxes_terminated=boxes,
                note='The session process is stopped and its researcher box terminated; its workers end at their own '
                     'time limits and are settled from provider receipts by `final`/`grade` recovery.')


def pause(campaign, out, note=None):
    """Ask a running cloud session to pause: its coordinator reads the request from the records Volume within
    REQUEST_SECONDS, snapshots the workspace, stops the researcher and its workers and ends without finalizing."""
    import io
    import uuid
    import modal
    records = modal.Volume.from_name(RECORDS_VOLUME, environment_name=ENVIRONMENT)
    try:
        state = json.loads(b''.join(records.read_file(f'{campaign}/{out}/session.json')))
    except Exception as exc:
        raise SystemExit(f'No recorded session {campaign}/{out} yet ({type(exc).__name__})')
    if state.get('finalized'):
        raise SystemExit('The session is finalized')
    request = dict(id=uuid.uuid4().hex, requested_at=time.time(), note=note)
    with records.batch_upload(force=True) as batch:
        batch.put_file(io.BytesIO((json.dumps(request, indent=2) + '\n').encode()), f'{campaign}/{out}/{PAUSE_REQUEST}')
    return dict(requested=request, session=f'{campaign}/{out}',
                note='The session pauses within about a minute: its coordinator call then finishes and '
                     f'{campaign}/{out}/pause.json reads state "paused". Resume with '
                     '`harness.cloud launch --campaign ... --game GAME -- resume --session OUT --execute`.')


def log_name(session):
    """The coordinator's own log and record for a session sit at the campaign root under this name."""
    return str(session).replace('/', '__')


def wanted(relative, out):
    """Whether a campaign-relative record belongs to session `out`: its folder, its binding, its coordinator log."""
    if out is None:
        return True
    out = str(PurePosixPath(out))
    return (relative.startswith(out + '/') or relative == out + '.harness-run.json'
            or relative.startswith(log_name(out) + '.coordinator.'))


def pull(campaign, to, out=None):
    """Download the campaign's records (or one session's) into `to/<campaign>/`."""
    import modal
    volume = modal.Volume.from_name(RECORDS_VOLUME, environment_name=ENVIRONMENT)
    count = 0
    for entry in volume.listdir(campaign, recursive=True):
        if entry.type != modal.volume.FileEntryType.FILE:
            continue
        relative = PurePosixPath(entry.path).relative_to(campaign)
        if not wanted(str(relative), out):
            continue
        target = Path(to) / campaign / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, 'wb') as stream:
            for chunk in volume.read_file(entry.path):
                stream.write(chunk)
        count += 1
    return dict(campaign=campaign, files=count, to=str(Path(to) / campaign))


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    rest = []
    if '--' in argv:
        index = argv.index('--')
        argv, rest = argv[:index], argv[index + 1:]
    parser = argparse.ArgumentParser(prog='python -m harness.cloud', description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('launch', help='run `python -m harness ARGS` (after --) in a Modal container')
    p.add_argument('--campaign', required=True)
    p.add_argument('--game', help='needed for final, grade and status')
    p.add_argument('--secret', action='append', default=[], help='Modal Secret (project environment) with model keys')
    p.add_argument('--dry-run', action='store_true', help='show what would run; build and upload nothing')
    for name in ('status', 'stop', 'pause'):
        p = sub.add_parser(name)
        p.add_argument('--campaign', required=True)
        p.add_argument('--out', required=name != 'status')
        if name == 'pause':
            p.add_argument('--note', help='why, recorded with the pause')
    p = sub.add_parser('pull', help='download records from the records Volume')
    p.add_argument('--campaign', required=True)
    p.add_argument('--out')
    p.add_argument('--to', type=Path, default=ROOT / 'runs' / 'cloud')
    args = parser.parse_args(argv)
    if args.command == 'launch':
        if not rest:
            parser.error('give the harness command after --')
        result = launch(args.campaign, rest, game=args.game, secrets=args.secret, dry=args.dry_run)
    elif args.command == 'status':
        result = status(args.campaign, args.out)
    elif args.command == 'stop':
        result = stop(args.campaign, args.out)
    elif args.command == 'pause':
        result = pause(args.campaign, args.out, note=args.note)
    else:
        result = pull(args.campaign, args.to, args.out)
    print(json.dumps(result, indent=2, default=str))


if __name__ == '__main__':
    main()
