"""Worker-to-host result hand-off through a per-session private Modal Volume.

A training worker writes what the host must collect (its exported network, logs, receipts) into this
session's private Volume and commits it before the worker ends. The host reads it afterwards, outside the
worker's paid time, with retries and no deadline. A slow or interrupted host link therefore never costs a
finished run: on Nov 13 every Polytopia training lost its network because the host had to stream a
10-16 MB archive out of the running worker within 19-100 s over a 16 KB/s link.

Pokémon already works this way with its own runtime (`harness/games/pokemon/provider.py`). This module is
the shared form for the games whose workers are driven by host `exec` calls.

Layout inside the Volume (mounted at `MOUNT` in a worker, root-only):

    <job>/files/<name>     one published file each, written by `PUBLISH` (temp file + rename)
    <job>/MANIFEST.json    {name: {bytes, sha256}}, written last; its presence means the job is published

`PUBLISH` runs inside the worker as root, after the trusted export step; it ends with `sync` on the mount,
which commits the Volume. The host calls `collect` after the publish exec has returned (or after the
worker has finished, when the exec's reply was lost) and verifies every byte against the manifest.

One Volume per backend root (a research session's `research/`, a final's `final/`): nothing is shared
between sessions or between a session and its fresh final.
"""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import time
import uuid

from harness.policy import ENVIRONMENT

MOUNT = '/handoff'
BINDING = 'handoff-binding.json'
MANIFEST = 'MANIFEST.json'
MAX_FILE_BYTES = 2 * 1024 * 1024 * 1024
NAME = re.compile(r'[A-Za-z0-9][A-Za-z0-9._-]{0,127}')

# Runs in the worker as root: python3 -c PUBLISH JOB NAME=PATH [NAME=PATH ...]
# Copies each file into MOUNT/JOB/files/NAME through a temporary name, writes the manifest last, makes the
# job folder root-only and commits the Volume with `sync`. Prints the manifest as one JSON line.
PUBLISH = r'''import hashlib,json,os,re,shutil,subprocess,sys
mount=%r;job=sys.argv[1];name_re=re.compile(r'[A-Za-z0-9][A-Za-z0-9._-]{0,127}')
if not name_re.fullmatch(job):raise SystemExit('bad job name')
os.makedirs(mount,exist_ok=True);os.chmod(mount,0o700)
folder=os.path.join(mount,job);files=os.path.join(folder,'files');os.makedirs(files,exist_ok=True)
os.chmod(folder,0o700);os.chmod(files,0o700)
manifest={}
for item in sys.argv[2:]:
    name,_,path=item.partition('=')
    if not name_re.fullmatch(name) or name=='MANIFEST.json':raise SystemExit('bad file name '+name)
    if os.path.islink(path) or not os.path.isfile(path):raise SystemExit('not a regular file: '+path)
    digest=hashlib.sha256();size=0;temporary=os.path.join(files,'.'+name+'.partial')
    with open(path,'rb') as source,open(temporary,'wb') as target:
        while True:
            chunk=source.read(1<<20)
            if not chunk:break
            digest.update(chunk);size+=len(chunk);target.write(chunk)
        target.flush();os.fsync(target.fileno())
    os.replace(temporary,os.path.join(files,name));manifest[name]=dict(bytes=size,sha256=digest.hexdigest())
temporary=os.path.join(folder,'.MANIFEST.json.partial')
with open(temporary,'w') as stream:json.dump(manifest,stream,sort_keys=True);stream.flush();os.fsync(stream.fileno())
os.replace(temporary,os.path.join(folder,'MANIFEST.json'))
subprocess.run(['sync',mount],check=True,timeout=120)
print(json.dumps(manifest,sort_keys=True))''' % MOUNT

# Runs in the worker as root before any researcher code: the mount is root-only from the start.
PREPARE = 'import os;os.makedirs(%r,exist_ok=True);os.chmod(%r,0o700)' % (MOUNT, MOUNT)


class NotPublished(RuntimeError):
    """The job's manifest is not (yet) in the Volume."""


def publish_command(job, files, python='python3'):
    """The worker command that publishes `files` ({name: path in the worker}) for `job`."""
    check_name(job)
    for name in files:
        check_name(name)
    return [python, '-c', PUBLISH, job, *(f'{name}={path}' for name, path in files.items())]


def prepare_command(python='python3'):
    return [python, '-c', PREPARE]


def check_name(name):
    if not NAME.fullmatch(name) or name == MANIFEST:
        raise ValueError('Hand-off names are short plain file names: ' + repr(name))
    return name


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    os.replace(temporary, path)


class SessionVolume:
    """This backend root's private hand-off Volume. Created on first use and recorded in `handoff-binding.json`."""

    def __init__(self, game, root, modal_module=None):
        self.game = game
        self.root = Path(root).resolve()
        self._modal = modal_module
        path = self.root / BINDING
        if path.is_file():
            self.binding = json.loads(path.read_text())
            if self.binding.get('environment') != ENVIRONMENT or self.binding.get('game') != game:
                raise ValueError('Hand-off Volume was bound for another game or environment')
        else:
            label = re.sub(r'[^a-z0-9]+', '-', (self.root.parent.name + '-' + self.root.name).lower()).strip('-')[:32]
            # A fresh random part: a session re-created at the same path (a cloud coordinator's /work starts
            # empty in every container) never sees an earlier session's published jobs.
            digest = hashlib.sha256((str(self.root) + uuid.uuid4().hex).encode()).hexdigest()[:12]
            self.binding = dict(schema=1, game=game, environment=ENVIRONMENT, mount=MOUNT,
                                volume=f'{game}-handoff-{label}-{digest}', root=str(self.root))
            _write_json(path, self.binding)

    @property
    def modal(self):
        if self._modal is None:
            import modal
            self._modal = modal
        return self._modal

    @property
    def name(self):
        return self.binding['volume']

    def volume(self, create=False):
        options = dict(version=2, create_if_missing=True) if create else {}
        return self.modal.Volume.from_name(self.name, environment_name=ENVIRONMENT, **options)

    def mounts(self):
        """`volumes=` for a worker of this session: the Volume at MOUNT (created if needed)."""
        return {MOUNT: self.volume(create=True)}

    def _read(self, volume, remote, limit):
        data = bytearray()
        for chunk in volume.read_file(remote):
            data.extend(chunk)
            if len(data) > limit:
                raise ValueError(f'Hand-off file exceeds {limit} bytes: {remote}')
        return bytes(data)

    def manifest(self, job, volume=None):
        check_name(job)
        volume = volume or self.volume()
        try:
            return json.loads(self._read(volume, f'{job}/{MANIFEST}', 1 << 20))
        except FileNotFoundError as exc:
            raise NotPublished(f'{job} has no published manifest') from exc
        except Exception as exc:  # the SDK raises its own not-found errors
            if 'not found' in str(exc).lower() or 'no such file' in str(exc).lower():
                raise NotPublished(f'{job} has no published manifest') from exc
            raise

    def collect(self, job, dest, names=None, attempts=6, wait=5.0, limit=MAX_FILE_BYTES):
        """Fetch a published job's files into `dest` and verify them against its manifest.

        Retries reads (a commit may still be landing) for about `attempts * wait` seconds. Returns
        {name: {bytes, sha256, path}}. `names` restricts which files are fetched (all by default).
        """
        check_name(job)
        dest = Path(dest)
        dest.mkdir(parents=True, exist_ok=True)
        last = None
        for attempt in range(attempts):
            try:
                volume = self.volume()
                manifest = self.manifest(job, volume)
                wanted = list(manifest) if names is None else list(names)
                missing = [name for name in wanted if name not in manifest]
                if missing:
                    raise NotPublished(f'{job} did not publish: ' + ', '.join(missing))
                result = {}
                for name in wanted:
                    check_name(name)
                    entry = manifest[name]
                    if int(entry['bytes']) > limit:
                        raise ValueError(f'Hand-off file exceeds {limit} bytes: {name}')
                    target = dest / name
                    if target.is_symlink():
                        raise ValueError('Symlinked hand-off target')
                    if target.is_file() and sha256(target) == entry['sha256']:
                        result[name] = dict(entry, path=str(target))
                        continue
                    temporary = target.with_name('.' + name + '.handoff-partial')
                    temporary.write_bytes(self._read(volume, f'{job}/files/{name}', int(entry['bytes'])))
                    if temporary.stat().st_size != int(entry['bytes']) or sha256(temporary) != entry['sha256']:
                        temporary.unlink()
                        raise ValueError(f'Hand-off file differs from its manifest: {name}')
                    os.replace(temporary, target)
                    result[name] = dict(entry, path=str(target))
                return result
            except ValueError:
                raise
            except Exception as exc:
                last = exc
                if attempt + 1 < attempts:
                    time.sleep(wait)
        raise last


def remote_path(job, name=None):
    """Where a published file sits for a worker that mounts the Volume."""
    check_name(job)
    if name is None:
        return str(PurePosixPath(MOUNT) / job)
    return str(PurePosixPath(MOUNT) / job / 'files' / check_name(name))
