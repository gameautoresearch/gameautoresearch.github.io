"""Archive campaign Volumes to the private Hugging Face dataset from a Modal box, so nothing is downloaded here.

    PY=.venv-harness/bin/python
    $PY tools/hf_archive_box.py launch NAME --capture 20261120T2300-complete --volume ar-nanny-NAME \
        [--volume OTHER ...] [--extra LOCAL_DIR ...] [--execute]
    $PY tools/hf_archive_box.py status NAME        # prints the box's log tail and, once done, the receipt

The box (CPU only, project environment) copies each Volume to /src/<volume>, copies each --extra folder to
/src/extra/<basename> (small local records: scoring results, launch records), and runs the repository's own
helpers unchanged: tools/stage_huggingface_artifacts.py (exclusion filter, credential scan, nested archives),
tools/package_huggingface_snapshot.py (one .tar.zst per Volume with per-file manifests), then uploads the package
to artifacts/NAME/CAPTURE/ and checks every remote file's size and SHA-256 (or git blob id) against the package.
Log and receipt go to Volume `ar-hf-archive-NAME` (/out/archive.log, /out/receipt.json), plus the package's
manifests (not the archives). The Hugging Face token of this machine's credential store is sent as the box's secret.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
ENVIRONMENT, PROFILE, APP = 'game-autoresearch-project', 'duper', 'game-autoresearch-archive'
REPO = 'game-autoresearch/game-autoresearch'
RECORDS = ROOT / 'runs/hf-archive'

INSIDE = r'''
import hashlib, json, os, shutil, subprocess, sys, time
from pathlib import Path
from huggingface_hub import HfApi

REPO = os.environ['ARCHIVE_REPO']; PREFIX = os.environ['ARCHIVE_PREFIX']
out = Path('/out'); work = Path('/work'); work.mkdir(exist_ok=True); Path('/src').mkdir(exist_ok=True)
api = HfApi(token=os.environ['HF_TOKEN'])
if not api.dataset_info(REPO).private:
    raise SystemExit('dataset is not private')
try:
    existing = list(api.list_repo_tree(REPO, PREFIX, repo_type='dataset'))
except Exception as exc:
    if type(exc).__name__ not in ('EntryNotFoundError', 'RemoteEntryNotFoundError'):
        raise
    existing = []
if existing:
    raise SystemExit('remote path exists; choose a new capture')
for v in sorted(Path('/mnt').iterdir()):  # a mounted Volume is a link elsewhere; the stager needs real paths under /src
    subprocess.run(['cp', '-a', str(v.resolve()), f'/src/{v.name}'], check=True)
    print('COPIED', v.name, flush=True)
roots = sorted(str(p) for p in Path('/src').iterdir())
run = lambda *a: subprocess.run([sys.executable, '-u', *a], check=True)
run('/opt/archive/stage_huggingface_artifacts.py', *sum((['--root', r] for r in roots), []),
    '--relative-to', '/src', '--output', str(work / 'snapshot'))
run('/opt/archive/package_huggingface_snapshot.py', '--snapshot', str(work / 'snapshot'), '--output', str(work / 'package'))
package = work / 'package'
for p in package.iterdir():
    if p.suffix != '.zst':
        shutil.copyfile(p, out / p.name)
print('UPLOAD', sum(p.stat().st_size for p in package.iterdir()), flush=True)
api.upload_folder(repo_id=REPO, repo_type='dataset', folder_path=str(package), path_in_repo=PREFIX,
                  commit_message='Archive ' + PREFIX)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 24), b''):
            h.update(block)
    return h.hexdigest()


info = api.dataset_info(REPO)
local = sorted(p for p in package.iterdir() if p.is_file())
remote = {p.path: p for p in api.get_paths_info(REPO, [f'{PREFIX}/{p.name}' for p in local], repo_type='dataset',
                                                 revision=info.sha)}
rows, ok = [], True
for p in local:
    r = remote.get(f'{PREFIX}/{p.name}')
    row = dict(file=p.name, bytes=p.stat().st_size, present=r is not None, match=False)
    if r is not None:
        row['remote_bytes'] = r.size
        if r.lfs:
            row.update(sha256=sha256(p), remote_sha256=r.lfs.sha256)
            row['match'] = r.size == row['bytes'] and r.lfs.sha256 == row['sha256']
        else:
            data = p.read_bytes()
            row.update(git_blob=hashlib.sha1(b'blob %d\0' % len(data) + data).hexdigest(), remote_blob=r.blob_id)
            row['match'] = r.size == row['bytes'] and r.blob_id == row['git_blob']
    ok &= row['match']
    rows.append(row)
receipt = dict(repo=REPO, private=info.private, commit=info.sha, prefix=PREFIX, checked_unix=time.time(),
               roots=roots, files=len(rows), bytes=sum(r['bytes'] for r in rows), all_match=ok,
               manifest=json.loads((package / 'MANIFEST.json').read_text()),
               snapshot=json.loads((package / 'snapshot-summary.json').read_text()), rows=rows)
(out / 'receipt.json').write_text(json.dumps(receipt, indent=1) + '\n')
print('DONE all_match', ok, 'commit', info.sha, flush=True)
'''


def modal_client():
    for line in (ROOT / 'credentials.env').read_text().splitlines():
        line = line.strip().removeprefix('export ').strip()
        if '=' in line and not line.startswith('#'):
            key, value = line.split('=', 1)
            if key in ('MODAL_TOKEN_ID', 'MODAL_TOKEN_SECRET'):
                os.environ[key] = value.strip().strip('"').strip("'")
    os.environ.update(MODAL_PROFILE=PROFILE, MODAL_ENVIRONMENT=ENVIRONMENT)
    os.environ.setdefault('MODAL_DISABLE_API_PROXY', '1')
    import modal
    return modal


def launch(args):
    prefix = f'artifacts/{args.name}/{args.capture}'
    plan = dict(name=args.name, prefix=prefix, volumes=args.volume, extra=[str(Path(e).resolve()) for e in args.extra])
    print(json.dumps(plan, indent=1))
    if not args.execute:
        print('Plan only; add --execute.', file=sys.stderr)
        return
    folder = RECORDS / args.name
    if (folder / 'launch.json').exists():
        raise SystemExit(f'{folder} exists; choose a new name or remove a failed launch record deliberately')
    modal = modal_client()
    token = (Path.home() / '.cache/huggingface/token').read_text().strip()
    image = (modal.Image.debian_slim(python_version='3.12').apt_install('zstd').pip_install('huggingface_hub==1.33.0')
             .add_local_file(ROOT / 'tools/stage_huggingface_artifacts.py', '/opt/archive/stage_huggingface_artifacts.py', copy=True)
             .add_local_file(ROOT / 'tools/package_huggingface_snapshot.py', '/opt/archive/package_huggingface_snapshot.py', copy=True))
    for extra in plan['extra']:
        image = image.add_local_dir(extra, f'/src/extra/{Path(extra).name}', copy=True)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / 'inside.py').write_text(INSIDE)
    image = image.add_local_file(folder / 'inside.py', '/opt/archive/inside.py', copy=True)
    volumes = {f'/mnt/{v}': modal.Volume.from_name(v, environment_name=ENVIRONMENT) for v in args.volume}
    out = 'ar-hf-archive-' + args.name
    volumes['/out'] = modal.Volume.from_name(out, environment_name=ENVIRONMENT, create_if_missing=True, version=2)
    app = modal.App.lookup(APP, environment_name=ENVIRONMENT, create_if_missing=True)
    with modal.enable_output():
        sandbox = modal.Sandbox.create(
            'bash', '-c', '(while true; do sleep 60; sync /out; done) & python3 -u /opt/archive/inside.py > /out/archive.log 2>&1; echo "exit $?" >> /out/archive.log; sync /out',
            app=app, image=image, volumes=volumes, timeout=int(args.hours * 3600), cpu=4, memory=16384,
            secrets=[modal.Secret.from_dict(dict(HF_TOKEN=token, ARCHIVE_REPO=REPO, ARCHIVE_PREFIX=prefix))])
    record = dict(plan, sandbox_id=sandbox.object_id, out_volume=out, launched=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
    (folder / 'launch.json').write_text(json.dumps(record, indent=1) + '\n')
    print(json.dumps(record, indent=1))


def status(args):
    modal = modal_client()
    folder = RECORDS / args.name
    record = json.loads((folder / 'launch.json').read_text())
    volume = modal.Volume.from_name(record['out_volume'], environment_name=ENVIRONMENT)
    names = {e.path for e in volume.listdir('/')}
    if 'archive.log' in names:
        print(b''.join(volume.read_file('archive.log')).decode(errors='replace')[-1500:])
    if 'receipt.json' in names:
        data = b''.join(volume.read_file('receipt.json'))
        (folder / 'receipt.json').write_bytes(data)
        receipt = json.loads(data)
        print(json.dumps({k: v for k, v in receipt.items() if k not in ('rows', 'manifest', 'snapshot')}, indent=1))
        print('receipt:', folder / 'receipt.json')
    else:
        print('no receipt yet; box', record['sandbox_id'], 'exit code', modal.Sandbox.from_id(record['sandbox_id']).poll())


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = p.add_subparsers(dest='command', required=True)
    a = sub.add_parser('launch')
    a.add_argument('name')
    a.add_argument('--capture', required=True)
    a.add_argument('--volume', action='append', required=True)
    a.add_argument('--extra', action='append', default=[])
    a.add_argument('--hours', type=float, default=6)
    a.add_argument('--execute', action='store_true')
    b = sub.add_parser('status')
    b.add_argument('name')
    args = p.parse_args()
    dict(launch=launch, status=status)[args.command](args)


if __name__ == '__main__':
    main()
