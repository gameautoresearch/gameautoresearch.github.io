#!/usr/bin/env python3
"""Copy full run artifacts into a new private-upload snapshot without changing originals.

No cloud jobs are started. Upload the output with `hf upload ... --repo-type dataset`.
The JSONL manifests retain original paths, sizes and SHA-256 identities.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import tarfile
import time


SKIP_PARTS = {'.git', '__pycache__', 'node_modules', '.pytest_cache', '.ruff_cache',
              '.jax-cache', '.gradle', 'target', 'gateway',
              'seed-banks', 'private-banks'}
ASSET_SUFFIXES = {'.png', '.jpg', '.jpeg', '.webp', '.gif', '.mp3', '.wav', '.ogg',
                  '.mp4', '.ttf', '.woff', '.woff2', '.ico'}
SECRET_PATTERNS = [
    re.compile(rb'hf_[A-Za-z0-9]{25,}'),
    re.compile(rb'\bsk-(?:or-v1-)?[A-Za-z0-9_-]{20,}'),
    re.compile(rb'\b(?:ghp_|github_pat_)[A-Za-z0-9_]{25,}'),
    re.compile(rb'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b'),
    re.compile(rb'(?i)bearer[ =:]+[A-Za-z0-9_.-]{24,}'),
    re.compile(rb'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----'),
]
SECRET_MARKERS = (b'hf_', b'sk-', b'ghp_', b'github_pat_', b'AKIA', b'ASIA', b'Bearer', b'bearer', b'PRIVATE KEY')


def exclusion(path: PurePosixPath, *, is_dir: bool = False) -> str | None:
    parts = set(path.parts)
    name = path.name.lower()
    if path.is_absolute() or '..' in path.parts:
        return 'unsafe path'
    if parts & SKIP_PARTS:
        return 'dependency/cache, gateway auth state, or private evaluation bank'
    # Dependency locks bind submitted source to exact crates/packages. They are
    # source artifacts, unlike process/advisory locks used during execution.
    dependency_lock = name in {'cargo.lock', 'uv.lock', 'poetry.lock', 'pdm.lock', 'yarn.lock', 'bun.lock'}
    if name == '.ds_store' or name.startswith('.env') or name.endswith(('.pyc', '.tmp', '.download-part')) or (name.endswith('.lock') and not dependency_lock):
        return 'local/transient state'
    if name in {'token', 'stored_tokens', 'credentials.json', 'gateway.json', 'gateway-state.json'}:
        return 'authentication state'
    if name.startswith(('private-bank', 'private-seeds', 'table-proposed-')) or 'private-bank' in name:
        return 'private evaluation bank'
    if parts & {'private', 'host-private-panel'}:
        # Reference weights are part of full run context. Do not confuse those
        # with the adjacent withheld evaluation maps, schedules and seed banks.
        if is_dir:
            return None
        model = path.suffix in {'.pt', '.pth', '.ckpt', '.safetensors'}
        model |= path.suffix == '.npz' and any(s in name for s in ('weights', 'policy', 'params', 'checkpoint', 'final', 'resume', 'model'))
        binding = name in {'binding.json', 'manifest.json', 'config.json', 'arch.json', 'network_spec.json', 'metadata.json', 'source-hashes.json'}
        code = path.suffix in {'.py', '.cpp', '.h', '.hpp', '.rs', '.cjs', '.js'}
        if not (model or binding or code):
            return 'withheld private evaluation material'
    if path.suffix in ASSET_SUFFIXES and parts & {'vendor', 'candidates', 'engines', 'assets', 'static'}:
        return 'raw game/dependency asset'
    return None


def scan_and_hash(stream, copy_to=None) -> tuple[str, int, bool]:
    h = hashlib.sha256(); size = 0; tail = b''; secret = False
    while chunk := stream.read(1024 * 1024):
        h.update(chunk); size += len(chunk)
        scan = tail + chunk
        if any(marker in scan for marker in SECRET_MARKERS) and any(p.search(scan) for p in SECRET_PATTERNS):
            secret = True
        tail = chunk[-512:]
        if copy_to is not None:
            copy_to.write(chunk)
    return h.hexdigest(), size, secret


def write_jsonl(stream, record):
    stream.write(json.dumps(record, ensure_ascii=False) + '\n'); stream.flush()


def stage(roots: list[Path], output: Path, relative_to: Path) -> dict:
    relative_to = relative_to.resolve()
    if output.exists():
        raise ValueError('Use a new output directory; snapshots must not overwrite earlier captures.')
    output.mkdir(parents=True)
    payload = output / 'files'; payload.mkdir()
    totals = {'files': 0, 'bytes': 0, 'excluded_entries': 0, 'expanded_archives': 0}
    last_report = time.monotonic()
    with (output/'files.jsonl').open('w') as records, (output/'exclusions.jsonl').open('w') as excluded:
        def omit(path, reason, **extra):
            totals['excluded_entries'] += 1
            write_jsonl(excluded, {'path': str(path), 'reason': reason, **extra})

        def add_stream(stream, rel: PurePosixPath, **extra):
            target = payload / str(rel)
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('wb') as dest:
                digest, size, secret = scan_and_hash(stream, dest)
            if secret:
                target.unlink(); omit(rel, 'recognizable credential literal; upload copy withheld', **extra)
                return
            totals['files'] += 1; totals['bytes'] += size
            write_jsonl(records, {'path': str(rel), 'bytes': size, 'sha256': digest, **extra})

        for root in roots:
            root = root.resolve()
            if not root.exists():
                omit(str(root), 'requested root not present'); continue
            # A root may be a single file (e.g. a run's sibling .log); walk it as its parent with only that file.
            walk = ([(str(root.parent), [], [root.name])] if root.is_file() and not root.is_symlink()
                    else os.walk(root, followlinks=False))
            for base, dirs, files in walk:
                # Record skipped directory entries, then avoid traversing their contents.
                for d in list(dirs):
                    p = Path(base)/d; rel = PurePosixPath(p.relative_to(relative_to).as_posix())
                    reason = exclusion(rel, is_dir=True)
                    if reason or p.is_symlink():
                        dirs.remove(d); omit(rel, reason or 'symlink; not dereferenced')
                for name in sorted(files):
                    p = Path(base)/name; rel = PurePosixPath(p.relative_to(relative_to).as_posix())
                    reason = exclusion(rel)
                    if reason or p.is_symlink() or not p.is_file():
                        omit(rel, reason or 'symlink/non-regular file'); continue
                    # Inspect nested tar archives. Preserve safe archives byte-for-byte;
                    # expose safe members separately if an archive contains excluded state.
                    is_tar = name.endswith(('.tar', '.tar.gz', '.tgz', '.tar.bz2', '.tar.xz'))
                    damaged = None
                    if is_tar:
                        # A truncated or corrupt archive is still run evidence: it is copied byte-for-byte below
                        # (the byte-level credential scan still applies) and marked in the manifest.
                        try:
                            with tarfile.open(p, 'r:*') as archive:
                                archive.getmembers()
                        except (tarfile.TarError, EOFError, OSError) as exc:
                            damaged = type(exc).__name__ + ': ' + str(exc)[:200]
                    if is_tar and damaged is None:
                        with tarfile.open(p, 'r:*') as archive:
                            members = archive.getmembers()
                            blocked = [m for m in members if exclusion(PurePosixPath(m.name), is_dir=m.isdir()) or m.issym() or m.islnk()]
                            secret_member = False
                            for m in members:
                                if m.isfile() and not exclusion(PurePosixPath(m.name)):
                                    with archive.extractfile(m) as s:
                                        _, _, found = scan_and_hash(s)
                                    secret_member |= found
                            if blocked or secret_member:
                                with p.open('rb') as s:
                                    original_sha, _, _ = scan_and_hash(s)
                                omit(rel, 'nested archive expanded to preserve safe members', source_sha256=original_sha)
                                totals['expanded_archives'] += 1
                                for m in members:
                                    inner = PurePosixPath(m.name)
                                    dest_rel = PurePosixPath(str(rel)+'.contents')/inner
                                    why = exclusion(inner, is_dir=m.isdir())
                                    if why or m.issym() or m.islnk():
                                        omit(dest_rel, why or 'nested link'); continue
                                    if m.isfile():
                                        with archive.extractfile(m) as s:
                                            add_stream(s, dest_rel, source_archive=str(rel), source_archive_sha256=original_sha)
                                continue
                    before = p.stat()
                    with p.open('rb') as s:
                        add_stream(s, rel, source_mtime_ns=before.st_mtime_ns,
                                   **(dict(damaged_archive=damaged) if damaged else {}))
                    after = p.stat()
                    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                        raise RuntimeError('Source changed during copy; retain this incomplete snapshot and recapture: '+str(rel))
                    if time.monotonic()-last_report > 30:
                        print('STAGED', totals, flush=True); last_report=time.monotonic()
    summary = {**totals, 'source_roots': [str(r) for r in roots],
               'snapshot_unix': time.time(), 'complete_copy': True,
               'note': 'Original run files are unchanged. Review exclusions.jsonl for omitted auth/cache/assets/banks and expanded nested archives.'}
    (output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', action='append', required=True, type=Path)
    p.add_argument('--relative-to', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    args=p.parse_args()
    print(json.dumps(stage(args.root, args.output.resolve(), args.relative_to.resolve()), indent=2))


if __name__ == '__main__':
    main()
