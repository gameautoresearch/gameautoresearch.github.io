"""Root-side export after stopping the unprivileged trainer. No symlink following."""
from pathlib import Path
import argparse
import json
import os
import stat

from .bundle import seal, archive_bundle


def safe_read(path, limit):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)  # a FIFO the trainer left must not block the export
    try:
        s = os.fstat(fd)
        if not stat.S_ISREG(s.st_mode) or s.st_size > limit:
            raise ValueError('Invalid trainer output file: ' + str(path))
        with os.fdopen(fd, 'rb', closefd=False) as f:
            return f.read(limit + 1)
    finally:
        os.close(fd)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prefix', default='/job')
    parser.add_argument('--allocated-elapsed', type=float)
    args = parser.parse_args()
    prefix = Path(args.prefix)
    if prefix != Path('/job'):
        raise ValueError('Invalid trusted export prefix')
    target = prefix / 'trusted-export'
    target.mkdir(mode=0o700)
    sources = [(n, '/job/input/' + n, 524288) for n in ('policy.py', 'learner.py', 'train.py')]
    sources += [('model.json', '/job/output/model.json', 65536), ('weights.pt', '/job/output/weights.pt', 134217728)]
    for name, origin, cap in sources:
        (target / name).write_bytes(safe_read(origin, cap))
    identity = seal(target, prefix / 'export', {
        'kind': 'allocated_worker_output', 'allocated_elapsed_seconds': args.allocated_elapsed,
        'recipe': json.loads(safe_read('/job/input/config.json', 65536)),
        'training_request': json.loads(safe_read('/job/input/request.json', 65536))})
    archive_bundle(prefix / 'export', prefix / 'export.tar')
    print(json.dumps({'bundle_sha256': identity}), flush=True)


if __name__ == '__main__':
    main()
