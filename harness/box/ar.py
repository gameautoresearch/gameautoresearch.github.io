#!/usr/bin/env python3
"""Researcher-side command transport, identical for every game and task variant.

Standard library only. This file holds no authority: it writes a request and
waits for the host broker's reply. Budgets, runs, submissions and grading are
decided outside the box.
"""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import uuid

AR = Path(os.environ.get('AR_ROOT', '/ar'))
WORKSPACE = Path(os.environ.get('AR_WORKSPACE', '/workspace'))
CONFIGS = ('configs/train.json', 'configs/smoke.json')
DRY_RUN_SECONDS = 600
# The longest one `ar wait` blocks the terminal. A researcher whose terminal is held for longer than its own
# turn cannot do anything else, and has lost half an hour trying to interrupt a 30-minute wait.
WAIT_SECONDS = 300.0


def box():
    return json.loads((AR / 'box.json').read_text())


def request(value, wait=180.0):
    name = uuid.uuid4().hex
    pending = AR / 'requests'
    temporary = pending / (name + '.tmp')
    temporary.write_text(json.dumps(value))
    temporary.replace(pending / (name + '.json'))
    reply = AR / 'replies' / (name + '.json')
    deadline = time.monotonic() + wait
    while time.monotonic() < deadline:
        if reply.exists():
            try:
                return json.loads(reply.read_text())
            except ValueError:  # reply still being written
                pass
        time.sleep(0.2)
    return {'error': 'The broker has not answered yet. Nothing is known about this command: '
                     'check `ar status` before repeating it.'}


def files():
    return {name: (WORKSPACE / name).read_text() for name in box()['writable']}


def config_path(value):
    path = Path(value)
    name = path.as_posix()
    if path.is_absolute():
        name = path.relative_to(WORKSPACE).as_posix()
    if name not in CONFIGS:
        raise SystemExit('--config must be configs/train.json or configs/smoke.json')
    return name


def dry_run(config):
    """The game's CPU dry run of the current editable files, in this box. None if the game has none."""
    script = AR / 'dryrun.py'
    if not script.is_file():
        return None
    command = [sys.executable, '-I', str(script), '--workspace', str(WORKSPACE), '--config', config]
    try:
        done = subprocess.run(command, capture_output=True, text=True, timeout=DRY_RUN_SECONDS)
        return json.loads(done.stdout)
    except subprocess.TimeoutExpired:
        return {'passed': False, 'error': f'The dry run did not finish in {DRY_RUN_SECONDS} seconds'}
    except ValueError:
        return {'passed': False, 'error': 'The dry run itself failed: ' + (done.stderr or done.stdout)[-1500:]}


def active(status, name=None):
    names = status.get('active', [])
    return name in names if name else bool(names)


def parser(variant, options=None):
    options = options or {}
    p = argparse.ArgumentParser(prog='ar', description='Research session commands. See /workspace/instructions.md.')
    sub = p.add_subparsers(dest='op', required=True)
    sub.add_parser('budget', help='remaining allocation, deadlines and what is graded')
    sub.add_parser('status', help='your runs and their progress')
    c = sub.add_parser('check', help='validate the editable files without spending allocation'
                       + (', then dry-run them here on CPU' if options.get('dry_run') else ''))
    c.add_argument('--config', default=CONFIGS[0], type=config_path)
    if not options.get('leases'):
        # A game with leases has one way to get compute: a worker the researcher leases and drives
        # (`training/launch.py` does the plain case). There the host alone starts fixed training runs.
        c = sub.add_parser('run', help='train the current editable files on a worker')
        c.add_argument('name')
        c.add_argument('--seconds', type=float, required=True, help='worker time granted to this run')
        c.add_argument('--config', default=CONFIGS[0], type=config_path)
        if options.get('run_seed'):
            c.add_argument('--seed', type=int, default=1)
    c = sub.add_parser('wait', help='block until active work finishes')
    c.add_argument('name', nargs='?')
    c.add_argument('--timeout', type=float, default=WAIT_SECONDS,
                   help=f'seconds to block here, at most {WAIT_SECONDS:g}; run it again to keep waiting')
    if options.get('leases'):
        c.add_argument('--started', action='store_true', help='return as soon as NAME is running (a lease is ready)')
        c = sub.add_parser('lease', help='hold a worker of a shape you choose and run your own commands on it')
        c.add_argument('name')
        c.add_argument('--seconds', type=float, required=True, help='how long the worker is held at most')
        c.add_argument('--gpu', type=int, default=1, help='1 for a GPU worker, 0 for CPU only')
        c.add_argument('--cpu', type=int, default=4, help='CPU cores')
        c.add_argument('--memory', type=int, default=16, help='memory in GiB')
        c = sub.add_parser('exec', help='run a shell command on a running lease: ar exec NAME -- COMMAND')
        c.add_argument('name')
        c.add_argument('--wait', type=float, default=60.0, help='seconds to wait here for the command to finish; '
                       'it keeps running on the lease after that')
        c.add_argument('--tail', type=int, default=6000, help='characters of output to show')
        c.add_argument('command', nargs='*', help='everything after -- is the command')
        c = sub.add_parser('poll', help='state and output of a command started with ar exec')
        c.add_argument('name')
        c.add_argument('exec')
        c.add_argument('--wait', type=float, default=0.0)
        c.add_argument('--tail', type=int, default=6000)
        c = sub.add_parser('push', help='copy your current editable files (training/, configs/ and the other files `ar status` lists) to a running lease')
        c.add_argument('name')
        c = sub.add_parser('sync', help="commit a lease's /scratch so other leases can see it")
        c.add_argument('name')
        c.add_argument('--reload', action='store_true', help='also bring in what other leases committed')
        c = sub.add_parser('package', help='seal a network a lease produced as a run you can eval and submit')
        c.add_argument('lease')
        c.add_argument('name')
        c.add_argument('--path', required=True, help='directory on the lease holding the network files a training run writes')
        c.add_argument('--config', default=CONFIGS[0], type=config_path)
    c = sub.add_parser('log', help='output and metrics from a run')
    c.add_argument('name')
    c.add_argument('--tail', type=int, default=80)
    c = sub.add_parser('stop', help='stop a queued or running job')
    c.add_argument('name')
    c = sub.add_parser('eval', help='public practice games for a completed run')
    c.add_argument('name')
    if options.get('eval_seconds'):
        c.add_argument('--seconds', type=float, help='worker time granted to this practice')
    if variant == 'recipe':
        c = sub.add_parser('submit', help='register the current training/ and configs/ as your recipe')
    else:
        c = sub.add_parser('submit', help='register a completed run as your network')
        c.add_argument('name')
    c.add_argument('--note', default='')
    sub.add_parser('finalize', help='close the session')
    sub.add_parser('_verify')
    return p


def command_text(parts):
    """`ar exec NAME -- python a.py --x 1` and `ar exec NAME -- 'cd /scratch && ls'` both mean what they say."""
    parts = list(parts)
    return parts[0] if len(parts) == 1 else shlex.join(parts)


def follow(name, exec_id, wait, tail):
    """Poll one lease command until it exits or `wait` seconds pass."""
    deadline = time.monotonic() + wait
    while True:
        result = request({'op': 'poll', 'name': name, 'exec': exec_id, 'tail': tail})
        if 'error' in result or result.get('state') != 'running' or time.monotonic() >= deadline:
            break
        time.sleep(2)
    if result.get('state') == 'running':
        result['note'] = f'Still running on the lease. ar poll {name} {exec_id} --wait S shows it again.'
    return result


def main(argv=None):
    try:
        return _main(argv)
    except KeyboardInterrupt:
        print(json.dumps({'interrupted': True, 'note': 'Only this ar client stopped; any run or request already '
                          'sent continues on the host. Check ar status.'}, indent=2))
        return 130


def _main(argv=None):
    options = box()
    variant = options['variant']
    argv = list(sys.argv[1:] if argv is None else argv)
    command = None
    if argv[:1] == ['exec'] and '--' in argv:
        # Everything after the first `--` is the lease command, dashes and all; `ar` options come before it.
        cut = argv.index('--')
        argv, command = argv[:cut], argv[cut + 1:]
    args = parser(variant, options).parse_args(argv)
    if command is not None:
        args.command = command
    value = {key: item for key, item in vars(args).items() if item is not None}
    if args.op in ('check', 'run', 'lease', 'push', 'package') or (args.op == 'submit' and variant == 'recipe'):
        value['files'] = files()
    if args.op == 'exec':
        result = request({'op': 'exec', 'name': args.name, 'command': command_text(args.command)})
        if 'error' not in result:
            result = follow(args.name, result['exec'], args.wait, args.tail)
    elif args.op == 'poll':
        result = follow(args.name, args.exec, args.wait, args.tail)
    elif args.op == 'wait':
        deadline = time.monotonic() + min(args.timeout, WAIT_SECONDS)
        while True:
            result = request({'op': 'status'})
            if 'error' in result or not active(result, args.name):
                break
            if getattr(args, 'started', False) and any(
                    job.get('name') == args.name and job.get('ready', job.get('status') == 'running')
                    for job in result.get('jobs', [])):
                break
            if time.monotonic() >= deadline:
                stuck = [job.get('name') for job in result.get('jobs', []) if job.get('status') == 'uncertain'
                         and (args.name is None or job.get('name') == args.name)]
                note = ('Still active after --timeout; the work continues. Run ar wait again.' if not stuck else
                        'Still unresolved after --timeout: ' + ', '.join(map(str, stuck)) + '. The host settles '
                        'these from provider records; ar stop NAME resolves one now.')
                result = dict(result, waiting=note)
                break
            time.sleep(5)
    else:
        value.pop('timeout', None)
        result = request(value, wait=900.0 if args.op in ('check', 'package') else 180.0)
        if args.op == 'check' and 'error' not in result:
            local = dry_run(args.config)
            if local is not None:
                result['dry_run'] = local
                if not local.get('passed'):
                    result['error'] = 'The CPU dry run failed; see dry_run. Nothing was charged.'
    print(json.dumps(result, indent=2))
    return 1 if 'error' in result else 0


if __name__ == '__main__':
    sys.exit(main())
