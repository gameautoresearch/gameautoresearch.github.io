#!/usr/bin/env python3
"""Train the current files once on a worker of your own, and seal the result as a run.

    python training/launch.py NAME --seconds S [--config configs/smoke.json] [--gpu 1 --cpu N --memory GIB]

This is the plain case, done with the same `ar` commands you can type yourself:

    ar check                                    the editable files are valid
    ar lease NAMEw --seconds S --gpu .. ...      a worker, held for at most S seconds
    ar exec NAMEw -- TRAINING COMMAND            the starter's training command, in /work on the worker
    ar package NAMEw NAME --path OUTPUT          seal the network it wrote as run NAME
    ar stop NAMEw                                end the worker; only the time it lived is charged

It returns once the training command has started and finishes the rest in the background, writing what it
does to LOG (printed below). `ar wait NAMEw` blocks until the worker has ended; then `ar eval NAME` and
`ar submit NAME`. The files are copied when you launch, so you can edit `training/` and `configs/` for the
next experiment while this one trains: the run is sealed with the files it was started from.

It is yours to change or replace: it holds no authority and is not part of what is trained or graded. Keep a
worker (`--keep`) to run more commands on it, continue from a checkpoint in /scratch, run several trainings on
one worker, or drive `ar lease` and `ar exec` by hand. Standard library only.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

# Filled in for this game when the release is built: the training command, where it writes, the worker shape.
SETTINGS = None

WORKSPACE = Path(os.environ.get('AR_WORKSPACE', '/workspace'))
STATE = Path(os.environ.get('AR_LAUNCH_STATE', '/tmp/launch'))
POLL_SECONDS = float(os.environ.get('AR_LAUNCH_POLL', 5.0))


def ar(*args, workspace=None, timeout=1200):
    """One `ar` command; its JSON reply. A reply with an `error` key means the command was refused."""
    env = dict(os.environ)
    if workspace is not None:
        env['AR_WORKSPACE'] = str(workspace)
    try:
        done = subprocess.run(['ar', *map(str, args)], capture_output=True, text=True, timeout=timeout, env=env)
        return json.loads(done.stdout)
    except subprocess.TimeoutExpired:
        return {'error': 'ar ' + str(args[0]) + ' did not answer in time'}
    except ValueError:
        return {'error': 'ar ' + str(args[0]) + ' printed no JSON: ' + (done.stderr or done.stdout)[-500:]}


def say(*parts):
    print(time.strftime('%H:%M:%S'), *parts, flush=True)


def job(name):
    for item in ar('status').get('jobs', []):
        if item.get('name') == name:
            return item
    return None


def snapshot(folder):
    """A copy of the editable files as they are now: what the worker gets and what the run is sealed with."""
    names = json.loads((Path(os.environ.get('AR_ROOT', '/ar')) / 'box.json').read_text())['writable']
    for name in names:
        target = folder / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(WORKSPACE / name, target)
    return folder


def command(settings, name, config, seed, seconds, files):
    values = dict(name=name, config=config, seed=seed, seconds=int(seconds),
                  trainer=json.loads((files / config).read_text()).get('trainer', {}))
    return settings['command'].format(**values), settings['output'].format(**values)


def finish(args, settings, worker, files):
    """Everything after the lease request: wait for the worker, train, package, stop."""
    started = None
    while True:
        state = job(worker)
        if state is None or state.get('status') not in ('queued', 'running'):
            say('The worker never became ready:', json.dumps(state))
            return 1
        if state['status'] == 'running' and started is None:
            started = time.time()  # the worker is charged from about now
        if state.get('ready'):
            break
        time.sleep(POLL_SECONDS)
    # Train for what the grant leaves once the worker's own start and end and the packaging are taken off.
    ends = started + args.seconds
    seconds = ends - time.time() - settings['reserve_seconds']
    if seconds < 10:
        say(f'--seconds {args.seconds:g} leaves no training time on this worker; nothing was trained')
        ar('stop', worker)
        return 1
    line, output = command(settings, args.name, args.config, args.seed, seconds, files)
    say('training:', line)
    reply = ar('exec', worker, '--wait', 0, '--', line)
    if 'error' in reply:
        say('The training command did not start:', reply['error'])
        ar('stop', worker)
        return 1
    # Follow the command until it exits, or until the worker is about to stop serving commands.
    code = None
    while time.time() < ends - settings['reserve_seconds'] / 2:
        state = ar('poll', worker, reply['exec'], '--wait', 60, '--tail', 600)
        if 'error' in state:
            say('poll:', state['error'])
            time.sleep(POLL_SECONDS)
        elif state.get('state') != 'running':
            code = state.get('exit_code')
            say(f'the training command exited with code {code}; its last output:\n' + state.get('output', ''))
            break
    else:
        say('the training command is still running close to the end of the grant; packaging what it has written')
    # Seal what the trainer wrote: its output folder, or failing that the newest whole export it left.
    paths = [output]
    if settings.get('fallback'):
        found = ar('exec', worker, '--wait', 30, '--', settings['fallback'].format(name=args.name))
        paths += [line for line in found.get('output', '').split() if line.startswith('/')][-1:]
    sealed = None
    for path in paths:
        sealed = ar('package', worker, args.name, '--path', path, '--config', args.config, workspace=files)
        if 'error' not in sealed:
            say(f'packaged {path} as run {args.name}:', json.dumps(sealed))
            break
        say(f'packaging {path} failed:', sealed['error'])
    if args.keep:
        say(f'the worker {worker} is kept and charged until its grant ends or `ar stop {worker}`')
    else:
        say('stopping the worker:', json.dumps(ar('stop', worker)))
    ok = sealed is not None and 'error' not in sealed
    say(f'done: ar eval {args.name}' if ok else 'done: no run was packaged; whatever the trainer wrote is under '
        + output + ' in /scratch, which a later lease can read')
    return 0 if ok else 1


def main(argv=None):
    settings = SETTINGS
    shape = settings['shape']
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('name', help='the run to create: 1-39 lowercase letters and digits; its worker is NAMEw')
    parser.add_argument('--seconds', type=float, required=True, help='how long the worker is held, all of it charged')
    parser.add_argument('--config', default='configs/train.json', choices=('configs/train.json', 'configs/smoke.json'))
    parser.add_argument('--seed', type=int, default=1, help='passed to the training command where it takes one')
    parser.add_argument('--gpu', type=int, default=shape['gpu'], help='1 for a GPU worker, 0 for CPU only')
    parser.add_argument('--cpu', type=int, default=shape['cpu'], help='CPU cores')
    parser.add_argument('--memory', type=int, default=shape['memory'], help='memory in GiB')
    parser.add_argument('--keep', action='store_true', help='leave the worker running after packaging')
    parser.add_argument('--no-check', action='store_true', help='skip ar check')
    parser.add_argument('--foreground', action='store_true', help='stay in this terminal until the run is packaged')
    args = parser.parse_args(argv)
    worker = args.name + 'w'
    folder = STATE / args.name
    if folder.exists():
        parser.error(f'{args.name} was launched before; run names are used once')
    files = snapshot(folder / 'files')
    if not args.no_check:
        checked = ar('check', '--config', args.config, workspace=files)
        if 'error' in checked:
            shutil.rmtree(folder)
            print(json.dumps(checked, indent=2))
            return 1
    leased = ar('lease', worker, '--seconds', args.seconds, '--gpu', args.gpu, '--cpu', args.cpu,
                '--memory', args.memory, workspace=files)
    if 'error' in leased:
        shutil.rmtree(folder)
        print(json.dumps(leased, indent=2))
        return 1
    log = folder / 'launch.log'
    print(json.dumps(dict(launched=args.name, worker=worker, log=str(log), lease=leased,
                          next=f'ar wait {worker}   (until the worker has ended), then: ar eval {args.name}'), indent=2))
    if args.foreground:
        return finish(args, settings, worker, files)
    if os.fork():
        return 0
    os.setsid()
    with log.open('a') as stream, open(os.devnull) as null:
        os.dup2(null.fileno(), 0)
        os.dup2(stream.fileno(), 1)
        os.dup2(stream.fileno(), 2)
    try:
        code = finish(args, settings, worker, files)
    except BaseException as exc:  # the log is the only place this is seen
        say('launch.py failed:', type(exc).__name__, exc)
        code = 1
    os._exit(code)


if __name__ == '__main__':
    sys.exit(main())
