#!/usr/bin/env python3
"""Live check that `ar stop` keeps a training run's network (any of the six games). Spends worker time.

A real network session on Modal with a scripted researcher instead of a model: it starts one smoke
training run through the real `ar` command, waits until the trainer has made progress, stops it, and
expects the run to finish `complete` with `stopped_early`, then practises, submits and finalizes it.

    MODAL_PROFILE=duper .venv-<game>-ar/bin/python tools/live_stop_check.py GAME --release R --out NEW_DIR

Everything the check saw is written to NEW_DIR/stop-check.json; the session's own records are beside it.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from harness import policy, registry, session  # noqa: E402
from harness.transport import LocalBox  # noqa: E402

# Run seconds, how long the trainer must have been running before the stop, and how to see its progress.
GAMES = {
    'clash': dict(seconds=270, settle=120),
    'tavern': dict(seconds=330, settle=75),
    # These three are watched through the researcher's own `ar status`: the stop comes `settle` seconds after
    # the job is reported running, long enough for setup, compilation and a first written policy.
    'generals': dict(seconds=170, settle=75),
    'pokemon': dict(seconds=300, settle=150),
    'polytopia': dict(seconds=420, settle=230),
    'doudizhu': dict(seconds=240, settle=90),
}


class Researcher:
    """The real in-box `ar` against a local directory standing in for the box."""

    def __init__(self, task_dir, box):
        self.ar, self.workspace = box / 'ar', box / 'workspace'
        shutil.copytree(task_dir / 'environment/ar', self.ar)
        shutil.copytree(task_dir / 'environment/workspace', self.workspace)
        for name in ('requests', 'replies'):
            (self.ar / name).mkdir(exist_ok=True)
        self.calls = []

    def __call__(self, *args):
        env = dict(os.environ, AR_ROOT=str(self.ar), AR_WORKSPACE=str(self.workspace))
        started = time.time()
        done = subprocess.run([sys.executable, '-I', str(self.ar / 'ar.py'), *args], env=env, capture_output=True,
                              text=True, timeout=1000, cwd=self.workspace)
        try:
            reply = json.loads(done.stdout)
        except ValueError:
            reply = dict(unparsed=done.stdout[-2000:], stderr=done.stderr[-2000:])
        self.calls.append(dict(args=list(args), at=started, seconds=round(time.time() - started, 2),
                               exit=done.returncode, reply=reply))
        return reply


class Script:
    """Stands in for the `harbor run` process: the researcher's script on a thread, waited for without limit."""

    def __init__(self, work):
        self.thread = threading.Thread(target=work, daemon=True)
        self.thread.start()

    def wait(self):
        self.thread.join()
        return 0


def job(status, name):
    return next((entry for entry in status.get('jobs', []) if entry.get('name') == name), {})


def trainer_progress(game, backend, blind=False):
    """How far the running trainer is, read from its worker by the host (the researcher cannot see this).

    Returns (training command running, updates finished or None when that could not be read).
    """
    controller, root = backend.controller, backend.root
    if game not in ('clash', 'tavern'):
        state = next((entry.get('status') for entry in backend.status().get('jobs', []) if entry.get('name') == 'a'), None)
        return state == 'running', None
    if game == 'tavern':
        key = ('train', 'a')
        sandbox = controller.sandboxes.get(key)
        if key not in controller.training or sandbox is None:
            return False, None
        try:
            return True, json.loads(controller.provider.read(sandbox, '/job/train/export.json'))['update']
        except Exception:
            return True, None
    record = root / 'jobs/a/provider.json'
    if not record.is_file():
        return False, None
    import modal
    from simulators.generals.autoresearch.provider import allocation_info
    provider = json.loads(record.read_text())
    receipt = allocation_info(provider['app_id'], provider['sandbox_id'])
    if not receipt or not receipt.get('started_at'):
        return False, None
    if blind:
        return True, None
    try:
        text = modal.Sandbox.from_id(provider['sandbox_id']).filesystem.read_text('/retained/output/artifact/progress.jsonl')
        return True, len(text.splitlines())
    except Exception:
        return True, None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('game', choices=sorted(GAMES))
    parser.add_argument('--release', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--skip-practice', action='store_true')
    parser.add_argument('--blind', action='store_true', help='never read the worker; stop after the settle time')
    parser.add_argument('--settle', type=float, help="seconds of training before the stop (default: the game's)")
    args = parser.parse_args()
    plan, out, box = dict(GAMES[args.game]), args.out.resolve(), args.out.resolve().with_name(args.out.name + '.box')
    if args.settle:
        plan['settle'] = args.settle
    policy.require_modal()
    policy.verify_release(args.game, args.release)
    report = dict(game=args.game, release=str(args.release.resolve()), started=time.time(), events=[], passed=False)
    backends = []

    class Probe(registry.backend(args.game)):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            backends.append(self)

    def note(event, **values):
        report['events'].append(dict(event=event, at=round(time.time() - report['started'], 1), **values))
        print(json.dumps(report['events'][-1], default=str), flush=True)

    def script():
        try:
            ar = Researcher(out / 'task', box)
            report['calls'] = ar.calls
            note('budget', reply=ar('budget'))
            note('check', reply=ar('check', '--config', 'configs/smoke.json'))
            queued = ar('run', 'a', '--seconds', str(plan['seconds']), '--config', 'configs/smoke.json')
            note('run', reply=queued)
            if 'error' in queued:
                return
            # Wait until the trainer has been training for a while and (where it can be read) finished an update.
            backend, running_since, updates, deadline = backends[0], None, None, time.time() + 2400
            while time.time() < deadline:
                status = ar('status')
                state = job(status, 'a').get('status')
                if state not in ('queued', 'running'):
                    note('ended-before-stop', job=job(status, 'a'))
                    return
                running, updates = trainer_progress(args.game, backend, args.blind)
                if running and running_since is None:
                    running_since = time.time()
                    note('trainer-running')
                if running_since and ((updates or 0) >= 2 or time.time() - running_since >= plan['settle']):
                    break
                time.sleep(5)
            note('stopping', trainer_seconds=round(time.time() - (running_since or time.time()), 1), updates_seen=updates)
            note('stop', reply=ar('stop', 'a'))
            deadline = time.time() + 600
            while time.time() < deadline:
                status = ar('status')
                if 'a' not in status.get('active', []):
                    break
                time.sleep(3)
            final = job(ar('status'), 'a')
            note('after-stop', job=final)
            log = ar('log', 'a', '--tail', '15')
            note('log', reply={key: value for key, value in log.items() if key not in ('progress',)})
            report['kept'] = final.get('status') == 'complete' and bool(final.get('stopped_early'))
            if not report['kept']:
                return
            if not args.skip_practice:
                practice = ar('eval', 'a')
                note('practice-queued', reply=practice)
                if 'error' not in practice:
                    deadline = time.time() + 1200
                    while time.time() < deadline:
                        status = ar('wait', '--timeout', '120')
                        if not status.get('active'):
                            break
                    done = job(status, practice.get('queued', ''))
                    note('practice', job=done)
                    report['practised'] = done.get('status') == 'complete'
            submitted = ar('submit', 'a', '--note', 'live stop check')
            note('submit', reply=submitted)
            finalized = ar('finalize')
            note('finalize', reply=finalized)
            report['submitted'] = 'error' not in submitted and bool(finalized.get('finalized'))
            report['passed'] = bool(report['kept'] and report['submitted'] and report.get('practised', True))
        except BaseException as exc:  # the host still settles every worker below
            report['script_error'] = type(exc).__name__ + ': ' + str(exc)
            report['script_traceback'] = traceback.format_exc()
            note('script-error', error=report['script_error'])

    original = registry.backend
    registry.backend = lambda game: Probe
    try:
        result = session.run(args.game, 'network', args.release, out, model='scripted/live-stop-check', smoke=True,
                             wall_seconds=3600, final=False, launcher=lambda session_dir: Script(script),
                             find_box=lambda: LocalBox(box / 'ar') if (box / 'ar').is_dir() else None)
    finally:
        registry.backend = original
    report.update(finished=time.time(), session=dict(status=result.get('status'), error=result.get('error'),
                                                    budget=result.get('budget'), submission=result.get('submission')))
    (out / 'stop-check.json').write_text(json.dumps(report, indent=2, default=str) + '\n')
    print(json.dumps(dict(passed=report['passed'], kept=report.get('kept'), practised=report.get('practised'),
                          submitted=report.get('submitted'), session=report['session']), indent=2, default=str))
    raise SystemExit(0 if report['passed'] else 1)


if __name__ == '__main__':
    main()
