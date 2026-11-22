#!/usr/bin/env python3
"""Live network smoke with a scripted researcher (no model calls). Spends worker time.

A real `network --smoke` session on Modal: the real in-box `ar` (against a local directory standing in for the
box) runs `budget`, `check`, one training run to completion, public practice, `submit`, `finalize`. The host
then grades the submission on a fresh private bank (`harness grade`). Shares its researcher plumbing with
`tools/live_stop_check.py`.

    MODAL_PROFILE=duper .venv-tavern-ar/bin/python tools/live_smoke_check.py tavern --release R --out NEW_DIR [--grade]

Everything the script saw is written to NEW_DIR/smoke-check.json; the session's own records are beside it.
"""
import argparse
import json
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from harness import policy, session  # noqa: E402
from harness.transport import LocalBox  # noqa: E402
from tools.live_stop_check import Researcher, Script, job  # noqa: E402

RUN_SECONDS = dict(tavern=330)  # the smoke's largest single run (180 s training + 150 s worker overhead)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('game', choices=sorted(RUN_SECONDS))
    parser.add_argument('--release', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--grade', action='store_true', help='grade the finalized submission on the host afterwards')
    parser.add_argument('--run-seconds', type=int, help="the training run's `ar run --seconds` (default: the smoke's largest)")
    args = parser.parse_args()
    out = args.out.resolve()
    box = out.with_name(out.name + '.box')
    policy.require_modal()
    policy.verify_release(args.game, args.release)
    report = dict(game=args.game, release=str(args.release.resolve()), started=time.time(), events=[], passed=False)

    def note(event, **values):
        report['events'].append(dict(event=event, at=round(time.time() - report['started'], 1), **values))
        print(json.dumps(report['events'][-1], default=str), flush=True)

    def wait_idle(ar, limit):
        deadline, status = time.time() + limit, {}
        while time.time() < deadline:
            status = ar('wait', '--timeout', '120')
            if not status.get('active'):
                break
        return status

    def script():
        try:
            ar = Researcher(out / 'task', box)
            report['calls'] = ar.calls
            note('budget', reply=ar('budget'))
            note('check', reply=ar('check', '--config', 'configs/smoke.json'))
            queued = ar('run', 'a', '--seconds', str(args.run_seconds or RUN_SECONDS[args.game]), '--config', 'configs/smoke.json')
            note('run', reply=queued)
            if 'error' in queued:
                return
            trained = job(wait_idle(ar, 2400), 'a')
            note('trained', job=trained)
            log = ar('log', 'a', '--tail', '15')
            note('log', reply={key: value for key, value in log.items() if key != 'progress'},
                 progress=(log.get('progress') or [])[-3:])
            report['trained'] = trained.get('status') == 'complete'
            if not report['trained']:
                return
            practice = ar('eval', 'a')
            note('practice-queued', reply=practice)
            if 'error' in practice:
                return
            done = job(wait_idle(ar, 1500), practice.get('queued', ''))
            note('practice', job=done, log=ar('log', practice.get('queued', '')))
            report['practised'] = done.get('status') == 'complete'
            submitted = ar('submit', 'a', '--note', 'live smoke check')
            note('submit', reply=submitted)
            finalized = ar('finalize')
            note('finalize', reply=finalized)
            report['submitted'] = 'error' not in submitted and bool(finalized.get('finalized'))
        except BaseException as exc:  # the host still settles every worker
            report['script_error'] = type(exc).__name__ + ': ' + str(exc)
            report['script_traceback'] = traceback.format_exc()
            note('script-error', error=report['script_error'])

    result = session.run(args.game, 'network', args.release, out, model='scripted/live-smoke-check', smoke=True,
                         wall_seconds=3600, final=False, launcher=lambda session_dir: Script(script),
                         find_box=lambda: LocalBox(box / 'ar') if (box / 'ar').is_dir() else None)
    report.update(session=dict(status=result.get('status'), error=result.get('error'), budget=result.get('budget'),
                               submission=result.get('submission')))
    if args.grade and report.get('submitted'):
        try:
            graded = session.grade(out)
            report['grade'] = {key: graded.get(key) for key in (
                'benchmark_id', 'players', 'complete_valid_schedule', 'expected_cases', 'observed_cases', 'raw_score',
                'mean_placement', 'top_half_rate', 'confidence_interval_95', 'failures', 'per_profile')}
            note('grade', result=report['grade'])
        except BaseException as exc:
            report['grade_error'] = type(exc).__name__ + ': ' + str(exc)
            report['grade_traceback'] = traceback.format_exc()
            note('grade-error', error=report['grade_error'])
    report['passed'] = bool(report.get('trained') and report.get('practised') and report.get('submitted')
                            and (not args.grade or (report.get('grade') or {}).get('complete_valid_schedule')))
    report['finished'] = time.time()
    (out / 'smoke-check.json').write_text(json.dumps(report, indent=2, default=str) + '\n')
    print(json.dumps(dict(passed=report['passed'], session=report['session'], grade=report.get('grade')), indent=2,
                     default=str))
    raise SystemExit(0 if report['passed'] else 1)


if __name__ == '__main__':
    main()
