#!/usr/bin/env python3
"""Train a config-only variant of a release's starter through the normal recipe path. Spends worker time.

A real recipe session on Modal with a scripted researcher instead of a model: it applies the given JSON
edits to the workspace's config files through the real `ar` command, checks, submits and finalizes without
any research run, and the host then trains that recipe from scratch exactly as it would a model's answer.
Used for host-side reference recipes (for example "the starter with a wider network") when measuring how
much room a game leaves above its starter.

    MODAL_PROFILE=duper .venv-<game>-ar/bin/python tools/scripted_recipe.py GAME --release R --out NEW_DIR \
        --set configs/train.json:width=64 [--set FILE:KEY=JSON ...] [--final-train-seconds S]

KEY may be dotted (`network.hidden=128`). The same recipe is replicated on other seeds with
`python -m harness retrain GAME --release R --session NEW_DIR --seed N`.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from harness import policy, session  # noqa: E402
from harness.transport import LocalBox  # noqa: E402
from tools.live_stop_check import Researcher, Script  # noqa: E402


def apply(workspace, edits):
    """Apply FILE:KEY=JSON edits in place; returns what changed."""
    changed = []
    for edit in edits:
        target, _, assignment = edit.partition(':')
        key, _, value = assignment.partition('=')
        if not target or not key or not value:
            raise ValueError('--set takes FILE:KEY=JSON, got ' + edit)
        path = workspace / target
        document = json.loads(path.read_text())
        node = document
        *parents, leaf = key.split('.')
        for part in parents:
            node = node[part]
        if leaf not in node:
            raise ValueError(f'{target} has no key {key}; a variant changes existing settings only')
        before, node[leaf] = node[leaf], json.loads(value)
        path.write_text(json.dumps(document, indent=2) + '\n')
        changed.append(dict(file=target, key=key, before=before, after=node[leaf]))
    return changed


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('game')
    parser.add_argument('--release', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--set', action='append', default=[], metavar='FILE:KEY=JSON', dest='edits')
    parser.add_argument('--final-train-seconds', type=float)
    parser.add_argument('--label', default='scripted/variant', help='recorded as the session model')
    args = parser.parse_args()
    out, box = args.out.resolve(), args.out.resolve().with_name(args.out.name + '.box')
    policy.require_modal()
    policy.verify_release(args.game, args.release)
    report = dict(game=args.game, edits=None, replies={})

    def script():
        try:
            ar = Researcher(out / 'task', box)
            report['edits'] = apply(ar.workspace, args.edits)
            for step in (('check',), ('submit', '--note', 'scripted variant: ' + '; '.join(args.edits)), ('finalize',)):
                report['replies'][step[0]] = reply = ar(*step)
                if 'error' in reply:
                    return
        except BaseException as exc:  # the host still closes the session
            report['script_error'] = type(exc).__name__ + ': ' + str(exc)

    result = session.run(args.game, 'recipe', args.release, out, model=args.label,
                         final_train_seconds=args.final_train_seconds, use_budget=False, restarts=0,
                         launcher=lambda session_dir: Script(script),
                         find_box=lambda: LocalBox(box / 'ar') if (box / 'ar').is_dir() else None)
    final = result.get('final_training') or {}
    report.update(status=result.get('status'), error=result.get('error'), final_status=final.get('status'),
                  final_error=final.get('error'))
    (out / 'scripted-recipe.json').write_text(json.dumps(report, indent=2, default=str) + '\n')
    print(json.dumps({key: report[key] for key in ('edits', 'status', 'error', 'final_status', 'final_error')},
                     indent=2, default=str))
    raise SystemExit(0 if final.get('status') == 'complete' else 1)


if __name__ == '__main__':
    main()
