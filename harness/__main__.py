"""python -m harness: the one entrypoint for preparing, running and grading research sessions.

Run it with the owning game's research interpreter (see `catalog`). Nothing is
spent unless `run --execute`, `final` or `grade` is requested explicitly.
"""
import argparse
import json
from pathlib import Path

from harness import policy, registry, session, task
from harness.broker import VARIANTS
from harness.workspace import GAMES


def options(items):
    result = {}
    for item in items or ():
        key, _, value = item.partition('=')
        try:
            result[key] = json.loads(value)
        except ValueError:
            result[key] = value
    return result


def rules(args):
    """Session rules the operator may relax: None leaves the default (on, except for a smoke)."""
    return dict(use_budget=False if args.allow_early_finalize else None, restarts=args.restarts)


def session_arguments(p):
    p.add_argument('game', choices=GAMES)
    p.add_argument('variant', choices=VARIANTS)
    p.add_argument('--release', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--smoke', action='store_true', help='short end-to-end session on the smoke configuration')
    p.add_argument('--budget-mode', choices=('seconds', 'usd'),
                   help="seconds: one job at a time within a number of worker seconds; usd: parallel jobs within "
                        "a dollar budget (default: implied by --research-seconds/--research-usd, else the game's)")
    p.add_argument('--research-seconds', type=float,
                   help="seconds mode: allocated worker seconds for research (default: the game's)")
    p.add_argument('--research-usd', type=float,
                   help="dollar mode: research budget in US dollars of worker time (default: the game's)")
    p.add_argument('--final-train-seconds', type=float, help='recipe variant: fresh final training time')
    p.add_argument('--wall-seconds', type=float, help='wall-clock limit for the researcher')
    p.add_argument('--allow-early-finalize', action='store_true',
                   help='accept the researcher\'s finalize with allocation unspent (the default for --smoke)')
    p.add_argument('--restarts', type=int, help='how often a researcher that stops with usable allocation left is '
                                                'started again in the same box (default 3; 0 for --smoke)')
    p.add_argument('--agent', help='researcher agent (default terminus-2); a game may enable others, e.g. claude-code')
    p.add_argument('--continue-from', type=Path, metavar='SEED',
                   help='start the workspace from an earlier session\'s editable files and notes (a seed written '
                        'by harness.task.make_seed); the session still has its own new ledger and allocation')
    p.add_argument('--full-session', action='store_true',
                   help='the researcher may finalize only once little allocation or session time remains')


def main(argv=None):
    parser = argparse.ArgumentParser(prog='python -m harness', description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('catalog', help='games, variants, interpreters and default budgets')
    p = sub.add_parser('prepare', help='construct and seal a new release for a game')
    p.add_argument('game', choices=GAMES)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--option', action='append', metavar='KEY=VALUE', help='game-specific constructor argument')
    p = sub.add_parser('verify', help='check a release against its binding and the current source')
    p.add_argument('game', choices=GAMES)
    p.add_argument('--release', required=True, type=Path)
    p = sub.add_parser('task', help='build the Harbor task directory only (offline)')
    session_arguments(p)
    p = sub.add_parser('run', help='run a research session with a researcher agent')
    session_arguments(p)
    p.add_argument('--model', required=True, help='LiteLLM model name, e.g. openrouter/deepseek/deepseek-v4-pro')
    p.add_argument('--api-base', help='inference endpoint override for a locally proxied model')
    p.add_argument('--key-file', type=Path, help='owner-only file holding the model key')
    p.add_argument('--key-env', help='environment variable the agent reads that key from (per-agent default)')
    p.add_argument('--reasoning-effort')
    p.add_argument('--model-info', type=json.loads, help='JSON LiteLLM model_info (Terminus-2 only)')
    p.add_argument('--agent-kwarg', action='append', default=[], metavar='KEY=VALUE', help='extra agent argument')
    p.add_argument('--backend-option', action='append', metavar='KEY=VALUE')
    p.add_argument('--skip-final', action='store_true', help='recipe variant: defer the fresh final training')
    p.add_argument('--execute', action='store_true', help='actually start the session; otherwise print the plan')
    p.add_argument('--scored', action='store_true', help='a scored session; requires --qualification')
    p.add_argument('--qualification', type=Path)
    p = sub.add_parser('pause', help='pause a running session: workspace kept, workers stopped, nothing finalized')
    p.add_argument('--session', required=True, type=Path)
    p.add_argument('--note', help='why, recorded with the pause')
    p = sub.add_parser('resume', help='continue a paused session where it stopped (same budget, runs and storage)')
    p.add_argument('--session', required=True, type=Path)
    p.add_argument('--api-base', help='default: as the session was started')
    p.add_argument('--key-file', type=Path, help='default: as the session was started')
    p.add_argument('--key-env', help='default: as the session was started')
    p.add_argument('--model-info', type=json.loads, help='default: as the session was started')
    p.add_argument('--agent-kwarg', action='append', metavar='KEY=VALUE', help='default: as the session was started')
    p.add_argument('--backend-option', action='append', metavar='KEY=VALUE')
    p.add_argument('--wall-seconds', type=float, help='wall-clock limit from now (default: what was left at the pause)')
    p.add_argument('--skip-final', action='store_true', help='recipe variant: defer the fresh final training')
    p.add_argument('--execute', action='store_true', help='actually resume; otherwise print the pause record')
    p = sub.add_parser('final', help='recipe variant: run the deferred fresh final training')
    p.add_argument('--session', required=True, type=Path)
    p.add_argument('--backend-option', action='append', metavar='KEY=VALUE')
    p = sub.add_parser('retrain', help='host-only replicate: train a recipe or the starter from scratch with a seed')
    p.add_argument('game', choices=GAMES)
    p.add_argument('--release', required=True, type=Path)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument('--session', type=Path, help="a recipe session whose last submission is retrained")
    source.add_argument('--starter', action='store_true', help="the release's own unmodified starter")
    p.add_argument('--seed', required=True, type=int)
    p.add_argument('--out', required=True, type=Path, help='new directory; tournaments take it as an entrant')
    p.add_argument('--label', help='model/entrant label recorded in the binding (default: the source model)')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--final-train-seconds', type=float)
    p.add_argument('--backend-option', action='append', metavar='KEY=VALUE')
    p.add_argument('--execute', action='store_true', help='actually train; otherwise bind nothing and print the plan')
    p = sub.add_parser('grade', help='host-private grading of a finalized session')
    p.add_argument('--session', required=True, type=Path)
    p.add_argument('--backend-option', action='append', metavar='KEY=VALUE')
    p = sub.add_parser('status', help="a session's result and completion evidence")
    p.add_argument('--session', required=True, type=Path)
    args = parser.parse_args(argv)

    if args.command == 'catalog':
        result = registry.catalog()
    elif args.command == 'prepare':
        if args.out.exists():
            parser.error('--out must be a new directory; releases are never rebuilt in place')
        registry.constructor(args.game)(args.out, **options(args.option))
        result = policy.bind_release(args.game, args.out)
        result = {k: v for k, v in result.items() if k != 'release_files'}
    elif args.command == 'verify':
        record = policy.verify_release(args.game, args.release)
        result = dict(valid=True, sha256=record['sha256'], workspace_sha256=record['workspace_sha256'])
    elif args.command == 'task':
        policy.verify_release(args.game, args.release)
        chosen = registry.research(args.game, args.smoke, args.budget_mode, args.research_seconds, args.research_usd)
        _, final_train = registry.budgets(args.game, args.smoke, final_train_seconds=args.final_train_seconds,
                                          budget_mode=chosen['mode'])
        result = task.build(args.game, args.variant, args.release, args.out, research_seconds=chosen['seconds'],
                            research_usd=chosen['usd'], budget_mode=chosen['mode'],
                            final_train_seconds=final_train,
                            wall_seconds=session.wall_limit(args.game, args.smoke, chosen['seconds'],
                                                            args.wall_seconds, chosen['mode']),
                            hardware=registry.spec(args.game)['hardware'], smoke=args.smoke, agent=args.agent,
                            full_session=args.full_session, continue_from=args.continue_from,
                            **{key: session.plan(args.game, args.variant, args.release, args.out, model=None,
                                                 smoke=args.smoke, **rules(args))[key]
                               for key in ('restarts', 'use_budget')})
    elif args.command == 'run':
        common = dict(model=args.model, agent=args.agent, smoke=args.smoke, research_seconds=args.research_seconds,
                      research_usd=args.research_usd, budget_mode=args.budget_mode,
                      final_train_seconds=args.final_train_seconds,
                      wall_seconds=args.wall_seconds, continue_from=args.continue_from, **rules(args))
        policy.verify_release(args.game, args.release)
        if args.scored or args.qualification:
            policy.require_scored(args.game, args.release, args.qualification)
        if not args.execute:
            result = dict(session.plan(args.game, args.variant, args.release, args.out, full_session=args.full_session,
                                       **common), executed=False)
        else:
            result = session.run(args.game, args.variant, args.release, args.out, api_base=args.api_base,
                                 key_file=args.key_file, key_env=args.key_env, reasoning_effort=args.reasoning_effort,
                                 model_info=args.model_info, agent_kwargs=args.agent_kwarg,
                                 backend_options=options(args.backend_option), final=not args.skip_final,
                                 full_session=args.full_session, **common)
    elif args.command == 'pause':
        result = session.request_pause(args.session, note=args.note)
    elif args.command == 'resume':
        if not args.execute:
            record = args.session.resolve() / session.PAUSE
            result = dict(json.loads(record.read_text()) if record.is_file() else {'state': 'not paused'},
                          executed=False)
        else:
            result = session.resume(args.session, api_base=args.api_base, key_file=args.key_file,
                                    key_env=args.key_env, model_info=args.model_info, agent_kwargs=args.agent_kwarg,
                                    backend_options=options(args.backend_option), final=not args.skip_final,
                                    wall_seconds=args.wall_seconds)
    elif args.command == 'final':
        policy.require_modal()
        result = session.final_training(args.session, backend_options=options(args.backend_option))
    elif args.command == 'retrain':
        policy.verify_release(args.game, args.release)
        common = dict(seed=args.seed, session=args.session, starter=args.starter, smoke=args.smoke,
                      final_train_seconds=args.final_train_seconds, label=args.label,
                      backend_options=options(args.backend_option))
        if not args.execute:
            research, final_train = registry.budgets(args.game, args.smoke, None, args.final_train_seconds)
            result = dict(game=args.game, release=str(args.release.resolve()), out=str(args.out.resolve()),
                          seed=args.seed, source='starter' if args.starter else str(args.session.resolve()),
                          final_train_seconds=final_train, retrain_seed=registry.spec(args.game)['retrain_seed'],
                          executed=False)
        else:
            policy.require_modal()
            result = session.retrain(args.game, args.release, args.out, **common)
    elif args.command == 'grade':
        result = session.grade(args.session, backend_options=options(args.backend_option))
    else:
        out = args.session.resolve()
        result = {name: json.loads((out / name).read_text()) if (out / name).is_file() else None
                  for name in ('session-result.json', 'final-train.json', 'grade.json', session.PAUSE)}
        result['completion'] = policy.completion(out)
    print(json.dumps(result, indent=2, default=str))
    if isinstance(result, dict) and result.get('status') == 'failed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
