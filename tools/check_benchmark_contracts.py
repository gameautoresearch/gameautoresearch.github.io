#!/usr/bin/env python3
"""Validate public benchmark declarations offline; never launch or import a game."""

import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# The declared research allocation per game (the harness SPEC's `research_seconds`); 7200 unless listed.
# The three small-form games research for four hours.
RESEARCH_SECONDS = {'generals': 16200, 'pokemon': 14400, 'polytopia': 14400}


def check(root, selected=None):
    root = Path(root).resolve()
    catalog = json.loads((root / 'simulators/catalog.json').read_text())['games']
    rows, errors = [], []
    for game in selected or catalog:
        base = root / catalog[game]['directory'] / 'autoresearch'
        prefix = f'{game}: '
        try:
            spec = json.loads((base / 'benchmark.json').read_text())
        except (OSError, ValueError) as exc:
            errors.append(prefix + str(exc))
            continue
        before = len(errors)

        def require(condition, message):
            if not condition:
                errors.append(prefix + message)

        def public_file(value, label):
            if not isinstance(value, str) or not value:
                require(False, f'{label} must name a relative public file')
                return
            path = Path(value)
            resolved = (base / path).resolve()
            if path.is_absolute() or not resolved.is_relative_to(base.resolve()):
                require(False, f'{label} escapes the game autoresearch directory')
                return
            if any(part in ('private', 'hidden', 'evidence', '.local') for part in path.parts):
                require(False, f'{label} refers to private/host-only material')
            require(resolved.is_file(), f'missing {label}: {value}')
            if resolved.is_file():
                require(resolved.stat().st_size > 0, f'empty {label}: {value}')

        require(spec.get('schema_version') == 1, 'unsupported schema_version')
        require(spec.get('game') == game, 'game does not match catalog')
        require(isinstance(spec.get('benchmark_id'), str) and bool(spec['benchmark_id']),
                'benchmark_id is required')
        require(isinstance(spec.get('task'), dict) and bool(spec['task']),
                'task settings must be explicit')
        resources = spec.get('resources', {})
        require(isinstance(resources.get('gpu'), str) and bool(resources['gpu']),
                'resources.gpu is required')
        for field in ('cpu_cores', 'memory_gib'):
            value = resources.get(field)
            require(isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0,
                    f'resources.{field} must be positive')
        budget = spec.get('budget', {})
        require(budget.get('accounting') == 'allocated_worker_wall_time',
                "budget.accounting must be 'allocated_worker_wall_time'")
        # Every game offers both research budget modes (harness/budget.py) and defaults to seconds; the amounts
        # must equal the harness SPEC's (checked by harness.tests.test_core, which may import the game).
        require(budget.get('default_mode') == 'seconds', "budget.default_mode must be 'seconds'")
        modes = (('seconds_mode', ('research_seconds', 'smoke_research_seconds')),
                 ('usd_mode', ('research_usd', 'smoke_research_usd', 'wall_seconds', 'smoke_wall_seconds')))
        for block, fields in modes:
            declared = budget.get(block)
            require(isinstance(declared, dict), f'budget.{block} is required')
            for field in fields if isinstance(declared, dict) else ():
                value = declared.get(field)
                require(isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0,
                        f'budget.{block}.{field} must be positive')
        if isinstance(budget.get('seconds_mode'), dict):
            expected = RESEARCH_SECONDS.get(game, 7200)
            require(budget['seconds_mode'].get('research_seconds') == expected,
                    f'budget.seconds_mode.research_seconds must be {expected!r}')
        starter = spec.get('starter', {})
        require(starter.get('initialization') == 'random', 'starter must initialize randomly')
        public_file(starter.get('recipe'), 'starter recipe')
        editable = starter.get('editable_files')
        require(isinstance(editable, list) and bool(editable), 'starter editable files are required')
        for value in editable if isinstance(editable, list) else []:
            public_file(value, 'editable starter file')
        # Instructions are the harness's common template; a game supplies its brief and rulebook.
        public_file(spec.get('agent', {}).get('brief'), 'agent brief')
        require('instruction' not in spec.get('agent', {}), 'agent.instruction is obsolete: the harness owns instructions')
        public_file('task/game_rules/README.md', 'rulebook entry')
        public_file('CONTRACT.md', 'contract')
        public_file('README.md', 'setup guide')
        public_file('recipes/smoke.json', 'smoke recipe')
        evaluation = spec.get('evaluation', {})
        require(isinstance(evaluation.get('primary_metric'), str) and bool(evaluation['primary_metric']),
                'evaluation.primary_metric is required')
        require(evaluation.get('score_scale') == [0, 100], 'primary score scale must be [0,100]')
        for name in ('prepare', 'smoke', 'grade'):
            command = spec.get('commands', {}).get(name)
            require(isinstance(command, str) and bool(command.strip()), f'commands.{name} is required')
        rows.append({'game': game, 'benchmark_id': spec.get('benchmark_id'),
                     'resources': resources, 'primary_metric': evaluation.get('primary_metric'),
                     'status': 'pass' if len(errors) == before else 'fail'})
    return rows, errors


def check_smokes(root, selected=None):
    """Check completion declarations and bounds; provider receipts still need review."""
    root = Path(root).resolve()
    catalog = json.loads((root / 'simulators/catalog.json').read_text())['games']
    rows, errors = [], []
    for game in selected or catalog:
        prefix = f'{game} smoke: '
        report = root / f'reports/{game}/benchmark-contract-smoke-2026-11-08.json'
        try:
            record = json.loads(report.read_text())
        except (OSError, ValueError) as exc:
            errors.append(prefix + str(exc))
            continue
        before = len(errors)

        def require(condition, message):
            if not condition:
                errors.append(prefix + message)

        def number(value):
            return (isinstance(value, (int, float)) and not isinstance(value, bool)
                    and math.isfinite(value) and value >= 0)

        require(record.get('game') == game, 'game does not match catalog')
        require(record.get('status') == 'passed', 'workflow has not passed')
        spec_path = root / catalog[game]['directory'] / 'autoresearch/benchmark.json'
        if spec_path.is_file():
            spec = json.loads(spec_path.read_text())
            require(record.get('benchmark_id') == spec.get('benchmark_id'),
                    'receipt and current contract identity differ')
        smoke = record.get('smoke', {})
        for field in ('agent_completed', 'learning_verified', 'practice_completed',
                      'submission_valid', 'private_grade_completed', 'owned_workers_stopped'):
            require(smoke.get(field) is True, f'{field} is not verified')
        score = smoke.get('raw_score_0_100')
        require(number(score) and score <= 100, 'raw score must be finite and within 0–100')
        require(smoke.get('normalized_reference_score', 'missing') is None,
                'uncalibrated reference normalization must remain null')
        gpu = smoke.get('gpu_allocated_seconds')
        holds = smoke.get('gpu_reserved_or_unresolved_seconds')
        require(number(gpu), 'allocated GPU time is unknown or invalid')
        require(number(holds), 'unresolved GPU time is unknown or invalid')
        require(smoke.get('gpu_limit_seconds') == 600, 'GPU campaign limit must be 600 seconds')
        if number(gpu) and number(holds):
            require(gpu + holds <= 600, 'actual plus held GPU time exceeds campaign allowance')
            require(holds == 0, 'unresolved GPU allocations remain')
        model = smoke.get('model_charged_or_reserved_usd')
        require(number(model) and model <= 2, 'model liability is unknown or over $2')
        require(smoke.get('model_limit_usd') == 2, 'model campaign limit must be $2')
        require(bool(record.get('local_checks')), 'local checks are missing')
        paths = record.get('evidence_paths')
        require(isinstance(paths, list) and bool(paths), 'evidence paths are missing')
        for path in paths if isinstance(paths, list) else []:
            require(isinstance(path, str) and (root / path).exists(), f'missing evidence: {path}')
        rows.append({'game': game, 'raw_score_0_100': score, 'gpu_allocated_seconds': gpu,
                     'model_charged_or_reserved_usd': model,
                     'status': 'pass' if len(errors) == before else 'fail'})
    return rows, errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--game', action='append', choices=('clash', 'doudizhu', 'generals', 'hex',
                                                         'pokemon', 'polytopia', 'tavern'))
    parser.add_argument('--json', action='store_true')
    parser.add_argument('--smoke-receipts', action='store_true',
                        help='also require complete bounded initial workflow smoke reports')
    args = parser.parse_args()
    rows, errors = check(args.root, args.game)
    smoke_rows = []
    if args.smoke_receipts:
        smoke_rows, smoke_errors = check_smokes(args.root, args.game)
        errors.extend(smoke_errors)
    if args.json:
        print(json.dumps({'contracts': rows, 'smokes': smoke_rows,
                          'errors': errors, 'valid': not errors}, indent=2))
    else:
        for row in rows:
            print(f'{row["status"].upper()}: {row["game"]} — {row["benchmark_id"]}')
        for row in smoke_rows:
            print(f'{row["status"].upper()}: {row["game"]} workflow smoke')
        for error in errors:
            print('FAIL: ' + error)
        if not errors:
            print(f'Validated {len(rows)} public contracts; no game code or paid commands executed.')
    raise SystemExit(bool(errors))


if __name__ == '__main__':
    main()
