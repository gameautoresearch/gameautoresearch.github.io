#!/usr/bin/env python3
"""Validate the six-game layout and historical aliases without game dependencies."""
import argparse
import json
from pathlib import Path
import re
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]


def check(root):
    root = Path(root).resolve()
    catalog = json.loads((root / 'simulators/catalog.json').read_text())
    errors = []
    if catalog.get('schema_version') != 1:
        errors.append('Unsupported simulator catalog version')
    games = catalog['games']
    actual = {p.name for p in (root / 'simulators').iterdir()
              if p.is_dir() and not p.name.startswith(('.', '__'))}
    if actual != set(games):
        errors.append(f'Simulator directories differ from catalog: {actual ^ set(games)}')
    upstream = set()
    for game, entry in games.items():
        if entry['directory'] != f'simulators/{game}':
            errors.append(f'{game}: nonuniform simulator directory')
        for key in ('directory', 'readme', 'tests', 'readout', 'working_directory', 'autoresearch',
                    'benchmark_contract', 'agent_brief', 'benchmark_readout', 'benchmark_smoke_receipt',
                    'game_rules', 'harness_backend', 'current_harness'):
            if entry[key] is not None and not (root / entry[key]).exists():
                errors.append(f'{game}: missing {key}: {entry[key]}')
        for dependency in entry['upstream']:
            upstream.add(dependency)
            if not (root / dependency).is_dir():
                errors.append(f'{game}: missing upstream source: {dependency}')
    metadata = json.loads((root / 'vendor/upstream-git/manifest.json').read_text())
    if upstream != {f'vendor/engines/{name}' for name in metadata}:
        errors.append('Catalog upstream dependencies differ from frozen metadata manifest')
    aliases = {f'integrations/{g}': f'simulators/{g}' for g in games if g != 'polytopia'}
    aliases.update({
        'integrations/generals_autoresearch': 'simulators/generals/autoresearch',
        'candidates': 'vendor/engines',
        'environments/polytopia': 'simulators/polytopia',
    })
    for old, target in aliases.items():
        link = root / old
        if not link.is_symlink() or link.readlink().is_absolute():
            errors.append(f'{old}: expected a relative compatibility link')
        elif link.resolve() != (root / target).resolve() or not link.is_dir():
            errors.append(f'{old}: broken or incorrect compatibility target')
    # Verify the maintained navigation, without rewriting historical documents.
    docs = ('README.md', 'simulators/README.md', 'harness/README.md', 'docs/repository-layout.md',
            'docs/benchmark-contracts.md', 'docs/research-task-layout.md', 'vendor/README.md')
    for name in docs:
        doc = root / name
        for link in re.findall(r'\]\(([^)]+)\)', doc.read_text()):
            target = unquote(link.split('#', 1)[0])
            if target and '://' not in target and not (doc.parent / target).exists():
                errors.append(f'{name}: broken link: {link}')
    return errors, dict(games=len(games), upstream_trees=len(upstream), aliases=len(aliases))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT, help='also validate a fresh checkout/export')
    args = parser.parse_args()
    errors, counts = check(args.root)
    for error in errors:
        print(f'FAIL: {error}')
    if not errors:
        print(f'PASS: {counts["games"]} games, {counts["upstream_trees"]} upstream trees, '
              f'{counts["aliases"]} portable compatibility links; navigation intact')
    raise SystemExit(bool(errors))


if __name__ == '__main__':
    main()
