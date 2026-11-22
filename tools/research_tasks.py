#!/usr/bin/env python3
"""Check and export each game's researcher rulebook (autoresearch/task/game_rules).

The research workspace itself is built by harness/workspace.py. This tool only
validates the Markdown-only rulebook boundary and exports a standalone copy.
No framework imports, private bank reads, network calls or compute allocations.
"""
import argparse
import hashlib
import json
import os
import re
from urllib.parse import unquote
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
RULE_SUFFIXES = {'.md'}
FORBIDDEN_PARTS = {'private', 'hidden', 'evidence', '.local', 'assets', 'checkpoints', '__pycache__'}

# Only owning rules/codec code and public rules documentation. No trajectories,
# checkpoints, game artwork, native binaries, credentials or evaluation schedules.
REFERENCE_PATTERNS = {
    'generals': ['vendor/engines/generals-bots/generals/core/*.py',
                 'simulators/generals/environment.py'],
    'doudizhu': ['vendor/engines/DouZero/douzero/env/*.py',
                'simulators/doudizhu/environment.py'],
    'clash': ['simulators/clash/native/rules/*.hpp',
              'simulators/clash/native/combat/*.hpp',
              'simulators/clash/native/pathfinding/*.hpp',
              'simulators/clash/native/observe/*.hpp',
              'simulators/clash/native/world*.hpp', 'simulators/clash/native/spec.hpp',
              'simulators/clash/native/deck.hpp', 'simulators/clash/native/holder.hpp',
              'simulators/clash/native/generated-v2/config.hpp'],
    'pokemon': ['vendor/engines/pokemon-showdown/sim/*.ts',
                'vendor/engines/pokemon-showdown/sim/SIM-PROTOCOL.md',
                'vendor/engines/pokemon-showdown/sim/SIMULATOR.md',
                'vendor/engines/pokemon-showdown/data/moves.ts',
                'vendor/engines/pokemon-showdown/data/abilities.ts',
                'vendor/engines/pokemon-showdown/data/items.ts',
                'vendor/engines/pokemon-showdown/data/pokedex.ts',
                'vendor/engines/pokemon-showdown/data/typechart.ts',
                'vendor/engines/pokemon-showdown/data/conditions.ts',
                'vendor/engines/pokemon-showdown/data/rulesets.ts',
                'vendor/engines/pokemon-showdown/data/scripts.ts',
                'vendor/engines/pokemon-showdown/data/random-battles/gen9/*.ts',
                'simulators/pokemon/view.cjs'],
    'polytopia': ['simulators/polytopia/engine-rs/src/*.rs',
                  'simulators/polytopia/src/polytopia_bot/rl/actions.py',
                  'simulators/polytopia/src/polytopia_bot/rl/observation.py'],
    'hex': ['simulators/hex/engine.py', 'simulators/hex/reference.py', 'simulators/hex/heuristic.py',
            'simulators/hex/mohex.py'],
    'tavern': ['vendor/engines/AutoTavern/crates/taverngym/src/*.rs',
               'vendor/engines/AutoTavern/crates/taverngym/src/content/*.rs',
               'vendor/engines/AutoTavern/docs/CONTENT.md',
               'vendor/engines/AutoTavern/docs/PATCH_37.0.2.md'],
}


def catalog(root=ROOT):
    return json.loads((root / 'simulators/catalog.json').read_text())['games']


def reference_files(game, root=ROOT):
    files = {p.relative_to(root).as_posix()
             for pattern in REFERENCE_PATTERNS[game]
             for p in root.glob(pattern) if p.is_file()}
    inventory = root / catalog(root)[game]['autoresearch'] / 'wiki_maintenance/metadata/reference.json'
    if inventory.is_file():
        record = json.loads(inventory.read_text())
        for name in record.get('files', []):
            if not isinstance(name, str):
                continue
            path = root / name
            # Optional inventories may add public owning rule/codec source. They
            # cannot turn this into an arbitrary host/private-file exporter.
            if not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
                continue
            if not (name.startswith('vendor/engines/') or name.startswith(f'simulators/{game}/')):
                continue
            if path.suffix not in {'.py', '.rs', '.ts', '.cjs', '.hpp', '.h', '.cpp', '.java', '.md'}:
                continue
            if any(any(word in part.lower() for word in ('private', 'hidden', 'checkpoint', 'credential', 'gateway', 'seeds', 'weights'))
                   or part in {'evidence', '.local', 'assets'} for part in Path(name).parts):
                continue
            files.add(name)
    return sorted(files)


def rulebook_files(game, root=ROOT):
    """Discover each game's own chapter/catalog structure; only README is common."""
    base = root / catalog(root)[game]['autoresearch'] / 'task/game_rules'
    result = []
    for path in sorted(base.rglob('*')):
        if not path.is_file():
            continue
        if path.suffix not in RULE_SUFFIXES:
            raise ValueError(f'Agent rulebook must contain Markdown only: {path}')
        relative = path.relative_to(base)
        if FORBIDDEN_PARTS.intersection(relative.parts):
            continue
        if path.is_symlink() or not path.resolve().is_relative_to(base.resolve()):
            raise ValueError(f'Rulebook entry escapes its folder: {path}')
        result.append(relative.as_posix())
    return sorted(result, key=lambda name: (name != 'README.md', name))


def check_rules(root=ROOT, games=None):
    """Validate the Markdown-only agent boundary and self-contained navigation."""
    errors = []
    for game in games or catalog(root):
        base = root / catalog(root)[game]['autoresearch'] / 'task/game_rules'
        try:
            files = rulebook_files(game, root)
        except ValueError as exc:
            errors.append(f'{game}: {exc}')
            continue
        if 'README.md' not in files:
            errors.append(f'{game}: missing game_rules/README.md')
        for name in files:
            path = base / name
            if not path.stat().st_size:
                errors.append(f'{game}: empty wiki file: {name}')
            if path.suffix != '.md':
                continue
            prose = path.read_text()
            if re.search(r'^\s*(?:```|~~~)|<details\b|<script\b|<a\s+(?:id|href)=', prose, re.M | re.I):
                errors.append(f'{game}: code/implementation block in agent Markdown: {name}')
            for link in re.findall(r'(?<!!)\[[^\]]*\]\(([^)]+)\)', prose):
                target = unquote(link.split('#', 1)[0].split(' "', 1)[0]).strip('<>')
                if not target or '://' in target or target.startswith('mailto:'):
                    continue
                resolved = (path.parent / target).resolve()
                if not resolved.is_relative_to(base.resolve()):
                    errors.append(f'{game}: agent wiki link escapes the Markdown packet: {name}: {target}')
                elif FORBIDDEN_PARTS.intersection(resolved.relative_to(root.resolve()).parts):
                    errors.append(f'{game}: wiki link refers to private/artifact material: {name}: {target}')
                elif not resolved.exists():
                    errors.append(f'{game}: broken wiki link: {name}: {target}')
    return errors


def export_rulebook(game, out, root=ROOT):
    """Copy the complete readable wiki with no trainer/runtime prerequisites."""
    errors = check_rules(root, [game])
    if errors:
        raise ValueError('\n'.join(errors))
    out = Path(out).resolve()
    base = root / catalog(root)[game]['autoresearch'] / 'task/game_rules'
    out.mkdir(parents=True, exist_ok=False)
    for name in rulebook_files(game, root):
        dest = out / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.suffix != '.md':
            shutil.copyfile(base / name, dest)
            continue
        def portable_link(match):
            label, link = match.group(1), match.group(2)
            target = unquote(link.split('#', 1)[0]).strip('<>')
            if not target or '://' in target or target.startswith('mailto:'):
                return match.group(0)
            resolved = (base / name).parent.joinpath(target).resolve()
            if resolved.is_relative_to(base.resolve()):
                return match.group(0)
            # The standalone wiki needs no repository checkout for provenance.
            relative = resolved.relative_to(root.resolve()).as_posix()
            return f'{label} (repository source: `{relative}`)'
        text = re.sub(r'(?<!!)\[([^\]]*)\]\(([^)]+)\)', portable_link, (base / name).read_text())
        dest.write_text(text)
    hashes = {name: hashlib.sha256((out / name).read_bytes()).hexdigest()
              for name in rulebook_files(game, root)}
    # Integrity bookkeeping belongs to the host, never the agent packet.
    return {'game': game, 'out': str(out), 'wiki_files': len(hashes), 'files_sha256': hashes}



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--game', choices=sorted(catalog()))
    parser.add_argument('--out', type=Path, help='export one game\'s rulebook to a new directory')
    parser.add_argument('--check', action='store_true', help='validate rulebooks (the default action)')
    parser.add_argument('--rules-only', action='store_true', help=argparse.SUPPRESS)  # accepted for old command lines
    args = parser.parse_args()
    if args.out:
        if not args.game:
            parser.error('--out requires --game')
        result = export_rulebook(args.game, args.out)
        print(json.dumps({k: v for k, v in result.items() if k != 'files_sha256'}, indent=2))
        return
    errors = check_rules(games=[args.game] if args.game else None)
    print('\n'.join(errors) if errors else 'PASS: rulebooks are Markdown-only with self-contained navigation')
    raise SystemExit(bool(errors))


if __name__ == '__main__':
    main()
