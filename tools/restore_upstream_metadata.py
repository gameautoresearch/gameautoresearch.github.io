#!/usr/bin/env python3
"""Restore a shallow Git pin for each vendored engine without changing source."""
import json
from pathlib import Path
import re
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = {
    'AutoTavern': 'MCG-NJU/AutoTavern',
    'DouZero': 'kwai/DouZero',
    'benzene': 'cgao3/benzene-vanilla-cmake',
    'crforge': 'voonhous/crforge',
    'generals-bots': 'strakam/generals-bots',
    'poke-env': 'hsahovic/poke-env',
    'pokemon-showdown': 'smogon/pokemon-showdown',
}

# Restore omitted dependency files only from the locked upstream tree.
LOCAL_UPSTREAM_FILES = {
    'benzene': ('regression/gtpregress.jar',),
    'crforge': ('gradle/wrapper/gradle-wrapper.jar',),
    'generals-bots': (
        'generals/assets/gifs/wide_gameplay.gif', 'generals/assets/gifs/wider_gameplay.gif',
        'generals/assets/images/citie.png', 'generals/assets/images/crownie.png',
        'generals/assets/images/game1.webp', 'generals/assets/images/game2.webp',
        'generals/assets/images/game3.webp', 'generals/assets/images/mountainie.png',
    ),
}


def git(repo, *args):
    subprocess.run(['git', '-C', str(repo), *args], check=True)


def main():
    manifest = ROOT / 'vendor/upstream-git/manifest.json'
    records = json.loads(manifest.read_text())
    if set(records) != set(UPSTREAM):
        raise ValueError('Upstream metadata manifest differs from the supported engine list')
    for name, record in records.items():
        repo = ROOT / 'vendor/engines' / name
        pin = record['commit']
        if not re.fullmatch(r'[a-f0-9]{40}', pin):
            raise ValueError(f'Invalid commit pin for {name}')
        if not (repo / '.git').exists():
            url = f'https://github.com/{UPSTREAM[name]}.git'
            with tempfile.TemporaryDirectory(prefix=f'.{name}-git-', dir=repo.parent) as temp:
                staged = Path(temp)
                git(staged, 'init', '-q')
                git(staged, 'remote', 'add', 'origin', url)
                git(staged, 'fetch', '--depth=1', '--no-tags', 'origin', pin)
                # A mixed reset fills the index at the pin and never touches vendored files.
                git(staged, 'reset', '--quiet', '--mixed', pin)
                if not (repo / '.git').exists():
                    (staged / '.git').rename(repo / '.git')
        missing = [path for path in LOCAL_UPSTREAM_FILES.get(name, ()) if not (repo / path).exists()]
        if missing:
            git(repo, 'restore', f'--source={pin}', '--worktree', '--', *missing)
        print(f'{name}: restored {pin}')


if __name__ == '__main__':
    main()
