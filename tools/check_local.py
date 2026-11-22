#!/usr/bin/env python3
"""Run the focused game and research-harness checks locally, serially; no cloud calls."""
import argparse
from datetime import datetime, timezone
import uuid
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
COMMANDS = {
    'pokemon': ['.venv-pokemon/bin/python', '-m', 'unittest', 'integrations.pokemon.tests.test_rl', '-v'],
    'clash': ['.venv-clash/bin/python', 'simulators/clash/tests/test_contract.py'],
    'doudizhu': ['.venv-doudizhu/bin/python', '-m', 'pytest', 'simulators/doudizhu/tests', '-q'],
    'generals': ['.venv-generals/bin/python', '-m', 'pytest', 'simulators/generals/tests', '-q'],
    'tavern': ['.venv-tavern/bin/python', '-m', 'pytest', 'simulators/tavern/tests', '-q'],
    'hex': ['.venv-hex/bin/python', '-m', 'pytest', '-q', 'simulators/hex/tests', 'simulators/hex/autoresearch/tests'],
    'polytopia': ['.venv-polytopia/bin/python', '-m', 'pytest', '-q',
                  'simulators/polytopia/autoresearch/tests'],
    # The research harness: shared core, then each game's backend under its research interpreter.
    'harness': ['python3', '-m', 'unittest', 'harness.tests.test_core', 'harness.tests.test_pause'],
    'harness-generals': ['.venv-generals-ar/bin/python', '-m', 'unittest', 'harness.tests.test_generals'],
    'harness-doudizhu': ['.venv-doudizhu-ar/bin/python', '-m', 'unittest', 'harness.tests.test_doudizhu'],
    'harness-pokemon': ['.venv-pokemon/bin/python', '-m', 'unittest', 'harness.tests.test_pokemon'],
    'harness-clash': ['.venv-clash-ar/bin/python', '-m', 'unittest', 'harness.tests.test_clash'],
    'harness-polytopia': ['.venv-polytopia-ar/bin/python', '-m', 'unittest', 'harness.tests.test_polytopia'],
    'harness-tavern': ['.venv-tavern-ar/bin/python', '-m', 'unittest', 'harness.tests.test_tavern'],
    'harness-hex': ['.venv-hex/bin/python', '-m', 'unittest', 'harness.tests.test_hex'],
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--game', choices=COMMANDS, action='append')
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output = ROOT / 'evidence/local-validation' / (stamp + '-' + uuid.uuid4().hex[:8])
    output.mkdir(parents=True, exist_ok=False)
    print(f'Evidence: {output}', flush=True)
    rows = []
    for game in args.game or COMMANDS:
        env = os.environ.copy()
        command = COMMANDS[game]
        started = time.monotonic()
        print(f'Checking {game}', flush=True)
        with (output / (game + '.log')).open('w') as log:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        rows.append(dict(game=game, command=command, exit_code=result.returncode,
                         seconds=round(time.monotonic() - started, 3)))
        print(f'{game}: {"PASS" if result.returncode == 0 else "FAIL"}', flush=True)
    (output / 'results.json').write_text(json.dumps(rows, indent=2) + '\n')
    raise SystemExit(int(any(r['exit_code'] for r in rows)))


if __name__ == '__main__':
    main()
