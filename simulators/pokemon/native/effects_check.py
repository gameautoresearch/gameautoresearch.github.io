"""Seeded differential checks of scalar native move effect families."""
import argparse
import json
from pathlib import Path

import numpy as np

from .bridge import Native, build
from .rules import DEVELOPMENT_MOVES, STATUSES, STATS, TYPES, compile_moves
from .validate import fill, oracle, seed64, oracle_identity


def load_fixture(native, data):
    fill(native, data['initial'])
    state = native.records[0]
    for side in range(2):
        source = data['initial']['mons'][side][0]
        mon = state.mons[side][0]
        mon.move_ids[0] = side
        mon.type[0] = TYPES.index(source['types'][0])
        mon.type[1] = TYPES.index(source['types'][1]) if len(source['types']) > 1 else -1
        mon.tera_type = TYPES.index(source['tera'])
        mon.status = STATUSES.index(source['status'])
        mon.status_turns = source['statusState']['time']
        mon.toxic_turns = source['statusState']['stage']
        for j, name in enumerate(STATS):
            mon.boosts[j] = source['boosts'][name]


def compare(state, reference):
    actual = dict(rng=state.rng, turn=state.turn, ended=bool(state.terminated), fault=state.fault)
    expected = dict(rng=seed64(reference['seed']), turn=reference['turn'], ended=reference['ended'], fault=0)
    actual['mons'] = []
    expected['mons'] = []
    for side in range(2):
        mon = state.mons[side][0]
        source = reference['mons'][side][0]
        actual['mons'].append(dict(hp=mon.hp, pp=mon.pp[0], status=mon.status,
                                   boosts=list(mon.boosts)))
        expected['mons'].append(dict(hp=source['hp'], pp=source['moves'][0]['pp'],
                                     status=STATUSES.index(source['status']),
                                     boosts=[source['boosts'][name] for name in STATS]))
    return None if actual == expected else dict(actual=actual, expected=expected)


def run(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    library = build(output / 'build')
    (output / 'oracle-identity.json').write_text(json.dumps(oracle_identity(), indent=2) + '\n')
    scenarios = [dict(moves=[move, 'tackle']) for move in sorted(DEVELOPMENT_MOVES)]
    scenarios += [
        dict(moves=['recover', 'tackle'], hp_fraction=[.31, 1]),
        dict(moves=['drainpunch', 'tackle'], hp_fraction=[.31, 1]),
        dict(moves=['drainingkiss', 'tackle'], hp_fraction=[.31, 1]),
        dict(moves=['doubleedge', 'tackle'], hp_fraction=[.31, 1]),
        dict(moves=['tackle', 'tackle'], boosts=[dict(accuracy=-2), dict(evasion=2)]),
        dict(moves=['tackle', 'tackle'], boosts=[dict(accuracy=6), dict(evasion=-6)]),
        dict(moves=['ember', 'tackle'], statuses=['', 'brn']),
        dict(moves=['ember', 'tackle'], statuses=['', 'frz']),
        dict(moves=['ember', 'tackle'], species=['Mew', 'Charmander']),
        dict(moves=['icebeam', 'tackle'], species=['Mew', 'Glalie']),
        dict(moves=['thundershock', 'tackle'], species=['Mew', 'Pikachu']),
        dict(moves=['willowisp', 'tackle'], species=['Mew', 'Charmander']),
        dict(moves=['sing', 'tackle'], species=['Mew', 'Blissey']),
        dict(moves=['poweruppunch', 'tackle'], hp_fraction=[1, .03]),
        dict(moves=['ember', 'tackle'], hp_fraction=[1, .03]),
        dict(moves=['recover', 'recover'], statuses=['tox', 'brn']),
        dict(moves=['recover', 'recover'], statuses=['brn', 'tox']),
        dict(moves=['recover', 'recover'], statuses=['tox', 'tox'], hp_fraction=[.03, .03]),
        dict(moves=['recover', 'recover'], statuses=['psn', 'psn'], hp_fraction=[.03, .03]),
    ]
    rows = []
    transitions = 0
    for case, options in enumerate(scenarios):
        for master in range(1, 9):
            spec = dict(op='duel', seed=f'{master},{master+1},{master+2},{master+3}', turns=32, **options)
            data = oracle(spec)
            with (output / 'oracle-fixtures.jsonl').open('a') as fixtures:
                fixtures.write(json.dumps(dict(spec=spec, oracle=data), separators=(',', ':')) + '\n')
            moves = compile_moves(data['moves'])
            with Native(library) as native:
                load_fixture(native, data)
                mismatch = None
                for frame, record in enumerate(data['frames']):
                    native.step([[0, 0]], moves, np.asarray(data['chart'], dtype=np.int8))
                    transitions += 1
                    mismatch = compare(native.records[0], record['state'])
                    if mismatch:
                        mismatch.update(frame=frame, oracle_calls=record['calls'])
                        break
            rows.append(dict(case=case, options=options, seed=master, exact=mismatch is None, mismatch=mismatch))
    result = dict(full_gen9_admission=False, cases=len(rows), exact=sum(r['exact'] for r in rows),
                  transitions=transitions, rows=rows)
    (output / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: value for key, value in result.items() if key != 'rows'}, indent=2))
    if result['exact'] != result['cases']:
        raise SystemExit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    run(parser.parse_args().output)
