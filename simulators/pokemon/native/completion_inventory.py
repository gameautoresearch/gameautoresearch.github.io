"""Unfiltered public development inventory; never a production admission filter.

Enumerate both teams from every task seed using the independent native generator,
retaining complete teams and every missing effect, rather than only the first
reset fault. The existing whole-battle oracle campaign is a separate obligation.
"""
import argparse
from collections import Counter
import ctypes as C
import hashlib
import json
import re
from pathlib import Path

from .bridge import Native, build
from .random_resident import task_seeds
from .reset import ResetTables, ident
from .rules import ABILITY_IDS, ITEM_IDS
from .teamgen import GeneratorTables, GenTeam


def run(output, generators, constructors, start=1109000, count=512):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    paths = [Path(generators), Path(constructors)]
    data, constructor = [json.loads(p.read_bytes()) for p in paths]
    generator = GeneratorTables(data['species'], data['moves'], data['status_moves'], data['items'])
    reset = ResetTables(generator, constructor['metadata'], data['moves'])
    library = build(output / 'build')
    frequency, affected = Counter(), Counter()
    rows = []
    with Native(library) as native:
        generator.bind(native)
        for master in range(start, start + count):
            seed = task_seeds(master)
            teams, gaps = [], []
            for side in range(2):
                rng, team = C.c_uint64(seed.teams[side]), GenTeam()
                if not native.lib.pkn_gen_random_team(C.byref(generator.rules), C.byref(rng), C.byref(team)):
                    raise RuntimeError(f'Native team generator fault {master}/{side}: {team.fault}')
                members = [generator.decode_set(team.pokemon[i]) for i in range(team.count)]
                teams.append(members)
                for slot, member in enumerate(members):
                    missing = []
                    for kind, name in [('ability', ident(member['ability'])), ('item', ident(member['item']))]:
                        if name not in (ABILITY_IDS if kind == 'ability' else ITEM_IDS):
                            missing.append((kind, name))
                    missing += [('move', name) for name in member['moves'] if name in reset.move_rejections]
                    for kind, name in missing:
                        key = f'{kind}:{name}'
                        frequency[key] += 1
                        gaps.append(dict(side=side, slot=slot, effect=key,
                                         reason=reset.move_rejections[name] if kind == 'move' else 'Unknown native effect'))
            affected.update(set(g['effect'] for g in gaps))
            rows.append(dict(master=master, teams=teams, gaps=gaps, statically_lowerable=not gaps))
    reachable_abilities = {ident(a) for s in data['species'] for spec in s['sets'] for a in spec['abilities']}
    # Over-approximate every item emitted by the frozen singles generator,
    # including rare random arrays absent from this finite seed sample.
    source=(Path(__file__).parent/'teamgen.hpp').read_text()
    item_body=source[source.index('PK_HD inline int gen_priority_item'):source.index('PK_HD inline bool gen_is_item')]
    reachable_items=set(re.findall(r'"([a-z0-9]+)"',item_body))&{row['id']for row in data['items']}
    unreachable=[]
    if 'gluttony'not in reachable_abilities:
        excluded={'aguavberry','figyberry','iapapaberry','magoberry','wikiberry'}
        reachable_items-=excluded;unreachable.append(dict(branch='Gluttony item array',reason='No frozen random set can select Gluttony',items=sorted(excluded)))
    for record in data['species']:
        required=record.get('required_items')or []
        if record['base']=='Arceus':required=required[:1]
        reachable_items.update(ident(name)for name in required)
    result = dict(schema_version=1, full_gen9_admission=False,
                  scope='Unfiltered public development task seeds; static acceptance is not semantic qualification',
                  start=start, seed_pairs=count, swapped_attempts=count * 2,
                  statically_lowerable_pairs=sum(r['statically_lowerable'] for r in rows),
                  missing_moves=reset.move_rejections,
                  reachable_missing_abilities=sorted(reachable_abilities - set(ABILITY_IDS)),
                  reachable_item_candidates=sorted(reachable_items),reachable_missing_items=sorted(reachable_items-set(ITEM_IDS)),unreachable_item_branches=unreachable,
                  gaps=[dict(effect=key, occurrences=value, affected_pairs=affected[key], severity='blocks_full_task')
                        for key, value in frequency.most_common()],
                  input_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                  source_signature=native.source_signature, rows=rows)
    (output / 'inventory.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'rows'}, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for key in ('output', 'generators', 'constructors'):
        parser.add_argument('--' + key, required=True)
    parser.add_argument('--start', type=int, default=1109000)
    parser.add_argument('--count', type=int, default=512)
    args = parser.parse_args()
    run(args.output, args.generators, args.constructors, args.start, args.count)
