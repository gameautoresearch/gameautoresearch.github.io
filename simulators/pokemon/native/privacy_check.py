"""Counterfactual public-information invariance checks for native observations."""
import argparse
import json
from pathlib import Path

import numpy as np

from .bridge import Native,build
from .teams_check import load_team_fixture
from .visible import PublicTables


def run(output,fixture_path):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    library=build(output/'build');cases=0;values=0
    for line in Path(fixture_path).read_text().splitlines():
        data=json.loads(line);ids={move['id']:i for i,move in enumerate(data['moves'])}
        tables=PublicTables(data['metadata'],data['chart'],ids)
        with Native(library) as native:
            load_team_fixture(native,data,ids)
            before=tables.observe(native)[0].copy();original=native.snapshot()
            for player in range(2):
                native.restore(original);s=native.records[0];foe=1-player
                for slot in range(6):
                    p=s.mons[foe][slot]
                    if p.maxhp<=0:continue
                    # Change private information without publishing an event.
                    p.hp=max(1,p.maxhp//7);p.ability=(p.ability+1)%len(data['metadata']['ability_tokens'])
                    p.item=(p.item+1)%len(data['metadata']['item_tokens']);p.tera_type=(p.tera_type+1)%18
                    for j in range(5):p.stats[j]=p.stats[j]*2+1
                    for j in range(7):p.boosts[j]=6 if j%2 else -6
                    for j in range(4):p.pp[j]=0;p.move_ids[j]=0
                    if slot not in list(s.public_order[foe])[:s.public_count[foe]]:
                        p.species=1+p.species%(len(data['metadata']['species'])-1)
                actual=tables.observe(native)[0,player]
                np.testing.assert_array_equal(actual,before[player],err_msg=f'Hidden opponent data leaked into observer {player}, fixture {cases}')
                values+=len(actual)
        cases+=1
    result=dict(full_gen9_admission=False,fixtures=cases,observer_counterfactuals=cases*2,exact_feature_values=values,
                invariant='Opponent private stats, HP, boosts, abilities, items, moves/PP, unrevealed tera and bench species do not alter a public observation without a visible event')
    (output/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);parser.add_argument('--fixtures',required=True)
    args=parser.parse_args();run(args.output,args.fixtures)
