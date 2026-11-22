"""Static lowering inventory, weighted by retained diagnostic random teams.

Acceptance by the data compiler is not battle parity or full-task admission.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from .bridge import HERE
from .rules import ABILITY_IDS,ITEM_IDS,compile_moves


def run(output,fixture_path):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    raw=Path(fixture_path).read_bytes();data=json.loads(raw);supported=set();rejections={}
    for source in data['moves']:
        record={key:value for key,value in source.items() if key not in ('power_callback','damage_callback','has_secondary')}
        try:compile_moves([record])
        except (ValueError,KeyError) as error:rejections[source['id']]=str(error)
        else:supported.add(source['id'])
    abilities=Counter();items=Counter();moves=Counter();pokemon=lowerable=teams=lowerable_teams=0
    def ident(name):return ''.join(char for char in name.lower() if char.isascii() and char.isalnum())
    for row in data['teams']:
        if row['fault']:continue
        complete=True;teams+=1
        for mon in row['result']:
            pokemon+=1;ability=ident(mon['ability']);item=ident(mon['item']);ok=True
            if ability not in ABILITY_IDS:abilities[ability]+=1;ok=False
            if item not in ITEM_IDS:items[item]+=1;ok=False
            for move in mon['moves']:
                if move not in supported:moves[move]+=1;ok=False
            lowerable+=ok;complete&=ok
        lowerable_teams+=complete
    result=dict(full_gen9_admission=False,scope='Static compiler inventory on diagnostic generated teams; not semantic coverage or completion percentage',
        fixture_sha256=hashlib.sha256(raw).hexdigest(),source_sha256={name:hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in ('sim.hpp','rules.py')},
        catalog_moves=len(data['moves']),statically_lowered_moves=len(supported),supported_move_ids=sorted(supported),rejections=rejections,
        diagnostic_teams=teams,diagnostic_pokemon=pokemon,statically_lowerable_pokemon=lowerable,statically_lowerable_teams=lowerable_teams,
        unsupported_abilities=abilities.most_common(),unsupported_items=items.most_common(),unsupported_moves=moves.most_common())
    (output/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('source_sha256','rejections','supported_move_ids')},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);parser.add_argument('--fixtures',required=True)
    args=parser.parse_args();run(args.output,args.fixtures)
