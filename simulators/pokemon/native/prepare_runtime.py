"""Build a local native runtime from the retained frozen public data extracts.

Only public Dex/generator tables enter the runtime bundle. Historical seed teams,
fixture states, outcomes and receipts remain outside it.
"""
import argparse
import hashlib
import json
from pathlib import Path

from .bridge import build


def format_id(team_size=6):
    """The Showdown format a native team size plays: gen9randombattle, or it with a smaller Max Team Size."""
    return 'gen9randombattle' if team_size==6 else f'gen9randombattle@@@Max Team Size = {team_size}'

EXPECTED={
    'generators':'b01188261ca32e9e665b2e4e20f6683d6fd89cc3dcb68a87099ea335f18bc66d',
    'constructors':'75b27fd6b977221b059d700f0ed47aa583d2b25158b3fa0f70df33d9787154e7',
}
# The public tables projected from those extracts, as sealed in releases; accepted in their place
# (projection is idempotent, so the runtime files come out byte-identical).
PROJECTED={
    'generators':'bafdacf37091f0ce84c2b4d91986a732fa6424e9f37843d9c72d4e30a9ccc623',
    'constructors':'72cda5afc6dca22df5af4f8cc30ca7cb5673ca4f5cc368b03cdb05b923d38d8b',
}


def run(output,generators,constructors,chart,admission=None,team_size=6):
    if type(team_size)is not int or not 1<=team_size<=6:raise ValueError('Team size must be 1..6')
    if admission is not None and json.loads(Path(admission).read_bytes()).get('team_size',6)!=team_size:raise ValueError('Admission certificate qualifies another team size')
    output=Path(output);output.mkdir(parents=True,exist_ok=False);inputs={};data={}
    for key,path in dict(generators=generators,constructors=constructors,chart=chart).items():
        raw=Path(path).read_bytes();inputs[key]=hashlib.sha256(raw).hexdigest();data[key]=json.loads(raw)
        if key in EXPECTED and inputs[key] not in (EXPECTED[key],PROJECTED[key]):raise ValueError(f'Frozen public {key} source extract changed')
    public=dict(generators={key:data['generators'][key]for key in('species','moves','status_moves','items')},constructors=dict(metadata=data['constructors']['metadata']),chart=dict(chart=data['chart']['chart']))
    paths={}
    for key,value in public.items():
        path=output/f'{key}.json';path.write_text(json.dumps(value,separators=(',',':'))+'\n');paths[key]=path
    paths['library']=build(output/'build',admission=admission)
    record=dict(schema_version=1,format=format_id(team_size),team_size=team_size,showdown_pin='567c1cedddeaa2f35576005d3bfd526cf4f8a13e',
        full_gen9_admission=admission is not None,source_extract_sha256=inputs,sha256={k:hashlib.sha256(p.read_bytes()).hexdigest()for k,p in paths.items()},**{k:str(p.relative_to(output))for k,p in paths.items()})
    (output/'runtime.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--generators',required=True);p.add_argument('--constructors',required=True);p.add_argument('--chart',required=True);p.add_argument('--admission');p.add_argument('--team-size',type=int,default=6);a=p.parse_args();run(a.output,a.generators,a.constructors,a.chart,a.admission,a.team_size)
