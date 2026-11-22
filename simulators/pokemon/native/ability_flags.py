"""Compile breakable-ability flags from the frozen public Dex metadata."""
import argparse
import json
from pathlib import Path
import subprocess

from .bridge import HERE
from .rules import ABILITY_IDS

PIN='567c1cedddeaa2f35576005d3bfd526cf4f8a13e'


def source():
    upstream=HERE.parents[2]/'candidates/pokemon-showdown'
    actual=subprocess.check_output(['git','-C',str(upstream),'rev-parse','HEAD'],text=True).strip()
    if actual!=PIN:raise ValueError('Ability metadata source pin changed')
    script="const {Dex}=require('./dist/sim');const names="+json.dumps(list(ABILITY_IDS))+";console.log(JSON.stringify(names.map(n=>{const a=Dex.abilities.get(n);if(!a.exists)throw Error(n);return {breakable:!!a.flags.breakable,traceable:!a.flags.notrace&&a.id!=='noability',notransform:!!a.flags.notransform,unassignable:!!a.flags.cantsuppress}})));"
    flags=json.loads(subprocess.check_output(['node','-e',script],cwd=upstream,text=True))
    cases=''.join(f'case {i}:'for i,flag in enumerate(flags)if flag['breakable'])
    trace_cases=''.join(f'case {i}:'for i,flag in enumerate(flags)if flag['traceable'])
    notransform_cases=''.join(f'case {i}:'for i,flag in enumerate(flags)if flag['notransform'])
    unassignable_cases=''.join(f'case {i}:'for i,flag in enumerate(flags)if flag['unassignable'])
    return '// SPDX-License-Identifier: MIT\n// Breakable flags from pinned Showdown '+PIN+'.\n#pragma once\nnamespace pkn {\nPK_HD inline bool breakable_ability(int id){switch(id){'+cases+'return true;default:return false;}}\nPK_HD inline bool traceable_ability(int id){switch(id){'+trace_cases+'return true;default:return false;}}\nPK_HD inline bool notransform_ability(int id){switch(id){'+notransform_cases+'return true;default:return false;}}\nPK_HD inline bool unassignable_ability(int id){switch(id){'+unassignable_cases+'return true;default:return false;}}\n}\n'


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--write',action='store_true');args=parser.parse_args()
    target=HERE/'ability_flags.hpp';expected=source()
    if args.write:target.write_text(expected)
    elif target.read_text()!=expected:raise SystemExit('Ability flags/header mapping differs from the frozen Dex')
    print(f'Exact frozen breakable flags: {len(ABILITY_IDS)} ability IDs')
