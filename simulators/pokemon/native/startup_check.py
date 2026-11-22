"""Native random generation/construction/startup, with explicit unknown faults."""
import argparse
import ctypes as C
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
from .bridge import Native,build
from .teamgen import GeneratorTables
from .reset import ResetTables,ResetSeeds,public_tables
from .reset_check import seed_value
from .teams_check import team_comparison
from .oracle_stage import stage_oracle


def flat_observations(state):
    keys=('tokens','mons','global','global_tokens','action_tokens','actions','mask')
    return np.asarray([np.concatenate([np.asarray(row[k],dtype=np.float64).ravel() for k in keys]) for row in state['observations']])

def run(output,generator_path,constructor_path,phases=0,replay=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    library=build(output/'build');staged=stage_oracle(output,output/'build/source')
    raw=Path(generator_path).read_bytes();data=json.loads(raw);constructor_raw=Path(constructor_path).read_bytes();constructor=json.loads(constructor_raw)
    generator=GeneratorTables(data['species'],data['moves'],data['status_moves'],data['items']);tables=ResetTables(generator,constructor['metadata'],data['moves'])
    # Static chart comes independently from the pinned Dex, along with startup
    # references only for cases whose effects already have native handlers.
    specs=[];indices=[];unsupported=0
    team_size=data.get('team_size',6);stored=0 if team_size==6 else team_size;sized={}if team_size==6 else {'team_size':team_size}
    with Native(library) as native:
        generator.bind(native);lib=tables.bind(native)
        blank=np.zeros((18,18),dtype=np.int8)
        for i,row in enumerate(data['teams']):
            if row['fault']:continue
            seed=ResetSeeds(seed_value('1,2,3,4'),(C.c_uint64*2)(seed_value(row['seed']),seed_value(row['seed'])),512,stored)
            ok=lib.pkn_reset_battle(C.byref(generator.rules),C.byref(tables.rules),C.byref(seed),C.byref(native.records[0]),tables.battle_moves,len(tables.battle_moves),blank.ctypes.data)
            if list(native.records[0].team_rng)!=[seed_value(row['after'])]*2:raise AssertionError('Random reset generator RNG mismatch')
            if not ok:
                if native.records[0].fault!=2:raise AssertionError('Unknown effect did not produce explicit unsupported fault')
                unsupported+=1;continue
            for battle_seed in ('1,2,3,4','7,8,9,10'):
                specs.append(dict(seed=battle_seed,teams=[row['result'],row['result']],**sized,generated=True,observe=True,phases=phases,policy_seed=173+i*37,tera=True,switch_every=13));indices.append(i)
    reference={'chart':None,'rows':[]}
    if replay:
        reference=json.loads(Path(replay).read_text())
        if reference['specs']!=specs or reference['indices']!=indices:raise ValueError('Replay case admission changed; regenerate reference')
        reference={key:reference[key] for key in ('chart','rows')}
    for begin in ([] if replay else range(0,len(specs),32)):
        if phases:
            for spec in specs[begin:begin+32]:
                value=json.loads(subprocess.check_output(['node',str(staged/'team_oracle.cjs')],input=json.dumps(dict(cases=[spec])),text=True))[0]
                reference['chart']=value['chart'];reference['rows'].append(value)
        else:
            result=json.loads(subprocess.check_output(['node',str(staged/'startup_oracle.cjs')],input=json.dumps(dict(cases=specs[begin:begin+32])),text=True))
            reference['chart']=result['chart'];reference['rows'].extend(result['rows'])
    if not reference['rows']:raise AssertionError('No implemented startup cases; full admission remains closed')
    (output/'oracle.json').write_text(json.dumps(dict(specs=specs,indices=indices,**reference),separators=(',',':')))
    chart=np.asarray(reference['chart'],dtype=np.int8);visible=public_tables(tables,chart)
    rows=[];transitions=0;natural=0
    with Native(library) as native:
        generator.bind(native);lib=tables.bind(native)
        for index,(spec,source_index,target) in enumerate(zip(specs,indices,reference['rows'])):
            row=data['teams'][source_index]
            seeds=ResetSeeds(seed_value(spec['seed']),(C.c_uint64*2)(seed_value(row['seed']),seed_value(row['seed'])),512,stored)
            ok=lib.pkn_reset_battle(C.byref(generator.rules),C.byref(tables.rules),C.byref(seeds),C.byref(native.records[0]),tables.battle_moves,len(tables.battle_moves),chart.ctypes.data)
            frames=0;mismatch=None
            states=[(None,target)] if not phases else [(None,target['initial'])]+[(frame['actions'],frame['state']) for frame in target['frames']]
            for actions,state in states:
                if actions is not None:
                    native.step([actions],tables.battle_moves,chart);frames+=1
                mismatch=team_comparison(native,state)
                actual=visible.observe(native)[0];expected=flat_observations(state)
                if mismatch is None and not np.array_equal(actual,expected):
                    diffs=np.argwhere(actual!=expected);mismatch=dict(observer_differences=[dict(side=int(a),index=int(b),actual=float(actual[a,b]),expected=float(expected[a,b])) for a,b in diffs[:20]])
                if mismatch is not None:
                    mismatch['frame']=frames;break
            transitions+=frames
            if phases:natural+=bool(target['frames'][-1]['state']['ended'])
            rows.append(dict(case=index,team=source_index,exact=bool(ok and mismatch is None),frames=frames,mismatch=mismatch))
    result=dict(full_gen9_admission=False,team_size=team_size,scope='Unfiltered native random reset attempts; unknown effects fault explicitly. Startup parity covers currently implemented complete teams only.',
        transitions=transitions,natural_oracle_games=natural,reset_attempts=len(data['teams']),unsupported_faults=unsupported,implemented_team_pairs=len(indices)//2,checks=len(rows),exact=sum(row['exact'] for row in rows),
        generator_fixture_sha256=hashlib.sha256(raw).hexdigest(),constructor_fixture_sha256=hashlib.sha256(constructor_raw).hexdigest(),rows=rows)
    (output/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))
    if result['exact']!=result['checks']:raise SystemExit(1)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--generators',required=True);p.add_argument('--constructors',required=True)
    p.add_argument('--phases',type=int,default=0);p.add_argument('--replay')
    a=p.parse_args();run(a.output,a.generators,a.constructors,a.phases,a.replay)
