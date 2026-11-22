"""Replay retained generated trajectories without changing their case admission."""
import argparse
import ctypes as C
import hashlib
import json
import itertools
from pathlib import Path
import numpy as np
from .bridge import Native,build
from .teamgen import GeneratorTables
from .reset import ResetTables,ResetSeeds,public_tables
from .reset_check import seed_value
from .resident_suite import flat_observations
from .teams_check import team_comparison


def run(output,generator_path,constructor_path,reference_path,include_unsupported=False,library_path=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=False);library=Path(library_path)if library_path else build(output/'build')
    paths=[Path(p)for p in (generator_path,constructor_path,reference_path)]
    data=json.loads(paths[0].read_bytes());constructor=json.loads(paths[1].read_bytes())
    inputs={}
    for path in paths:
        digest=hashlib.sha256()
        with path.open('rb')as stream:
            for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
        inputs[str(path)]=digest.hexdigest()
    if paths[2].suffix=='.jsonl':
        def retained_cases():
            with paths[2].open()as stream:
                for line in stream:
                    target=json.loads(line);yield target['spec'],target
        cases=iter(retained_cases());first=next(cases)
        chart_data=first[1]['chart'];cases=itertools.chain([first],cases);reference=None
    else:
        reference=json.loads(paths[2].read_bytes());chart_data=reference['chart']
        cases=zip(reference['specs'],reference['rows'])
    generator=GeneratorTables(data['species'],data['moves'],data['status_moves'],data['items']);reset=ResetTables(generator,constructor['metadata'],data['moves'])
    chart=np.asarray(chart_data,dtype=np.int8);visible=public_tables(reset,chart);rows=[];transitions=natural=draws=bounded=violations=0;sizes=set()
    with Native(library)as native:
        generator.bind(native);lib=reset.bind(native)
        for i,(spec,target)in enumerate(cases):
            terminal=target['frames'][-1]['state'];natural+=bool(terminal['ended']);draws+=bool(terminal['ended']and terminal['winner']not in('p1','p2'));bounded+=not terminal['ended'];violations+=len(target.get('schedule_violations',[]))
            if reference is not None and 'indices'in reference:seeds=[data['teams'][reference['indices'][i]]['seed']]*2
            else:seeds=spec['team_seeds']
            team_size=spec.get('team_size',6);sizes.add(team_size)
            seeds=ResetSeeds(seed_value(spec['seed']),(C.c_uint64*2)(*[seed_value(v)for v in seeds]),spec['phases']+1,0 if team_size==6 else team_size)
            if not lib.pkn_reset_battle(C.byref(generator.rules),C.byref(reset.rules),C.byref(seeds),C.byref(native.records[0]),reset.battle_moves,len(reset.battle_moves),chart.ctypes.data):
                if not include_unsupported:raise AssertionError(f'Retained case {i} became unsupported')
                rows.append(dict(case=i,exact=False,unsupported=True,fault=native.records[0].fault,frames=0))
                continue
            mismatch=None;frames=0
            for phase in range(-1,len(target['frames'])):
                if phase>=0:native.step([target['frames'][phase]['actions']],reset.battle_moves,chart);frames+=1
                state=target['initial']if phase<0 else target['frames'][phase]['state'];mismatch=team_comparison(native,state)
                if mismatch is None:
                    actual=visible.observe(native)[0];expected=flat_observations(state)
                    if not np.array_equal(actual,expected):
                        mismatch=dict(observer_differences=[dict(side=int(a),index=int(b),actual=float(actual[a,b]),expected=float(expected[a,b]))for a,b in np.argwhere(actual!=expected)[:20]])
                if mismatch is not None:mismatch['phase']=phase;break
            transitions+=frames;rows.append(dict(case=i,exact=mismatch is None,frames=frames,mismatch=mismatch))
    if len(sizes)>1:raise ValueError('One bank holds one team size')
    result=dict(full_gen9_admission=False,library_sha256=native.binary_sha256,source_signature=native.source_signature,**({}if sizes<={6}else {'team_size':sizes.pop()}),natural_oracle_games=natural,official_draws=draws,bounded_oracle_games=bounded,schedule_violations=violations,attempts=len(rows),unsupported_faults=sum(bool(row.get('unsupported'))for row in rows),cases=sum(not row.get('unsupported',False)for row in rows),exact=sum(row['exact']for row in rows),transitions=transitions,inputs=inputs,rows=rows)
    (output/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items()if k not in ('rows','inputs')},indent=2))
    if result['cases']!=result['exact']:raise SystemExit(1)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--generators',required=True);p.add_argument('--constructors',required=True);p.add_argument('--reference',required=True);p.add_argument('--include-unsupported',action='store_true');p.add_argument('--library');a=p.parse_args();run(a.output,a.generators,a.constructors,a.reference,a.include_unsupported,a.library)
