"""Native reset/rollout against original task-domain seeds and independent teams."""
import argparse
import ctypes as C
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from .bridge import Native,build
from .teamgen import GeneratorTables
from .reset import ResetTables,public_tables
from .random_resident import task_seeds
from .startup_check import flat_observations
from .teams_check import team_comparison
from .oracle_stage import stage_oracle
from .validate import oracle_identity


def seed_text(value):return ','.join(str((value>>shift)&65535) for shift in (48,32,16,0))


def run(output,generator_path,constructor_path,chart_path,count=2048,start=930000,phases=512):
    output=Path(output);output.mkdir(parents=True,exist_ok=False);library=build(output/'build');staged=stage_oracle(output,output/'build/source')
    raw=Path(generator_path).read_bytes();constructor_raw=Path(constructor_path).read_bytes();data=json.loads(raw);constructor=json.loads(constructor_raw)
    chart=np.asarray(json.loads(Path(chart_path).read_bytes())['chart'],dtype=np.int8)
    generator=GeneratorTables(data['species'],data['moves'],data['status_moves'],data['items']);reset=ResetTables(generator,constructor['metadata'],data['moves']);visible=public_tables(reset,chart)
    attempts=[];specs=[];seeds=[]
    with Native(library) as native:
        generator.bind(native);lib=reset.bind(native)
        for master in range(start,start+count):
            for swapped in (False,True):
                seed=task_seeds(master,phases+1,swapped)
                ok=lib.pkn_reset_battle(C.byref(generator.rules),C.byref(reset.rules),C.byref(seed),C.byref(native.records[0]),reset.battle_moves,len(reset.battle_moves),chart.ctypes.data)
                attempts.append(dict(master=master,swapped=swapped,fault=native.records[0].fault))
                if not ok:
                    if native.records[0].fault!=2:raise AssertionError('Unsupported task reset was not an explicit fault')
                    continue
                specs.append(dict(master=master,swapped=swapped,seed=seed_text(seed.battle),team_seeds=[seed_text(value) for value in seed.teams],generated=True,observe=True,phases=phases,policy_seed=master*37,tera=True,switch_every=13));seeds.append(seed)
    references=[]
    for begin in range(0,len(specs),8):references.extend(json.loads(subprocess.check_output(['node',str(staged/'team_oracle.cjs')],input=json.dumps(dict(cases=specs[begin:begin+8])),text=True)))
    if not references:raise AssertionError('No complete implemented task-domain pairs; retain the explicit faults')
    reference=dict(chart=chart.tolist(),specs=specs,rows=references)
    (output/'oracle.json').write_text(json.dumps(reference,separators=(',',':')))
    rows=[];transitions=0;feature_values=0
    with Native(library)as native:
        generator.bind(native);lib=reset.bind(native)
        for index,(seed,row)in enumerate(zip(seeds,references)):
            ok=lib.pkn_reset_battle(C.byref(generator.rules),C.byref(reset.rules),C.byref(seed),C.byref(native.records[0]),reset.battle_moves,len(reset.battle_moves),chart.ctypes.data)
            if not ok:raise AssertionError('Task reset admission changed during qualification')
            mismatch=None;frames=0
            for actions,state in [(None,row['initial'])]+[(frame['actions'],frame['state'])for frame in row['frames']]:
                if actions is not None:native.step([actions],reset.battle_moves,chart);frames+=1
                mismatch=team_comparison(native,state)
                actual=visible.observe(native)[0];expected=flat_observations(state)
                if mismatch is None and not np.array_equal(actual,expected):
                    indices=np.argwhere(actual!=expected);mismatch=dict(observer_differences=[dict(side=int(a),index=int(b),actual=float(actual[a,b]),expected=float(expected[a,b]))for a,b in indices[:20]])
                if mismatch is not None:mismatch['frame']=frames;break
                feature_values+=actual.size
            transitions+=frames;rows.append(dict(case=index,master=specs[index]['master'],swapped=specs[index]['swapped'],frames=frames,exact=mismatch is None,mismatch=mismatch))
    result=dict(full_gen9_admission=False,scope='Unfiltered task-domain native reset attempts; differential completion checks only for implemented independent team pairs, no production filtering',attempts=attempts,checks=len(rows),exact=sum(row['exact']for row in rows),transitions=transitions,exact_public_feature_values=feature_values,natural_oracle_games=sum(row['frames'][-1]['state']['ended']for row in references),generator_fixture_sha256=hashlib.sha256(raw).hexdigest(),constructor_fixture_sha256=hashlib.sha256(constructor_raw).hexdigest(),oracle_identity=oracle_identity(),rows=rows)
    (output/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({key:value for key,value in result.items()if key not in ('attempts','rows','oracle_identity')},indent=2))
    if result['checks']!=result['exact']:raise SystemExit(1)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);parser.add_argument('--generators',required=True);parser.add_argument('--constructors',required=True);parser.add_argument('--chart',required=True);parser.add_argument('--count',type=int,default=2048);parser.add_argument('--start',type=int,default=930000);parser.add_argument('--phases',type=int,default=512)
    args=parser.parse_args();run(args.output,args.generators,args.constructors,args.chart,args.count,args.start,args.phases)
