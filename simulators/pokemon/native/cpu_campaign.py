"""Bounded-memory original-task seed qualification of native CPU battles.

Every attempted seed remains in the audit. Unsupported mechanics are explicit
faults, never a filtered production environment or a replacement task.
"""
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
from .task_reset_check import seed_text
from .teams_check import team_comparison
from .startup_check import flat_observations
from .oracle_stage import stage_oracle
from .validate import oracle_identity


def run(output,generator_path,constructor_path,chart_path,count=512,start=940000,phases=512,include_unsupported=False,policy="scripted_rotation",checkpoint_path=None,library_path=None,team_size=6):
    output=Path(output);output.mkdir(parents=True,exist_ok=False);library=Path(library_path)if library_path else build(output/'build');staged=stage_oracle(output,Path(library).parent/'source')
    raw=Path(generator_path).read_bytes();ctor_raw=Path(constructor_path).read_bytes();data=json.loads(raw);metadata=json.loads(ctor_raw)['metadata'];chart=np.asarray(json.loads(Path(chart_path).read_bytes())['chart'],dtype=np.int8)
    generator=GeneratorTables(data['species'],data['moves'],data['status_moves'],data['items']);reset=ResetTables(generator,metadata,data['moves']);visible=public_tables(reset,chart)
    specs=[];attempts=[];rows=[];transitions=features=natural=schedule_violations=0;reference_hash=hashlib.sha256()
    learned=None;policy_receipt=None
    if policy=='retained_learned':
        if not checkpoint_path:raise ValueError('Retained learned policy requires its checkpoint artifact')
        import torch
        from ..network import Policy
        from ..efficient import unpack_tensor
        torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
        checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
        integration=hashlib.sha256(b''.join((Path(__file__).parent.parent/name).read_bytes()for name in ('network.py','efficient.py'))).hexdigest()
        if checkpoint['integration_signature']!=integration:raise ValueError('Retained policy network integration changed')
        learned=Policy(hidden=64,embedding=8);learned.load_state_dict(checkpoint['model']);learned.eval()
        policy_receipt=dict(checkpoint_sha256=hashlib.sha256(Path(checkpoint_path).read_bytes()).hexdigest(),updates=checkpoint['updates'],integration_signature=integration,mode='Frozen policy weights for differential actions; no optimizer or environment resume')

    with Native(library)as native:
        generator.bind(native);lib=reset.bind(native)
        for master in range(start,start+count):
            for swapped in (False,True):
                seed=task_seeds(master,phases+1,swapped,team_size);ok=lib.pkn_reset_battle(C.byref(generator.rules),C.byref(reset.rules),C.byref(seed),C.byref(native.records[0]),reset.battle_moves,len(reset.battle_moves),chart.ctypes.data)
                attempts.append(dict(master=master,swapped=swapped,fault=native.records[0].fault))
                if not ok:
                    if native.records[0].fault!=2:raise AssertionError('Missing effect did not produce unsupported fault')
                    if not include_unsupported:continue
                specs.append(dict(master=master,swapped=swapped,seed=seed_text(seed.battle),team_seeds=[seed_text(value)for value in seed.teams],generated=True,observe=True,phases=phases,policy_seed=master*37,policy=policy,tera=True,switch_every=13,**({}if team_size==6 else {'team_size':team_size})))
        (output/'attempts.json').write_text(json.dumps(attempts,indent=2)+'\n')
        with(output/'oracle-fixtures.jsonl').open('xb')as retained:
            for begin in range(0,len(specs),8):
                batch=specs[begin:begin+8]
                if learned is not None:
                    for spec in batch:
                        seed=task_seeds(spec['master'],phases+1,spec['swapped'],team_size)
                        if not lib.pkn_reset_battle(C.byref(generator.rules),C.byref(reset.rules),C.byref(seed),C.byref(native.records[0]),reset.battle_moves,len(reset.battle_moves),chart.ctypes.data):
                            spec['schedule']=[];continue
                        torch.manual_seed(spec['master']*2+int(spec['swapped']));schedule=[]
                        with torch.no_grad():
                            while len(schedule)<phases and not(native.records[0].terminated or native.records[0].truncated or native.records[0].fault):
                                obs=torch.from_numpy(visible.observe(native)[0].astype(np.float32))
                                logits,_=learned(unpack_tensor(obs));ready=obs[:,686:].bool().any(-1)
                                actions=torch.distributions.Categorical(logits=logits,validate_args=False).sample();actions=torch.where(ready,actions,-1)
                                schedule.append(actions.tolist());native.step([schedule[-1]],reset.battle_moves,chart)
                        spec['schedule']=schedule;spec['native_schedule_fault']=native.records[0].fault
                (output/f'specs-{begin:06d}.json').write_text(json.dumps(batch)+'\n')
                reference=json.loads(subprocess.check_output(['node',str(staged/'team_oracle.cjs')],input=json.dumps(dict(cases=batch)),text=True))
                for spec,target in zip(batch,reference):
                    schedule_violations+=len(target.get('schedule_violations',[]))
                    encoded=json.dumps(target,separators=(',',':')).encode()+b'\n';retained.write(encoded);reference_hash.update(encoded)
                    seed=task_seeds(spec['master'],phases+1,spec['swapped'],team_size)
                    ok=lib.pkn_reset_battle(C.byref(generator.rules),C.byref(reset.rules),C.byref(seed),C.byref(native.records[0]),reset.battle_moves,len(reset.battle_moves),chart.ctypes.data)
                    if not ok:
                        if not include_unsupported:raise AssertionError('Previously supported seed changed admission')
                        ended=bool(target['frames'][-1]['state']['ended'])
                        rows.append(dict(case=len(rows),master=spec['master'],swapped=spec['swapped'],exact=False,frames=0,natural_oracle_game=ended,unsupported_reset=True,mismatch=dict(fault=native.records[0].fault,phase=-1)))
                        continue
                    mismatch=None;frames=0
                    for actions,state in [(None,target['initial'])]+[(frame['actions'],frame['state'])for frame in target['frames']]:
                        if actions is not None:native.step([actions],reset.battle_moves,chart);frames+=1
                        mismatch=team_comparison(native,state)
                        if mismatch is None:
                            actual=visible.observe(native)[0];expected=flat_observations(state)
                            if not np.array_equal(actual,expected):
                                where=np.argwhere(actual!=expected);mismatch=dict(observer_differences=[dict(side=int(a),index=int(b),actual=float(actual[a,b]),expected=float(expected[a,b]))for a,b in where[:20]])
                        if mismatch is not None:mismatch['phase']=frames-1;break
                        features+=1400
                    transitions+=frames;ended=bool(target['frames'][-1]['state']['ended']);natural+=ended
                    rows.append(dict(case=len(rows),master=spec['master'],swapped=spec['swapped'],exact=mismatch is None,frames=frames,natural_oracle_game=ended,mismatch=mismatch))
                (output/'partial.json').write_text(json.dumps(dict(full_gen9_admission=False,checked=len(rows),exact=sum(r['exact']for r in rows),transitions=transitions,failure_cases=[r['case']for r in rows if not r['exact']]),indent=2)+'\n')
                print(f'CPU_CAMPAIGN {len(rows)}/{len(specs)} exact={sum(r["exact"]for r in rows)}',flush=True)
    checked=[r for r in rows if not r.get('unsupported_reset')]
    result=dict(full_gen9_admission=False,library_sha256=native.binary_sha256,source_signature=native.source_signature,team_size=team_size,policy=policy,policy_receipt=policy_receipt,scope='Unfiltered original-task seed attempts; unsupported teams explicitly retained; no production seed filtering',oracle_includes_unsupported=include_unsupported,schedule_violations=schedule_violations,attempts=len(attempts),unsupported_faults=sum(a['fault']!=0 for a in attempts),cases=len(checked),oracle_cases=len(rows),oracle_natural_games=sum(r['natural_oracle_game']for r in rows),exact=sum(r['exact']for r in rows),transitions=transitions,exact_public_feature_values=features,natural_oracle_games=natural,bounded_oracle_games=len(checked)-natural,reference_sha256=reference_hash.hexdigest(),generator_sha256=hashlib.sha256(raw).hexdigest(),constructor_sha256=hashlib.sha256(ctor_raw).hexdigest(),oracle_identity=oracle_identity(),rows=rows)
    (output/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items()if k not in ('rows','oracle_identity')},indent=2))
    if result['exact']!=result['cases']:raise SystemExit(1)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--generators',required=True);p.add_argument('--constructors',required=True);p.add_argument('--chart',required=True);p.add_argument('--count',type=int,default=512);p.add_argument('--start',type=int,default=940000);p.add_argument('--phases',type=int,default=512);p.add_argument('--include-unsupported',action='store_true');p.add_argument('--policy',choices=('scripted_rotation','random','scripted_public_power','retained_learned'),default='scripted_rotation');p.add_argument('--checkpoint');p.add_argument('--team-size',type=int,default=6);p.add_argument('--library');a=p.parse_args();run(a.output,a.generators,a.constructors,a.chart,a.count,a.start,a.phases,a.include_unsupported,a.policy,a.checkpoint,a.library,a.team_size)
