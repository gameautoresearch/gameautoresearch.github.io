"""Independent construction checks for all retained generated sets and teams."""
import argparse
import ctypes as C
import hashlib
import json
from pathlib import Path
import subprocess

from .bridge import Mon,Native,build
from .teamgen import GeneratorTables,GenSet,GenTeam
from .reset import ResetTables
from .validate import seed64,oracle_identity
from .oracle_stage import stage_oracle


def seed_value(seed):return seed64(','.join(map(str,seed)) if isinstance(seed,list) else seed)

def run(output,generation_fixtures,replay=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    library=build(output/'build');staged=stage_oracle(output,output/'build/source')
    raw=Path(generation_fixtures).read_bytes();data=json.loads(raw)
    generator=GeneratorTables(data['species'],data['moves'],data['status_moves'],data['items'])
    sets=[row for row in data['full_sets'] if not row['fault']];teams=[row for row in data['teams'] if not row['fault']]
    source_teams=[[row['result']] for row in sets]+[row['result'] for row in teams]
    if replay:oracle_raw=Path(replay).read_bytes()
    else:
        result={'metadata':None,'rows':[]}
        for begin in range(0,len(source_teams),128):
            request=dict(formes=generator.formes if begin==0 else [],teams=source_teams[begin:begin+128],team_size=data.get('team_size',6))
            response=json.loads(subprocess.check_output(['node',str(staged/'reset_oracle.cjs')],input=json.dumps(request),text=True))
            if begin==0:result['metadata']=response['metadata']
            result['rows'].extend(response['rows'])
        oracle_raw=json.dumps(result,separators=(',',':')).encode()
    (output/'oracle.json').write_bytes(oracle_raw);expected=json.loads(oracle_raw)
    if len(expected['rows'])!=len(source_teams):raise AssertionError('Constructor fixture count changed')
    tables=ResetTables(generator,expected['metadata'],data['moves']);count=0;unsupported=0
    with Native(library) as native:
        gen=generator.bind(native);lib=tables.bind(native)
        for index,row in enumerate(sets+teams):
            rng=C.c_uint64(seed_value(row['seed']));is_team=index>=len(sets)
            if is_team:
                value=GenTeam();ok=gen.pkn_gen_random_team_sized(C.byref(generator.rules),C.byref(rng),C.byref(value),data.get('team_size',6))
                pokemon=(Mon*6)();ok=ok and lib.pkn_materialize_team_sized(C.byref(tables.rules),C.byref(value),pokemon,data.get('team_size',6))
            else:
                value=GenSet();ok=gen.pkn_gen_random_set(C.byref(generator.rules),generator.species_ids[row['species']],row['team'],row['lead'],C.byref(rng),C.byref(value))
                pokemon=(Mon*1)();ok=ok and lib.pkn_materialize_set(C.byref(tables.rules),C.byref(value),pokemon)
            if not ok or rng.value!=seed_value(row['after']):raise AssertionError(f'Native construction/generation failed: {index}')
            pokemon=pokemon[:value.count] if is_team else pokemon  # beyond a smaller team's size the slots stay empty
            actual=[tables.decode(p,generator.move_names) for p in pokemon]
            target=expected['rows'][index]
            if actual!=target['result']:raise AssertionError(f'Native constructed Pokemon mismatch {index}: actual={actual}; expected={target["result"]}')
            if seed_value(target['seed'])!=seed64('1,2,3,4'):raise AssertionError('Constructor consumed unexpected battle RNG')
            count+=len(pokemon);unsupported+=sum(bool(p.unsupported) for p in pokemon)
    result=dict(full_gen9_admission=False,scope='Complete frozen generated-set stat/PP construction; startup callback dispatch and RL reset remain separate',
        team_size=data.get('team_size',6),standalone_sets=len(sets),teams=len(teams),pokemon_exact=count,pokemon_with_unimplemented_battle_effects=unsupported,
        move_rejections=tables.move_rejections,generator_fixture_sha256=hashlib.sha256(raw).hexdigest(),oracle_sha256=hashlib.sha256(oracle_raw).hexdigest(),oracle_identity=oracle_identity())
    (output/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ('move_rejections','oracle_identity')},indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);parser.add_argument('--fixtures',required=True);parser.add_argument('--replay')
    args=parser.parse_args();run(args.output,args.fixtures,args.replay)
