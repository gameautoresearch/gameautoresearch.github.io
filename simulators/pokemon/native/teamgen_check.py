"""Differential checks of native frozen-format generation primitives."""
import argparse
import ctypes as C
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from .bridge import HERE,Native,build
from .teamgen import GeneratorTables,GenPool,GenCounter,GenMoveList,GenSet,GenTeam,COUNTER_NAMES,ABILITY_FLAGS,ROLE_NAMES
from .validate import oracle_identity,seed64
from .oracle_stage import stage_oracle


def run(output,replay=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    library=build(output/'build')
    oracle_source=stage_oracle(output,output/'build/source')
    raw=Path(replay).read_bytes() if replay else subprocess.check_output(['node',str(oracle_source/'teamgen_oracle.cjs')])
    if replay:(output/'replay-source.json').write_text(json.dumps(dict(path=str(Path(replay).resolve()),sha256=hashlib.sha256(raw).hexdigest()),indent=2)+'\n')
    (output/'oracle.json').write_bytes(raw)
    (output/'oracle-identity.json').write_text(json.dumps(oracle_identity(),indent=2)+'\n')
    data=json.loads(raw);tables=GeneratorTables(data['species'],data['moves'],data['status_moves'],data['items'])
    pool_count=forme_count=compatibility_count=0
    with Native(library) as native:
        lib=tables.bind(native)
        for record in data['pools']:
            excluded=np.zeros(len(data['species']),dtype=np.uint8)
            for name in record['excluded']:excluded[tables.species_ids[name]]=1
            pool=GenPool()
            if not lib.pkn_gen_pool(C.byref(tables.rules),C.byref(pool),record['type'],excluded.ctypes.data):raise AssertionError('Native generator pool failed')
            groups=[tables.bases[i] for i in pool.groups[:pool.group_count]]
            members={}
            for species_id in pool.species[:pool.species_count]:
                row=data['species'][species_id];members.setdefault(row['base'],[]).append(row['id'])
            if groups!=record['groups'] or members!=record['members']:raise AssertionError(f'Generator pool mismatch: type={record["type"]}')
            pool_count+=1
        for record in data['formes']:
            rng=C.c_uint64(seed64(','.join(map(str,record['seed']))))
            index=lib.pkn_gen_forme(C.byref(tables.rules),tables.species_ids[record['species']],C.byref(rng))
            if index<0 or tables.formes[index]!=record['forme'] or rng.value!=seed64(record['after']):
                raise AssertionError(f'Native forme or RNG mismatch: {record}')
            forme_count+=1
        for candidate,expected in enumerate(data['compatibility']):
            for other,compatible in enumerate(expected):
                team=(C.c_int32*1)(other)
                actual=bool(lib.pkn_gen_compatible(C.byref(tables.rules),candidate,team,1))
                if actual!=compatible:raise AssertionError(f'Native compatibility mismatch: {candidate},{other}')
                compatibility_count+=1
        counters=[]
        for record in data['queries']:
            moves=(C.c_int32*len(record['moves']))(*[tables.move_ids[name] for name in record['moves']])
            abilities=sum(ABILITY_FLAGS.get(name,0) for name in set(record['abilities']))
            result=GenCounter()
            ok=lib.pkn_gen_query_moves(C.byref(tables.rules),tables.species_ids[record['species']],moves,len(moves),abilities,data['types'].index(record['tera']),C.byref(result))
            counts=[record['counts'].get(name,0) for name in COUNTER_NAMES]
            damaging=[tables.move_names[i] for i in result.damaging_moves[:result.damaging_count]]
            power=[tables.move_names[i] for i in result.base_power_moves[:result.base_power_count]]
            if not ok or list(result.counts)!=counts or damaging!=record['damaging'] or power!=record['power']:
                raise AssertionError(f'Generator move query mismatch: {record}; actual={dict(zip(COUNTER_NAMES,result.counts))}, damaging={damaging}, power={power}')
            counters.append(result)
        for record in data['ability_queries']:
            query=data['queries'][record['query']]
            options=(C.c_int32*len(query['abilities']))(*[tables.ability_ids[name] for name in query['abilities']])
            rng=C.c_uint64(seed64(','.join(map(str,record['seed']))))
            moves=sum(flag for flag,name in ((1,'defog'),(2,'flipturn'),(4,'snowscape')) if name in query['moves'])
            selected=lib.pkn_gen_ability(C.byref(tables.rules),tables.species_ids[query['species']],options,len(options),
                C.byref(counters[record['query']]),record['weather'],moves,data['types'].index(query['tera']),C.byref(rng))
            if selected<0 or tables.ability_names[selected]!=record['chosen'] or rng.value!=seed64(record['after']):
                raise AssertionError(f'Generator ability or RNG mismatch: {query}; expected={record}; actual={selected},{rng.value}')
        for record in data['culls']:
            chosen=GenMoveList();pool=GenMoveList();chosen.count=len(record['chosen']);pool.count=len(record['pool'])
            for i,name in enumerate(record['chosen']):chosen.values[i]=tables.move_ids[name]
            for i,name in enumerate(record['pool']):pool.values[i]=tables.move_ids[name]
            abilities=sum(ABILITY_FLAGS.get(name,0) for name in set(record['abilities']))
            ok=lib.pkn_gen_cull_moves(C.byref(tables.rules),tables.species_ids[record['species']],C.byref(chosen),C.byref(pool),abilities,record['team'],data['types'].index(record['tera']))
            actual=[tables.move_names[i] for i in pool.values[:pool.count]]
            if bool(ok)==bool(record['fault']) or (ok and actual!=record['result']):
                raise AssertionError(f'Generator move cull mismatch: {record}; actual={actual}; fault={pool.fault}')
        for record in data['movesets']:
            pool=GenMoveList();pool.count=len(record['pool'])
            for i,name in enumerate(record['pool']):pool.values[i]=tables.move_ids[name]
            abilities=sum(ABILITY_FLAGS.get(name,0) for name in set(record['abilities']))
            rng=C.c_uint64(seed64(','.join(map(str,record['seed']))));chosen=GenMoveList()
            ok=lib.pkn_gen_moveset(C.byref(tables.rules),tables.species_ids[record['species']],C.byref(pool),abilities,record['team'],
                data['types'].index(record['tera']),ROLE_NAMES.index(record['role']),C.byref(rng),C.byref(chosen))
            actual=[tables.move_names[i] for i in chosen.values[:chosen.count]]
            remaining=[tables.move_names[i] for i in pool.values[:pool.count]]
            if bool(ok)==bool(record['fault']) or rng.value!=seed64(record['after']) or (ok and (actual!=record['chosen'] or remaining!=record['remaining'])):
                raise AssertionError(f'Generator moveset or RNG mismatch: {record}; actual={actual}; remaining={remaining}; rng={rng.value}; fault={chosen.fault}')
        for record in data['set_stages']:
            pool=GenMoveList();pool.count=len(record['pool'])
            for i,name in enumerate(record['pool']):pool.values[i]=tables.move_ids[name]
            flags=sum(ABILITY_FLAGS.get(name,0) for name in set(record['abilities']))
            options=(C.c_int32*len(record['abilities']))(*[tables.ability_ids[name] for name in record['abilities']])
            rng=C.c_uint64(seed64(','.join(map(str,record['seed']))));chosen=GenMoveList();selected=(C.c_int32*2)(-1,-1)
            ok=lib.pkn_gen_set_stage(C.byref(tables.rules),tables.species_ids[record['species']],C.byref(pool),options,len(options),flags,record['team'],
                data['types'].index(record['tera']),ROLE_NAMES.index(record['role']),record['lead'],C.byref(rng),C.byref(chosen),selected)
            actual=[tables.move_names[i] for i in chosen.values[:chosen.count]]
            ability=tables.ability_names[selected[0]] if selected[0]>=0 else None
            item=tables.item_names[selected[1]] if selected[1]>=0 else None
            if bool(ok)==bool(record['fault']) or rng.value!=seed64(record['after']) or (ok and (actual!=record['chosen'] or ability!=record['ability'] or item!=record['item'])):
                raise AssertionError(f'Generator combined moves/ability/item mismatch: {record}; actual={actual},{ability},{item}; rng={rng.value}; fault={chosen.fault}')
        for record in data['full_sets']:
            rng=C.c_uint64(seed64(','.join(map(str,record['seed']))));result=GenSet()
            ok=lib.pkn_gen_random_set(C.byref(tables.rules),tables.species_ids[record['species']],record['team'],record['lead'],C.byref(rng),C.byref(result))
            actual=tables.decode_set(result) if ok else None
            if bool(ok)==bool(record['fault']) or rng.value!=seed64(record['after']) or (ok and actual!=record['result']):
                raise AssertionError(f'Native full random set mismatch: {record}; actual={actual}; rng={rng.value}')
        for record in data['teams']:
            rng=C.c_uint64(seed64(record['seed']));result=GenTeam()
            ok=lib.pkn_gen_random_team_sized(C.byref(tables.rules),C.byref(rng),C.byref(result),data.get('team_size',6))
            actual=[tables.decode_set(result.pokemon[i]) for i in range(result.count)] if ok else None
            if bool(ok)==bool(record['fault']) or rng.value!=seed64(record['after']) or (ok and actual!=record['result']):
                raise AssertionError(f'Native full random team mismatch: {record}; actual={actual}; rng={rng.value}; fault={result.fault}')
        try:native.require_full_task()
        except RuntimeError:pass
        else:raise AssertionError('Generation primitives must not open full-task admission')
    result=dict(full_gen9_admission=False,scope='Native species pools, cosmetic formes, compatibility; full randomSet/randomTeam unfinished',
                species=len(data['species']),pools_exact=pool_count,formes_and_rng_exact=forme_count,
                compatibility_pairs_exact=compatibility_count,move_queries_exact=len(data['queries']),ability_queries_and_rng_exact=len(data['ability_queries']),
                move_culls_exact=len(data['culls']),oracle_cull_faults=sum(bool(record['fault']) for record in data['culls']),
                movesets_and_rng_exact=len(data['movesets']),moves_ability_item_stages_and_rng_exact=len(data['set_stages']),
                full_sets_and_rng_exact=len(data['full_sets']),full_teams_and_rng_exact=len(data['teams']),
                oracle_full_set_faults=sum(bool(row['fault']) for row in data['full_sets']),oracle_team_faults=sum(bool(row['fault']) for row in data['teams']),
                team_size=data.get('team_size',6),oracle_sha256=hashlib.sha256(raw).hexdigest())
    (output/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);parser.add_argument('--replay')
    args=parser.parse_args();run(args.output,args.replay)
