"""Actual CUDA comparison of complete frozen-format native set/team generation."""
import ctypes as C
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .bridge import Native,Mon
from .teamgen import GeneratorTables,GenRules,GenSet,GenTeam
from .validate import seed64


def run(cpu_library,cuda_library,fixture_path,constructor_path=None):
    raw=Path(fixture_path).read_bytes();data=json.loads(raw)
    tables=GeneratorTables(data['species'],data['moves'],data['status_moves'],data['items'])
    device_rules=tables.to_cuda();cuda=C.CDLL(str(cuda_library));from .cuda_runtime import prepare
    runtime_limits=prepare(cuda);stream=torch.cuda.current_stream().cuda_stream
    cuda.pkn_gen_sets_device.argtypes=[C.POINTER(GenRules)]+[C.c_void_p]*6+[C.c_int,C.c_void_p]
    cuda.pkn_gen_teams_device.argtypes=[C.POINTER(GenRules)]+[C.c_void_p]*3+[C.c_int,C.c_void_p]
    def upload(array):return torch.from_numpy(np.ascontiguousarray(array).view(np.uint8)).cuda()
    result=dict(full_gen9_admission=False,runtime_limits=runtime_limits,fixture_sha256=hashlib.sha256(raw).hexdigest(),scope='Complete native set/team generation; complete battle mechanics still unfinished')
    with Native(cpu_library) as native:
        lib=tables.bind(native);reset_tables=None
        if constructor_path:
            from .reset import ResetTables,ResetRules
            constructor_raw=Path(constructor_path).read_bytes();constructor=json.loads(constructor_raw)
            reset_tables=ResetTables(tables,constructor['metadata'],data['moves']);reset_tables.bind(native)
            device_reset=reset_tables.to_cuda();constructor_cursor=0
            for mode in ('sets','teams'):getattr(cuda,'pkn_materialize_'+mode+'_device').argtypes=[C.POINTER(ResetRules)]+[C.c_void_p]*3+[C.c_int,C.c_void_p]
            result['constructor_fixture_sha256']=hashlib.sha256(constructor_raw).hexdigest()

        for mode,kind in (('sets',GenSet),('teams',GenTeam)):
            records=data['full_sets'] if mode=='sets' else data['teams'];count=len(records)
            seeds=np.asarray([seed64(','.join(map(str,row['seed']))) if isinstance(row['seed'],list) else seed64(row['seed']) for row in records],dtype=np.uint64)
            source_seeds=seeds.copy();expected=(kind*count)();expected_ok=np.empty(count,dtype=np.int32)
            species=np.asarray([tables.species_ids[row['species']] for row in records],dtype=np.int32) if mode=='sets' else None
            flags=np.asarray([row['team'] for row in records],dtype=np.uint32) if mode=='sets' else None
            leads=np.asarray([row['lead'] for row in records],dtype=np.int32) if mode=='sets' else None
            for i,row in enumerate(records):
                rng=C.c_uint64(int(seeds[i]))
                ok=lib.pkn_gen_random_set(C.byref(tables.rules),int(species[i]),int(flags[i]),int(leads[i]),C.byref(rng),C.byref(expected[i])) if mode=='sets' else lib.pkn_gen_random_team(C.byref(tables.rules),C.byref(rng),C.byref(expected[i]))
                seeds[i]=rng.value;expected_ok[i]=ok
                decoded=tables.decode_set(expected[i]) if mode=='sets' and ok else [tables.decode_set(expected[i].pokemon[j]) for j in range(expected[i].count)] if ok else None
                if bool(ok)==bool(row['fault']) or rng.value!=seed64(row['after']) or (ok and decoded!=row['result']):raise AssertionError(f'CPU generator oracle mismatch: {mode} {i}')
            device_seeds=upload(source_seeds);output=torch.empty(count*C.sizeof(kind),dtype=torch.uint8,device='cuda');ok=torch.empty(count,dtype=torch.int32,device='cuda')
            if mode=='sets':
                inputs=[upload(array) for array in (species,flags,leads)]
                code=cuda.pkn_gen_sets_device(C.byref(device_rules),*[x.data_ptr() for x in inputs],device_seeds.data_ptr(),output.data_ptr(),ok.data_ptr(),count,stream)
            else:code=cuda.pkn_gen_teams_device(C.byref(device_rules),device_seeds.data_ptr(),output.data_ptr(),ok.data_ptr(),count,stream)
            if code:raise RuntimeError(f'CUDA generation launch failed: {code}')
            torch.cuda.synchronize()
            np.testing.assert_array_equal(ok.cpu().numpy(),expected_ok)
            np.testing.assert_array_equal(device_seeds.cpu().numpy().view(np.uint64),seeds)
            if output.cpu().numpy().tobytes()!=bytes(expected):raise AssertionError(f'Complete native CPU/CUDA generator state mismatch: {mode}')
            result[mode+'_exact']=count;result[mode+'_source_faults']=int((expected_ok==0).sum())
            if reset_tables:
                width=1 if mode=='sets' else 6;mons=(Mon*(count*width))();host_ok=np.zeros(count,dtype=np.int32)
                for i,row in enumerate(records):
                    target=C.cast(C.byref(mons,i*width*C.sizeof(Mon)),C.POINTER(Mon))
                    host_ok[i]=getattr(lib,'pkn_materialize_'+('set' if mode=='sets' else 'team'))(C.byref(reset_tables.rules),C.byref(expected[i]),target)
                    if expected_ok[i]:
                        actual=[reset_tables.decode(mons[i*width+j],tables.move_names) for j in range(width)]
                        reference=constructor['rows'][constructor_cursor]['result'];constructor_cursor+=1
                        if actual!=reference:raise AssertionError(f'CPU constructor oracle mismatch: {mode} {i}')
                np.testing.assert_array_equal(host_ok,expected_ok)
                device_mons=torch.zeros(count*width*C.sizeof(Mon),dtype=torch.uint8,device='cuda');device_ok=torch.empty(count,dtype=torch.int32,device='cuda')
                code=getattr(cuda,'pkn_materialize_'+mode+'_device')(C.byref(device_reset),output.data_ptr(),device_mons.data_ptr(),device_ok.data_ptr(),count,stream)
                if code:raise RuntimeError(f'CUDA constructor launch failed: {code}')
                torch.cuda.synchronize();np.testing.assert_array_equal(device_ok.cpu().numpy(),host_ok)
                if device_mons.cpu().numpy().tobytes()!=bytes(mons):raise AssertionError(f'Complete CPU/CUDA constructed Pokemon mismatch: {mode}')
                result[mode+'_constructed_pokemon_exact']=int(host_ok.sum())*width

    return result
