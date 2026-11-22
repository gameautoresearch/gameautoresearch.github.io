"""CUDA native generation-to-completion qualification; unknown resets stay faults."""
import ctypes as C
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .bridge import Native,State
from .teamgen import GeneratorTables,GenRules
from .reset import ResetTables,ResetRules,ResetSeeds,public_tables
from .reset_check import seed_value
from .teams_check import team_comparison
from .visible import PublicRules
from .resident_suite import flat_observations


def run(cpu_library,cuda_library,generator_path,constructor_path,reference_path):
    raw=Path(reference_path).read_bytes();reference=json.loads(raw)
    generator_raw=Path(generator_path).read_bytes();constructor_raw=Path(constructor_path).read_bytes()
    # The replay's source-bound construction and generation fixtures are supplied
    # independently; every attempted random reset, including faults, is retained.
    data=json.loads(generator_raw);constructor=json.loads(constructor_raw)
    generator=GeneratorTables(data['species'],data['moves'],data['status_moves'],data['items'])
    reset=ResetTables(generator,constructor['metadata'],data['moves']);chart=np.asarray(reference['chart'],dtype=np.int8)
    visible=public_tables(reset,chart);device_generator=generator.to_cuda();device_reset=reset.to_cuda();device_visible=visible.to_cuda()
    cuda=C.CDLL(str(cuda_library));from .cuda_runtime import prepare
    runtime_limits=prepare(cuda);stream=torch.cuda.current_stream().cuda_stream
    cuda.pkn_random_reset_device.argtypes=[C.POINTER(GenRules),C.POINTER(ResetRules)]+[C.c_void_p]*3+[C.c_int,C.c_void_p,C.c_int,C.c_void_p,C.c_void_p]
    cuda.pkn_step_device.argtypes=[C.c_void_p,C.c_void_p,C.c_int,C.c_void_p,C.c_int,C.c_void_p,C.c_void_p]
    cuda.pkn_observe_device.argtypes=[C.c_void_p,C.c_void_p,C.c_int,C.POINTER(PublicRules),C.c_void_p]
    cuda.pkn_masks_device.argtypes=[C.c_void_p,C.c_void_p,C.c_int,C.c_void_p]
    def upload(value):return torch.from_numpy(np.frombuffer(bytes(value),dtype=np.uint8).copy()).cuda()
    move_data=upload(reset.battle_moves);chart_data=torch.from_numpy(chart).cuda()
    counts=dict(reset_attempts=0,unsupported_faults=0,battles=0,oracle_transitions=0,complete_state_mask_comparisons=0,exact_oracle_feature_values=0,natural_oracle_games=0)
    for mode in ('unfiltered','trajectories'):
        specs=[dict(seed='1,2,3,4',index=i) for i in range(len(data['teams']))] if mode=='unfiltered' else [dict(seed=spec['seed'],index=i) for spec,i in zip(reference['specs'],reference['indices'])] if 'indices' in reference else reference['specs']
        n=len(specs);seeds=(ResetSeeds*n)()
        with Native(cpu_library,n) as native:
            generator.bind(native);lib=reset.bind(native);expected_ok=np.zeros(n,dtype=np.int32)
            for i,spec in enumerate(specs):
                if 'index' in spec:
                    row=data['teams'][spec['index']];seeds[i]=ResetSeeds(seed_value(spec['seed']),(C.c_uint64*2)(seed_value(row['seed']),seed_value(row['seed'])),512)
                else:seeds[i]=ResetSeeds(seed_value(spec['seed']),(C.c_uint64*2)(*[seed_value(value)for value in spec['team_seeds']]),spec['phases']+1)
                expected_ok[i]=lib.pkn_reset_battle(C.byref(generator.rules),C.byref(reset.rules),C.byref(seeds[i]),C.byref(native.records[i]),reset.battle_moves,len(reset.battle_moves),chart.ctypes.data)
                if mode=='unfiltered':
                    counts['reset_attempts']+=1;counts['unsupported_faults']+=not bool(expected_ok[i])
                    if not expected_ok[i] and native.records[i].fault!=2:raise AssertionError('Reset fault classification changed')
                    if list(native.records[i].team_rng)!=[seed_value(row['after'])]*2:raise AssertionError('Reset generator oracle RNG changed')
            device_seeds=upload(seeds);state=torch.empty(n*C.sizeof(State),dtype=torch.uint8,device='cuda');ok=torch.empty(n,dtype=torch.int32,device='cuda')
            code=cuda.pkn_random_reset_device(C.byref(device_generator),C.byref(device_reset),device_seeds.data_ptr(),state.data_ptr(),ok.data_ptr(),n,move_data.data_ptr(),len(reset.battle_moves),chart_data.data_ptr(),stream)
            if code:raise RuntimeError(f'CUDA random reset launch: {code}')
            torch.cuda.synchronize();np.testing.assert_array_equal(ok.cpu().numpy(),expected_ok)
            if state.cpu().numpy().tobytes()!=native.snapshot():raise AssertionError(f'CPU/CUDA random reset bytes differ: {mode}')
            counts['complete_state_mask_comparisons']+=n
            if mode=='unfiltered':continue
            if not np.all(expected_ok):raise AssertionError('Previously implemented replay reset became unsupported')
            counts['battles']=n
            obs=torch.empty((n,2,700),device='cuda');masks=torch.empty((n,2,14),dtype=torch.uint8,device='cuda')
            for phase in range(-1,max(len(row['frames']) for row in reference['rows'])):
                if phase>=0:
                    actions=np.asarray([row['frames'][phase]['actions'] if phase<len(row['frames']) else [-1,-1] for row in reference['rows']],dtype=np.int32)
                    native.step(actions,reset.battle_moves,chart);device_actions=torch.from_numpy(actions).cuda()
                    code=cuda.pkn_step_device(state.data_ptr(),device_actions.data_ptr(),n,move_data.data_ptr(),len(reset.battle_moves),chart_data.data_ptr(),stream)
                    if code:raise RuntimeError(f'CUDA generated step launch: {code}')
                code=cuda.pkn_observe_device(state.data_ptr(),obs.data_ptr(),n,C.byref(device_visible),stream)
                if code:raise RuntimeError(f'CUDA generated observer launch: {code}')
                code=cuda.pkn_masks_device(state.data_ptr(),masks.data_ptr(),n,stream)
                if code:raise RuntimeError(f'CUDA generated mask launch: {code}')
                torch.cuda.synchronize()
                if state.cpu().numpy().tobytes()!=native.snapshot():raise AssertionError(f'Generated CPU/CUDA state diverged at {phase}')
                cpu_masks=native.masks();np.testing.assert_array_equal(masks.cpu().numpy(),cpu_masks)
                cpu_obs=visible.observe(native);np.testing.assert_array_equal(obs.cpu().numpy(),cpu_obs.astype(np.float32));counts['complete_state_mask_comparisons']+=n
                for i,row in enumerate(reference['rows']):
                    if phase>=len(row['frames']):continue
                    target=row['initial'] if phase<0 else row['frames'][phase]['state']
                    mismatch=team_comparison(native,target,i,cpu_masks)
                    if mismatch:raise AssertionError(f'Generated oracle mechanics case {i} phase {phase}: {mismatch}')
                    np.testing.assert_array_equal(cpu_obs[i],flat_observations(target));counts['exact_oracle_feature_values']+=1400;counts['oracle_transitions']+=phase>=0
            counts['natural_oracle_games']=sum(row['frames'][-1]['state']['ended'] for row in reference['rows'])
    return dict(full_gen9_admission=False,scope='Unfiltered reset faults plus implemented complete generated-team trajectories; no filtered production distribution',reference_sha256=hashlib.sha256(raw).hexdigest(),runtime_limits=runtime_limits,**counts)
