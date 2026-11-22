"""CPU qualification of generated auto-reset, rewards, final views and cursors."""
import argparse
import ctypes as C
import hashlib
import json
from pathlib import Path

import numpy as np

from .bridge import Native,State,build
from .teamgen import GeneratorTables,GenRules
from .reset import ResetTables,ResetRules,ResetSeeds,public_tables
from .random_resident import task_seeds
from .visible import PublicRules


def run(output,generator_path,constructor_path,reference_path):
    output=Path(output);output.mkdir(parents=True,exist_ok=False);library=build(output/'build')
    data=json.loads(Path(generator_path).read_bytes());metadata=json.loads(Path(constructor_path).read_bytes())['metadata'];reference=json.loads(Path(reference_path).read_bytes())
    generator=GeneratorTables(data['species'],data['moves'],data['status_moves'],data['items']);reset=ResetTables(generator,metadata,data['moves']);chart=np.asarray(reference['chart'],dtype=np.int8);visible=public_tables(reset,chart)
    schedules=[]
    for i,limit in enumerate((2,3,2,1)):
        schedules.append([task_seeds(reference['specs'][(i+j)%len(reference['specs'])]['master'],limit,reference['specs'][(i+j)%len(reference['specs'])]['swapped'])for j in range(3)])
    # Locate a negative fixture only; no experiment distribution is filtered.
    with Native(library)as search:
        generator.bind(search);lib=reset.bind(search)
        for master in range(940000,941024):
            candidate=task_seeds(master,2)
            if not lib.pkn_reset_battle(C.byref(generator.rules),C.byref(reset.rules),C.byref(candidate),C.byref(search.records[0]),reset.battle_moves,len(reset.battle_moves),chart.ctypes.data):
                if search.records[0].fault!=2:raise AssertionError('Unexpected initial reset fault')
                unsupported=candidate;break
        else:raise AssertionError('No missing-mechanics negative fixture remains')
    schedules[2][1]=unsupported;n=len(schedules);length=3;schedule=(ResetSeeds*(n*length))(*[row for game in schedules for row in game]);episodes=np.zeros(n,dtype=np.int32)
    control_episodes=episodes.copy();boundaries=0;resets=0;rewards_checked=0;final_values=0
    signature=hashlib.sha256(bytes(schedule)).hexdigest()
    with Native(library,n,2)as env,Native(library,n)as control:
        for arena in (env,control):
            generator.bind(arena);lib=reset.bind(arena)
            for i in range(n):lib.pkn_reset_battle(C.byref(generator.rules),C.byref(reset.rules),C.byref(schedule[i*length]),C.byref(arena.records[i]),reset.battle_moves,len(reset.battle_moves),chart.ctypes.data)
        env.lib.pkn_random_env_step.argtypes=[C.c_void_p,C.c_void_p,C.c_void_p,C.c_int,C.c_void_p,C.POINTER(GenRules),C.POINTER(ResetRules),C.c_void_p,C.c_int,C.c_void_p,C.POINTER(PublicRules)]+[C.c_void_p]*4+[C.c_double]
        obs=np.zeros((n,2,700),dtype=np.float32);final=np.zeros_like(obs);rewards=np.empty((n,2),dtype=np.float32);flags=np.empty((n,3),dtype=np.int32)
        checkpoints=[];histories=[]
        def step(phase):
            nonlocal boundaries,resets,rewards_checked,final_values
            masks=control.masks();actions=np.full((n,2),-1,dtype=np.int32)
            for i in range(n):
                for side in range(2):
                    legal=np.flatnonzero(masks[i,side]);moves=legal[legal<4]
                    if len(legal):actions[i,side]=int((moves if len(moves)else legal)[phase%(len(moves)or len(legal))])
            before=visible.observe(control)[:,:,439];old_faults=[record.fault for record in control.records]
            control.step(actions,reset.battle_moves,chart);expected_obs=visible.observe(control);expected_rewards=np.zeros((n,2),dtype=np.float32);expected_flags=np.zeros((n,3),dtype=np.int32)
            for i,record in enumerate(control.records):
                if old_faults[i]:expected_flags[i,2]=record.fault;continue
                expected_flags[i]=[record.terminated,record.truncated,record.fault]
                expected_rewards[i]=.1*(expected_obs[i,:,439]-before[i])
                if record.terminated and record.winner>=0:expected_rewards[i,record.winner]+=1;expected_rewards[i,1-record.winner]-=1
                if record.truncated:
                    final_values+=1400
                if (record.terminated or record.truncated)and not record.fault:
                    boundaries+=1;control_episodes[i]+=1
                    if control_episodes[i]>=length:record.fault=4
                    else:
                        resets+=1;control.lib.pkn_reset_battle(C.byref(generator.rules),C.byref(reset.rules),C.byref(schedule[i*length+int(control_episodes[i])]),C.byref(record),reset.battle_moves,len(reset.battle_moves),chart.ctypes.data)
                    expected_flags[i,2]=record.fault
            env.lib.pkn_random_env_step(env.ptr,actions.ctypes.data,C.addressof(reset.battle_moves),len(reset.battle_moves),chart.ctypes.data,C.byref(generator.rules),C.byref(reset.rules),C.addressof(schedule),length,episodes.ctypes.data,C.byref(visible.rules),obs.ctypes.data,final.ctypes.data,rewards.ctypes.data,flags.ctypes.data,.1)
            if env.snapshot()!=control.snapshot():raise AssertionError(f'Generated auto-reset complete state mismatch: {phase}')
            np.testing.assert_array_equal(episodes,control_episodes);np.testing.assert_array_equal(flags,expected_flags);np.testing.assert_array_equal(rewards,expected_rewards);np.testing.assert_array_equal(obs,visible.observe(control).astype(np.float32))
            selected=flags[:,1].astype(bool);np.testing.assert_array_equal(final[selected],expected_obs[selected].astype(np.float32));rewards_checked+=n*2
            histories.append(dict(states=env.snapshot(),episodes=episodes.copy(),obs=obs.copy(),flags=flags.copy(),rewards=rewards.copy()))
        for phase in range(10):
            step(phase)
            if phase==2:checkpoint=(env.snapshot(),episodes.copy(),control.snapshot(),control_episodes.copy())
        expected=histories[3:];env.restore(checkpoint[0]);episodes[:]=checkpoint[1];control.restore(checkpoint[2]);control_episodes[:]=checkpoint[3];histories=[]
        for phase in range(3,10):step(phase)
        for a,b in zip(histories,expected):
            if a['states']!=b['states']:raise AssertionError('Generated state continuation changed')
            for key in ('episodes','obs','flags','rewards'):np.testing.assert_array_equal(a[key],b[key])
    result=dict(full_gen9_admission=False,scope='Development random auto-reset lifecycle, including explicit unsupported and schedule-exhaustion faults',environments=n,schedule_sha256=signature,reset_boundaries=boundaries,native_generated_resets=resets,rewards_checked=rewards_checked,final_observation_values_checked=final_values,exact_state_and_cursor_resume=True,unsupported_reset_fault=2,schedule_exhaustion_fault=4)
    (output/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--generators',required=True);p.add_argument('--constructors',required=True);p.add_argument('--reference',required=True);a=p.parse_args();run(a.output,a.generators,a.constructors,a.reference)
