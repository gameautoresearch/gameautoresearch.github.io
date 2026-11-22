"""Locked matched CPU simulation/encoding profile on retained diagnostic seeds.

The selected complete battles exercise implemented mechanics only. This is not a
full-format distribution, admission claim, PPO benchmark or predicted speedup.
"""
import argparse
import ctypes as C
import fcntl
import hashlib
import json
from pathlib import Path
import platform
import statistics
import time

import numpy as np
import torch

from ..env import Vector
from .bridge import build
from .cpu_env import CPURandomBattles
from .random_resident import task_seeds
from .reset import ResetSeeds


def run(output,generator_path,constructor_path,reference_path,n=96,repeats=5,library_path=None):
    if n<3 or n%3 or repeats<3:raise ValueError('Use a positive multiple of three environments and at least three repetitions')
    root=Path(__file__).resolve().parents[3]
    with(root/'evidence/local-performance.lock').open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        output=Path(output);output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
        tick=time.perf_counter();library=Path(library_path)if library_path else build(output/'build');compile_seconds=time.perf_counter()-tick
        raw=Path(reference_path).read_bytes();reference=json.loads(raw);indices=[i%len(reference['specs'])for i in range(n)]
        specs=[reference['specs'][i]for i in indices];trajectories=[reference['rows'][i]for i in indices]
        schedule=[[task_seeds(s['master'],s['phases']+1,s['swapped'])]for s in specs]
        actions=[np.asarray([r['frames'][t]['actions']if t<len(r['frames'])else[-1,-1]for r in trajectories],dtype=np.int32)for t in range(max(len(r['frames'])for r in trajectories))]
        js_actions=[[[int(a)if a>=0 else None for a in row]for row in phase]for phase in actions]
        samples=sum(int((phase>=0).sum())for phase in actions);rows=[];exact_values=0
        tick=time.perf_counter();native=CPURandomBattles(library,generator_path,constructor_path,reference['chart'],schedule,threads=3,developer_scope='Matched CPU profile on retained diagnostic independent-team games; no full-format admission');native_setup=time.perf_counter()-tick
        tick=time.perf_counter();js=Vector(3,n//3,packed=True,binary=True,shared=True,lazy=True);js_setup=time.perf_counter()-tick
        def reset_native():
            lib=native.arena.lib
            for i,row in enumerate(schedule):lib.pkn_reset_battle(C.byref(native.generator.rules),C.byref(native.constructor.rules),C.byref(row[0]),C.byref(native.arena.records[i]),native.moves,len(native.moves),native.chart.ctypes.data)
            native.refresh_observation()
        def reset_js():return js.reset([s['master']for s in specs],specs[0]['phases']+1,[s['swapped']for s in specs])
        def flatten(states):return np.asarray([[obs.flat for obs in state['observations']]for state in states],dtype=np.float32)
        try:
            # Untimed paired audit: every public feature and input mask, every
            # decision phase, matching natural outcomes and no faults.
            reset_native();states=reset_js();np.testing.assert_array_equal(native.obs.numpy(),flatten(states));exact_values+=n*1400
            for phase,chosen in enumerate(actions):
                native.arena.step(chosen,native.moves,native.chart);native.refresh_observation();states=js.step(js_actions[phase]);np.testing.assert_array_equal(native.obs.numpy(),flatten(states));exact_values+=n*1400
            for i,state in enumerate(states):
                record=native.arena.records[i]
                if not state['terminated']or state['truncated']or not record.terminated or record.truncated or record.fault:raise AssertionError('Profile did not finish naturally')
                if state['winner']!=('p1'if record.winner==0 else'p2'if record.winner==1 else None):raise AssertionError('Winner differs')
            for repeat in range(repeats):
                order=('showdown_cpu','native_cpp_cpu')if repeat%2==0 else('native_cpp_cpu','showdown_cpu')
                for backend in order:
                    started=time.perf_counter()
                    if backend=='showdown_cpu':
                        states=reset_js();flatten(states)
                        for phase in range(len(actions)):states=js.step(js_actions[phase]);flatten(states)
                    else:
                        reset_native()
                        for chosen in actions:native.arena.step(chosen,native.moves,native.chart);native.refresh_observation()
                    seconds=time.perf_counter()-started;rows.append(dict(repeat=repeat,backend=backend,seconds=seconds,decision_samples=samples,samples_per_second=samples/seconds))
                    print(f'CPU_PROFILE {repeat} {backend} {seconds:.6f}s',flush=True)
            medians={backend:statistics.median(r['seconds']for r in rows if r['backend']==backend)for backend in ('showdown_cpu','native_cpp_cpu')}
            result=dict(full_gen9_admission=False,scope='Selected previously qualified independent-team diagnostic battles; simulation+generation+encoding+host conversion only; excludes PPO and compilation; not full-format speedup',hardware=platform.platform(),cpu_workers=3,main_threads=1,environments=n,decision_samples_per_repeat=samples,exact_public_feature_values=exact_values,natural_games_per_repeat=n,compile_seconds=compile_seconds,one_time_native_setup_seconds=native_setup,one_time_js_setup_seconds=js_setup,reference_sha256=hashlib.sha256(raw).hexdigest(),source_signature=native.source_signature,median_seconds=medians,matched_component_speedup=medians['showdown_cpu']/medians['native_cpp_cpu'],rows=rows)
            (output/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items()if k!='rows'},indent=2))
        finally:native.close();js.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--generators',required=True);p.add_argument('--constructors',required=True);p.add_argument('--reference',required=True);p.add_argument('--n',type=int,default=96);p.add_argument('--repeats',type=int,default=5);p.add_argument('--library');a=p.parse_args();run(a.output,a.generators,a.constructors,a.reference,a.n,a.repeats,a.library)
