"""Locked CPU worker scheduling comparison with exact environment trajectories."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import platform
import statistics
import time

import numpy as np
import torch

from .bridge import build
from .cpu_env import CPURandomBattles
from .random_resident import task_seeds


def run(output,generators,constructors,reference,environments=(96,384),phases=256,repeats=7):
    root=Path(__file__).resolve().parents[3]
    with(root/'evidence/local-performance.lock').open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        output=Path(output);output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
        raw=Path(reference).read_bytes();data=json.loads(raw);chunks=(32,16,8,4,0)
        libraries={chunk:build(output/f'build-{chunk}',cpu_chunk=chunk)for chunk in chunks};rows=[];audits=[]
        for n in environments:
            specs=data['specs'];schedule=[[task_seeds(specs[(i+j)%len(specs)]['master'],128,specs[(i+j)%len(specs)]['swapped'])for j in range(64)]for i in range(n)]
            envs={};initial={};actions=[];expected=[]
            try:
                for chunk in chunks:
                    envs[chunk]=CPURandomBattles(libraries[chunk],generators,constructors,data['chart'],schedule,threads=3,developer_scope='CPU worker chunk diagnostic; no complete-format admission')
                    envs[chunk].allocate_rollout(1);initial[chunk]=envs[chunk].snapshot()
                def signature(env):
                    digest=hashlib.sha256(env.arena.snapshot())
                    for value in (env.episodes,env.obs.numpy(),env.flags.numpy(),env.rewards.numpy(),env.final_obs[0,env.flags[0,:,1].bool()].numpy()):digest.update(value.tobytes())
                    return digest.hexdigest()
                baseline=envs[32]
                # Freeze actual legal actions once, including unavailable-choice
                # decisions, truncations and real native generation/resets.
                for phase in range(phases):
                    masks=baseline.obs.numpy()[...,-14:]>0
                    chosen=np.where(masks.any(axis=-1),masks.argmax(axis=-1),-1).astype(np.int32)
                    for side in range(2):
                        switch=masks[:,side,9:].any(axis=-1)&(phase%13==side)
                        chosen[switch,side]=9+masks[switch,side,9:].argmax(axis=-1)
                    action=torch.from_numpy(chosen.copy());actions.append(action)
                    baseline.step(action,0)
                    if baseline.flags[...,2].any():raise AssertionError('Diagnostic schedule encountered a simulator fault')
                    expected.append(signature(baseline))
                for chunk,env in envs.items():
                    env.restore(initial[chunk])
                    for phase,action in enumerate(actions):
                        env.step(action,0)
                        if signature(env)!=expected[phase]:raise AssertionError(('Chunk changed trajectory',n,chunk,phase))
                audits.append(dict(environments=n,exact_phases=phases,variants=len(chunks),state_observations_flags_rewards_cursors_and_boundary_views_exact=True))
                for repeat in range(repeats):
                    order=chunks[repeat%len(chunks):]+chunks[:repeat%len(chunks)]
                    if repeat%2:order=tuple(reversed(order))
                    for chunk in order:
                        env=envs[chunk];env.restore(initial[chunk])
                        started=time.perf_counter()
                        for action in actions:env.step(action,0)
                        seconds=time.perf_counter()-started
                        rows.append(dict(environments=n,repeat=repeat,chunk=chunk,seconds=seconds,decisions=sum(int((a>=0).sum())for a in actions),final_signature=signature(env)))
                    print(f'WORKER_PROFILE n={n} repeat={repeat} complete',flush=True)
            finally:
                for env in envs.values():env.close()
        medians={str(n):{str(chunk):statistics.median(row['seconds']for row in rows if row['environments']==n and row['chunk']==chunk)for chunk in chunks}for n in environments}
        result=dict(full_gen9_admission=False,scope='Diagnostic implemented-team CPU simulation, generation/reset and fused float32 observation writes; excludes inference and PPO; no full-format speedup',
                    hardware=platform.platform(),cpu_workers=3,main_threads=1,phases=phases,repeats=repeats,
                    reference_sha256=hashlib.sha256(raw).hexdigest(),median_seconds=medians,audits=audits,rows=rows)
        (output/'results.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps({k:v for k,v in result.items()if k!='rows'},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);parser.add_argument('--generators',required=True);parser.add_argument('--constructors',required=True);parser.add_argument('--reference',required=True)
    args=parser.parse_args();run(args.output,args.generators,args.constructors,args.reference)
