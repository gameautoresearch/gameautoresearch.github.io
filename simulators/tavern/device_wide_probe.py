"""Wide, history-preserving replay qualification for the selected runtime."""
import hashlib
import json
import time
import numpy as np
import torch
import taverngym as oracle
from device_env import DeviceEnv
from device_state_schema import native_episode


def qualify(out,rules):
    started=time.perf_counter();seed=12345;n=128;steps=300
    cpu=oracle.BatchEnv(n,seed=seed,controlled=[0],config=rules,threads=4)
    env=DeviceEnv(n,seed=seed,controlled=(0,),config=rules)
    natural=0;digest=hashlib.sha256()
    try:
        for step in range(steps):
            a=cpu.observe();b={k:v.cpu().numpy() for k,v in env.observe().items()}
            ids=np.flatnonzero(a['ready'])
            assert np.array_equal(a['ready'],b['ready']),('wide ready',step)
            for key in ['features','cards','action_mask']:
                assert np.array_equal(a[key][ids],b[key][ids]),('wide visible state',key,step)
                digest.update(a[key][ids].tobytes())
            actions=np.zeros(n,np.int64)
            for gi in ids:actions[gi]=cpu.game(int(gi)).bot_action(0)
            ar,ad,ai=cpu.step(actions)
            br,bd,bi=env.step(torch.from_numpy(actions).cuda())
            assert np.array_equal(ar,br.cpu().numpy()) and np.array_equal(ad,bd.cpu().numpy()),('wide results',step)
            assert all(np.array_equal(ai[k],bi[k].cpu().numpy()) for k in ai),('wide metadata',step)
            natural+=int((ai['stop_reason']==1).sum())
            if step%25==0:print(json.dumps({'wide_replay_step':step,'natural_episodes':natural}),flush=True)
        actual=[native_episode(json.loads(s),seed=seed,controlled=(0,)) for s in env.save()]
        expected=[json.loads(cpu.save(gi)) for gi in range(n)]
        if actual!=expected:
            (out/'wide-gpu.states.json').write_text(json.dumps(actual)+'\n')
            (out/'wide-cpu.states.json').write_text(json.dumps(expected)+'\n')
            raise AssertionError('wide full hidden Game/RNG mismatch')
        result={'passed':True,'n_games':n,'steps':steps,'seed':seed,'block_threads':env.block_threads,
            'natural_episodes':natural,'views_sha256':digest.hexdigest(),
            'final_state_rng_sha256':hashlib.sha256(json.dumps(actual,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
            'wall_seconds':time.perf_counter()-started}
        (out/'wide-replay.json').write_text(json.dumps(result,indent=2)+'\n')
        return result
    finally:env.close()
