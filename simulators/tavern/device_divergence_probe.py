"""Recorded independent two-lobby regression for divergent refresh/buy execution."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import taverngym as oracle
from device_env import DeviceEnv
from device_state_schema import native_episode


def qualify_fixture(out,source):
    raw=source.read_bytes();fixture=json.loads(raw);rows=[]
    if oracle.build_info()['source_hash']!=fixture['source']['source_hash']:
        raise RuntimeError('independent regression oracle source changed')
    for width in [1,2,4,8,16,32]:
        cpu=oracle.BatchEnv(2,seed=fixture['seed'],controlled=[0],config=fixture['rules'],threads=1)
        for gi,state in enumerate(fixture['before']):cpu.load(gi,json.dumps(state))
        env=DeviceEnv(2,seed=fixture['seed'],controlled=(0,),config=fixture['rules'],block_threads=width)
        try:
            imported=[]
            for state in fixture['before']:
                imported.append(json.dumps(dict(state,seed=fixture['seed'],controlled=[0]*8,
                    k=1,finished=state['finished']+[False]*7)))
            env.load(imported)
            x=cpu.observe();y={k:v.cpu().numpy() for k,v in env.observe().items()}
            assert all(np.array_equal(x[k],y[k]) for k in x),('divergent fixture initial views',width)
            actions=np.array(fixture['actions'],np.int64)
            ar,ad,ai=cpu.step(actions);br,bd,bi=env.step(torch.from_numpy(actions).cuda())
            assert np.array_equal(ar,br.cpu().numpy()) and np.array_equal(ad,bd.cpu().numpy()),width
            assert all(np.array_equal(ai[k],bi[k].cpu().numpy()) for k in ai),width
            actual=[native_episode(json.loads(s),seed=fixture['seed'],controlled=(0,)) for s in env.save()]
            expected=[json.loads(cpu.save(gi)) for gi in range(2)]
            assert expected==fixture['after'],'independent recorded fixture changed'
            if actual!=expected:
                (out/f'{source.stem}-width-{width}-gpu.json').write_text(json.dumps(actual)+'\n')
                (out/f'{source.stem}-width-{width}-cpu.json').write_text(json.dumps(expected)+'\n')
                raise AssertionError(('divergent refresh/buy full state/RNG',width))
            rows.append({'block_threads':width,'initial_views_masks_outcomes_final_state_rng_exact':True})
        finally:env.close()
    result={'fixture_sha256':hashlib.sha256(raw).hexdigest(),'source_trace_sha256':fixture['trace_sha256'],
            'source_game_ids':fixture['source_game_ids'],'step':fixture['step'],'checks':rows}
    return result


def qualify(out):
    sources=sorted((Path(__file__).parent/'device/fixtures').glob('dispatch-step*.json'))
    if not sources:raise RuntimeError('independent divergence fixtures missing')
    result={'fixtures':[qualify_fixture(out,source) for source in sources]}
    (out/'divergent-control-regression.json').write_text(json.dumps(result,indent=2)+'\n')
    return result
