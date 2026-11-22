"""Bounded full-rule batch scaling against the strongest native CPU transport.

Every arm executes the same complete scripted trace and checks complete final
game/RNG state. The 128-game runs additionally check every player-visible row.
No faulty launch mode participates in the throughput comparison.
"""
import json
import os
from pathlib import Path
import time

from device_simulation import replay as device_replay
from original_simulation import record, replay as native_replay


def qualify(out,sizes):
    rows=[];started=time.perf_counter()
    for n in sizes:
        if n not in [128,512,1024,2048,4096]:raise ValueError('bounded batch scaling only')
        trace=out/f'scaling-{n}.json';record(trace,n,300)
        cpu=native_replay(trace,compact=True,verify=n==128)
        (out/f'scaling-{n}-native.json').write_text(json.dumps(cpu,indent=2)+'\n')
        for width in [1,32]:
            os.environ['TAVERN_DEVICE_BLOCK_THREADS']=str(width)
            print(json.dumps({'stage':'scaling_start','n_games':n,'block_threads':width}),flush=True)
            gpu=device_replay(trace,verify=n==128)
            fields=['trace_sha256','decisions','natural_episodes','final_state_rng_sha256']
            if n==128:fields.append('views_masks_results_sha256')
            exact=all(cpu[k]==gpu[k] for k in fields)
            row={'n_games':n,'block_threads':width,'passed':exact,'cpu':cpu,'gpu':gpu,
                 'complete_protocol_speedup':cpu['complete_protocol_seconds']/gpu['complete_protocol_seconds']}
            rows.append(row)
            (out/'scaling-progress.json').write_text(json.dumps(rows,indent=2)+'\n')
            print(json.dumps({'stage':'scaling_complete','n_games':n,'block_threads':width,
                'passed':exact,'native_seconds':cpu['complete_protocol_seconds'],
                'gpu_seconds':gpu['complete_protocol_seconds'],
                'speedup':row['complete_protocol_speedup']}),flush=True)
            if not exact:raise AssertionError(('full-rule scaling differs from independent oracle',n,width))
    return {'rows':rows,'all_cases_passed':all(row['passed'] for row in rows),
            'wall_seconds':time.perf_counter()-started,
            'scope':'Same B200/four physical CPU cores/16GiB. Full initialization, observe/step, transfers and checkpoint IO; GPU and native keep all eight-player rules. Independent original-source and neural trainer comparisons remain separate.'}
