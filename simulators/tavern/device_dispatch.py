"""Source-bound CUDA launch-width sweep followed by full selected-mode qualification."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

from device_env import DeviceEnv
from device_simulation import replay


def qualify(out):
    start=time.perf_counter()
    reference=Path('/results/native-state-schema-v29')
    trace=out/'scripted-128.json';gold=out/'native-reference.json'
    shutil.copyfile(reference/'scripted-128.json',trace)
    shutil.copyfile(reference/'n128-original-validation.json',gold)
    expected=json.loads(gold.read_text());rows=[]
    # A separately reported common warmup initializes immutable caches. Every
    # mode gets the same warm state; the outer allocation charges all warmup.
    env=DeviceEnv(1,seed=12345,controlled=(0,),config=json.loads(trace.read_text())['rules'])
    resources=env.kernel_resources();env.close();warmup=time.perf_counter()-start
    for width in [1,2,4,8,16,32]:
        os.environ['TAVERN_DEVICE_BLOCK_THREADS']=str(width)
        print(json.dumps({'stage':'dispatch_validation_start','block_threads':width}),flush=True)
        row=replay(trace,verify=True,state_out=out/f'width-{width}.states.json')
        for field in ['trace_sha256','decisions','natural_episodes','views_masks_results_sha256','final_state_rng_sha256']:
            if row[field]!=expected[field]:raise AssertionError(('dispatch mode differs from original',width,field))
        row['block_threads']=width;rows.append(row)
        (out/'dispatch-progress.json').write_text(json.dumps(rows,indent=2)+'\n')
        print(json.dumps({'stage':'dispatch_validation_passed','block_threads':width,
                         'diagnostic_protocol_seconds':row['complete_protocol_seconds']}),flush=True)
    best=min(rows,key=lambda r:r['complete_protocol_seconds'])['block_threads']
    os.environ['TAVERN_DEVICE_BLOCK_THREADS']=str(best)
    measured=replay(trace,verify=False)
    if measured['final_state_rng_sha256']!=expected['final_state_rng_sha256']:
        raise AssertionError('selected-mode timed replay state changed')
    (out/'selected-width-timing.json').write_text(json.dumps(measured,indent=2)+'\n')
    # Re-run the full game/adversarial/GAE/PPO/resume gate on the actual selected
    # Python runtime and width. No module or rule-source substitutions occur.
    print(json.dumps({'stage':'full_selected_mode_qualification','block_threads':best}),flush=True)
    nested=out/'selected-mode'
    argv=sys.argv;sys.argv=['device_probe.py','--out',str(nested)]
    try:
        from device_probe import main
        main()
    finally:sys.argv=argv
    proof=json.loads((nested/'differential.json').read_text())
    if proof['runtime_options']['block_threads']!=best:raise AssertionError('selected runtime binding changed')
    for path in [nested/'differential.json',*nested.glob('tavern*.ptx'),*nested.glob('tavern*.cubin')]:
        path.replace(out/path.name)
    return {'rows':rows,'selected_block_threads':best,'selected_timing':measured,
        'warmup_seconds':warmup,'kernel_resources':resources,'wall_seconds':time.perf_counter()-start,
        'source_trace_sha256':hashlib.sha256(trace.read_bytes()).hexdigest(),
        'full_selected_runtime_qualified':True,
        'scope':'Every launch width preserves all frozen original observations/masks/outcomes/full game/RNG. Sweep timings include identical validation overhead and warm caches; selected timing excludes validation. Cold setup/full allocation and original/native full-process comparisons remain separate.'}
