"""Memory diagnostics on a frozen full-rule replay, never performance evidence."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def qualify(out):
    tool=Path('/usr/local/cuda-13.3/compute-sanitizer/compute-sanitizer')
    if not tool.is_file():raise RuntimeError('required CUDA memory diagnostic missing')
    trace=json.loads(Path('/results/native-state-schema-v29/scripted-128.json').read_text())
    trace['actions']=trace['actions'][:34];trace['steps']=34
    # Regenerate the completion count with the independent CPU oracle rather
    # than altering any GPU state or suppressing a mismatch.
    import numpy as np
    import taverngym as oracle
    cpu=oracle.BatchEnv(trace['n_games'],seed=trace['seed'],controlled=[0],config=trace['rules'],threads=4)
    natural=0
    for action in trace['actions']:
        _,d,_=cpu.step(np.asarray(action,np.int64));natural+=int(d.sum())
    states=[json.loads(cpu.save(gi)) for gi in range(trace['n_games'])]
    expected=hashlib.sha256(json.dumps(states,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    del states,cpu
    trace['natural_episodes']=natural;trace['decisions']=34*trace['n_games']
    source=out/'sanitizer-trace.json';source.write_text(json.dumps(trace)+'\n')
    result=out/'sanitizer-replay.json';started=time.perf_counter()
    check=os.environ.get('TAVERN_SANITIZER_TOOL','memcheck')
    if check not in ['memcheck','initcheck']:raise ValueError('unsupported memory diagnostic')
    command=[str(tool),'--tool',check,'--error-exitcode','91','--log-file',str(out/'memcheck.log'),
        sys.executable,'-u',str(Path(__file__).with_name('device_simulation.py')),
        '--trace',str(source),'--verify','--out',str(result)]
    with (out/'replay.log').open('x') as log:
        proc=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=3200,
            env=dict(os.environ,TAVERN_DEVICE_BLOCK_THREADS='2',TAVERN_DEVICE_TRACE_KERNELS='1'))
    actual=json.loads(result.read_text())['final_state_rng_sha256'] if result.exists() else None
    data={'returncode':proc.returncode,'wall_seconds':time.perf_counter()-started,
          'diagnostic_tool':check,'memory_check_passed':proc.returncode==0,'state_exact':actual==expected,
          'all_cases_passed':proc.returncode==0 and actual==expected,
          'complete_replay':result.exists(),'scope':'Full-rule diagnostic of known width2 hidden-state mismatch; no throughput or qualification claim.'}
    print(json.dumps(data),flush=True)
    return data
