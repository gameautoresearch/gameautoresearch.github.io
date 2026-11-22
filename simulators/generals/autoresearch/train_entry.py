"""Root-owned child supervisor. The host controller owns allocation time."""
from pathlib import Path
import argparse
import json
import os
import signal
import subprocess
import sys
import time

from .training_limits import drop_training_privileges as _drop_privileges, wait_monitored
from .contract import DEFAULT,ROOT,write_json,unprivileged_paths

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',default='/job/input');p.add_argument('--output',default='/job/output')
    p.add_argument('--seconds',required=True,type=float);a=p.parse_args()
    source=Path(a.input);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    os.chown(out,65534,65534);out.chmod(0o700)
    env={'PATH':os.environ['PATH'],'HOME':str(out),'TMPDIR':str(out),'PYTHONPATH':':'.join(unprivileged_paths()),
        'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','JAX_PLATFORMS':'cuda','XLA_PYTHON_CLIENT_PREALLOCATE':'false','JAX_COMPILATION_CACHE_DIR':str(out/'jax-cache')}
    # Builds are public source-bound caches. The worker can write its own cache,
    # never frozen code, host receipts, the broker ledger or grader inputs.
    cache=Path('/job/cache');cache.mkdir(parents=True,exist_ok=True)
    os.chown(cache,65534,65534);cache.chmod(0o700)
    request=json.loads((source/'request.json').read_text())
    command=[sys.executable,str(source/'train.py'),'--config',str(source/'config.json'),'--out',str(out),
        '--seed',str(request['seed']),'--seconds',str(max(0.,a.seconds-2))]
    proc=subprocess.Popen(command,cwd=source,env=env,preexec_fn=_drop_privileges,start_new_session=True)
    reason='completed';runtime={}
    try:rc=wait_monitored(proc,a.seconds,runtime)
    except subprocess.TimeoutExpired:
        reason='deadline';os.killpg(proc.pid,signal.SIGTERM)
        try:rc=proc.wait(timeout=.5)
        except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);rc=proc.wait()
    write_json('/job/supervisor.json',{'status':reason,'returncode':rc,'child_pid':proc.pid,'training_runtime':runtime,
        'scope':'diagnostic only; authoritative provider allocation timestamps are recorded outside this sandbox'})
    return 0

if __name__=='__main__':raise SystemExit(main())
