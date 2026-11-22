"""Interleaved exact native replay comparisons against a frozen Python/native package."""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--trace',type=Path,required=True)
    p.add_argument('--baseline-package',type=Path);p.add_argument('--worker',action='store_true')
    p.add_argument('--verify',action='store_true');p.add_argument('--trials',type=int,default=5)
    a=p.parse_args()
    if a.worker:
        from original_simulation import replay
        rows=[replay(a.trace,compact=True,verify=a.verify) for _ in range(1 if a.verify else 3)]
        a.out.write_text(json.dumps(rows,indent=2)+'\n');return
    if not a.baseline_package:raise ValueError('frozen baseline package required')
    a.out.mkdir(parents=True,exist_ok=False)
    import fcntl
    rows=[];proof=[]
    with (ROOT/'evidence/local-performance.lock').open('a+') as lock:
        print('waiting for local performance lock',flush=True);fcntl.flock(lock,fcntl.LOCK_EX)
        for trial in range(-1,a.trials):
            for arm in (['baseline','candidate'] if trial%2 else ['candidate','baseline']):
                dest=a.out/f'{arm}-{trial}.json'
                command=[sys.executable,str(Path(__file__)), '--worker','--out',str(dest.resolve()),'--trace',str(a.trace.resolve())]
                if trial==-1:command+=['--verify']
                path=a.baseline_package if arm=='baseline' else ROOT/'candidates/AutoTavern/python'
                started=time.perf_counter()
                subprocess.run(command,check=True,env=dict(os.environ,PYTHONPATH=str(path.resolve()),
                    TAVERNGYM_WORKERS='pooled',TAVERNGYM_SCHEDULER='dynamic'),timeout=120)
                results=json.loads(dest.read_text())
                if trial==-1:proof.append(results[0])
                else:rows.append({'arm':arm,'trial':trial,'process_seconds':time.perf_counter()-started,'rows':results})
        fields=['trace_sha256','decisions','natural_episodes','views_masks_results_sha256','final_state_rng_sha256']
        checks={key:proof[0][key]==proof[1][key] for key in fields}
        if not all(checks.values()):raise AssertionError(checks)
    metrics={}
    for arm in ['baseline','candidate']:
        selected=[r for row in rows if row['arm']==arm for r in row['rows']]
        metrics[arm]={key:statistics.median(r[key] for r in selected) for key in
            ['complete_protocol_seconds','observation_seconds','simulation_step_seconds']}
        metrics[arm]['observe_step_seconds']=statistics.median(r['observation_seconds']+r['simulation_step_seconds'] for r in selected)
        metrics[arm]['source']=selected[0]['source']
    speedup=metrics['baseline']['observe_step_seconds']/metrics['candidate']['observe_step_seconds']
    result={'checks':checks,'metrics':metrics,'observe_step_speedup':speedup,'rows':rows,
            'scope':'Local CPU, four native workers, identical full-eight-player trace. Interleaved; exclusive local performance lock. No GPU throughput claim.'}
    (a.out/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'checks':checks,'metrics':metrics,'observe_step_speedup':speedup}),flush=True)


if __name__=='__main__':main()
