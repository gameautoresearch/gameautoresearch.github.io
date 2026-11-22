#!/usr/bin/env python3
"""Bounded exact update-boundary resume check on the same immutable source/recipe."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--config', type=Path, help='explicit recipe including optimization flags')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    here = Path(__file__).resolve().parent
    config=args.config or here/'recipes/starter-local.json'
    if args.device.startswith('cuda') and not args.config:
        c=json.loads(config.read_text());c.update(device=args.device,native_threads=4,torch_threads=1,worker_backend='pooled',fast_inference=True,skip_finished_lobbies=True,defer_learner_metrics=True)
        c['machine'].update(label='Modal-L4-fixed-four-core-resume',attached_cpu_threads=4,attached_ram_gib=16,gpu_model='NVIDIA L4')
        config=args.out/'cuda-recipe.json';config.write_text(json.dumps(c,indent=2)+'\n')
    for name, extra in [('whole', []), ('resumed', ['--resume', str(args.out/'whole/opponent-000002.pt')])]:
        with (args.out/f'{name}.log').open('w') as log:
            subprocess.run([sys.executable, str(here/'run.py'), 'train', '--config', str(config),
                            '--out', str(args.out/name), *extra], stdout=log, stderr=subprocess.STDOUT, check=True)
    import torch
    import numpy as np
    a = torch.load(args.out/'whole/latest.pt', map_location='cpu', weights_only=False)
    b = torch.load(args.out/'resumed/latest.pt', map_location='cpu', weights_only=False)
    def same(x,y):
        if isinstance(x,np.ndarray): return np.array_equal(x,y)
        if isinstance(x,torch.Tensor): return torch.equal(x,y)
        if isinstance(x,dict): return x.keys()==y.keys() and all(same(x[k],y[k]) for k in x)
        if isinstance(x,(list,tuple)): return len(x)==len(y) and all(same(u,v) for u,v in zip(x,y))
        return x == y
    checks = {k: same(a[k],b[k]) for k in ['policy','optimizer','torch_rng','cuda_rng','mps_rng','numpy_rng','python_rng','env_states','environment_cohort_seed','next_cohort_seed','stats']}
    aa = [json.loads(s) for s in (args.out/'whole/metrics.jsonl').read_text().splitlines()][2:]
    bb = [json.loads(s) for s in (args.out/'resumed/metrics.jsonl').read_text().splitlines()]
    checks['episodes_and_learning'] = all(all(x[k]==y[k] for k in ['cohort_seed','counts','learning','episodes']) for x,y in zip(aa,bb)) and len(aa)==len(bb)
    whole=json.loads((args.out/'whole/accounting.json').read_text())
    resumed=json.loads((args.out/'resumed/accounting.json').read_text())
    checks['prior_full_session_charged'] = resumed['prior_allocated_worker_seconds'] >= whole['total_allocated_worker_seconds']
    result={'checks':checks,'passed':all(checks.values()),'outer_allocated_worker_seconds':time.perf_counter()-started,'cuda_measured':args.device.startswith('cuda')}
    (args.out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    if not result['passed']: raise RuntimeError('resume mismatch')


if __name__ == '__main__':
    main()
