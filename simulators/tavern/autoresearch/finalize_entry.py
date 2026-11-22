"""Trusted packaging of a trainer that ended before it wrote its summary.

The host runs this after ending the trainer process: the researcher stopped the run, or the run was cut off
at its time limit. It reads only what train_entry.py last exported whole (weights.npz, identity.json,
metrics.json, export.json) and writes the summary.json a finished run writes, marked `ended_early`. Nothing
is trained, executed or repaired here: a run without one complete export stays a failure.
"""
import argparse,json
from pathlib import Path
import numpy as np
from contract import sha,write

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 if (a.out/'summary.json').exists():raise ValueError('This run already wrote its own summary')
 missing=[f for f in ('weights.npz','identity.json','metrics.json','export.json') if not (a.out/f).is_file()]
 if missing:raise ValueError('No complete export to package; missing '+', '.join(missing))
 history=json.loads((a.out/'metrics.json').read_text());marker=json.loads((a.out/'export.json').read_text())
 if not history:raise ValueError('No completed PPO update')
 with np.load(a.out/'weights.npz',allow_pickle=False) as arrays:
  values=[arrays[k] for k in arrays.files]
 if not all(np.isfinite(v).all() for v in values) or sum(v.size for v in values)>500000:raise ValueError('Nonfinite/oversized export')
 summary=dict(updates=history[-1]['update'],samples=sum(x['learning']['samples'] for x in history),finite_losses=all(np.isfinite(x['learning']['loss']) for x in history),changed_weights=any(x['learning']['parameter_delta_l2']>0 for x in history),accounted_seconds=marker['accounted_seconds'],device=json.loads((a.out/'identity.json').read_text())['config']['device'],identity_sha256=sha(a.out/'identity.json'),weights_sha256=sha(a.out/'weights.npz'),ended_early=True)
 write(a.out/'summary.json',summary)
 print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
