#!/usr/bin/env python3
"""Paired candidate improvement, bootstrapped by deal across roles and panels."""
import argparse
import json
from pathlib import Path

import numpy as np


def compare(before, after, replicates=4000, seed=83):
    def keyed(rows):
        result = {(r['opponent_id'], r['deal_sha256'], r['candidate_role']): r for r in rows}
        if len(result) != len(rows):
            raise ValueError('Duplicate evaluation schedule rows')
        return result
    a, b = keyed(before), keyed(after)
    if a.keys() != b.keys() or not a:
        raise ValueError('Evaluation schedules must be nonempty and exactly paired')
    rng = np.random.default_rng(seed)
    groups = {}
    for role in ('landlord', 'landlord_down', 'landlord_up', 'role_balanced_mean'):
        differences = {}
        for key in a:
            if role == 'role_balanced_mean' or key[2] == role:
                differences.setdefault(key[1], []).append(float(b[key]['won']) - float(a[key]['won']))
        # Keep the dependence of all role/opponent games sharing a deal.
        values = np.array([np.mean(v) for _, v in sorted(differences.items())])
        samples = values[rng.integers(len(values), size=(replicates,len(values)))].mean(axis=1)
        groups[role] = {'paired_win_rate_delta': float(values.mean()),
                        'paired_deal_bootstrap95': np.quantile(samples,[.025,.975]).tolist(),
                        'unique_deals':len(values)}
    return {'groups':groups,'bootstrap_replicates':replicates,'bootstrap_seed':seed,
            'interpretation':'functional local learning evidence; a CI crossing zero does not establish improvement'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('before',type=Path)
    parser.add_argument('after',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args()
    result = compare(json.loads(args.before.read_text()),json.loads(args.after.read_text()))
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
