"""Exact imported optimizer/RNG/population resume and dense/sparse equivalence."""
import copy,json
from pathlib import Path
import numpy as np
import torch
from .common import write
from .lineage import parity
from .train import run,rng_state,restore_rng
from ..validate import equal
from .. import collection
from .population import schedule

def validate(config,inputs,root,seconds=120):
 root=Path(root);root.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1);c=copy.deepcopy(config);c.update(n_games=2,population_snapshot_every=1)
 parity(inputs,root/'import-parity.json',device=c['device'])
 full=run(c,root/'full',inputs,seconds,limit_updates=4)
 half=run(c,root/'half',inputs,seconds,limit_updates=2)
 resumed=run(c,root/'resumed',inputs,seconds,resume=half,limit_updates=2)
 a=torch.load(full,map_location='cpu',weights_only=False);b=torch.load(resumed,map_location='cpu',weights_only=False)
 keys=['policy','optimizer','population','rng','history','env_states','next_cohort_seed','update','binding','config','lineage']
 results={k:equal(a[k],b[k]) for k in keys}
 if not all(results.values()):raise ValueError('Resume differs: '+str(results))
 b['binding']['native_source_hash']='old-identity';torch.save(b,root/'bad.pt')
 try:run(c,root/'must-reject',inputs,seconds,resume=root/'bad.pt',limit_updates=1)
 except ValueError as exc:results['old_identity_rejected']='Exact source/config/identity' in str(exc)
 else:raise ValueError('Old identity admitted')
 # Imported optimizer tensors are preserved before the first new update.
 original=torch.load(Path(inputs)/'parent-resume.pt',map_location='cpu',weights_only=False)
 imported=torch.load(root/'full/initial.pt',map_location='cpu',weights_only=False)
 results['exact_parent_policy']=equal(original['policy'],imported['policy'])
 results['exact_parent_optimizer']=equal(original['optimizer'],imported['optimizer'])
 # PPO snapshots, not speculative/deferred responses, enter the loss.
 results['no_unexecuted_rows']=all(row['learning']['samples']==row['counts']['accepted_learning_samples'] for row in a['history'])
 results['all_passed']=all(results.values());write(root/'validation.json',results)
 if not results['all_passed']:raise ValueError('Proof failed: '+str(results))
 return results
