import json,shutil
import numpy as np
from simulators.clash.autoresearch.common import HERE,dump,load_module
from simulators.clash.autoresearch.bundle import seal

def test_pair_parallelism_preserves_every_public_trajectory(tmp_path):
 from simulators.clash.autoresearch.evaluation_v3 import evaluate as sequential
 from simulators.clash.autoresearch.evaluation_v3_parallel import evaluate as parallel
 source=tmp_path/'source';source.mkdir()
 for name in ('policy.py','learner.py','train.py'):shutil.copyfile(HERE/'starter'/name,source/name)
 dump(source/'config.json',{'width':8});model=load_module(source/'policy.py','pair_fixture').create({'width':8});np.savez_compressed(source/'weights.npz',**{k:v.detach().numpy().astype(np.float32) for k,v in model.state_dict().items()});seal(source,tmp_path/'bundle')
 schedule=tmp_path/'schedule.json';dump(schedule,{'engine_seeds':[2100101],'policy_seed':20261109,'opponents':[{'name':'standard','kind':'scripted'},{'name':'pressure','kind':'scripted'}]})
 a=sequential(tmp_path/'bundle',schedule,tmp_path/'sequential.json',development=True,max_seconds=90,trajectory_digest=True);b=parallel(tmp_path/'bundle',schedule,tmp_path/'parallel.json',development=True,max_seconds=90,pair_workers=2,trajectory_digest=True)
 key=lambda r:(r['opponent'],r['seed_index'],r['seat'])
 assert sorted(a['rows'],key=key)==sorted(b['rows'],key=key)
 assert a['raw_score_0_100']==b['raw_score_0_100'] and a['schedule_commitment']==b['schedule_commitment']
 assert all('public_trajectory_sha256' in r for r in b['rows'])
