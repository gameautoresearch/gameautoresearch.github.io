"""Sequential paired complete-loop measurements with dense/trimmed correctness checks."""
from pathlib import Path
import json,subprocess,sys,statistics
import torch
HERE=Path(__file__).resolve().parents[1];ROOT=HERE.parent.parent
import fcntl
_perf_lock=(ROOT/'evidence/local-performance.lock').open('a')
fcntl.flock(_perf_lock,fcntl.LOCK_EX)
OUT=ROOT/(sys.argv[1] if len(sys.argv)>1 else 'evidence/clash-implementation/profile-comparison')
OUT.mkdir(exist_ok=True)
rows=[]
for trial in range(3):
 modes=('dense','trimmed') if trial%2==0 else ('trimmed','dense')
 pair={};checkpoints={}
 for mode in modes:
  name=f'{mode}-{trial+1}';target=OUT/name
  cmd=[sys.executable,str(HERE/'runner.py'),'profile','--config',str(HERE/f'configs/profile-{mode}.json'),'--out',str(target)]
  if not (target/'result.json').exists():
   with (OUT/(name+'.log')).open('w') as log:subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
  result=json.loads((target/'result.json').read_text());pair[mode]=result
  checkpoints[mode]=torch.load(target/'checkpoint.pt',weights_only=False,map_location='cpu')
  assert result['updates']==80,(mode,result['updates'])
  assert result['counts']['illegal_actions']==result['counts']['unsupported_actions']==0
  assert not result['fault_signatures']
  rows.append({'trial':trial+1,'mode':mode,'elapsed_seconds':result['allocated_worker_seconds'],
   'counts':result['counts'],'phase_seconds':{k:v['seconds'] for k,v in result['timing'].items()}})
 a,b=checkpoints['dense'],checkpoints['trimmed']
 assert a['session']['observation_sha256']==b['session']['observation_sha256'],'simulation trajectories differ'
 assert a['session']['histories']==b['session']['histories']
 maximum=max(float((a['model'][k]-b['model'][k]).abs().max()) for k in a['model'])
 # Shape-dependent floating-point gradients accumulate under Adam. Compare behavior too.
 assert maximum<1e-3,maximum
 sys.path.insert(0,str(HERE))
 from game import Session,make_vector
 from runner import network,setup
 import numpy as np
 c=a['config'];setup(c)
 with make_vector(c) as v:
  Session(v,c,np.random.default_rng(1)).restore(a['session'])
  ma,mb=network(v,c),network(v,c);ma.load_state_dict(a['model']);mb.load_state_dict(b['model'])
  packed=ma.pack(v.observations,v.masks)
  with torch.no_grad():
   da,va=ma.distribution(packed);db,vb=mb.distribution(packed)
   kl=float(torch.distributions.kl_divergence(da,db).abs().max());value_error=float((va-vb).abs().max());prob_error=float((da.probs-db.probs).abs().max())
  assert kl<1e-6 and value_error<1e-3 and prob_error<1e-4,(kl,value_error,prob_error)
 print(json.dumps({'trial':trial+1,'dense_seconds':pair['dense']['allocated_worker_seconds'],
  'trimmed_seconds':pair['trimmed']['allocated_worker_seconds'],'max_parameter_difference':maximum,'max_policy_kl':kl,'max_value_error':value_error,'max_action_probability_error':prob_error}),flush=True)
result={'workload':'80 full masked PPO updates; 8 both-seat self-play envs; 4 Java threads; 1 torch thread; 64 decisions/rollout; identical seed',
 'device':'CPU (no GPU measurement)','trials':rows,
 'median_elapsed_seconds':{m:statistics.median(r['elapsed_seconds'] for r in rows if r['mode']==m) for m in ('dense','trimmed')},
 'correctness':'all three pairs: identical engine observation/mask/info digests and action histories; final parameter max error <1e-3, policy KL <1e-6, value error <1e-3 and action-probability error <1e-4 on reconstructed engine states; zero faults/unsupported/illegal',
 'scope':'includes initialization, manifests, rollouts, transfers, inference, simulator, PPO and checkpoints; local timing only'}
result['median_speedup']=result['median_elapsed_seconds']['dense']/result['median_elapsed_seconds']['trimmed']
(OUT/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
