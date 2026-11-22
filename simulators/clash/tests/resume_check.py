"""Process-level PPO resume equivalence including frozen-opponent lazy construction."""
from pathlib import Path
import subprocess,sys,json
import torch,numpy as np
HERE=Path(__file__).resolve().parents[1];ROOT=HERE.parent.parent
out=ROOT/(sys.argv[1] if len(sys.argv)>1 else 'evidence/clash-implementation/resume-equivalence')
out.mkdir(exist_ok=False)
c=json.loads((HERE/'configs/smoke.json').read_text())
c.update(envs=4,updates=6,rollout_steps=64,opponent_mode='population',scripted_fraction=0.,snapshot_interval=2,warmstart_steps=0)
if '--cuda' in sys.argv:
 c.update(device='cuda',graph_inference=True,inference_entity_bucket=200,device_indices=True,pinned_transfer=True,combined_readback=True,cache_rollout_device=True)
 c['hardware_allocation']={'gpu_model':torch.cuda.get_device_name(),'gpu_count':1,'cpu_threads':4,'ram_gib':16,'qualification':'actual CUDA graph resume check'}
if '--structural' in sys.argv:
 c.update(sparse_ppo=True,deferred_values=True,preallocated_rollout=True,eval_skip_forced=True,actor_batch_bucket=8,inference_graph_cache_size=8,source_compresslevel=1)
 if c['device']=='cpu':c.update(ragged_entities=True,sparse_policy_head=True)
if '--graph-native' in sys.argv:
 if c['device']!='cuda':raise ValueError('--graph-native requires --cuda')
 c.update(deferred_values=False,graph_sampling=True,optimizer_fused=True,cache_rollout_fields=True,combined_learner_metrics=True,actor_chunk_size=2,vector_bookkeeping=True,synchronize_timing=False)
config=out/'recipe.json';config.write_text(json.dumps(c,indent=2)+'\n')
def run(name,updates,resume=None):
 cmd=[sys.executable,str(HERE/'runner.py'),'train','--config',str(config),'--out',str(out/name),'--updates',str(updates)]
 if resume:cmd.extend(['--resume',str(out/resume/'checkpoint.pt')])
 with (out/(name+'.log')).open('w') as log:subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
run('uninterrupted',6);run('partial',3);run('resumed',6,'partial')
a=torch.load(out/'uninterrupted/checkpoint.pt',weights_only=False,map_location='cpu')
b=torch.load(out/'resumed/checkpoint.pt',weights_only=False,map_location='cpu')
assert a['session']['observation_sha256']==b['session']['observation_sha256']
assert a['session']['histories']==b['session']['histories']
assert a['session']['rng']==b['session']['rng']
assert torch.equal(a['torch_rng'],b['torch_rng'])
assert a['python_rng']==b['python_rng']
if c['device']=='cuda':
 assert len(a['cuda_rng'])==len(b['cuda_rng'])
 for x,y in zip(a['cuda_rng'],b['cuda_rng']):assert torch.equal(x,y),'CUDA RNG differs'
if c['device']=='mps':assert torch.equal(a['mps_rng'],b['mps_rng'])
for k in a['model']:assert torch.equal(a['model'][k],b['model'][k]),k
for k in a['optimizer']['state']:
 for field in a['optimizer']['state'][k]:
  assert torch.equal(a['optimizer']['state'][k][field],b['optimizer']['state'][k][field]),(k,field)
result={'status':'passed','checks':['bit-identical model tensors','bit-identical Adam state','engine observation/mask/info digest','ongoing episode histories','NumPy RNG','CPU Torch RNG','Python RNG']+(['CUDA RNG tensors'] if c['device']=='cuda' else []),
        'updates':6,'opponent_mode':'frozen population','source':'tests/resume_check.py','full_elapsed_seconds':a['clock']['allocated_worker_seconds'],'resumed_cumulative_elapsed_seconds':b['clock']['allocated_worker_seconds']}
(out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
