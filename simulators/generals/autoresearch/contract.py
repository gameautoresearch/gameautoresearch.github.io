"""Versioned benchmark contract; independent of historical simulator checkpoints."""
from pathlib import Path
import hashlib
import json
import math
import subprocess

HERE=Path(__file__).resolve().parent
# Prepared research boxes retain the historical sibling-package layout.
ROOT=HERE.parents[2] if HERE.parent.name=='generals' else HERE.parents[1]
GAME=HERE.parent if HERE.parent.name=='generals' else HERE.parent/'generals'
EDITABLE=('policy.py','learner.py','train.py')
RUNTIME=('__init__.py','environment.py','packed.py')
PIN='954927f508aa47899019d66317cc313af3e9cc19'
DEFAULT={
 'version':'generals-autoresearch-v3','environment_id':'generals-research-1v1-medium-v2',
 'gpu':'L40S','budget_usd':10.0,'cpu_cores':4,'memory_mib':16384,'min_run_seconds':120,'max_run_seconds':1500,
 # Compute per move, not parameters, limits a submitted policy (flops.py): XLA's FLOP count of one forward pass
 # for one board. The tensor cap is only a storage ceiling (the weights file must fit the 128 MiB export).
 'policy_flops_cap':1_000_000_000,'policy_tensor_cap':32_000_000,'source_bytes_cap':524288,'artifact_bytes_cap':134217728,
 'grading_points':[1/3,2/3,1.0],'practice_maps':16,'grade_maps':128,
 'inference_batch':32,'inference_timeout_seconds':15,'bootstrap_replicates':2000,
 'grading_inference':'cuda','grading_gpu':'L40S','grading_cpu_cores':4,
 # Leases: workers of a shape the researcher chooses, running the researcher's own commands. These bound
 # what one session may hold at once and what one lease may ask for.
 'max_concurrent_jobs':4,'max_concurrent_gpus':2,'max_concurrent_cpu_cores':64,
 'lease_max_cpu_cores':32,'lease_min_memory_mib':2048,'lease_max_memory_mib':65536,
 'lease_command_bytes':65536,'lease_running_commands':16,
 'allocation_rule':'full provider-allocated worker seconds, including setup, compilation, simulation, transfers, learning, evaluation, saving, idle and teardown, charged in US dollars at the list price per second of the worker shape; all failed allocations charged',
 'evaluation':'paired seats on each private map against each frozen panel bot; capture win=1, capture loss=0, cap draw=.5; equal opponent weight',
 'submission':'latest explicitly submitted immutable completed-run policy, exact source, architecture and weights; finalization requires no active/unreconciled jobs',
 'reward':'primary=mean paired panel game points in [0,1]; 100*reward is panel points; separately report unclipped normalized improvement only after multi-seed full-budget calibration passes',
}

# The policy-package format. The session contract (DEFAULT['version']) moved to v2 with leases and the dollar
# budget; packages did not change, so panel opponents sealed under v1 still verify.
PACKAGE='generals-autoresearch-v2'
# A panel opponent sealed with its own source (kind `policy`) may be larger than an entrant may be.
PANEL_TENSOR_CAP=1_000_000
# Panel entries that carry a sealed bundle: `checkpoint` has the starter architecture and runs inside the
# evaluator; `policy` has its own source and runs in an isolated process, as a submitted policy does.
BUNDLED=('checkpoint','policy')

# One dollar budget for every worker shape. A worker second is charged what its shape costs at Modal's sandbox
# list prices in USD per second (modal.com/pricing, read 2026-11-13). These frozen prices are the contract's
# unit of account; Modal's invoice may differ from them.
PRICE={'gpu':0.000542,'cpu_core':0.00003942,'memory_gib':0.00000667}
TRAIN_SHAPE={'gpu':1,'cpu_cores':DEFAULT['cpu_cores'],'memory_mib':DEFAULT['memory_mib']}
# Practice runs the submitted network on the GPU, on a worker of the training shape.
PRACTICE_SHAPE=dict(TRAIN_SHAPE)

def _cost(shape):
 return shape['gpu']*PRICE['gpu']+shape['cpu_cores']*PRICE['cpu_core']+shape['memory_mib']/1024*PRICE['memory_gib']

def rate(shape):
 """US dollars charged per wall second of a worker of this shape."""
 return round(_cost(shape),10)

def lease_shape(config,gpu,cpu_cores,memory_gib):
 """A validated lease shape within the session's limits."""
 for value,what in ((gpu,'--gpu'),(cpu_cores,'--cpu'),(memory_gib,'--memory')):
  if isinstance(value,bool) or not isinstance(value,int):raise ValueError(what+' must be a whole number')
 c=dict(DEFAULT,**config)
 if gpu not in (0,1):raise ValueError('--gpu is 0 or 1: a lease has at most one '+c['gpu'])
 if not 1<=cpu_cores<=c['lease_max_cpu_cores']:raise ValueError(f'--cpu must be between 1 and {c["lease_max_cpu_cores"]} cores')
 memory_mib=memory_gib*1024
 if not c['lease_min_memory_mib']<=memory_mib<=c['lease_max_memory_mib']:
  raise ValueError(f'--memory must be between {c["lease_min_memory_mib"]//1024} and {c["lease_max_memory_mib"]//1024} GiB')
 return {'gpu':gpu,'cpu_cores':cpu_cores,'memory_mib':memory_mib}

def unprivileged_paths(root=None):
 """Import roots of an unprivileged process (a researcher's command, a training run, a policy being played).

 A release whose starter is empty stages `unprivileged/` beside the full frozen tree: the engine's rule modules,
 the wrapper without its scripted players and the few control modules such a process needs. The image hides the
 rest from that user. Other releases import the frozen tree itself."""
 root=Path(root or ROOT);public=root/'unprivileged'
 return [str(public)] if public.is_dir() else [str(root),str(root/'candidates/generals-bots')]

def digest(value):
 return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def seed_manifest():
 """Read a local historical partition when available, or the public train/practice inputs."""
 historical = GAME / 'seeds.json'
 if historical.exists():
  return json.loads(historical.read_text())
 return {**json.loads((GAME / 'public-seeds.json').read_text()), 'heldout': []}


def public_seeds():
 """Return training and practice seeds, extending practice without reusing the known partitions."""
 seeds=seed_manifest();practice=list(seeds['practice'])
 used=set(seeds['train'])|set(practice)|set(seeds['heldout']);seed=max(practice)
 while len(practice)<DEFAULT['practice_maps']:
  seed+=1
  if seed not in used:practice.append(seed);used.add(seed)
 return {'train':list(seeds['train']),'practice':practice,'heldout':list(seeds['heldout'])}

def frozen_manifest():
 from integrations.generals.environment import RULESET
 repo=ROOT/'candidates/generals-bots'
 commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
 if commit!=PIN:raise ValueError('Pinned upstream commit mismatch')
 if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=repo,text=True):raise ValueError('Pinned upstream source changed')
 files=subprocess.check_output(['git','ls-files'],cwd=repo,text=True).splitlines()
 return {'commit':commit,'rules':RULESET,'core':{n:sha(GAME/n) for n in RUNTIME},
  'upstream':{n:sha(repo/n) for n in files if (repo/n).is_file()}}

# The two research budget modes (harness/budget.py). The ledger always keeps US dollars at PRICE: a seconds-mode
# session's allocation is `budget_seconds` of the training worker, held as budget_seconds x rate(TRAIN_SHAPE), so a
# worker of any shape is charged its seconds x rate(shape)/rate(TRAIN_SHAPE) training-worker seconds. A ledger that
# names no mode is a dollar one (every ledger written before the modes existed).
MODES=('usd','seconds')

def validate_session(config):
 c=dict(DEFAULT,**config)
 if c['gpu']!='L40S' or c['cpu_cores']!=4 or c['memory_mib']!=16384:raise ValueError('The training worker is one strict L40S +4 physical CPU cores +16GiB')
 if 'gpu_minutes' in c:raise ValueError('This session was opened under the earlier contract budgeted in seconds; the dollar-budget contract cannot continue or grade it')
 c['budget_mode']=c.get('budget_mode','usd')
 if c['budget_mode'] not in MODES:raise ValueError('budget_mode is usd or seconds')
 c['prices']=dict(PRICE)
 if c['budget_mode']=='seconds':
  if not math.isfinite(float(c.get('budget_seconds') or 0)) or float(c.get('budget_seconds') or 0)<=0:raise ValueError('A seconds-mode session needs a positive budget_seconds')
  if c['max_concurrent_jobs']!=1:raise ValueError('A seconds-mode session runs one job at a time')
  c['budget_usd']=float(c['budget_seconds'])*rate(TRAIN_SHAPE)
 elif 'budget_seconds' in c:raise ValueError('A dollar-mode session has no budget_seconds')
 if any(not math.isfinite(float(c[k])) or float(c[k])<=0 for k in ('budget_usd','min_run_seconds')):raise ValueError('Invalid bounded allocation')
 if c['budget_usd']>1000:raise ValueError('Session exceeds the infrastructure limit')
 return c

def write_json(path,value,exclusive=False):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
 data=json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n'
 if exclusive:
  with p.open('x') as f:f.write(data)
 else:
  tmp=p.with_name(p.name+'.tmp');tmp.write_text(data);tmp.replace(p)
