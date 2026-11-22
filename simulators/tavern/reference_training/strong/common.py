from pathlib import Path
import hashlib,json,sys
HERE=Path(__file__).resolve().parent
GAME=HERE.parents[1]
ROOT=GAME.parents[1]
ENGINE=ROOT/'vendor/engines/AutoTavern'
sys.path.insert(0,str(GAME));sys.path.insert(1,str(GAME/'autoresearch'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');t.replace(p)
def admission():
 import taverngym as tg
 from contract import binding
 if tg.build_info()['source_hash']!='217577b83fc4525e' or not tg.build_is_current(ENGINE):raise ValueError('Unqualified native runtime')
 return binding()
def binding():
 b=admission()
 files=[*HERE.glob('*.py'),GAME/'model.py',GAME/'device_rollout.py',GAME/'run.py',GAME/'reference_training/collection.py',GAME/'reference_training/diagnostic_trace.py',GAME/'reference_training/bootstrap.py',*(ENGINE/'python').rglob('*.py'),ROOT/'tools/open_credit_campaign.py',ROOT/'tools/overnight_budget.py']
 return dict(schema='tavern-strong-population-v1',admission=b,files={str(p.relative_to(ROOT)):sha(p) for p in sorted(files)},native_source_hash=b['native_source_hash'])
def validate_config(c):
 from contract import RULES,validate
 validate(c)
 if c['rules']!=RULES or c['mode'] not in ('plain','curriculum'):raise ValueError('Frozen task rules and declared treatment required')
 if c['native_threads']>4 or c.get('torch_threads',1)>4:raise ValueError('Fixed4CPU allocation')
 if c['device'].startswith('cuda') and c['machine'] not in ({'gpu':gpu,'cpu_cores':4,'memory_gib':16} for gpu in ('L4','H100')):raise ValueError('Pinned L4/H100,4CPU,16GiB topology')
 if c['device'].startswith('cuda') and any(c.get(k,False) for k in ('graph_inference','validated_sampler','ppo_backward_graph')):
  import torch
  if torch.__version__.split('+')[0]!='2.9.1':raise ValueError('CUDA graph/sampler paths require qualified PyTorch2.9.1')
