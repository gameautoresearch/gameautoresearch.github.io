"""The benchmark's host side: the frozen game mode, sealed manifests and frozen configuration fields.

The session contract (dollar budget, worker shapes, leases) is ../contract.py and its ledger ../ledger.py.
"""
import hashlib,importlib.util,json,os
from pathlib import Path
HERE=Path(__file__).resolve().parents[1]
ROOT=HERE.parent
def _mode():
 """The frozen game mode. It is defined once, in the worker package's own module, and read here by path: the host
 interpreter has no JAX and never imports the package."""
 spec=importlib.util.spec_from_file_location('polytopia_autoresearch_mode',HERE/'starter/src/polytopia_bot/rl/mode.py')
 module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
MODE=_mode()
EDITABLE=('src/polytopia_bot/rl/network.py','src/polytopia_bot/rl/ppo.py','src/polytopia_bot/rl/gpu_resident.py','experiments/gpu_resident_train.py')
BENCHMARK='polytopia-bardur-1v1-11-drylands-v1'
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(d,indent=2)+'\n');os.replace(t,p)
def manifest(root):return {str(p.relative_to(root)):digest(p) for p in sorted(Path(root).rglob('*')) if p.is_file() and p.name!='manifest.json' and '__pycache__' not in p.parts}
def seal(root):
 d=manifest(root);write(Path(root)/'manifest.json',{'benchmark_id':BENCHMARK,'files':d,'sha256':hashlib.sha256(json.dumps(d,sort_keys=True).encode()).hexdigest()});return verify(root)
def verify(root):
 d=json.loads((Path(root)/'manifest.json').read_text());files=manifest(root)
 if d['benchmark_id']!=BENCHMARK or files!=d['files'] or d['sha256']!=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest():raise ValueError('Source/config/checkpoint manifest mismatch')
 return d['sha256']
def validate_config(c):
 for k,v in MODE.FROZEN.items():
  if k in c and (type(c[k])!=type(v) or c[k]!=v):raise ValueError('Frozen task field '+k+': this task keeps '+repr(v))
 if not 0<float(c['learning_rate'])<=.01 or not 0<=float(c['entropy_coef'])<=.1:raise ValueError('Invalid optimizer settings')
 return c
