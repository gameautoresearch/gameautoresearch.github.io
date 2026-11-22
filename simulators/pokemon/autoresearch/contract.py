"""Frozen Pokémon task, narrow editable contract and content identities."""
from pathlib import Path
import hashlib
import json
import math
import os
import tempfile
from functools import lru_cache

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
EDITABLE = ('policy.py', 'learner.py', 'train.py')
RUNTIME = ('env.py', 'network.py', 'rl.py', 'efficient.py', 'worker.cjs', 'view.cjs', 'engine_fast.cjs')
PIN = '567c1cedddeaa2f35576005d3bfd526cf4f8a13e'
BENCHMARK_ID = 'pokemon-gen9-random-6v6-v1'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

def write(path, value, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content=json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n'
    if exclusive:
        with path.open('x') as f:f.write(content)
        path.chmod(0o444)
    else:
        fd,temporary=tempfile.mkstemp(prefix='.'+path.name+'-',dir=path.parent)
        try:
            with os.fdopen(fd,'w') as f:f.write(content);f.flush();os.fsync(f.fileno())
            os.replace(temporary,path)
        finally:
            if os.path.exists(temporary):os.unlink(temporary)

@lru_cache(maxsize=1)
def runtime_signature():
    root = HERE.parent
    files={name:sha(root/name) for name in RUNTIME}
    fingerprint=ROOT/'FROZEN.json'
    if fingerprint.exists():files['FROZEN.json']=sha(fingerprint)
    engine=ROOT/'candidates/pokemon-showdown'
    for directory in ('dist','node_modules'):
        for path in sorted((engine/directory).rglob('*')):
            if path.is_file() and path.suffix in ('.js','.cjs','.json','.node'):
                files['engine/'+str(path.relative_to(engine))]=sha(path)
    return identity(files)

def source_signature(directory):
    return identity({name: sha(Path(directory)/name) for name in EDITABLE})

def validate_recipe(config):
    required = {'seed', 'training_seed_start', 'iterations', 'environments_per_worker',
                'network', 'learning_rate', 'gamma', 'gae_lambda', 'ppo_epochs', 'minibatch',
                'clip', 'value_coef', 'entropy_coef', 'max_grad_norm'}
    if set(config) != required:
        raise ValueError('Recipe fields differ from the editable contract: '+str(set(config)^required))
    for name in ('seed', 'training_seed_start', 'iterations', 'environments_per_worker', 'ppo_epochs', 'minibatch'):
        if type(config[name]) is not int or config[name] < 1:
            raise ValueError('Positive integer required: '+name)
    if config['environments_per_worker'] > 64 or config['ppo_epochs'] > 16 or config['minibatch'] > 16384:
        raise ValueError('Recipe exceeds resource limits')
    if config['training_seed_start'] + 3*config['environments_per_worker']*config['iterations'] >= 800_000_000:
        raise ValueError('Training seed range overlaps reserved evaluation domain')
    if set(config['network']) != {'hidden', 'embedding'}:
        raise ValueError('network requires hidden and embedding')
    if any(type(v) is not int or not 1 <= v <= 512 for v in config['network'].values()):
        raise ValueError('Network dimension outside contract')
    for name in required-{'network', 'seed', 'training_seed_start', 'iterations', 'environments_per_worker', 'ppo_epochs', 'minibatch'}:
        if not isinstance(config[name], (int, float)) or not math.isfinite(config[name]) or config[name] < 0:
            raise ValueError('Invalid finite recipe field: '+name)
    if not 0 < config['gamma'] <= 1 or not 0 <= config['gae_lambda'] <= 1 or config['learning_rate'] <= 0:
        raise ValueError('Invalid discount/learning rate')
    return config

def runtime_config(config, device='cpu'):
    validate_recipe(config)
    return dict(config, hardware={'device':device, 'cpu_threads':1, 'workers':3,
        'environments_per_worker':config['environments_per_worker'], 'attached_cpu_count':4,
        'attached_ram_gb':8, 'machine':'L4-4CPU-8GiB'}, max_decision_phases=512,
        faint_reward=0.0, optimization='shared', lazy_observations=True, batched_scripted=True)
