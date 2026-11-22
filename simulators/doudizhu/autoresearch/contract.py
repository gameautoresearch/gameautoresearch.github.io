from pathlib import Path
import hashlib
import json

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
GAME=HERE.parent
EDITABLE=('policy.py','learner.py','train.py')
DEFAULT={
    'version':'doudizhu-autoresearch-v1','gpu':'L40S','gpu_minutes':120,
    'cpu_cores':4,'memory_mib':16384,'min_run_seconds':120,
    'policy_tensor_cap':500_000,'source_bytes_cap':524288,
    'artifact_bytes_cap':134217728,'grading_points':[1/3,2/3,1.0],
    'practice_deals':32,'grade_deals':256,'inference_batch':32,
    'inference_timeout_seconds':10,'bootstrap_replicates':2000,
    'allocation_rule':'all allocated worker wall time, including setup, compile, simulation, transfers, learning, evaluation, checkpoints, idle GPU and teardown; failed allocations are charged',
    'evaluation':'role-balanced win probability: candidate separately controls each role; both other seats use the same panel bot, including the candidate teammate',
    'submission':'immutable checkpoint + exact policy source + architecture; latest explicitly submitted completed run',
    'reward':'clip((candidate_quality-starter_quality)/(reference_quality-starter_quality),0,1); also retain unclipped normalized quality',
}
RUNTIME=(
    '__init__.py','environment.py','model.py','graph.py','artifacts.py','workers.py','rollout.py',
    'native_backend.py','native_rollout.py','native/engine.cpp',
    'cuda/backend.py','cuda/reference.py','cuda/actor_rng.py','cuda/rollout.py',
    'cuda/rules.h','cuda/engine.cu','cuda/cpu_reference.cpp',
    'source.lock.json','rules.lock.json','public-data-seeds.json','LICENSE-DouZero','NOTICE.md',
)

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def frozen_manifest():
    return {name:sha(GAME/name) for name in RUNTIME}

def validate_session(config):
    c=dict(DEFAULT,**config)
    if c['gpu']!='L40S' or c['cpu_cores']!=4 or c['memory_mib']!=16384:
        raise ValueError('v1 resource contract is one L40S + four physical CPU cores +16GiB')
    if not 0<c['gpu_minutes']<=1440 or c['min_run_seconds']<=0:raise ValueError('Invalid bounded allocation')
    return c

def write_json(path,value,exclusive=False):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    data=json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n'
    if exclusive:
        with p.open('x') as f:f.write(data)
    else:
        tmp=p.with_name(p.name+'.tmp');tmp.write_text(data);tmp.replace(p)
