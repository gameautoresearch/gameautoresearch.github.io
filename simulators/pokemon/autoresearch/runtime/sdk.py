"""Read-only native-runtime identity and the admitted batched environment interface."""
import hashlib
import json
import os
from pathlib import Path

from ..contract import identity, sha, source_signature, write, EDITABLE

HERE=Path(__file__).resolve().parent
# The three-Pokémon form. The six-Pokémon id was 'pokemon-gen9-random-6v6-native-h100-pilot4h-v1' (tag full-size-20261113).
BENCHMARK_ID='pokemon-3v3-v1'
STARTER=HERE/'starter'
STARTER_CONFIG=HERE.parent/'recipes/starter.json'
# Trusted worker-side files; their bytes are the runtime identity every checkpoint and bundle binds.
PROTECTED=('sdk.py','fastenv.py','env_service.py','env_client.py','graph_export.py','graph_worker.py','bundle.py','policy_process.py','supervisor.py','worker.py','practice.py','qualification.py','arena.py','bootstrap.py')


def runtime_signature():
    # The unprivileged learner cannot read private supervisor/native sources.
    # Its declared binding is independently recomputed by the root sealer.
    inherited=os.environ.get('POKEMON_PILOT_RELEASE_SHA256')
    if inherited and os.geteuid()!=0:return inherited
    return identity({n:sha(HERE/n)for n in PROTECTED})


def validate_recipe(config):
    # Editable algorithms/architectures are source, recipe remains JSON and bounded.
    if not isinstance(config,dict) or len(json.dumps(config))>65536:
        raise ValueError('Bounded object recipe required')
    for name in ('seed','training_seed_start','environments','rollout_steps','minibatch','ppo_epochs'):
        if type(config.get(name)) is not int or config[name]<1:raise ValueError('Positive integer '+name)
    if not 1<=config['environments']<=4096 or not 1<=config['rollout_steps']<=256:
        raise ValueError('Environment/rollout memory ceiling')
    if not config['training_seed_start']<800_000_000:raise ValueError('Public training seeds only')
    return config


def vector_for(config,device='cuda:0'):
    from .env_client import Client
    return Client(config,device)
