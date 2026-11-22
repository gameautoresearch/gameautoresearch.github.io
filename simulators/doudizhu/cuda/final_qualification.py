"""Qualify the final upload packing against the prior real BF16 learning runs."""
import json
from pathlib import Path
import subprocess
import sys
import time

from .experiment import ROOT,configuration,validate


def run(out):
    from ..artifacts import write_json,load_checkpoint
    from ..structural_profile import tensor_hash
    validation=out/'validation';validation.mkdir()
    validate(validation)
    previous=Path('/retained/cuda-precision-20261108T070129-a427be')
    rows=[]
    for seed in (17,29,43):
        config=configuration()
        config.update(seed=seed,deal_seed=20261108+seed,vector_games=4096,batch_size=4096,
            max_episodes=65536,max_seconds=600,matmul_precision='ieee',compute_dtype='bfloat16')
        recipe=out/f'{seed}.json';dest=out/f'{seed}-bulk-pack'
        write_json(recipe,config)
        start=time.perf_counter()
        subprocess.run([sys.executable,str(ROOT/'integrations/doudizhu/launch.py'),'train','--recipe',str(recipe),'--out',str(dest)],cwd=ROOT,check=True,timeout=630)
        cold=time.perf_counter()-start
        policy,data=load_checkpoint(dest/'latest.pt','cpu')
        oracle,old=load_checkpoint(previous/f'{seed}-bfloat16-ieee/latest.pt','cpu')
        assert tensor_hash(policy)==tensor_hash(oracle),seed
        for key in ('decisions','samples','updates','used_deal_hashes','next_deal','numpy_rng'):
            assert data['state'][key]==old['state'][key],(seed,key)
        rows.append({'seed':seed,'training_cold_seconds':cold,'decisions':data['state']['decisions'],
            'decisions_per_second':data['state']['decisions']/cold,'timing':data['state']['timing'],
            'exact_previous_full_training_weights_samples_updates_deals':True,'weights_sha256':tensor_hash(policy)})
        write_json(out/'final-learning.json',rows)
    write_json(out/'result.json',{'latest_source_resident_parity_and_resume_qualified':True,
        'full_learning_games':3*65536,'bulk_packing_semantics_exact':True,
        'earlier_canonical_gpu_rules_source_unchanged':True,'bf16_is_optional_numerical_recipe':True})
