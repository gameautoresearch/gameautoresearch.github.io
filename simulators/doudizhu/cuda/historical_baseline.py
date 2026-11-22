"""Run the literal committed pre-rewrite integration, without editing its source."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import tarfile
import time

from .experiment import ROOT,configuration


def run(out):
    from ..artifacts import write_json,load_checkpoint
    from ..structural_profile import tensor_hash
    source=out/'original-workspace';source.mkdir()
    archive=Path('/tmp/original-doudizhu.tar')
    provenance=json.loads(Path('/tmp/original-doudizhu-manifest.json').read_text())
    if hashlib.sha256(archive.read_bytes()).hexdigest()!=provenance['archive_sha256']:
        raise ValueError('Historical source archive drift')
    with tarfile.open(archive) as f:f.extractall(source,filter='data')
    (source/'candidates').mkdir()
    (source/'candidates/DouZero').symlink_to(ROOT/'candidates/DouZero',target_is_directory=True)
    write_json(out/'original-source.json',{'git_commit':provenance['git_commit'],
        'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),
        'original_sources_unmodified':True,'original_upstream_commit':'718a5c920bf3361e34178a38f3b80458e176b351'})
    rows=[]
    for seed in (17,29,43):
        variants=[('original-w0',source,0),('original-w4',source,4),('native-w4',ROOT,4)]
        if seed==29:variants.reverse()
        for label,workspace,workers in variants:
            config=configuration()
            for key in ('engine_backend','encoding_backend','native_fast','actor_rng','learner_shuffle','inference_backend','record_games'):
                config.pop(key,None)
            config.update(seed=seed,deal_seed=20261108+seed,workers=workers,max_episodes=2048,max_seconds=600,
                          vector_games=64,batch_size=256)
            if label=='native-w4':config.update(engine_backend='native',native_fast=True,record_games=64)
            recipe=out/(f'{seed}-{label}.json');dest=out/f'{seed}-{label}'
            write_json(recipe,config)
            started=time.perf_counter()
            subprocess.run([sys.executable,str(workspace/'integrations/doudizhu/launch.py'),'train',
                            '--recipe',str(recipe),'--out',str(dest)],cwd=workspace,check=True,timeout=630)
            elapsed=time.perf_counter()-started
            policy,data=load_checkpoint(dest/'latest.pt','cpu')
            row={'seed':seed,'label':label,'training_cold_seconds':elapsed,'decisions':data['state']['decisions'],
                 'samples':data['state']['samples'],'updates':data['state']['updates'],
                 'weights_sha256':tensor_hash(policy),'used_deal_hashes':data['state']['used_deal_hashes'],
                 'timing':data['state']['timing'],'command':[sys.executable,str(workspace/'integrations/doudizhu/launch.py'),'train','--recipe',str(recipe),'--out',str(dest)]}
            rows.append(row);write_json(out/'historical-summary.json',rows)
    for seed in (17,29,43):
        selected=[r for r in rows if r['seed']==seed]
        for key in ('decisions','samples','updates','weights_sha256','used_deal_hashes'):
            assert all(r[key]==selected[0][key] for r in selected),('literal original/native parity',seed,key)
    write_json(out/'result.json',{'literal_pre_structural_full_training_executed':True,'trials':len(rows),
        'matched_original_native_legacy_actor_exact_weights_samples_deals':True,
        'training_baseline':'committed runnable integration, not invented upstream DouZero training throughput'})
