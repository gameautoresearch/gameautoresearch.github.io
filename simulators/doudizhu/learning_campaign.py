"""Independent full launcher validation and explicit large-wave algorithm comparison.

Practice panels are public development data; no competitive-strength claim. The
large-wave arm changes policy refresh cadence and is never called exact parity.
"""
from pathlib import Path
import copy
import json
import subprocess
import sys
import time


def run(base, mode='learning'):
    root=Path('/opt/arg')
    source=root/'integrations/doudizhu'
    recipe=json.loads((source/'recipes/validated-l4-smoke.json').read_text())
    recipe.pop('budget_gpu_hours',None)
    recipe.update(architecture={'hidden':512,'depth':5,'history':128},max_seconds=900,
                  workers=0,vector_games=64,max_episodes=4096,engine_backend='upstream',encoding_backend='upstream',
                  inference_backend='eager',skip_forced=False)
    if mode=='frontier-learning':
        recipe.update(record_games=4,max_episodes=8192)
        recipe['opponents']['selfplay_probability']=1.
    rows=[]
    for seed in (17,29,43):
        variants=[('baseline',64,'upstream','upstream','eager'),
                  ('exact',64,'view','direct','eager'),
                  ('large-wave',512,'view','direct','graph')]
        if mode=='frontier-learning':
            variants=[('baseline',64,'upstream','upstream','eager'),
                      ('exact',64,'native','direct','eager'),
                      ('frontier',2048,'native','direct','eager')]
        if seed==29:
            variants.reverse()
        for label,batch,engine,codec,inference in variants:
            config=copy.deepcopy(recipe)
            config.update(seed=seed,deal_seed=20261108+seed,vector_games=batch,
                          engine_backend=engine,encoding_backend=codec,inference_backend=inference)
            if mode=='frontier-learning' and label!='baseline':
                config.update(native_fast=True,workers=4)
            if label=='frontier':
                config.update(batch_size=4096,max_episodes=65536,max_candidates=16384)
            target=base/f'learning-{seed}-{label}'
            config_path=base/f'learning-{seed}-{label}.json'
            config_path.write_text(json.dumps(config,indent=2)+'\n')
            started=time.perf_counter()
            command=[sys.executable,str(source/'launch.py'),'train','--recipe',str(config_path),'--out',str(target)]
            subprocess.run(command,cwd=root,check=True,timeout=930)
            training_cold_seconds=time.perf_counter()-started
            eval_config=copy.deepcopy(config)
            # Same per-game opponent RNG and all three candidate roles on each deal.
            # Keep evaluation batch fixed so arm differences reflect only learning.
            eval_config.update(split='practice',deals=32,deal_seed=761943,vector_games=64,
                               inference_backend='eager',opponents_panel=[{'kind':'random'},{'kind':'heuristic'}])
            eval_path=base/f'eval-{seed}-{label}.json'
            eval_path.write_text(json.dumps(eval_config,indent=2)+'\n')
            eval_rows={}
            for checkpoint in ('initial','latest'):
                out=base/f'eval-{seed}-{label}-{checkpoint}'
                command=[sys.executable,str(source/'launch.py'),'evaluate','--recipe',str(eval_path),
                         '--checkpoint',str(target/f'{checkpoint}.pt'),'--out',str(out)]
                subprocess.run(command,cwd=root,check=True,timeout=180)
                eval_rows[checkpoint]=json.loads((out/'games.json').read_text())
            from integrations.doudizhu.compare import compare
            from integrations.doudizhu.artifacts import load_checkpoint
            from integrations.doudizhu.structural_profile import tensor_hash
            policy,data=load_checkpoint(target/'latest.pt')
            row={'seed':seed,'label':label,'vector_games':batch,'inference':inference,
                 'training_cold_seconds':training_cold_seconds,'train_evaluate_cold_seconds':time.perf_counter()-started,
                 'weights_sha256':tensor_hash(policy),'state':data['state'],
                 'initial_evaluation_sha256':__import__('hashlib').sha256(json.dumps(eval_rows['initial'],sort_keys=True).encode()).hexdigest(),
                 'final_evaluation_sha256':__import__('hashlib').sha256(json.dumps(eval_rows['latest'],sort_keys=True).encode()).hexdigest(),
                 'paired_practice_change':compare(eval_rows['initial'],eval_rows['latest']),
                 'algorithm_change':label in ('large-wave','frontier')}
            rows.append(row)
            (base/'learning-summary.json').write_text(json.dumps(rows,indent=2)+'\n')
            print(json.dumps({k:row[k] for k in ('seed','label','training_cold_seconds','train_evaluate_cold_seconds')}),flush=True)
        exact=[r for r in rows if r['seed']==seed and not r['algorithm_change']]
        for key in ('weights_sha256','initial_evaluation_sha256','final_evaluation_sha256'):
            assert exact[0][key]==exact[1][key],(seed,key)
    return rows
