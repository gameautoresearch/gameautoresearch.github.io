"""Bounded follow-up: rare rules, original baseline, tensor-core learning and trace."""
from pathlib import Path
import copy
import json
import subprocess
import sys
import time

from .experiment import ROOT,configuration


def targeted(out):
    import numpy as np
    import torch
    from collections import Counter
    from ..environment import Game,deck
    from .backend import CudaBatch
    from .reference import CPUReference
    hand=sum(([r]*3 for r in range(3,7)),[])+[7]*4+[8]*2+[20,30]
    rest=sorted((Counter(deck)-Counter(hand)).elements())
    cards={'landlord':sorted(hand),'landlord_down':rest[:17],'landlord_up':rest[17:],
           'three_landlord_cards':sorted(hand)[:3]}
    airplane=tuple(sum(([r]*3 for r in range(3,7)),[])+[7]*4)
    checks=0
    for opening in (airplane,(20,30),(7,7,7,7)):
        gpu=CudaBatch(1)
        gpu.reset([cards])
        oracle=Game(cards,'adp',engine_backend='upstream')
        reference=CPUReference(cards)
        for turn in range(4):
            gpu.generate()
            _,actions,counts,features=gpu.debug()
            obs=oracle.observe()
            legal,x,z=reference.observe()
            actual=tuple(tuple(rank for rank,copies in zip((3,4,5,6,7,8,9,10,11,12,13,14,17,20,30),row[:15]) for _ in range(int(copies))) for row in actions[0,:counts[0]])
            assert actual==legal==obs.legal
            a,b,_=obs.encode(backend='upstream')
            np.testing.assert_array_equal(features[0,:len(a)],a)
            np.testing.assert_array_equal(features[0,len(a):len(a)+810].reshape(5,162),b)
            if turn==3:
                assert obs.role=='landlord' and () not in legal
                break
            selected=legal.index(opening if turn==0 else ())
            oracle.step(selected);reference.step(selected)
            gpu.step(torch.tensor([selected],device='cuda'))
            checks+=1
        gpu.close()
    swapped={k:list(v) for k,v in cards.items()}
    swapped['landlord_down'],swapped['landlord_up']=swapped['landlord_up'],swapped['landlord_down']
    gpu=CudaBatch(2);gpu.reset([cards,swapped]);gpu.generate()
    _,actions,counts,features=gpu.debug()
    np.testing.assert_array_equal(features[0],features[1])
    assert counts[0]==counts[1]
    np.testing.assert_array_equal(actions[0,:counts[0]],actions[1,:counts[1]])
    gpu.close()
    (out/'targeted-cuda-rules.json').write_text(json.dumps({'rare_airplane_type15_rocket_bomb_pass_reset_transitions':checks,
        'initial_hidden_hand_repartition_public_features_identical':True})+'\n')


def run(out):
    import torch
    from ..artifacts import write_json,load_checkpoint
    from ..compare import compare
    from ..structural_profile import tensor_hash
    targeted(out)
    # Explicit original pinned simulator and earliest upstream-backed integration
    # settings, current faithful oracle path; all software/hardware knobs declared.
    checkpoint=Path('/retained/structural-frontier-learning-gpu-20261108T054525-54f17a/learning-17-frontier/latest.pt')
    baseline=[]
    for repeat in range(3):
        cases=[('upstream',64),('native',64),('cuda',64),('native',2048),('cuda',2048)]
        if repeat%2:cases.reverse()
        for backend,batch in cases:
            dest=out/f'baseline-{repeat}-{backend}-{batch}'
            command=[sys.executable,str(ROOT/'integrations/doudizhu/structural_profile.py'),'--device','cuda:0',
                '--checkpoint',str(checkpoint),'--games','2048','--batch',str(batch),'--engine',backend,
                '--codec','upstream' if backend=='upstream' else 'direct','--workers','4' if backend=='native' else '0',
                '--actor-rng','splitmix-counter-v1','--learner-shuffle','torch','--max-candidates','16384',
                '--no-sample-hash','--record-first-only','--record-games','4','--out',str(dest)]
            if backend=='native':command+=['--native-fast']
            subprocess.run(command,cwd=ROOT,check=True,timeout=300)
            row=json.loads((dest/'result.json').read_text());row['repeat']=repeat
            baseline.append(row);write_json(out/'baseline-matrix.json',baseline)
    # BF16 qualification includes real cuDNN backward and exact midpoint resume.
    config=configuration();config['compute_dtype']='bfloat16'
    states={}
    try:
        for label,games,resume in [('full',256,None),('partial',128,None),('resumed',256,'partial')]:
            recipe=out/f'bf16-{label}.json';dest=out/f'bf16-{label}'
            write_json(recipe,dict(config,max_episodes=games))
            command=[sys.executable,str(ROOT/'integrations/doudizhu/launch.py'),'train','--recipe',str(recipe),'--out',str(dest)]
            if resume:command+=['--resume',str(out/f'bf16-{resume}'/'latest.pt')]
            subprocess.run(command,cwd=ROOT,check=True,timeout=330)
            states[label]=load_checkpoint(dest/'latest.pt')
        a,aa=states['full'];b,bb=states['resumed']
        for k,v in a.state_dict().items():torch.testing.assert_close(v,b.state_dict()[k],atol=0,rtol=0)
        for key in ('episodes','next_deal','decisions','samples','updates','used_deal_hashes','numpy_rng'):
            assert aa['state'][key]==bb['state'][key],key
        assert all(torch.equal(x,y) for x,y in zip(aa['cuda_rng'],bb['cuda_rng']))
    except Exception as exc:
        write_json(out/'bf16-failure.json',{'type':type(exc).__name__,'error':str(exc),'qualified':False})
        raise
    rows=[]
    for seed in (17,29,43):
        variants=[('tf32','float32'),('ieee','bfloat16')]
        if seed==29:variants.reverse()
        for precision,dtype in variants:
            config=configuration()
            config.update(seed=seed,deal_seed=20261108+seed,vector_games=4096,batch_size=4096,
                max_episodes=65536,max_seconds=600,matmul_precision=precision,compute_dtype=dtype)
            label=f'{seed}-{dtype}-{precision}'
            dest=out/label;recipe=out/(label+'.json');write_json(recipe,config)
            started=time.perf_counter()
            subprocess.run([sys.executable,str(ROOT/'integrations/doudizhu/launch.py'),'train','--recipe',str(recipe),'--out',str(dest)],cwd=ROOT,check=True,timeout=630)
            train=time.perf_counter()-started
            evaluation=copy.deepcopy(config)
            evaluation.update(engine_backend='native',native_fast=True,workers=4,vector_games=64,
                split='practice',deals=32,deal_seed=761943,compute_dtype='float32',matmul_precision='ieee',
                opponents_panel=[{'kind':'random'},{'kind':'heuristic'}])
            evaluation_recipe=out/('eval-'+label+'.json');write_json(evaluation_recipe,evaluation)
            games={}
            for name in ('initial','latest'):
                target=out/f'eval-{label}-{name}'
                subprocess.run([sys.executable,str(ROOT/'integrations/doudizhu/launch.py'),'evaluate','--recipe',str(evaluation_recipe),
                    '--checkpoint',str(dest/(name+'.pt')),'--out',str(target)],cwd=ROOT,check=True,timeout=180)
                games[name]=json.loads((target/'games.json').read_text())
            policy,data=load_checkpoint(dest/'latest.pt','cpu')
            result=json.loads((dest/'result.json').read_text())
            row={'seed':seed,'dtype':dtype,'matmul_precision':precision,'batch':4096,
                'training_cold_seconds':train,'train_evaluate_cold_seconds':time.perf_counter()-started,
                'decisions':data['state']['decisions'],'samples':data['state']['samples'],'timing':data['state']['timing'],
                'updates':data['state']['updates'],'weights_sha256':tensor_hash(policy),
                'parameter_l2_change':result['parameter_l2_change_this_invocation'],
                'paired_practice_change':compare(games['initial'],games['latest'])}
            rows.append(row);write_json(out/'precision-summary.json',rows)
    # Operator-level trace on the actual resident actor+learner after qualification.
    from ..runner import initialize,make_optimizers
    from ..model import Policy,update
    from ..rollout import Policies,collect
    from ..workers import VectorGames
    from ..environment import deal,ROLES
    config=configuration();config.update(compute_dtype='bfloat16',vector_games=2048)
    rng=initialize(config);policy=Policy(config['architecture']).to('cuda');policy.compute_dtype='bfloat16'
    optimizers=make_optimizers(policy,config);policies=Policies(policy,'cuda:0',max_candidates=16384)
    policies.actor_rng='splitmix-counter-v1'
    cards=[deal('train',7293,i) for i in range(2048)]
    assignments=[{r:{'kind':'learner'} for r in ROLES} for _ in cards]
    with VectorGames(0,engine_backend='cuda') as vector:
        collect(vector,policies,cards,assignments,rng,training=True)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=True) as trace:
            result=collect(vector,policies,cards,assignments,rng,training=True)
            update(policy,optimizers,result['samples'],4096,1,rng,shuffle_backend='torch')
            torch.cuda.synchronize()
        trace.export_chrome_trace(str(out/'operators.json'))
        (out/'operators.txt').write_text(trace.key_averages().table(sort_by='self_cuda_time_total',row_limit=40))
    write_json(out/'result.json',{'baseline_trials':len(baseline),'precision_learning_arms':len(rows),
        'bf16_cuda_training_and_resume_bit_exact':True,'fp32_scalar_head':True,
        'numerical_semantics_change':True,'practice_panel':'public inspected weak opponents; not competitive qualification'})
