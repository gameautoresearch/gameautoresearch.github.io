"""Actual CUDA parity, resident full-loop profiles and learning campaigns."""
from pathlib import Path
import argparse
import ctypes as C
import json
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))


def configuration():
    config=json.loads((ROOT/'integrations/doudizhu/recipes/validated-l4-smoke.json').read_text())
    config.pop('budget_gpu_hours',None)
    config.update(architecture={'hidden':512,'depth':5,'history':128},engine_backend='cuda',
                  native_fast=False,workers=0,vector_games=64,record_games=4,actor_rng='splitmix-counter-v1',
                  learner_shuffle='torch',inference_backend='eager',encoding_backend='direct',
                  max_episodes=256,max_seconds=300,max_candidates=16384)
    config['opponents']['selfplay_probability']=1.
    return config


def validate(out):
    import numpy as np
    import torch
    from integrations.doudizhu.environment import Game,deal,ROLES,heuristic
    from integrations.doudizhu.cuda.backend import CudaBatch,source_hash
    from integrations.doudizhu.cuda.reference import CPUReference,RawState
    from integrations.doudizhu.runner import initialize
    from integrations.doudizhu.artifacts import write_json,load_checkpoint
    import subprocess
    config=configuration()
    initialize(config)
    rng=np.random.default_rng(72931)
    cases=[]
    total_decisions=0
    retained=[]
    for wave in range(4):
        cards=[deal('train',72931,wave*32+i) for i in range(32)]
        oracles=[Game(c,('wp','adp','logadp')[wave%3],record=True,engine_backend='upstream') for c in cards]
        references=[CPUReference(c) for c in cards]
        gpu=CudaBatch(len(cards))
        gpu.reset(cards)
        seed_values=rng.integers(0,np.iinfo(np.uint64).max,size=len(cards),dtype=np.uint64)
        seeds=torch.as_tensor(seed_values,device='cuda')
        maximum=late=0
        for turn in range(163):
            count,peak,active=gpu.generate()
            maximum=max(maximum,peak)
            states,actions,counts,features=gpu.debug()
            if not count:break
            kinds=(np.arange(len(cards))+turn)%4+1
            greedy=np.zeros(len(cards),dtype=np.int64)
            for i,oracle in enumerate(oracles):
                obs=oracle.observe()
                raw=RawState.from_buffer_copy(states[i].tobytes())
                np.testing.assert_array_equal(states[i,:C.sizeof(RawState)],np.frombuffer(references[i].state,dtype=np.uint8)[:C.sizeof(RawState)])
                if obs is None:
                    assert raw.done and counts[i]==0
                    continue
                legal,x,z=references[i].observe()
                actual=tuple(tuple(rank for rank,copies in zip((3,4,5,6,7,8,9,10,11,12,13,14,17,20,30),row[:15]) for _ in range(int(copies))) for row in actions[i,:counts[i]])
                assert actual==obs.legal==legal
                assert raw.role==ROLES.index(obs.role) and raw.last_pid==ROLES.index(obs.last_pid)
                xx,zz,_=obs.encode(backend='upstream')
                np.testing.assert_array_equal(features[i,:len(xx)],xx)
                np.testing.assert_array_equal(features[i,len(xx):len(xx)+810].reshape(5,162),zz)
                greedy[i]=int(rng.integers(len(legal)))
                late+=min(obs.remaining)<=5
            choices=gpu.choose(torch.as_tensor(greedy,device='cuda'),torch.as_tensor(kinds,dtype=torch.int8,device='cuda'),seeds,.37)
            selected=choices.cpu().numpy()
            for i,oracle in enumerate(oracles):
                obs=oracle.observe()
                if obs is None:continue
                expected=references[i].choose(len(obs.legal),greedy[i],int(kinds[i]),seed_values[i],turn,.37)
                assert selected[i]==expected,('counter_rng',wave,turn,i,selected[i],expected)
                references[i].step(int(selected[i]))
                oracle.step(int(selected[i]))
                total_decisions+=1
            gpu.step(choices)
        flags=gpu.outcomes()
        for i,oracle in enumerate(oracles):
            expected=oracle.outcome()
            assert flags[i].tolist()==[1,int(expected['landlord_won']),expected['bombs'],expected['decisions']]
        retained.extend(oracle.trajectory() for oracle in oracles[:2])
        cases.append({'games':len(cards),'max_legal_candidates':maximum,'late_decisions':late,'all_shared_cpu_original_cpu_cuda_states_and_features_match':True})
        gpu.close()
    write_json(out/'private-replays.json',retained)
    write_json(out/'simulator-parity.json',{'cases':cases,'decisions':total_decisions,'source_sha256':source_hash(),
        'counter_rng_choices_match':True,'faults':0,'truncations':0})
    # Real train/save/resume through the ordinary whole-process launcher.
    runs={}
    for label,games,resume in [('full',256,None),('partial',128,None),('resume',256,'partial')]:
        local=dict(config,max_episodes=games)
        recipe=out/(label+'.json')
        write_json(recipe,local)
        command=[sys.executable,str(ROOT/'integrations/doudizhu/launch.py'),'train','--recipe',str(recipe),'--out',str(out/label)]
        if resume:command+=['--resume',str(out/resume/'latest.pt')]
        subprocess.run(command,cwd=ROOT,check=True,timeout=330)
        runs[label]=load_checkpoint(out/label/'latest.pt')
    a,aa=runs['full'];b,bb=runs['resume']
    for key,value in a.state_dict().items():torch.testing.assert_close(value,b.state_dict()[key],atol=0,rtol=0)
    for key in ('episodes','next_deal','decisions','samples','updates','used_deal_hashes','numpy_rng'):
        assert aa['state'][key]==bb['state'][key],key
    assert torch.equal(aa['torch_rng'],bb['torch_rng'])
    for x,y in zip(aa['cuda_rng'],bb['cuda_rng']):assert torch.equal(x,y)
    write_json(out/'validation.json',{'simulator_games':128,'simulator_decisions':total_decisions,
        'original_cpu_shared_cpu_cuda_full_transition_legal_feature_outcome_parity':True,
        'counter_rng_parity':True,'full_resident_neural_training_executed':True,
        'resident_torch_shuffle_checkpoint_resume_bit_exact':True,
        'source_sha256':source_hash(),'device':torch.cuda.get_device_name(0),
        'faults':0,'truncations':0,'simulator_illegals':0})



def profile(out):
    import subprocess
    import numpy as np
    from integrations.doudizhu.artifacts import write_json
    checkpoint=Path('/retained/structural-frontier-learning-gpu-20261108T054525-54f17a/learning-17-frontier/latest.pt')
    if not checkpoint.exists():raise FileNotFoundError('Retained trained full-size policy is required')
    rows=[]
    # Same GPU, CPU/RAM, weights, game/deal/RNG protocols, learner and batch within
    # each comparison. Scaling changes policy refresh cadence and is labelled.
    for repeat in range(3):
        cases=[(backend,batch) for batch in (64,256,1024,2048) for backend in ('native','cuda')]
        if repeat%2:cases.reverse()
        for backend,batch in cases:
            target=out/f'online-{repeat}-{backend}-b{batch}'
            command=[sys.executable,str(ROOT/'integrations/doudizhu/structural_profile.py'),
                '--device','cuda:0','--checkpoint',str(checkpoint),'--games','4096','--batch',str(batch),
                '--engine',backend,'--codec','direct','--workers','4' if backend=='native' else '0',
                '--actor-rng','splitmix-counter-v1','--learner-shuffle','torch','--max-candidates','16384',
                '--no-sample-hash','--record-first-only','--record-games','4','--out',str(target)]
            if backend=='native':command+=['--native-fast']
            subprocess.run(command,cwd=ROOT,check=True,timeout=360)
            row=json.loads((target/'result.json').read_text())
            row.update(repeat=repeat,protocol='online')
            rows.append(row)
            write_json(out/'matrix.json',rows)
    # Independent fixed-policy collection removes batch-dependent training/deal
    # interleaving: each game has its own counter seed; one identical learner pass.
    for repeat in range(3):
        cases=[(backend,batch) for batch in (64,2048) for backend in ('native','cuda')]
        if repeat%2:cases.reverse()
        for backend,batch in cases:
            target=out/f'fixed-{repeat}-{backend}-b{batch}'
            command=[sys.executable,str(ROOT/'integrations/doudizhu/structural_profile.py'),
                '--device','cuda:0','--checkpoint',str(checkpoint),'--games','2048','--batch',str(batch),
                '--engine',backend,'--codec','direct','--workers','4' if backend=='native' else '0',
                '--actor-rng','splitmix-counter-v1','--learner-shuffle','torch','--max-candidates','16384',
                '--no-sample-hash','--fixed-policy','--record-first-only','--record-games','4','--out',str(target)]
            if backend=='native':command+=['--native-fast']
            subprocess.run(command,cwd=ROOT,check=True,timeout=300)
            row=json.loads((target/'result.json').read_text())
            row.update(repeat=repeat,protocol='fixed-policy')
            rows.append(row)
            write_json(out/'matrix.json',rows)
    write_json(out/'result.json',{'trials':len(rows),'source':'all per-trial manifests and full checkpoints retained',
        'comparison':'native CPU versus resident CUDA simulation; same L4 +4 CPU +16GiB; full-size genuinely trained policy',
        'cold_process_accounting':True,'simulator_and_training_timings_separate':True})


def learning(out):
    import copy
    import hashlib
    import subprocess
    from integrations.doudizhu.artifacts import write_json,load_checkpoint
    from integrations.doudizhu.structural_profile import tensor_hash
    from integrations.doudizhu.compare import compare
    rows=[]
    for seed in (17,29,43):
        variants=[('native-ieee','native','ieee'),('resident-ieee','cuda','ieee'),('resident-tf32','cuda','tf32')]
        if seed==29:variants.reverse()
        for label,backend,precision in variants:
            config=configuration()
            config.update(seed=seed,deal_seed=20261108+seed,engine_backend=backend,
                native_fast=backend=='native',workers=4 if backend=='native' else 0,
                vector_games=2048,batch_size=4096,max_episodes=65536,max_seconds=900,
                matmul_precision=precision,max_candidates=16384)
            target=out/f'{seed}-{label}'
            recipe=out/f'{seed}-{label}.json'
            write_json(recipe,config)
            started=time.perf_counter()
            command=[sys.executable,str(ROOT/'integrations/doudizhu/launch.py'),'train',
                     '--recipe',str(recipe),'--out',str(target)]
            subprocess.run(command,cwd=ROOT,check=True,timeout=930)
            training_seconds=time.perf_counter()-started
            # Same compiled CPU evaluator and actor RNG for all arms, including
            # a fixed opponent teammate in each peasant-role test.
            evaluation=copy.deepcopy(config)
            evaluation.update(engine_backend='native',native_fast=True,workers=4,vector_games=64,
                split='practice',deals=32,deal_seed=761943,matmul_precision='ieee',
                opponents_panel=[{'kind':'random'},{'kind':'heuristic'}])
            eval_recipe=out/f'eval-{seed}-{label}.json'
            write_json(eval_recipe,evaluation)
            games={}
            for checkpoint in ('initial','latest'):
                dest=out/f'eval-{seed}-{label}-{checkpoint}'
                subprocess.run([sys.executable,str(ROOT/'integrations/doudizhu/launch.py'),'evaluate',
                    '--recipe',str(eval_recipe),'--checkpoint',str(target/(checkpoint+'.pt')),'--out',str(dest)],
                    cwd=ROOT,check=True,timeout=180)
                games[checkpoint]=__import__('json').loads((dest/'games.json').read_text())
            policy,data=load_checkpoint(target/'latest.pt','cpu')
            state=data['state']
            row={'seed':seed,'label':label,'backend':backend,'matmul_precision':precision,
                 'training_cold_seconds':training_seconds,'train_evaluate_cold_seconds':time.perf_counter()-started,
                 'weights_sha256':tensor_hash(policy),'episodes':state['episodes'],'decisions':state['decisions'],
                 'samples':state['samples'],'updates':state['updates'],'timing':state['timing'],
                 'used_deal_hashes_sha256':hashlib.sha256(''.join(state['used_deal_hashes']).encode()).hexdigest(),
                 'parameter_l2_change':__import__('json').loads((target/'result.json').read_text())['parameter_l2_change_this_invocation'],
                 'paired_practice_change':compare(games['initial'],games['latest']),
                 'final_evaluation_sha256':hashlib.sha256(__import__('json').dumps(games['latest'],sort_keys=True).encode()).hexdigest(),
                 'numerical_semantics_change':precision=='tf32'}
            rows.append(row)
            write_json(out/'learning-summary-compact.json',rows)
            print(__import__('json').dumps({k:row[k] for k in ('seed','label','training_cold_seconds','train_evaluate_cold_seconds')}),flush=True)
    write_json(out/'result.json',{'arms':len(rows),'learning_executed':True,'panel':'public development only',
        'comparison':'same full-size model/deals/actor and learner RNG/hardware; TF32 is a separately labelled arithmetic change'})


def long_matched(out,repeats=3):
    import subprocess
    from integrations.doudizhu.artifacts import write_json
    checkpoint=Path('/retained/structural-frontier-learning-gpu-20261108T054525-54f17a/learning-17-frontier/latest.pt')
    rows=[]
    for repeat in range(repeats):
        cases=['upstream','native','cuda']
        if repeat%2:cases.reverse()
        for backend in cases:
            dest=out/f'{repeat}-{backend}'
            command=[sys.executable,str(ROOT/'integrations/doudizhu/structural_profile.py'),
                '--device','cuda:0','--checkpoint',str(checkpoint),'--games','8192','--sim-games','128',
                '--batch','2048','--engine',backend,'--codec','upstream' if backend=='upstream' else 'direct',
                '--workers','4' if backend=='native' else '0','--actor-rng','splitmix-counter-v1',
                '--learner-shuffle','torch','--max-candidates','16384','--no-sample-hash',
                '--record-first-only','--record-games','4','--out',str(dest)]
            if backend=='native':command+=['--native-fast']
            subprocess.run(command,cwd=ROOT,check=True,timeout=600)
            row=json.loads((dest/'result.json').read_text());row['repeat']=repeat
            rows.append(row);write_json(out/'matrix.json',rows)
    for key in ('decisions','candidates','trajectory_sha256','weights_sha256','evaluation_sha256'):
        assert len({row[key] for row in rows})==1,('long matched FP32 parity',key)
    write_json(out/'result.json',{'trials':len(rows),'all_three_backends_exact_training_hashes':True,
        'training_games':8192,'batch':2048,'separate_simulation_diagnostic_games':128,
        'cold_process_scope':'interpreter, imports, native/CUDA compilation, simulation diagnostic, full collection and learner pass per wave, final checkpoint, practice24 and teardown',
        'configuration':'FP32 full-size model, IEEE head matmuls and unchanged deterministic cuDNN TF32 permission; same trained start/deals/counter actor/Torch learner shuffle/resources; no batch or precision scaling'})

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',required=True,choices=['validate','profile','learning','precision','historical','long-matched','sim-original','scalar-matched','final'])
    parser.add_argument('--out',required=True,type=Path)
    args=parser.parse_args()
    started=time.time()
    try:
        if args.stage=='validate':validate(args.out)
        elif args.stage=='profile':profile(args.out)
        elif args.stage=='learning':learning(args.out)
        elif args.stage=='long-matched':long_matched(args.out)
        elif args.stage=='scalar-matched':long_matched(args.out,repeats=1)
        elif args.stage=='final':
            from integrations.doudizhu.cuda.final_qualification import run
            run(args.out)
        elif args.stage=='sim-original':
            from integrations.doudizhu.cuda.simulation_baseline import run
            run(args.out)
        elif args.stage=='precision':
            from integrations.doudizhu.cuda.precision_experiment import run
            run(args.out)
        else:
            from integrations.doudizhu.cuda.historical_baseline import run
            run(args.out)
    except BaseException as exc:
        (args.out/'failure.json').write_text(json.dumps({'error_type':type(exc).__name__,'error':str(exc),
            'started_unix':started,'ended_unix':time.time()},indent=2)+'\n')
        raise


if __name__=='__main__':main()
