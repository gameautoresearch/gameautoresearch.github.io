"""Three independent GPU learning seeds with paired public-practice evaluation.

This is development evidence against seven scripted opponents, not private grading
or a strong-reference claim. Full worker allocation remains the budget unit.
"""
import copy
import json
from pathlib import Path
import random
import time
import numpy as np
import torch

from device_training import train,source_hash,RULES
from model import Policy
from rollout import Timer
from run import evaluation


def evaluate(policy,config):
    random.seed(271828);np.random.seed(271828)
    torch.manual_seed(271828);torch.cuda.manual_seed_all(271828)
    timer=Timer('cuda:0');started=time.perf_counter()
    result=evaluation(policy,config,torch.device('cuda:0'),timer,float('inf'),None)
    result['wall_seconds']=time.perf_counter()-started
    result['stage_seconds']=dict(timer.seconds)
    if not result['complete_valid_schedule']:raise AssertionError('public evaluation contains simulator fault or truncation')
    return result


def qualify(out):
    started=time.perf_counter();rows=[]
    config=json.loads(Path(__file__).with_name('recipes').joinpath('gpu-throughput.json').read_text())
    config['rules']=RULES
    config['evaluation'].update(seeds=[10101,10102],games_per_seat=2,opponent='scripted',deterministic=False)
    protocol={'seeds':[12345,45678,78901],'updates':64,'n_games':32,'learner_seats':[0],
        'training_opponents':'seven unchanged native-scripted bots executing on the GPU',
        'evaluation':config['evaluation'],'source_sha256':source_hash(),
        'scope':'Fixed public practice, all eight evaluated positions; complete natural episodes required. No hidden grading or strong-reference claim.'}
    (out/'learning-protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    for seed in protocol['seeds']:
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        initial=Policy().cuda()
        before=evaluate(initial,copy.deepcopy(config))
        del initial
        print(json.dumps({'stage':'learning_seed_start','seed':seed,'baseline_mean_placement':before['mean_placement']}),flush=True)
        dest=out/f'seed-{seed}'
        history=train(dest,n_games=protocol['n_games'],updates=protocol['updates'],seed=seed)
        if any(row['collection']['invalid_lobbies'] or row['collection']['natural_episodes']!=protocol['n_games'] for row in history):
            raise AssertionError(('invalid or incomplete GPU learning cohort',seed))
        if not all(np.isfinite(row['learning']['loss']) and row['learning']['parameter_delta_l2']>0 for row in history):
            raise AssertionError(('nonfinite or absent neural learning',seed))
        checkpoint=torch.load(dest/'latest.pt',map_location='cpu',weights_only=False)
        if checkpoint['source_sha256']!=protocol['source_sha256']:raise AssertionError('learning source drift')
        policy=Policy().cuda();policy.load_state_dict(checkpoint['policy'])
        after=evaluate(policy,copy.deepcopy(config));del policy
        row={'seed':seed,'before':before,'after':after,
             'mean_placement_improvement':before['mean_placement']-after['mean_placement'],
             'accepted_samples':sum(h['collection']['samples'] for h in history),
             'natural_training_episodes':sum(h['collection']['natural_episodes'] for h in history),
             'training_seconds':sum(h['seconds'] for h in history)}
        rows.append(row)
        (out/'learning-seeds-progress.json').write_text(json.dumps(rows,indent=2)+'\n')
        print(json.dumps({'stage':'learning_seed_complete','seed':seed,
            'samples':row['accepted_samples'],'placement_improvement':row['mean_placement_improvement']}),flush=True)
    return {'rows':rows,'protocol':protocol,'all_cases_passed':True,
        'all_seeds_improved':all(r['mean_placement_improvement']>0 for r in rows),
        'mean_placement_improvement':float(np.mean([r['mean_placement_improvement'] for r in rows])),
        'wall_seconds':time.perf_counter()-started,
        'scope':'Genuine complete GPU simulation and CUDA PPO with three seeds; development-only paired public-practice evaluation uses the unchanged CPU oracle, with CUDA policy inference. All costs charged to the same fixed GPU allocation.'}
