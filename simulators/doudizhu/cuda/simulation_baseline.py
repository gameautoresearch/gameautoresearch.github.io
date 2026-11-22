"""Direct pinned GameEnv baseline, without the integration Game/Observation wrapper."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace
import copy
import json
import multiprocessing
import time


class PublicAgent:
    def __init__(self,role):self.role=role
    def act(self,info):
        from integrations.doudizhu.environment import heuristic
        legal=tuple(sorted(set(tuple(sorted(a)) for a in info.legal_actions)))
        public=SimpleNamespace(role=self.role,hand=info.player_hand_cards,last_pid=info.last_pid,legal=legal)
        return list(legal[heuristic(public)])


def raw_games(cards):
    from integrations.doudizhu.environment import ROLES,verify_checkout
    from douzero.env.game import GameEnv
    verify_checkout()
    rows=[]
    for c in cards:
        game=GameEnv({r:PublicAgent(r) for r in ROLES});game.card_play_init(copy.deepcopy(c))
        steps=0
        while not game.game_over:
            game.step();steps+=1
            if steps>162:raise RuntimeError('upstream guard fault')
        rows.append((int(game.get_winner()=='landlord'),game.bomb_num,steps))
    return rows


def run(out):
    import torch
    import numpy as np
    from integrations.doudizhu.environment import deal,digest
    from integrations.doudizhu.workers import VectorGames
    from integrations.doudizhu.artifacts import write_json,manifest
    from integrations.doudizhu.cuda.backend import CudaBatch
    from integrations.doudizhu.cuda.experiment import configuration
    config=configuration()
    write_json(out/'manifest.json',manifest(config))
    cards=[deal('train',68193,i) for i in range(2048)]
    started=time.perf_counter()
    pool=ProcessPoolExecutor(4,mp_context=multiprocessing.get_context('spawn'))
    # Warm persistent interpreters before repeated throughput measurements; retain
    # startup separately and charge it in the complete allocated worker receipt.
    list(pool.map(raw_games,[[cards[0]] for _ in range(4)]))
    setup=time.perf_counter()-started
    rows=[]
    try:
        for repeat in range(3):
            cases=['raw-original-serial','raw-original-4workers','native-1thread','native-4threads','cuda']
            if repeat%2:cases.reverse()
            for backend in cases:
                started=time.perf_counter()
                if backend=='raw-original-serial':outcomes=raw_games(cards)
                elif backend=='raw-original-4workers':
                    parts=list(pool.map(raw_games,[cards[i*512:(i+1)*512] for i in range(4)]))
                    outcomes=[item for part in parts for item in part]
                elif backend.startswith('native'):
                    with VectorGames(1 if '1thread' in backend else 4,engine_backend='native') as vector:
                        vector.reset(cards)
                        result=vector.native.simulate_heuristic()
                        outcomes=[(int(result[i][1]['landlord_won']),result[i][1]['bombs'],result[i][1]['decisions']) for i in range(len(cards))]
                else:
                    gpu=CudaBatch(len(cards));gpu.reset(cards)
                    greedy=torch.zeros(len(cards),dtype=torch.int64,device='cuda')
                    kinds=torch.full((len(cards),),3,dtype=torch.int8,device='cuda')
                    seeds=torch.zeros(len(cards),dtype=torch.uint64,device='cuda')
                    for turn in range(163):
                        total,_,_=gpu.generate(encode=False)
                        if not total:break
                        gpu.step(gpu.choose(greedy,kinds,seeds,0.))
                    flags=gpu.outcomes();assert np.all(flags[:,0]==1)
                    outcomes=[tuple(map(int,r[1:])) for r in flags];gpu.close()
                seconds=time.perf_counter()-started
                decisions=sum(r[2] for r in outcomes)
                rows.append({'repeat':repeat,'backend':backend,'games':len(cards),'decisions':decisions,
                    'seconds':seconds,'decisions_per_second':decisions/seconds,'outcomes_sha256':digest(outcomes),
                    'worker_pool_setup_seconds':setup,'rules':'full natural games; canonical public cooperating heuristic',
                    'faults':0,'truncations':0})
                write_json(out/'simulation-matrix.json',rows)
                print(json.dumps(rows[-1]),flush=True)
    finally:pool.shutdown(wait=True,cancel_futures=True)
    assert len({r['outcomes_sha256'] for r in rows})==1
    write_json(out/'result.json',{'direct_pinned_original_GameEnv':True,'integration_game_wrapper_used_in_original':False,
        'all_outcomes_decisions_bombs_identical':True,'trials':len(rows),'upstream_training_rate_invented':False,
        'source_pin':'718a5c920bf3361e34178a38f3b80458e176b351'})
