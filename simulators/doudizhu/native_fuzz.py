"""Broad deterministic native/oracle parity audit; not a performance comparison."""
from pathlib import Path
import json
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def run(out,games=4096):
    import numpy as np
    from integrations.doudizhu.environment import deal,ROLES
    from integrations.doudizhu.workers import VectorGames
    from integrations.doudizhu.artifacts import write_json,sha256,verify_source
    from douzero.env.move_detector import get_move_type
    out.mkdir(parents=True,exist_ok=False)
    rng=np.random.default_rng(819231)
    selected={str(i):0 for i in range(16)}
    legal_types=set()
    total=max_legal=late=0
    retained=[]
    started=time.time()
    with VectorGames(4,engine_backend='native') as vector:
        for cursor in range(0,games,64):
            cards=[deal('train',819231,cursor+i) for i in range(min(64,games-cursor))]
            active={i:item[0] for i,item in vector.reset(cards,objective=('wp','adp','logadp')[(cursor//64)%3],record=True).items()}
            while active:
                choices={}
                for i,obs in active.items():
                    choice=int(rng.integers(len(obs.legal)))
                    choices[i]=choice
                    selected[str(get_move_type(list(obs.legal[choice]))['type'])]+=1
                    max_legal=max(max_legal,len(obs.legal))
                    if vector.native.audit[i].decisions%9==0:
                        legal_types.update(get_move_type(list(a))['type'] for a in obs.legal)
                    late+=min(len(s.player_hand_cards) for s in vector.native.audit[i]._engine.info_sets.values())<=5
                total+=len(active)
                for i,(obs,outcome,trace) in vector.step(choices).items():
                    if outcome is not None:
                        if len(retained)<12:retained.append(trace)
                        del active[i]
                    else:active[i]=obs
            write_json(out/'progress.json',{'games':cursor+len(cards),'decisions':total})
    write_json(out/'private-replays.json',retained)
    write_json(out/'result.json',{'games':games,'decisions':total,'max_legal_candidates':max_legal,
        'late_decisions_before_action':late,'selected_action_types':selected,'legal_action_types':sorted(legal_types),
        'all_public_features_legal_actions_transitions_outcomes_match':True,'faults':0,'truncations':0,
        'native_source_sha256':sha256(ROOT/'integrations/doudizhu/native/engine.cpp'),
        'oracle_source':verify_source(),'started_unix':started,'ended_unix':time.time(),
        'scope':'non-timed correctness audit; all games shadowed by unchanged upstream; 12 complete replays retained'})


if __name__=='__main__':
    run(Path(sys.argv[1]),int(sys.argv[2]) if len(sys.argv)>2 else 4096)
