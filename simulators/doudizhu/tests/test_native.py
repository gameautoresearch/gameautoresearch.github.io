"""The unmodified frozen DouZero engine remains the native port's rules oracle."""
import numpy as np
import pytest
from integrations.doudizhu.environment import Game,deal,heuristic
from integrations.doudizhu.workers import VectorGames


@pytest.mark.parametrize('objective',['wp','adp','logadp'])
@pytest.mark.parametrize('threads',[1,4])
def test_native_all_legal_features_outcomes_replay(objective,threads):
    rng=np.random.default_rng(819)
    with VectorGames(threads,engine_backend='native') as vector:
        for wave in range(3):
            cards=[deal('train',781923,32*wave+i) for i in range(32)]
            active={i:item[0] for i,item in vector.reset(cards,objective,record=True).items()}
            completed=0
            while active:
                actions={i:int(rng.integers(len(obs.legal))) for i,obs in active.items()}
                replies=vector.step(actions)
                for i,(obs,outcome,trace) in replies.items():
                    if obs is None:
                        assert outcome==trace['outcome']
                        del active[i]
                        completed+=1
                    else:
                        active[i]=obs
            assert completed==32


def test_native_bulk_heuristic_matches_natural_oracle():
    cards=[deal('practice',13923,i) for i in range(100)]
    with VectorGames(4,engine_backend='native') as vector:
        vector.reset(cards)
        outcomes=vector.native.simulate_heuristic()
    for i,c in enumerate(cards):
        game=Game(c,engine_backend='upstream')
        while (obs:=game.observe()) is not None:
            game.step(heuristic(obs))
        assert outcomes[i][1]==game.outcome()


def test_native_real_learning_samples_and_weights_exact():
    import torch
    from integrations.doudizhu.model import Policy,update
    from integrations.doudizhu.rollout import Policies,collect
    from integrations.doudizhu.environment import ROLES
    from integrations.doudizhu.runner import make_optimizers
    torch.set_num_threads(1)
    config={'architecture':{'hidden':32,'depth':2,'history':16},'learning_rate':.001}
    cards=[deal('train',9123,i) for i in range(32)]
    assignments=[{r:{'kind':'learner'} for r in ROLES} for _ in cards]
    rows=[]
    for backend,threads in [('upstream',0),('native',4)]:
        torch.manual_seed(78)
        policy=Policy(config['architecture'])
        rng=np.random.default_rng(8923)
        with VectorGames(threads,engine_backend=backend) as vector:
            result=collect(vector,Policies(policy,'cpu'),cards,assignments,rng,epsilon=.1,training=True,record=True)
        update(policy,make_optimizers(policy,config),result['samples'],256,1,rng)
        rows.append((result,{k:v.clone() for k,v in policy.state_dict().items()},rng.bit_generator.state))
    assert rows[0][0]['traces']==rows[1][0]['traces']
    assert rows[0][2]==rows[1][2]
    for role in ROLES:
        for a,b in zip(rows[0][0]['samples'][role],rows[1][0]['samples'][role]):
            for x,y in zip(a,b):
                np.testing.assert_array_equal(x,y)
    for key,value in rows[0][1].items():
        torch.testing.assert_close(value,rows[1][1][key],atol=0,rtol=0)


def test_bulk_native_collect_matches_python_rng_samples_weights_and_abort():
    import torch
    from integrations.doudizhu.model import Policy,update
    from integrations.doudizhu.rollout import Policies,collect
    from integrations.doudizhu.environment import ROLES
    from integrations.doudizhu.runner import make_optimizers
    torch.set_num_threads(1)
    config={'architecture':{'hidden':32,'depth':2,'history':16},'learning_rate':.001}
    cards=[deal('train',89231,i) for i in range(32)]
    assignments=[{r:{'kind':'learner'} for r in ROLES} for _ in cards]
    rows=[]
    for backend,threads,fast in [('upstream',0,False),('native',4,True)]:
        torch.manual_seed(713)
        policy=Policy(config['architecture'])
        policies=Policies(policy,'cpu')
        policies.native_fast=fast
        rng=np.random.default_rng(1923)
        with VectorGames(threads,engine_backend=backend,record_limit=4) as vector:
            result=collect(vector,policies,cards,assignments,rng,epsilon=.37,training=True,record=True)
        update(policy,make_optimizers(policy,config),result['samples'],256,1,rng)
        rows.append((result,{k:v.clone() for k,v in policy.state_dict().items()},rng.bit_generator.state))
    assert rows[0][0]['traces']==rows[1][0]['traces']
    assert rows[0][2]==rows[1][2]
    for role in ROLES:
        for a,b in zip(rows[0][0]['samples'][role],rows[1][0]['samples'][role]):
            for x,y in zip(a,b):
                np.testing.assert_array_equal(x,y)
    for key,value in rows[0][1].items():
        torch.testing.assert_close(value,rows[1][1][key],atol=0,rtol=0)
    class Expired:
        def expired(self):return True
    with VectorGames(4,engine_backend='native') as vector:
        result=collect(vector,policies,cards,assignments,np.random.default_rng(4),budget=Expired(),training=True)
    assert result['aborted'] and not result['outcomes']
    assert all(not len(x) for x in result['samples'].values())


def test_native_composite_hands_and_hidden_partition_boundary():
    from collections import Counter
    from integrations.doudizhu.environment import deck,ROLES
    targeted=[sum(([r]*3 for r in range(3,8)),[])+[8]*4+[20],
              sum(([r]*3 for r in range(3,7)),[])+[7]*4+[8]*2+[20,30],
              sum(([r]*4 for r in range(3,8)),[])]
    for hand in targeted:
        rest=sorted((Counter(deck)-Counter(hand)).elements())
        cards={'landlord':sorted(hand),'landlord_down':rest[:17],'landlord_up':rest[17:],
               'three_landlord_cards':sorted(hand)[:3]}
        altered={k:list(v) for k,v in cards.items()}
        altered['landlord_up'],altered['landlord_down']=altered['landlord_down'],altered['landlord_up']
        observations=[]
        for c in (cards,altered):
            with VectorGames(4,engine_backend='native') as vector:
                obs=vector.reset([c],record=True)[0][0]
                observations.append(obs)
        assert tuple(observations[0].legal)==tuple(observations[1].legal)
        for a,b in zip(observations[0].encode(),observations[1].encode()):
            np.testing.assert_array_equal(a,b)


def test_native_checkpoint_binds_port_source(tmp_path,monkeypatch):
    import torch
    import integrations.doudizhu.artifacts as artifacts
    from integrations.doudizhu.model import Policy
    from integrations.doudizhu.runner import make_optimizers
    config={'architecture':{'hidden':16,'depth':1,'history':8},'learning_rate':.001,'engine_backend':'native'}
    policy=Policy(config['architecture'])
    path=tmp_path/'native-source.pt'
    artifacts.save_checkpoint(path,policy,make_optimizers(policy,config),{},config)
    artifacts.load_checkpoint(path)
    original=artifacts.sha256
    monkeypatch.setattr(artifacts,'sha256',lambda p:'0'*64 if p.name=='engine.cpp' else original(p))
    with pytest.raises(ValueError,match='native source mismatch'):
        artifacts.load_checkpoint(path)


def test_native_reachable_airplane_quirk_rocket_passes_and_bomb():
    from collections import Counter
    from integrations.doudizhu.environment import deck
    from douzero.env.move_detector import get_move_type
    hand=sum(([r]*3 for r in range(3,7)),[])+[7]*4+[8]*2+[20,30]
    rest=sorted((Counter(deck)-Counter(hand)).elements())
    cards={'landlord':sorted(hand),'landlord_down':rest[:17],'landlord_up':rest[17:],
           'three_landlord_cards':sorted(hand)[:3]}
    # Upstream permits this generated airplane even though its detector calls it
    # TYPE_15_WRONG. Preserve the resulting bomb/pass-only response semantics.
    airplane=tuple(sum(([r]*3 for r in range(3,7)),[])+[7]*4)
    assert get_move_type(list(airplane))['type']==15
    for opening in (airplane,(20,30),(7,7,7,7)):
        with VectorGames(4,engine_backend='native') as vector:
            obs=vector.reset([cards],objective='adp',record=True)[0][0]
            obs=vector.step({0:obs.legal.index(opening)})[0][0]
            if opening==(20,30):
                assert tuple(obs.legal)==((),)
            obs=vector.step({0:obs.legal.index(())})[0][0]
            obs=vector.step({0:obs.legal.index(())})[0][0]
            assert obs.role=='landlord' and () not in obs.legal
