import numpy as np
from integrations.doudizhu.environment import Game,deal
from integrations.doudizhu.cuda.reference import CPUReference


def test_bulk_deal_packing_preserves_all_rank_counts_and_validation():
    from collections import Counter
    import pytest
    from integrations.doudizhu.cuda.reference import packed_deals
    from integrations.doudizhu.environment import ROLES
    from integrations.doudizhu.native_backend import RANKS
    cards=[deal('train',771,i) for i in range(129)]
    hands,bottom=packed_deals(cards)
    for g,c in enumerate(cards):
        for r,role in enumerate(ROLES):
            assert hands[g,r].tolist()==[Counter(c[role])[rank] for rank in RANKS]+[0]
        assert bottom[g].tolist()==[Counter(c['three_landlord_cards'])[rank] for rank in RANKS]+[0]
    assert packed_deals([])[0].shape==(0,3,16)
    cards[0]['landlord'][0]=99
    with pytest.raises(ValueError):packed_deals(cards)


def test_shared_cpu_cuda_header_matches_frozen_engine():
    rng=np.random.default_rng(7923)
    for i in range(128):
        cards=deal('train',7923,i)
        oracle=Game(cards,engine_backend='upstream')
        native=CPUReference(cards)
        while (obs:=oracle.observe()) is not None:
            legal,x,z=native.observe()
            assert legal==obs.legal
            a,b,_=obs.encode(backend='upstream')
            np.testing.assert_array_equal(a,x)
            np.testing.assert_array_equal(b,z)
            index=int(rng.integers(len(legal)))
            native.step(index)
            oracle.step(index)
        assert native.observe() is None
        assert bool(native.raw.won)==oracle.outcome()['landlord_won']
        assert native.raw.bombs==oracle.outcome()['bombs']
        assert native.raw.steps==oracle.outcome()['decisions']


def test_counter_rng_host_vector_matches_shared_cpu_device_rule():
    from integrations.doudizhu.cuda.actor_rng import words,uniform,explore,CounterGameRNG
    from integrations.doudizhu.environment import heuristic
    rng=np.random.default_rng(7293)
    cards=deal('train',7293,0)
    oracle=Game(cards,engine_backend='upstream')
    reference=CPUReference(cards)
    for turn in range(20):
        obs=oracle.observe()
        legal,_,_=reference.observe()
        seeds=rng.integers(0,np.iinfo(np.uint64).max,size=64,dtype=np.uint64)
        counts=np.full(64,len(legal),dtype=np.int64)
        greedy=np.full(64,0,dtype=np.int32)
        choices=explore(seeds,turn,counts,greedy,.37)
        random=uniform(seeds,turn,counts)
        for i,seed in enumerate(seeds):
            scalar=CounterGameRNG(seed,turn)
            assert scalar.word(1)==words(seeds,turn,1)[i]
            assert scalar.integers(len(legal))==random[i]
            assert (scalar.random()<.37)==bool(words(seeds,turn,1)[i]<int(.37*(1<<32)))
            assert choices[i]==reference.choose(len(legal),0,1,seed,turn,.37)
            assert random[i]==reference.choose(len(legal),0,2,seed,turn,.37)
            assert heuristic(obs)==reference.choose(len(legal),0,3,seed,turn,.37)
        # Exercise rejection, not just power-of-two bounds.
        big=np.full(64,1073741825,dtype=np.int64)
        wide=uniform(seeds,turn,big)
        for i,seed in enumerate(seeds):
            assert CounterGameRNG(seed,turn).integers(int(big[i]))==wide[i]
            assert wide[i]==reference.choose(int(big[i]),0,2,seed,turn,.37)
        chosen=int(random[0])
        reference.step(chosen)
        oracle.step(chosen)
        if oracle.observe() is None:break
