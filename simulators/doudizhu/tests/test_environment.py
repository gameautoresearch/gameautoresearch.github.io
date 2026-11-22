from collections import Counter
from copy import deepcopy
from dataclasses import fields
import json
from pathlib import Path

import numpy as np
import pytest

from integrations.doudizhu.environment import (
    ROLES, Game, Observation, action_bits, deal, deck, digest, heuristic, replay, validate_deal,
)
from douzero.env.env import get_obs
from douzero.env.game import GameEnv


def test_seed_domains_and_conservation():
    hashes = set()
    for split in ('train', 'practice', 'heldout'):
        for index in range(100):
            cards = deal(split, 111, index)
            validate_deal(cards)
            assert cards == deal(split, 111, index)
            hashes.add(digest(cards))
    assert len(hashes) == 300
    with pytest.raises(ValueError):
        deal('private', 111, 0)
    with pytest.raises(ValueError):
        validate_deal({r: [] for r in ROLES})


def test_public_codec_matches_upstream_all_roles_and_history():
    seen = set()
    game = Game(deal('train', 17, 5))
    while (obs := game.observe()) is not None:
        raw = game._engine.game_infoset
        oracle = get_obs(raw)
        x, z, actions = obs.encode()
        np.testing.assert_array_equal(x, oracle['x_no_action'])
        np.testing.assert_array_equal(z, oracle['z'])
        for i, action in enumerate(obs.legal):
            j = raw.legal_actions.index(list(action))
            np.testing.assert_array_equal(np.concatenate((x, actions[i])), oracle['x_batch'][j])
        seen.add(obs.role)
        game.step(heuristic(obs))
    assert seen == set(ROLES)


def test_hidden_repartition_invariance_for_each_role():
    game = Game(deal('train', 99, 0), cache_observation=False)
    for _ in range(3):
        obs = game.observe()
        others = [r for r in ROLES if r != obs.role]
        engine = game._engine
        # Reassign opponent cards without changing either public hand count or own
        # cards. Preserve the publicly known landlord bottom cards when applicable.
        left = engine.info_sets[others[0]].player_hand_cards
        right = engine.info_sets[others[1]].player_hand_cards
        bottom = set(engine.three_landlord_cards)
        a = next(i for i, c in enumerate(left) if others[0] != 'landlord' or c not in bottom)
        b = next(i for i, c in enumerate(right) if c != left[a] and (others[1] != 'landlord' or c not in bottom))
        left[a], right[b] = right[b], left[a]
        left.sort()
        right.sort()
        engine.game_infoset = engine.get_infoset()
        altered = game.observe()
        assert obs == altered
        for before, after in zip(obs.encode(), altered.encode()):
            np.testing.assert_array_equal(before, after)
        # Raw infosets do expose hands; the public type must not carry any handle.
        assert hasattr(engine.game_infoset, 'all_handcards')
        assert not hasattr(altered, 'all_handcards')
        assert not ({'seed', 'deal_id', 'deal', 'engine', 'other_hand_cards'} & {f.name for f in fields(Observation)})
        game.step(heuristic(altered))


def test_invalid_actions_and_terminal_errors():
    game = Game(deal('practice', 3, 0))
    with pytest.raises(ValueError):
        game.step(-1)
    with pytest.raises(ValueError):
        game.step(len(game.observe().legal))
    with pytest.raises(ValueError):
        game.step(0.5)
    with pytest.raises(RuntimeError):
        game.outcome()
    while (obs := game.observe()) is not None:
        game.step(heuristic(obs))
    with pytest.raises(RuntimeError):
        game.step(0)


@pytest.mark.parametrize('objective', ['wp', 'adp', 'logadp'])
def test_rewards_replay_and_engine_differential(objective):
    rng = np.random.default_rng(72)
    for index in range(8):
        cards = deal('practice', 3, index)
        game = Game(cards, objective, record=True)
        # Independent unchanged upstream simulator receives exactly the chosen card plays.
        oracle = GameEnv(game.players)
        oracle.card_play_init(deepcopy(cards))
        while (obs := game.observe()) is not None:
            legal = {tuple(sorted(a)) for a in oracle.game_infoset.legal_actions}
            assert set(obs.legal) == legal
            action = int(rng.integers(len(obs.legal)))
            role = obs.role
            cards_played = list(obs.legal[action])
            game.step(action)
            game.players[role].set_action(cards_played)
            oracle.step()
            assert game._engine.game_over == oracle.game_over
            for r in ROLES:
                assert game._engine.info_sets[r].player_hand_cards == oracle.info_sets[r].player_hand_cards
        outcome = game.outcome()
        sign = 1 if oracle.get_winner() == 'landlord' else -1
        bombs = oracle.bomb_num
        multiplier = {'wp': 1, 'adp': 2**bombs, 'logadp': bombs+1}[objective]
        assert outcome['returns']['landlord'] == sign * multiplier
        assert outcome['returns']['landlord_up'] == outcome['returns']['landlord_down'] == -sign * multiplier
        assert sum(outcome['scores'].values()) == 0
        assert outcome['scores']['landlord'] == sign*2*2**bombs
        assert replay(game.trajectory()) == game.trajectory()
        broken = deepcopy(game.trajectory())
        broken['steps'][0]['observation_sha256'] = '0'*64
        with pytest.raises(ValueError):
            replay(broken)


def test_bombs_rocket_pass_reset_and_card_codec():
    # Put both jokers in the landlord's hand via swaps while conserving a real deck.
    cards = deal('train', 11, 1)
    for joker in (20, 30):
        if joker not in cards['landlord']:
            r = next(r for r in ROLES[1:] if joker in cards[r])
            replacement = next(c for c in cards['landlord'] if c not in cards['three_landlord_cards'] and c < 20)
            cards['landlord'].remove(replacement)
            cards['landlord'].append(joker)
            cards[r].remove(joker)
            cards[r].append(replacement)
    cards = {k: sorted(v) for k,v in cards.items()}
    game = Game(cards, 'adp')
    game.step(game.observe().legal.index((20, 30)))
    assert game.observe().legal == ((),)
    assert game.observe().bombs == 1
    game.step(0)
    assert game.observe().legal == ((),)
    game.step(0)
    obs = game.observe()
    assert obs.role == 'landlord'
    assert () not in obs.legal
    assert len(obs.legal) > 1
    assert action_bits((20,30)).sum() == 2
    assert action_bits(()).sum() == 0
    assert action_bits((3,3,3,3)).sum() == 4


def test_canonical_unique_actions_and_public_inferable_union():
    for index in range(10):
        game = Game(deal('train', 39, index))
        while (obs := game.observe()) is not None:
            assert list(obs.legal) == sorted(set(obs.legal))
            assert Counter(obs.public_infoset().other_hand_cards) == Counter(game._engine.game_infoset.other_hand_cards)
            game.step(heuristic(obs))


def test_public_feature_and_view_caches_preserve_replay_and_are_readonly():
    traces=[]
    for cached in (False,True):
        game=Game(deal('train',103,0),record=True,cache_observation=cached)
        while (obs:=game.observe()) is not None:
            before=obs.hash()
            a=obs.encode(cache=cached)
            b=obs.encode(cache=False)
            assert before==obs.hash()
            for x,y in zip(a,b):
                np.testing.assert_array_equal(x,y)
                if cached:
                    assert not x.flags.writeable
            game.step(heuristic(obs))
        traces.append(game.trajectory())
    assert traces[0]==traces[1]


@pytest.mark.parametrize('seed', [17, 91, 293])
def test_direct_encoder_and_live_view_full_natural_game_differential(seed):
    rng = np.random.default_rng(seed)
    max_candidates = 0
    for index in range(12):
        reference = Game(deal('train',seed,index),engine_backend='upstream',record=True)
        candidate = Game(deal('train',seed,index),engine_backend='view',record=True)
        old = []
        while (a := reference.observe()) is not None:
            b = candidate.observe()
            assert a == b
            for x,y in zip(a.encode(backend='upstream'),b.encode(backend='direct')):
                np.testing.assert_array_equal(x,y)
                assert x.dtype == y.dtype == np.int8
            old.append((b,b.hash(),[x.copy() for x in b.encode()]))
            max_candidates = max(max_candidates,len(b.legal))
            action = int(rng.integers(len(a.legal)))
            reference.step(action)
            candidate.step(action)
        assert candidate.observe() is None
        assert reference.trajectory() == candidate.trajectory()
        # Live private aliases cannot mutate previously exported observations/features.
        for obs,sha,arrays in old:
            assert obs.hash() == sha
            for x,y in zip(obs.encode(),arrays):
                np.testing.assert_array_equal(x,y)
    assert max_candidates >= 50
