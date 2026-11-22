"""Private CPU engine, explicit public policy boundary, deterministic deal/replay codec."""
from __future__ import annotations

import hashlib
import json
import random
import subprocess
import sys
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass, fields
from functools import lru_cache, cached_property
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'candidates' / 'DouZero'
PIN = '718a5c920bf3361e34178a38f3b80458e176b351'


def verify_checkout():
    lock = json.loads((Path(__file__).resolve().parent/'source.lock.json').read_text())
    if lock['commit'] != PIN:
        raise RuntimeError('DouZero lockfile commit pin mismatch')
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=SOURCE,text=True).strip() != PIN:
        raise RuntimeError('DouZero source commit pin mismatch')
    for name, expected in lock['files'].items():
        if hashlib.sha256((SOURCE/name).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f'DouZero source drift: {name}')
    actual = {str(p.relative_to(SOURCE)) for p in SOURCE.rglob('*.py') if '.git' not in p.parts}
    if actual != {name for name in lock['files'] if name.endswith('.py')}:
        raise RuntimeError('Unpinned Python file in DouZero checkout')
    return lock


verify_checkout()  # Before executing any pinned simulator source, including in workers.
sys.path.insert(0, str(SOURCE))
from douzero.env.game import GameEnv
from douzero.env.env import DummyAgent, deck, get_obs, _cards2array

ROLES = ('landlord', 'landlord_down', 'landlord_up')  # actual clockwise action order
RULESET = 'douzero-fixed-role-v1'
SPLITS = ('train', 'practice', 'heldout')


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def seed_value(split: str, seed: int, index: int) -> int:
    if split not in SPLITS or seed < 0 or index < 0:
        raise ValueError('Invalid split, seed or deal index')
    # Independent domains; shuffle algorithm is explicitly implemented below.
    return int(digest([RULESET, split, seed, index]), 16)


def deal(split: str, seed: int, index: int) -> dict:
    cards = deck.copy()
    rng = random.Random(seed_value(split, seed, index))
    # Fisher-Yates with Python MT19937 randrange; pinned Python for exact reproduction.
    for i in range(len(cards) - 1, 0, -1):
        j = rng.randrange(i + 1)
        cards[i], cards[j] = cards[j], cards[i]
    return {k: sorted(v) for k, v in {
        'landlord': cards[:20], 'landlord_up': cards[20:37],
        'landlord_down': cards[37:54], 'three_landlord_cards': cards[17:20],
    }.items()}


def validate_deal(cards):
    if set(cards) != set(ROLES) | {'three_landlord_cards'}:
        raise ValueError('Invalid deal fields')
    if [len(cards[r]) for r in ROLES] != [20, 17, 17]:
        raise ValueError('Invalid role hand sizes')
    if Counter(sum((cards[r] for r in ROLES), [])) != Counter(deck):
        raise ValueError('Deck conservation failed')
    if len(cards['three_landlord_cards']) != 3 or (
        Counter(cards['three_landlord_cards']) - Counter(cards['landlord'])
    ):
        raise ValueError('Invalid landlord bottom cards')


@lru_cache(maxsize=32768)
def action_bits(action: tuple[int, ...]):
    bits = _cards2array(action)
    bits.flags.writeable = False
    return bits


@dataclass(frozen=True)
class Observation:
    """Only player-visible data. No seed, deal ID, engine object or opponent hand."""
    role: str
    hand: tuple[int, ...]
    remaining: tuple[int, int, int]
    played: tuple[tuple[int, ...], ...]
    history: tuple[tuple[int, ...], ...]
    last_by_role: tuple[tuple[int, ...], ...]
    last_move: tuple[int, ...]
    last_pid: str
    bombs: int
    bottom_remaining: tuple[int, ...]
    legal: tuple[tuple[int, ...], ...]

    def public_infoset(self):
        # The union of opponents' remaining cards is fully inferable from the deck,
        # one's own hand and all public played cards. Never copy engine other_hand_cards.
        unseen = Counter(deck) - Counter(self.hand)
        unseen.subtract(Counter(c for cards in self.played for c in cards))
        if any(n < 0 for n in unseen.values()):
            raise ValueError('Invalid public card accounting')
        return SimpleNamespace(
            player_position=self.role, player_hand_cards=list(self.hand),
            other_hand_cards=sorted(unseen.elements()),
            num_cards_left_dict=dict(zip(ROLES, self.remaining)),
            played_cards={r: list(v) for r, v in zip(ROLES, self.played)},
            last_move=list(self.last_move),
            last_move_dict={r: list(v) for r, v in zip(ROLES, self.last_by_role)},
            card_play_action_seq=[list(a) for a in self.history],
            bomb_num=self.bombs, legal_actions=[[]],
        )

    def _encode_fresh(self):
        """Unchanged upstream oracle, retained for differential profiles/tests."""
        obs = get_obs(self.public_infoset())
        actions = np.stack([action_bits(a) for a in self.legal])
        return obs['x_no_action'], obs['z'], actions

    def _encode_direct(self, include_actions=True):
        # Identical upstream feature order without constructing unused x_batch,
        # z_batch, mutable infosets, Counters of history or repeated history codecs.
        counts = np.bincount(self.hand, minlength=31)
        for played in self.played:
            counts += np.bincount(played, minlength=31)
        remaining = _DECK_COUNTS - counts
        if (remaining < 0).any():
            raise ValueError('Invalid public card accounting')
        unseen = tuple(np.repeat(_RANKS, remaining[_RANKS]))
        bits = action_bits
        role = ROLES.index(self.role)
        x = np.zeros(319 if role == 0 else 430, dtype=np.int8)
        if role == 0:
            card_fields = (self.hand, unseen, self.last_move, self.played[2], self.played[1])
            count_fields = ((self.remaining[2],17),(self.remaining[1],17))
        else:
            teammate = 3-role
            card_fields = (self.hand, unseen, self.played[0], self.played[teammate],
                           self.last_move, self.last_by_role[0], self.last_by_role[teammate])
            count_fields = ((self.remaining[0],20),(self.remaining[teammate],17))
        cursor = 0
        for cards in card_fields:
            x[cursor:cursor+54] = bits(cards)
            cursor += 54
        for number, width in count_fields:
            x[cursor+number-1] = 1
            cursor += width
        x[cursor+self.bombs] = 1
        z = np.zeros((15,54),dtype=np.int8)
        tail = self.history[-15:]
        for i, action in enumerate(tail, 15-len(tail)):
            z[i] = bits(action)
        return x, z.reshape(5,162), np.stack([bits(a) for a in self.legal]) if include_actions else None

    @cached_property
    def _encoding(self):
        encoded = self._encode_direct()
        for array in encoded:
            array.flags.writeable = False
        return encoded

    @cached_property
    def _upstream_encoding(self):
        encoded = self._encode_fresh()
        for array in encoded:
            array.flags.writeable = False
        return encoded

    def encode(self, cache=True, backend='direct'):
        if backend == 'direct':
            return self._encoding if cache else self._encode_direct()
        if backend == 'upstream':
            return self._upstream_encoding if cache else self._encode_fresh()
        raise ValueError('Unknown public encoder')

    def hash(self):
        # Encoding caches contain only derived public features and are deliberately
        # excluded from public replay identities.
        return digest({f.name:getattr(self,f.name) for f in fields(self)})


@dataclass(frozen=True)
class CompactObservation:
    """Public worker wire format; public static/history features, no private state.

    Full public replay hashing stays in Game, before this transport projection.
    Byte-backed x/z remain read-only after pickling and exclude mutable cache state.
    Legal multisets preserve canonical action indices and heuristic compatibility.
    """
    role: str
    hand: tuple[int, ...]
    last_pid: str
    legal: tuple[tuple[int, ...], ...]
    features: bytes

    @classmethod
    def from_observation(cls, obs):
        if obs is None:
            return None
        x,z,_ = obs._encode_direct(include_actions=False)
        return cls(obs.role,obs.hand,obs.last_pid,obs.legal,x.tobytes()+z.tobytes())

    def _encode(self):
        width = 319 if self.role=='landlord' else 430
        data = np.frombuffer(self.features,dtype=np.int8)
        actions = np.stack([action_bits(a) for a in self.legal])
        actions.flags.writeable = False
        return data[:width],data[width:].reshape(5,162),actions

    @cached_property
    def _encoding(self):
        return self._encode()

    def encode(self, cache=True, backend='direct'):
        if backend!='direct':
            raise ValueError('Compact observations require the direct encoder')
        return self._encoding if cache else self._encode()


_RANKS = np.array(sorted(set(deck)))
_DECK_COUNTS = np.bincount(deck, minlength=31)


class ViewGameEnv(GameEnv):
    """Private live infoset: upstream transitions and legal generation, no deepcopy.

    The only agent in this engine is our DummyAgent. It reads legal_actions before
    mutation. Game.observe freezes the policy boundary into immutable tuples; live
    infosets never cross that boundary or leave the simulator process.
    """
    def get_infoset(self):
        s = self.info_sets[self.acting_player_position]
        s.last_pid = self.last_pid
        s.legal_actions = self.get_legal_card_play_actions()
        s.bomb_num = self.bomb_num
        s.last_move = self.get_last_move()
        s.last_two_moves = self.get_last_two_moves()
        s.last_move_dict = self.last_move_dict
        s.num_cards_left_dict = {r:len(self.info_sets[r].player_hand_cards) for r in ROLES}
        # Retain private compatibility fields for the upstream differential oracle.
        s.other_hand_cards = sum((self.info_sets[r].player_hand_cards for r in ROLES
                                  if r != self.acting_player_position), [])
        s.played_cards = self.played_cards
        s.three_landlord_cards = self.three_landlord_cards
        s.card_play_action_seq = self.card_play_action_seq
        s.all_handcards = {r:self.info_sets[r].player_hand_cards for r in ROLES}
        return s


class Game:
    def __init__(self, cards: dict, objective='wp', record=False, cache_observation=True,
                 engine_backend='view'):
        if objective not in ('wp', 'adp', 'logadp'):
            raise ValueError('Invalid objective')
        validate_deal(cards)
        self.cards = deepcopy(cards)
        self.objective = objective
        self.record = record
        self.trace = []
        self.decisions = 0
        self.cache_observation = cache_observation
        self._observation = None
        self.players = {r: DummyAgent(r) for r in ROLES}
        if engine_backend not in ('view', 'upstream'):
            raise ValueError('Unknown engine backend')
        self._engine = (ViewGameEnv if engine_backend == 'view' else GameEnv)(self.players)
        self._engine.card_play_init(deepcopy(cards))

    def observe(self):
        if self._engine.game_over:
            return None
        if self.cache_observation and self._observation is not None:
            return self._observation
        s = self._engine.game_infoset
        # Canonicalize and deduplicate legal card multisets; upstream ordering includes
        # set iteration and can contain repeated wing combinations. Actions use cards,
        # not persistent global indices. Lexicographic order gives stable tie breaks.
        legal = tuple(sorted(set(tuple(sorted(a)) for a in s.legal_actions)))
        observation = Observation(
            s.player_position, tuple(s.player_hand_cards),
            tuple(s.num_cards_left_dict[r] for r in ROLES),
            tuple(tuple(s.played_cards[r]) for r in ROLES),
            tuple(tuple(a) for a in s.card_play_action_seq),
            tuple(tuple(s.last_move_dict[r]) for r in ROLES),
            tuple(s.last_move), s.last_pid, s.bomb_num,
            tuple(s.three_landlord_cards), legal,
        )
        if self.cache_observation:
            self._observation = observation
        return observation

    def step(self, index):
        obs = self.observe()
        if obs is None:
            raise RuntimeError('Step after natural terminal')
        if not isinstance(index, (int, np.integer)) or not 0 <= index < len(obs.legal):
            raise ValueError('Illegal action index')
        action = list(obs.legal[index])
        if self.record:
            self.trace.append({'role': obs.role, 'observation_sha256': obs.hash(), 'action': action})
        self.players[obs.role].set_action(action)
        self._engine.step()
        self._observation = None
        self.decisions += 1
        # Every non-pass removes >=1 card; at most two passes between non-passes.
        # A guard failure is a fault, never scored as a game terminal.
        if self.decisions > 162:
            raise RuntimeError('Engine exceeded finite shedding-game bound')
        return self.observe(), self.outcome() if self._engine.game_over else None

    def outcome(self):
        if not self._engine.game_over:
            raise RuntimeError('No return before natural terminal')
        sign = 1 if self._engine.get_winner() == 'landlord' else -1
        bombs = self._engine.bomb_num
        unit = 2 ** bombs
        target = {'wp': 1, 'adp': unit, 'logadp': bombs + 1}[self.objective]
        return {
            'landlord_won': sign == 1, 'bombs': bombs, 'decisions': self.decisions,
            'returns': {r: float(sign * target * (1 if r == 'landlord' else -1)) for r in ROLES},
            'scores': {r: sign * unit * (2 if r == 'landlord' else -1) for r in ROLES},
            'terminal': 'natural_empty_hand', 'truncated': False,
        }

    def trajectory(self):
        return {'format': 1, 'ruleset': RULESET, 'objective': self.objective,
                'deal': self.cards, 'deal_sha256': digest(self.cards),
                'steps': self.trace, 'outcome': self.outcome()}


def replay(trajectory):
    if trajectory['ruleset'] != RULESET or digest(trajectory['deal']) != trajectory['deal_sha256']:
        raise ValueError('Replay rules/deal mismatch')
    game = Game(trajectory['deal'], trajectory['objective'], record=True)
    for step in trajectory['steps']:
        obs = game.observe()
        if obs is None or obs.role != step['role'] or obs.hash() != step['observation_sha256']:
            raise ValueError('Replay public observation mismatch')
        game.step(obs.legal.index(tuple(step['action'])))
    if game.outcome() != trajectory['outcome']:
        raise ValueError('Replay return mismatch')
    return game.trajectory()


def heuristic(obs):
    # Public-only, simple cooperating shedding baseline, not the stronger RLCard bot.
    if obs.role != 'landlord' and obs.last_pid != 'landlord' and () in obs.legal:
        # Take an immediate win even when teammate leads.
        for i, a in enumerate(obs.legal):
            if len(a) == len(obs.hand):
                return i
        return obs.legal.index(())
    # Shed most cards; preserve high cards on ties. Deterministic and transparent.
    return min(range(len(obs.legal)), key=lambda i: (-len(obs.legal[i]), sum(obs.legal[i]), obs.legal[i]))
