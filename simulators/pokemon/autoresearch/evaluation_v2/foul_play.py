"""Foul Play offline adapter. Input is one player's request and channel history only."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import random
import re
import sys
import time


def normalize(value):
    return ''.join(c for c in value.lower() if c.isalnum())


def translate(decision,request,mask,battle=None):
    """Exactly one of the existing14 actions; an unmappable/illegal search action is a fault."""
    if decision.startswith('switch '):
        name=decision[len('switch '):]
        if name.isdigit():index=int(name)
        elif battle is not None:
            matches=[p.index for p in battle.user.reserve if p.name==name]
            if len(matches)!=1:raise ValueError('Ambiguous Foul Play switch choice')
            index=matches[0]
        else:
            matches=[i+1 for i,p in enumerate(request['side']['pokemon']) if normalize(p['details'].split(',')[0])==normalize(name)]
            if len(matches)!=1:raise ValueError('Unmapped Foul Play switch choice')
            index=matches[0]
        action=7+index
    else:
        tera=decision.endswith('-tera');move=decision.removesuffix('-tera')
        moves=(request.get('active') or [{}])[0].get('moves',[])
        matches=[i for i,m in enumerate(moves) if normalize(m.get('id',m['move']))==normalize(move)]
        if len(matches)!=1:raise ValueError('Unmapped Foul Play move choice: '+decision)
        action=matches[0]+(4 if tera else 0)
    if type(action) is not int or not 0<=action<14 or not mask[action]:
        raise ValueError('Foul Play emitted a masked action: '+decision)
    return action


def published_team_sizes(lines):
    """Public `|teamsize|pN|K` lines; Showdown sends both before `|start`."""
    sizes={}
    for line in lines:
        if line.startswith('|teamsize|'):
            _,_,side,size=line.split('|',3);sizes[side]=int(size)
    return sizes


def team_size_mode(base,team_size):
    """Upstream random-battle beliefs fill the opponent to six members. For a Max Team
    Size format fill to the published size instead, with the same sampler/constraints."""
    from fp.search import random_battles
    def populate(battle):
        existing=list(battle.opponent.reserve)+([battle.opponent.active] if battle.opponent.active is not None else [])
        while len(existing)<team_size:
            pkmn=random_battles.sample_randombattle_pokemon(existing,battle.mode.datasets)
            existing.append(pkmn);battle.opponent.reserve.append(pkmn)
    class TeamSizeRandomBattleMode(base):
        def prepare_battles(self,battle,num_battles):
            upstream=random_battles.populate_randombattle_unrevealed_pkmn
            random_battles.populate_randombattle_unrevealed_pkmn=populate
            try:return random_battles.prepare_random_battles(battle,num_battles)
            finally:random_battles.populate_randombattle_unrevealed_pkmn=upstream
    return TeamSizeRandomBattleMode


class FoulPlay:
    def __init__(self,source,sets,mode='fixed-work',iterations=1000,time_ms=10,samples=2,seed=20261109):
        source=Path(source)
        sys.path.insert(0,str(source))
        from fp.battle.state import Battle
        from fp.modes.random_battle import RandomBattleMode
        from fp.config import FoulPlayConfig
        FoulPlayConfig.pokemon_format='gen9randombattle'
        self.Battle=Battle;self.mode_class=RandomBattleMode
        self.raw_sets=json.loads(Path(sets).read_text());self.battle=None
        self.mode=mode;self.iterations=iterations;self.time_ms=time_ms;self.samples=samples;self.seed=seed
        self.decisions=0;self.search_seconds=0.;self.search_visits=[];self.input_hashes=[]

    def update(self,visible):
        from fp.battle.protocol import process_battle_updates
        request=deepcopy(visible['request'])
        # The local BattleStream omits server transport rqid. Derive it only from
        # this player's protocol cursor, never from engine internals or RNG/seed.
        request['rqid']=visible['protocol_offset']
        lines=[line for text in visible['protocol'] for line in text.split('\n')]
        lines=[line for line in lines if not line.startswith('|request|')]
        if self.battle is None:
            b=self.Battle('offline-public-channel');b.pokemon_format='gen9randombattle';b.generation='gen9'
            own=request['side']['id'];sizes=published_team_sizes(lines);team_size=sizes.get('p2' if own=='p1' else 'p1',6)
            if team_size!=6 and (sizes!={'p1':team_size,'p2':team_size} or len(request['side']['pokemon'])!=team_size):
                raise ValueError('Published team sizes differ from own request')
            b.battle_type=b.format_spec.battle_type;b.mode=(self.mode_class if team_size==6 else team_size_mode(self.mode_class,team_size))()
            b.mode.datasets.raw_pkmn_sets=deepcopy(self.raw_sets);b.mode.datasets._initialize_pkmn_sets()
            b.mode.datasets.pkmn_mode='gen9randombattle'
            b.user.name=request['side']['id'];b.opponent.name='p2' if b.user.name=='p1' else 'p1'
            b.user.initialize_first_turn_user_from_json(request);b.started=True
            # Upstream random-battle initialization skips its own initial switch;
            # own complete team/state was supplied by the legal own request.
            start=next((i for i,line in enumerate(lines) if line=='|start'),-1)
            lines=lines[start+1:]
            lines=[line for line in lines if not line.startswith('|switch|'+b.user.name)]
            self.battle=b
        b=self.battle;b.msg_list=lines
        # Public form-change callbacks may consult the current own request (e.g.
        # initial Minior Shields Down). It must exist before protocol replay,
        # rather than being None on the first view or stale on subsequent views.
        b.request_json=request;b.rqid=request['rqid']
        process_battle_updates(b)
        b.force_switch=any(request.get('forceSwitch',[]));b.wait=bool(request.get('wait'))
        b.user.update_from_request_json(request)
        return request

    def choose(self,visible):
        if set(visible)!={'seat','request','protocol','protocol_offset','mask','turn','termination_policy'}:
            raise ValueError('Only the explicit player-channel interface is accepted')
        if visible['request']['side']['id'] != ('p1' if visible['seat']==0 else 'p2'):
            raise ValueError('Foreign player request rejected')
        visible=deepcopy(visible)
        visible['protocol']=[re.sub(r'^\|t:\|[0-9]+$', '|t:|0',text,flags=re.MULTILINE) for text in visible['protocol']]
        from fp.search.poke_engine_helpers import battle_to_poke_engine_state
        from fp.battle.state import LastUsedMove
        from poke_engine import monte_carlo_tree_search
        request=self.update(visible)
        digest=hashlib.sha256(json.dumps(visible,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        self.input_hashes.append(digest)
        previous=random.getstate();random.seed(int(digest[:16],16)^self.seed)
        started=time.monotonic();policies={};counts=[];sample_hashes=[]
        try:
            beliefs=self.battle.mode.prepare_battles(deepcopy(self.battle),self.samples)
            for index,(belief,chance) in enumerate(beliefs):
                state=battle_to_poke_engine_state(belief)
                sample_hashes.append(hashlib.sha256(state.to_string().encode()).hexdigest())
                os.environ['POKE_ENGINE_TEST_SEED']=str((int(digest[:16],16)^self.seed^index)&((1<<64)-1))
                result=monte_carlo_tree_search(state,duration_ms=self.time_ms,
                    iterations=self.iterations if self.mode=='fixed-work' else 0,threads=1)
                counts.append(result.total_visits)
                for option in result.side_one:
                    policies[option.move_choice]=policies.get(option.move_choice,0.)+chance*option.visits/result.total_visits
            # Deterministic argmax aggregation; no hidden fallback or rescue action.
            decision=min(policies,key=lambda key:(-policies[key],key))
            action=translate(decision,request,visible['mask'],self.battle)
            self.battle.user.last_selected_move=LastUsedMove(self.battle.user.active.name,decision.removesuffix('-tera'),self.battle.turn)
        finally:
            random.setstate(previous)
        elapsed=time.monotonic()-started;self.decisions+=1;self.search_seconds+=elapsed;self.search_visits.append(counts)
        return action,{'decision':decision,'seconds':elapsed,'visits':counts,'input_sha256':digest,
            'belief_state_sha256':sample_hashes,'mode':self.mode,'samples':self.samples,'threads':1}
