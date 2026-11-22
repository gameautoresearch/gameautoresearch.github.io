"""Fresh public scripted Battlegrounds films, verified by independent action replay.

Only projected visible card/board summaries and combat frames enter the website.
Full states/RNG, card database, private seeds, checkpoints and art stay out.
"""
import argparse
import functools
import hashlib
import json
from pathlib import Path
import taverngym as tg

ROOT = Path(__file__).resolve().parents[2]
RULES = dict(num_players=8, content='Implemented', turn_action_caps=[30],
             max_actions_per_turn=30, max_turns=100, record_log=True)
KEYWORDS = ['Taunt','Divine shield','Windfury','Reborn','Venomous','Poisonous',
            'Stealth','Magnetic','Mega windfury','Cleave','Immune','Immune attacking']
RACES = ['Beast','Demon','Dragon','Elemental','Mech','Murloc','Pirate','Quilboar','Undead','Aberration','Naga']

def digest(v):
    return hashlib.sha256(json.dumps(v, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

def write(p,v):
    p.write_text(json.dumps(v, separators=(',', ':'), allow_nan=False)+'\n')

@functools.lru_cache(None)
def name(card):
    return tg.card_info((card & 0x0fff) + 1)[1] if card is not None else 'Choosing hero'

def minion(m):
    return dict(eid=m['eid'], name=name(m['card']), attack=m['atk'], health=m['hp'],
                maxHealth=m['max_hp'], tier=m['tier'], golden=m['golden'],
                shield=m['shield_hits'] > 0, dead=m['destroyed'] or m['hp'] <= 0,
                keywords=[s for i,s in enumerate(KEYWORDS) if m['kw'] & (1 << i)],
                races=[s for i,s in enumerate(RACES) if m['races'] & (1 << i)])

def players(s):
    return [dict(seat=i,hero=name(p['hero']),health=p['health'],armor=p['armor'],
                 alive=p['alive'],placement=p['placement'],tier=p['tier'],gold=p['gold'],
                 board=[minion(m) for m in p['board']]) for i,p in enumerate(s['players'])]

def film_round(before, after):
    return dict(turn=before['turn'],before=players(before),after=players(after),
        combats=[dict(a=c['a'],b=c['b'],winner=c['winner'],damage=c['damage'],
            frames=[dict(boards=[[minion(m) for m in board] for board in f['boards']],
                         attacks=f['attacks']) for f in c['frames']]) for c in after['last_combats']])

def valid(g):
    s=json.loads(g.state_json())
    if (s.get('stopped_reason') or s['turn_limit_end'] or s['truncated_combats']
        or s.get('truncated_recruit_contexts',0)):
        raise AssertionError('Fault/truncation cannot be published as a natural demo')
    return s

def generate(out):
    if not tg.build_is_current(ROOT/'candidates/AutoTavern'):
        raise RuntimeError('Rebuild the source-bound native engine first')
    out.mkdir(parents=True,exist_ok=False)
    entries=[]; receipts=[]
    for i,seed in enumerate([2026110821,2026110822,2026110823],1):
        g=tg.Game(seed,json.dumps(RULES)); actions=[]; rounds=[]; boundaries=[]
        for _ in range(30000):
            if g.is_over(): break
            acted=False
            for seat in range(8):
                while g.needs_action(seat):
                    acted=True; a=g.bot_action(seat)
                    if not g.legal_mask(seat)[a]: raise AssertionError('Illegal scripted action')
                    before=json.loads(g.state_json())
                    turn=g.turn;g.step(seat,a);actions.append([seat,a])
                    if before['phase'] != 'HeroSelect' and (g.turn!=turn or g.is_over()):
                        after=valid(g); rounds.append(film_round(before,after))
                        boundaries.append([len(actions),digest(after)])
                    if len(actions)>30000: raise RuntimeError('Action guard')
            if not acted and not g.is_over(): raise RuntimeError('No progress')
        else: raise RuntimeError('Driver guard')
        final=valid(g); places=list(g.placements())
        if sorted(places)!=list(range(1,9)): raise AssertionError('Incomplete natural placements')
        # Separate simulator instance; policies are not called during verification.
        check=tg.Game(seed,json.dumps(RULES)); boundary=dict(boundaries); checked_rounds=0
        for step,(seat,a) in enumerate(actions,1):
            if not check.needs_action(seat) or not check.legal_mask(seat)[a]:
                raise AssertionError('Closed replay action')
            before=json.loads(check.state_json())
            turn=check.turn;check.step(seat,a)
            if before['phase'] != 'HeroSelect' and (check.turn!=turn or check.is_over()):
                after=valid(check)
                assert digest(after)==boundary[step]
                assert film_round(before,after)==rounds[checked_rounds]
                checked_rounds+=1
        assert check.is_over() and digest(valid(check))==digest(final) and checked_rounds==len(rounds)
        winner=places.index(1); hero=name(final['players'][winner]['hero']); identifier=f'tavern-scripted-{i:02}'
        title=f'{hero} takes the lobby'
        movie=dict(schemaVersion=1,gameId='tavern',id=identifier,title=title,
            subtitle=f'Eight scripted bots · {len(rounds)} rounds',
            provenance=dict(kind='fresh-public-demo',demoSeed=seed,engineSource=tg.build_info()['source_hash'],
                ruleset='AutoTavern-37.0.2-eight-player',policy='Native scripted bots; no trained-policy strength claim',
                verification='Every action independently replayed; exact full state/RNG at every round and final state; exact projected combat frames; all actions legal; natural finish',
                actionCount=len(actions),trajectorySha256=digest(actions),finalStateSha256=digest(final),
                verifiedRounds=len(rounds)),
            races=[s for j,s in enumerate(RACES) if final['lobby_races'] & (1 << j)],
            players=[dict(seat=p['id'],hero=name(p['hero'])) for p in final['players']],
            rounds=rounds,outcome=dict(kind='natural',winner=winner,placements=places))
        write(out/f'{identifier}.json',movie)
        entries.append(dict(gameId='tavern',id=identifier,title=title,subtitle=movie['subtitle'],
                            moves=len(rounds),url=f'replays/tavern/{identifier}.json'))
        receipts.append(dict(id=identifier,actions=len(actions),rounds=len(rounds),
            combatFrames=sum(len(c['frames']) for r in rounds for c in r['combats']),
            winner=winner,placements=places,sha256=hashlib.sha256((out/f'{identifier}.json').read_bytes()).hexdigest()))
        print(json.dumps(receipts[-1]),flush=True)
    write(out/'index.json',entries)
    write(out/'verification.json',dict(status='passed',native=tg.build_info(),exporterSha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),recordings=receipts))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True,type=Path)
    generate(p.parse_args().output)
