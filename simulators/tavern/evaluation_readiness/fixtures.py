"""Synthetic public-seed native admission/guard fixtures; never used as evaluation seeds."""
import copy,json
import numpy as np
import taverngym as tg

def recruit_game(seed=2026,rules=None):
    """Default eight-player rules, or `rules` (for example the frozen four-player form)."""
    g=tg.Game(seed,json.dumps(rules)) if rules else tg.Game(seed)
    for p in range(int((rules or {}).get('num_players',8))):
        g.step(p,next(i for i,b in enumerate(g.legal_mask(p)) if b))
        g.debug(json.dumps(dict(cmd='set_hero',p=p,hero='TB_BaconShop_HERO_34')))
        for zone in ('board','hand'):g.debug(json.dumps(dict(cmd='clear_zone',p=p,zone=zone)))
    return g

def cross_seat_fixture(threads=1,existing_menu=False):
    g=recruit_game()
    g.debug(json.dumps(dict(cmd='add_minion',p=3,zone='board',card='BG26_524')))
    g.debug(json.dumps(dict(cmd='set_hero',p=4,hero='TB_BaconShop_HERO_70')))
    for _ in range(2):g.debug(json.dumps(dict(cmd='add_minion',p=4,zone='board',card='BG26_524')))
    s=json.loads(g.state_json())
    for pl in s['players']:
        pl['pending']=[];pl['gold']=10;pl['health']=30;pl['armor']=0;pl['skipping']=False
    s['players'][3]['health']=1;s['players'][3]['board'][0]['counter']=1
    if existing_menu:
        m=copy.deepcopy(s['players'][4]['board'][0]);m['eid']=s['next_eid'];s['next_eid']+=1
        s['players'][4]['pending']=[{'Discover':dict(source=s['players'][4]['powers'][0]['card'],options=[{'Minion':m}])}]
    env=tg.BatchEnv(1,seed=2026,controlled=range(8),config=s['cfg'],threads=threads)
    env.load(0,json.dumps(dict(game=s,index=0,finished=[False]*8)))
    return env
