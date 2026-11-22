"""Adversarial full-rule CPU/device transition fixtures, never training distributions.

Construct with the oracle's editor, serialize into BOTH engines, then compare every
real transition/view/mask/RNG. No editor or CPU callback runs inside GPU simulation.
All cases retain eight players and the frozen production rules/action bounds.
"""
import json
import numpy as np
import torch
import taverngym as oracle
from device_env import DeviceEnv


def debug(g,**kw):return json.loads(g.debug(json.dumps(kw)))


def fixtures(rules):
    def base():
        g=oracle.Game(541,json.dumps(rules))
        for p in range(8):g.step(p,json.loads(g.view(p))['actions']['CHOOSE'])
        s=json.loads(g.state_json())
        for p in s['players']:
            for k in ['board','hand','pending','effects','quests','rewards','trinkets','powers','turn_data','counters','secrets']:p[k]=[]
            p.update(skipping=False,ended_turn=False,actions_this_turn=0,gold=30,tier=6,health=40,armor=0)
        return oracle.Game.from_json(json.dumps(s))
    def add(g,card,p=0,zone='board',**kw):debug(g,cmd='add_minion',p=p,zone=zone,card=card,**kw)
    def pack(name,g,first=None):
        game=json.loads(g.state_json())
        return {'name':name,'game':game,'first_actions':first}
    g=base()
    for p in range(8):
        for card in ['BG36_511','BG30_125','BGS_039','BG21_015','BG35_155']:add(g,card,p=p)
    yield pack('populated-deathrattle-reborn-avenge',g)
    g=base();add(g,'BGS_039',hp=2147483647);add(g,'TB_BaconShop_HERO_39_Buddy')
    debug(g,cmd='add_spell',p=0,zone='hand',card='BG20_GEM')
    view=json.loads(g.view(0));first=np.zeros(8,np.int64);first[0]=view['actions']['PLAY']
    yield pack('capped-stat-gain-and-titanic-recursion',g,first.tolist())
    g=base();debug(g,cmd='set_turn',turn=12)
    for p in range(8):
        add(g,'BG26_525',p=p);add(g,'BG22_403',p=p)
        debug(g,cmd='set_player',p=p,health=1,armor=0)
    yield pack('end-turn-lethal-before-combat',g,[0]*8)
    g=base();add(g,'BGS_115');add(g,'BGS_115');add(g,'BG_EX1_564',zone='hand')
    for _ in range(9):debug(g,cmd='add_spell',p=0,zone='hand',card='BG28_810')
    debug(g,cmd='give_trinket',p=0,card='BG30_MagicItem_706')
    view=json.loads(g.view(0));first=np.zeros(8,np.int64);first[0]=view['actions']['PLAY']+2
    yield pack('queued-triple-pending-target-menu',g,first.tolist())
    g=base()
    for p in range(8):
        for card in ['BG25_008','BGS_039','BG21_015','BG36_345','BG31_HERO_811t7','BG30_125','BG35_155']:
            add(g,card,p=p)
    yield pack('full-boards-mixed-effects-and-combat',g,[0]*8)


def qualify(rules):
    result=[]
    for case in fixtures(rules):
        cpu=oracle.BatchEnv(1,seed=541,controlled=list(range(8)),config=rules,threads=1)
        ref=json.loads(cpu.save(0));ref['game']=case['game'];ref['finished']=[False]*8
        cpu.load(0,json.dumps(ref))
        env=DeviceEnv(1,seed=541,controlled=list(range(8)),config=rules)
        gpu={'game':case['game'],'index':ref['index'],'seed':541,'finished':[False]*8,'controlled':list(range(8)),'k':8}
        env.load([json.dumps(gpu)])
        steps=0
        try:
            for step in range(180):
                a=cpu.observe();b={k:v.cpu().numpy() for k,v in env.observe().items()}
                assert np.array_equal(a['ready'],b['ready']) and np.array_equal(a['action_mask'],b['action_mask'])
                assert all(np.array_equal(a[k][a['ready']],b[k][a['ready']]) for k in ['features','cards']),('fixture view',case['name'],step)
                actions=np.zeros(8,np.int64)
                for p in np.flatnonzero(a['ready']):actions[p]=cpu.game(0).bot_action(int(p))
                if step==0 and case['first_actions'] is not None:
                    candidate=np.array(case['first_actions'],np.int64)
                    for p in np.flatnonzero(a['ready']):
                        if a['action_mask'][p,candidate[p]]:actions[p]=candidate[p]
                ar,ad,ai=cpu.step(actions);br,bd,bi=env.step(torch.from_numpy(actions).cuda())
                assert np.array_equal(ar,br.cpu().numpy()) and np.array_equal(ad,bd.cpu().numpy()),('fixture outcomes',case['name'],step)
                assert all(np.array_equal(ai[k],bi[k].cpu().numpy()) for k in ai),('fixture stop classification',case['name'],step)
                gs=json.loads(env.save()[0]);cs=json.loads(cpu.save(0))
                assert gs['game']==cs['game'] and gs['finished']==cs['finished'],('fixture full state/RNG',case['name'],step)
                steps+=1
                if cpu.episode_index(0)>ref['index']:break
            result.append({'case':case['name'],'steps':steps,'exact':True})
        finally:env.close()
    return result
