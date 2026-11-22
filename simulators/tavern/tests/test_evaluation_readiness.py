import json,sys
from pathlib import Path
import numpy as np
import pytest
import taverngym as tg
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from evaluation_readiness.opponents import act_visible,VisibleScript,BF,HF,SF,CF,BC,HC,SC,CC
from evaluation_readiness.fixtures import cross_seat_fixture,recruit_game
from evaluation_readiness.lobby import run_lobby
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'autoresearch'))
from contract import RULES
N=RULES['num_players']
STYLES=('tempo','economy','synergy')

@pytest.mark.parametrize('menu',[False,True])
def test_cross_seat_death_defers_old_move_or_same_choose_index(menu):
    outputs=[]
    for threads in (1,4):
        env=cross_seat_fixture(threads,menu);obs=env.observe();a=obs['action_mask'].argmax(-1);a[3]=1;a[4]=30 if menu else 203
        assert obs['action_mask'][4,a[4]]
        _,_,info=env.step(a)
        assert info['deferred'][4] and not info['applied'][4] and not info['rejected'].any()
        new=env.observe();_,_,fresh=env.step(new['action_mask'].argmax(-1))
        assert fresh['applied'][4] and not fresh['rejected'].any();outputs.append(env.save(0))
    assert outputs[0]==outputs[1]

def test_initial_illegal_choice_is_not_retroactively_admitted():
    e=cross_seat_fixture();o=e.observe();a=o['action_mask'].argmax(-1);a[3]=1;a[4]=30
    assert not o['action_mask'][4,30]
    _,_,i=e.step(a);assert i['rejected'][4] and not i['applied'][4] and not i['deferred'][4]

def test_all_encoded_action_responses_and_pending_kinds_stay_masked():
    f=np.zeros(3227,np.float32);c=np.zeros(124,np.int32)
    for style in ('tempo','economy','synergy'):
        for action in range(245):
            mask=np.zeros(245,bool);mask[action]=True
            assert act_visible(f,c,mask,style)==action
        for pending in range(1,9):
            f[59:68]=0;f[59+pending]=1;mask=np.zeros(245,bool);mask[[30,31,7]]=True
            assert act_visible(f,c,mask,style) in (30,31)
        with pytest.raises(ValueError):act_visible(f,c,np.zeros(245,bool),style)

def test_economy_and_tempo_make_distinct_public_leveling_decisions():
    f=np.zeros(3227,np.float32);c=np.zeros(124,np.int32);mask=np.zeros(245,bool);mask[[0,3,8]]=True
    f[0]=4/20;f[10]=2/6;f[5]=1;f[7]=1;f[11]=.4
    f[BF:HF].reshape(7,31)[:5,0]=1;shop=f[SF:CF].reshape(8,40);shop[0,[0,1,9,10,11]]=1
    assert act_visible(f,c,mask,'economy')==3
    assert act_visible(f,c,mask,'tempo')==8

def test_no_damage_empty_board_lobby_hits_fault_guard_without_placement():
    g=recruit_game(rules=RULES);s=json.loads(g.state_json())
    # Diagnostic continuation after the scheduled trinket turns; production rules stay fixed.
    s['turn']=10
    for p in s['players']:
        for key in ('pending','board','hand','powers','effects','scheduled','trinkets','quests','rewards','secrets','hand_queue'):p[key]=[]
        p['health']=30;p['armor']=0;p['greater_offered']=True;p['skipping']=False
    env=tg.BatchEnv(1,seed=2026,controlled=range(N),config=RULES);env.load(0,json.dumps(dict(game=s,index=0,finished=[False]*N)))
    for _ in range(110):
        rewards,done,info=env.step(np.zeros(N,np.int64))
        if info['truncated'].any():
            assert (info['stop_reason']==4).all() and not info['placement'].any() and not rewards.any();break
    else:pytest.fail('No-damage cycle was not faulted at fixed100turn guard')

@pytest.mark.parametrize('seed',[30101,30102])
def test_visible_mixed_scripts_complete_natural_lobbies(seed):
    policies=[VisibleScript(STYLES[i%3]) for i in range(N)]
    r=run_lobby(policies,seed,RULES)
    assert r['complete_valid_lobby'] and len(r['terminals'])==N and sorted(t['placement'] for t in r['terminals'].values())==list(range(1,N+1)) and not r['counts']['rejected']
    assert all(r['counts'].get('action_'+a,0)>0 for a in ('Buy','Play','Sell','Refresh','Upgrade','HeroPower','Choose','Move'))

def test_script_environment_exact_save_resume():
    a=tg.BatchEnv(1,seed=30103,controlled=range(N),config=RULES,threads=1)
    for _ in range(80):
        o=a.observe();actions=np.zeros(N,np.int64)
        for i in np.flatnonzero(o['ready']):actions[i]=act_visible(o['features'][i],o['cards'][i],o['action_mask'][i],'synergy')
        a.step(actions)
    b=tg.BatchEnv(1,seed=1,controlled=range(N),config=RULES,threads=4);b.load(0,a.save(0))
    for _ in range(30):
        x,y=a.observe(),b.observe();assert all(np.array_equal(x[k],y[k]) for k in x)
        actions=np.zeros(N,np.int64)
        for i in np.flatnonzero(x['ready']):actions[i]=act_visible(x['features'][i],x['cards'][i],x['action_mask'][i],'synergy')
        ar,ad,ai=a.step(actions);br,bd,bi=b.step(actions)
        assert np.array_equal(ar,br) and np.array_equal(ad,bd) and all(np.array_equal(ai[k],bi[k]) for k in ai)
    assert a.save(0)==b.save(0)

def test_compact_script_mixture_preserves_original_slot_roles(monkeypatch):
    import evaluation_readiness.opponents as opponents
    monkeypatch.setattr(opponents,'act_visible',lambda f,c,m,style:opponents.STYLES.index(style))
    styles=('tempo','economy','synergy','tempo','economy','synergy','tempo','economy')
    mixture=opponents.ScriptMixture(styles,rotate_by_lobby=True)
    slots=np.array([1,4,8,11]);obs=dict(features=np.zeros((4,3227)),cards=np.zeros((4,124)),action_mask=np.ones((4,245),bool),slots=slots)
    actions,_,_=mixture.remote_act(obs,np.arange(4))
    assert actions.tolist()==[1,1,1,1]

@pytest.mark.parametrize('source',["import time;time.sleep(2)","import sys,time;sys.stdout.write('[1');sys.stdout.flush();time.sleep(2)"])
def test_candidate_deadline_covers_blocked_input_and_partial_reply(source):
    import subprocess,sys
    from simulators.tavern.evaluation_readiness.policies import visible_exchange
    p=subprocess.Popen([sys.executable,'-c',source],stdin=subprocess.PIPE,stdout=subprocess.PIPE)
    try:
        with pytest.raises(TimeoutError):visible_exchange(p,b'x'*1000000+b'\n',seconds=.1)
    finally:p.kill();p.wait()


def test_candidate_visible_exchange_complete_response():
    import subprocess,sys
    from simulators.tavern.evaluation_readiness.policies import visible_exchange
    p=subprocess.Popen([sys.executable,'-c',"import sys;sys.stdin.buffer.readline();sys.stdout.buffer.write(b'[1]\\n');sys.stdout.buffer.flush()"],stdin=subprocess.PIPE,stdout=subprocess.PIPE)
    try:assert visible_exchange(p,b'visible-only\n',seconds=2)==b'[1]'
    finally:p.wait(timeout=2)


def test_private_bank_excludes_native_training_streams():
    from simulators.tavern.evaluation_readiness.fresh_bank import sm,training_overlap
    cfg=dict(seed=119101,updates=100000,n_games=16)
    for update,game in [(0,0),(43,15),(99999,7)]:
        base=(cfg['seed']+update*1000003)%(2**64)
        equivalent=base^sm(game)^sm(0)
        assert training_overlap(equivalent,cfg)
        a=tg.BatchEnv(16,seed=base,controlled=range(N),config=RULES,threads=1)
        b=tg.BatchEnv(1,seed=equivalent,controlled=range(N),config=RULES,threads=1)
        assert json.loads(a.save(game))['game']==json.loads(b.save(0))['game']
    assert not training_overlap(30101,cfg)


def test_early_candidate_death_does_not_salvage_later_fault(monkeypatch):
    from simulators.tavern.evaluation_readiness import lobby
    class Actor:
        def remote_act(self,obs,indices):return np.zeros(len(indices),np.int64),None,None
    class Env:
        def __init__(self,*args,**kwargs):self.frame=0
        def save(self,*args):return '{}'
        def observe(self):
            mask=np.zeros((N,245),bool);mask[:,0]=True
            return dict(ready=np.ones(N,bool),action_mask=mask,features=np.zeros((N,3227),np.float32))
        def step(self,actions):
            done=np.zeros(N,bool);reason=np.zeros(N,np.int64);placement=np.zeros(N,np.int64);reward=np.zeros(N,np.float32)
            seat=self.frame;done[seat]=True;reason[seat]=1 if self.frame==0 else 4
            if self.frame==0:placement[0]=N;reward[0]=-1
            self.frame+=1
            info={k:np.zeros(N,bool) for k in ('rejected','deferred','applied')};info.update(stop_reason=reason,placement=placement)
            return reward,done,info
    monkeypatch.setattr(lobby.tg,'BatchEnv',Env)
    result=lobby.run_lobby([Actor()]*N,1,RULES)
    assert result['terminals'][0]['placement']==N
    assert result['failure']['kind']=='simulator_fault'
    assert not result['complete_valid_lobby'] and not result['score_eligible']


def test_illegal_visible_policy_has_no_fallback():
    from simulators.tavern.evaluation_readiness.lobby import run_lobby
    class Illegal:
        def remote_act(self,obs,indices):return np.full(len(indices),245,np.int64),None,None
    result=run_lobby([Illegal()]*N,30101,RULES)
    assert result['failure']['kind']=='illegal_action'
    assert result['counts'].get('frames',0)==0 and not result['score_eligible']
