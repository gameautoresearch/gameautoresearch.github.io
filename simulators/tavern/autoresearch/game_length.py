"""Typical game length of a Tavern rules form: turns and learner decisions per episode.

One learner seat (0) against the native scripted bots, as in the starter's training lobbies. The learner is
either uniformly random over the legal mask (an untrained starter) or the public `tempo` visible script.
Also checks that every opponent slot beyond the lobby's players is zero-padded in every observation.

    .venv-tavern/bin/python simulators/tavern/autoresearch/game_length.py --players 8 --lobbies 64
"""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
import taverngym as tg

HERE=Path(__file__).resolve().parent
L=dict(tg.layout())
OPP_FLOATS=L['N_OPP']*L['OPP_F']+L['N_OPP']*L['N_BOARD']*L['MINION_F']
OPP_CARDS=L['N_OPP']+L['N_OPP']*L['N_BOARD']


def padding_ok(features,cards,players):
    """Opponent slots players-1..6 (heroes and last-seen boards) must be all zero."""
    real=players-1;n,b=L['N_OPP'],L['N_BOARD']
    heroes=features[:,-OPP_FLOATS:-OPP_FLOATS+n*L['OPP_F']].reshape(len(features),n,L['OPP_F'])
    boards=features[:,-OPP_FLOATS+n*L['OPP_F']:].reshape(len(features),n,b*L['MINION_F'])
    hc=cards[:,-OPP_CARDS:-OPP_CARDS+n];bc=cards[:,-OPP_CARDS+n:].reshape(len(cards),n,b)
    return not (heroes[:,real:].any() or boards[:,real:].any() or hc[:,real:].any() or bc[:,real:].any())


def measure(rules,lobbies,seed,learner,threads=4):
    from simulators.tavern.evaluation_readiness.opponents import act_visible
    env=tg.BatchEnv(lobbies,seed=seed,controlled=(0,),config=rules,threads=threads);rng=np.random.default_rng(seed)
    decisions=np.zeros(lobbies,int);turns=np.zeros(lobbies);done_mask=np.zeros(lobbies,bool);placements=np.zeros(lobbies,int);padding=True;frames=0;started=time.perf_counter()
    while not done_mask.all():
        obs=env.observe();ready=obs['ready']&~done_mask
        padding&=padding_ok(obs['features'][ready],obs['cards'][ready],rules['num_players'])
        actions=np.zeros(lobbies,np.int64)
        for i in np.flatnonzero(ready):
            legal=np.flatnonzero(obs['action_mask'][i])
            actions[i]=rng.choice(legal) if learner=='random' else act_visible(obs['features'][i],obs['cards'][i],obs['action_mask'][i],'tempo')
            decisions[i]+=1;turns[i]=round(float(obs['features'][i,0])*20)
        _,done,info=env.step(actions);frames+=1
        if (info['stop_reason']>1).any():raise RuntimeError('Simulator stop '+str(info['stop_reason'].tolist()))
        for i in np.flatnonzero(done&~done_mask):placements[i]=int(info['placement'][i]);done_mask[i]=True
    q=lambda x:dict(mean=float(np.mean(x)),p10=float(np.quantile(x,.1)),median=float(np.median(x)),p90=float(np.quantile(x,.9)))
    return dict(players=rules['num_players'],max_regular_tier=rules.get('max_regular_tier',6),lobby_races=rules['lobby_races'],learner=learner,lobbies=lobbies,
                learner_turns=q(turns),learner_decisions=q(decisions),mean_placement=float(placements.mean()),opponent_padding_all_zero=bool(padding),
                frames=frames,seconds=time.perf_counter()-started,learner_decisions_per_second=float(decisions.sum()/(time.perf_counter()-started)))


def lobby(rules,lobbies,seed,threads=4):
    """Whole lobbies, every seat a public visible script (tempo/economy/synergy in turn): the evaluator's unit."""
    from simulators.tavern.evaluation_readiness.opponents import act_visible,STYLES
    n=rules['num_players'];env=tg.BatchEnv(lobbies,seed=seed,controlled=range(n),config=rules,threads=threads)
    finished=np.zeros(lobbies*n,bool);last_turn=np.zeros(lobbies);decisions=np.zeros(lobbies,int);frames=0;started=time.perf_counter()
    while not finished.all():
        obs=env.observe();ready=obs['ready']&~finished;actions=np.zeros(lobbies*n,np.int64)
        for i in np.flatnonzero(ready):
            actions[i]=act_visible(obs['features'][i],obs['cards'][i],obs['action_mask'][i],STYLES[i%n%3])
            decisions[i//n]+=1;last_turn[i//n]=max(last_turn[i//n],round(float(obs['features'][i,0])*20))
        _,done,info=env.step(actions);frames+=1
        if (info['stop_reason']>1).any():raise RuntimeError('Simulator stop')
        finished|=done
    q=lambda x:dict(mean=float(np.mean(x)),p10=float(np.quantile(x,.1)),median=float(np.median(x)),p90=float(np.quantile(x,.9)))
    seconds=time.perf_counter()-started
    return dict(players=n,mode='whole lobby, all seats visible scripts',lobbies=lobbies,lobby_last_turn=q(last_turn),lobby_decisions=q(decisions),frames=frames,seconds=seconds,seconds_per_lobby_cpu=seconds/lobbies)


def main():
    p=argparse.ArgumentParser();p.add_argument('--players',type=int,choices=(4,8),default=4);p.add_argument('--lobbies',type=int,default=64);p.add_argument('--seed',type=int,default=7);p.add_argument('--learner',choices=('random','tempo','lobby'),default='random');a=p.parse_args()
    rules=json.loads((HERE/'recipes/starter.json').read_text())['rules']
    if a.players==8:rules=dict(rules,num_players=8,lobby_races=5,max_regular_tier=6)  # the full-size form
    print(json.dumps(lobby(rules,a.lobbies,a.seed) if a.learner=='lobby' else measure(rules,a.lobbies,a.seed,a.learner),indent=2))

if __name__=='__main__':
    sys.path.insert(0,str(HERE.parents[2]));main()
