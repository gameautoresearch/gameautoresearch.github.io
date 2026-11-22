"""Recover terminal Game state alongside the batch API's automatic reset."""
import json,hashlib
from pathlib import Path
import numpy as np
import taverngym as tg


def shadow_from_episode(serialized):
    return tg.Game.from_json(json.dumps(json.loads(serialized)['game']))


def shadow_step(game,obs,actions,finished):
    """The pinned batch admission rule: preframe ready/mask/pending, seat order."""
    before=json.loads(game.state_json());ready=np.flatnonzero(obs['ready'] & ~finished)
    pending={int(p):before['players'][p]['pending'] for p in ready}
    applied=np.zeros(len(finished),bool);deferred=np.zeros(len(finished),bool)
    if not obs['action_mask'][ready,actions[ready]].all():raise ValueError('Shadow replay illegal preframe response')
    for p in ready:
        state=json.loads(game.state_json());a=int(actions[p])
        if game.is_over() or not game.needs_action(int(p)) or state['players'][p]['pending']!=pending[int(p)] or not game.legal_mask(int(p))[a]:deferred[p]=True;continue
        game.step(int(p),a);applied[p]=True
    return applied,deferred


def verify_frame(shadow,env,obs,actions,finished,info,done):
    applied,deferred=shadow_step(shadow,obs,actions,finished)
    if not np.array_equal(applied,info['applied']) or not np.array_equal(deferred,info['deferred']):raise ValueError('Native applied/deferred frame differs from shadow')
    state=json.loads(shadow.state_json())
    if (finished|done).all():
        if not shadow.is_over() or sorted(shadow.placements())!=list(range(1,len(finished)+1)):raise ValueError('Missing natural terminal Game state')
    elif state!=json.loads(env.save(0))['game']:raise ValueError('Native batch/Game frame state differs')
    return state


def audit_dir(path):
    """Recorded responses only; preserve actual terminal and post-reset identities."""
    path=Path(path);result=json.loads((path/'result.json').read_text());initial=(path/'initial.json').read_text()
    if not result['complete_valid_lobby'] or result['native']!=tg.build_info()['source_hash']:raise ValueError('Natural current-native recorded lobby required')
    cfg=json.loads(initial)['game']['cfg'];n=int(cfg.get('num_players',8))
    env=tg.BatchEnv(1,seed=result['seed'],controlled=range(n),config=cfg,threads=1);env.load(0,initial);shadow=shadow_from_episode(initial);finished=np.zeros(n,bool);terminals={}
    with np.load(path/'actions.npz',allow_pickle=False) as z:actions=z['actions']
    for joint in actions:
        obs=env.observe();reward,done,info=env.step(joint)
        if info['rejected'].any() or (info['stop_reason']>1).any():raise ValueError('Recorded natural replay fault')
        verify_frame(shadow,env,obs,joint,finished,info,done)
        for p in np.flatnonzero(done & ~finished):terminals[str(p)]=dict(placement=int(info['placement'][p]),stop_reason='natural',natural=True,reward=float(reward[p]))
        finished|=done
    if not finished.all() or terminals!=result['terminals'] or hashlib.sha256(env.save(0).encode()).hexdigest()!=result['final_state_sha256']:raise ValueError('Recorded terminal/post-reset state mismatch')
    final=shadow.state_json();target=path/'actual-terminal-game.json'
    if target.exists():
        if target.read_text()!=final:raise ValueError('Retained actual terminal Game differs from independent replay')
    else:target.write_text(final);target.chmod(0o600)
    return dict(trace=path.name,frames=len(actions),exact_all8_natural_terminals=True,exact_post_reset_state=True,exact_true_terminal_game_state=True,actual_terminal_game_sha256=hashlib.sha256(final.encode()).hexdigest(),post_step_sha256=result['final_state_sha256'])
