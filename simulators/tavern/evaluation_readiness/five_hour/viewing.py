"""PUBLIC mixed-lobby films with independent native replay; no grading bank input."""
import argparse,json,time,hashlib
from pathlib import Path
import numpy as np
import torch
import taverngym as tg
from ..policies import NeuralVisible
from ..opponents import VisibleScript
from ..lobby import checked_actions
from simulators.tavern.export_demo_replays import film_round,digest,name,RACES
from simulators.tavern.autoresearch.evaluation_v2.release import write,sha
from .terminal_replay import shadow_from_episode,verify_frame

PUBLIC_SEEDS=(202611090701,202611090702,202611090703)


def game_state(env):
    state=json.loads(env.state_json())
    if state.get('stopped_reason') or state['turn_limit_end'] or state['truncated_combats'] or state.get('truncated_recruit_contexts',0):raise ValueError('Faulted lobby cannot become a natural viewing film')
    return state


def record(registry,candidate,out,index=0,profile='incumbent-mixed',seat=0):
    registry,out=Path(registry),Path(out)
    if index not in range(3) or seat not in range(8):raise ValueError('Predeclared PUBLIC seed and physical seat required')
    manifest=json.loads((registry/'manifest.json').read_text())
    if any(sha(registry/f)!=h for f,h in manifest.items()):raise ValueError('Viewing registry mutated')
    reg=json.loads((registry/'registry.json').read_text());items={r['policy_id']:r for r in reg['opponents']};lineup=[None]*8;lineup[seat]=candidate
    for offset,identity in enumerate(reg['profiles'][profile],1):lineup[(seat+offset)%8]=identity
    out.mkdir(parents=True,exist_ok=False);out.chmod(0o700)
    seed=PUBLIC_SEEDS[index];rules={**reg['binding']['rules'],'record_log':True}
    write(out/'precommit.json',dict(candidate=candidate,candidate_seat=seat,public_seed=seed,profile=profile,lineup=lineup,registry_manifest_sha256=sha(registry/'manifest.json'),exporter_sha256=sha(__file__),private_schedule_used=False,outcome_cherry_picking=False,recording_only_rule='record_log=true; gameplay rules unchanged'))
    cache={};torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    for identity in set(lineup):
        item=items[identity]
        cache[identity]=VisibleScript(item['style']) if item['kind']=='script' else NeuralVisible(registry/item['path'],identity)
    policies=[cache[x] for x in lineup];env=tg.BatchEnv(1,seed=seed,controlled=range(8),config=rules,threads=1);initial=env.save(0);shadow=shadow_from_episode(initial);rounds=[];actions=[];boundaries={};finished=np.zeros(8,bool);terminals={};started=time.monotonic();responses=0
    for frame in range(12000):
        if time.monotonic()-started>180:raise TimeoutError('Finite public viewing limit')
        obs=env.observe();ready=np.flatnonzero(obs['ready'] & ~finished);joint=np.zeros(8,np.int64);groups={}
        for p in ready:groups.setdefault(id(policies[p]),(policies[p],[]))[1].append(int(p))
        for policy,ids in groups.values():joint[ids]=checked_actions(policy,obs,ids)
        before=game_state(shadow);reward,done,info=env.step(joint);actions.append(joint.copy());responses+=len(ready)
        if info['rejected'].any() or (info['stop_reason']>1).any():raise ValueError('Illegal/faulted public viewing frame')
        verify_frame(shadow,env,obs,joint,finished,info,done);after=game_state(shadow)
        if before['phase']!='HeroSelect' and (after['turn']!=before['turn'] or (finished|done).all()):
            rounds.append(film_round(before,after));boundaries[frame+1]=digest(after)
        for p in np.flatnonzero(done & ~finished):terminals[str(p)]={'placement':int(info['placement'][p]),'reward':float(reward[p]),'stop_reason':int(info['stop_reason'][p])}
        finished|=done
        if finished.all():break
    else:raise RuntimeError('External viewing frame guard')
    batch_post_step=env.save(0);final=shadow.state_json();state=game_state(shadow);places=[terminals[str(p)]['placement'] for p in range(8)]
    if sorted(places)!=list(range(1,9)):raise ValueError('Incomplete public viewing placements')
    check=tg.BatchEnv(1,seed=seed,controlled=range(8),config=rules,threads=1);check.load(0,initial);shadow_check=shadow_from_episode(initial);checked=0;replay_finished=np.zeros(8,bool);replay_terminals={}
    for step,joint in enumerate(actions,1):
        obs=check.observe();ready=np.flatnonzero(obs['ready'] & ~replay_finished)
        if not obs['action_mask'][ready,joint[ready]].all():raise ValueError('Replay preframe legality differs')
        before=game_state(shadow_check);reward,done,info=check.step(joint);verify_frame(shadow_check,check,obs,joint,replay_finished,info,done);after=game_state(shadow_check)
        if info['rejected'].any() or (info['stop_reason']>1).any():raise ValueError('Replay native fault')
        if step in boundaries:
            if digest(after)!=boundaries[step] or film_round(before,after)!=rounds[checked]:raise ValueError('Public combat projection replay differs')
            checked+=1
        for p in np.flatnonzero(done & ~replay_finished):replay_terminals[str(p)]={'placement':int(info['placement'][p]),'reward':float(reward[p]),'stop_reason':int(info['stop_reason'][p])}
        replay_finished|=done
    if shadow_check.state_json()!=final or check.save(0)!=batch_post_step or replay_terminals!=terminals or checked!=len(rounds) or not replay_finished.all():raise ValueError('Exact final native state/terminals replay differs')
    identifier=f'tavern-strong-{candidate}-{index+1:02}';title=f'{candidate} in a mixed eight-player lobby';subtitle=f'PUBLIC viewing game · seat {seat+1} · placement {places[seat]}/8 · {len(rounds)} rounds'
    identities={identity:sha(registry/items[identity]['path']/'identity.json') for identity in set(lineup) if items[identity]['kind']=='learned'}
    movie=dict(schemaVersion=1,gameId='tavern',id=identifier,title=title,subtitle=subtitle,provenance=dict(kind='fresh-public-demo',demoSeed=seed,engineSource=tg.build_info()['source_hash'],ruleset='AutoTavern-37.0.2-eight-player',policy='Visible-only neural or declared static scripts, greedy, no search or LLM; fresh public viewing example, not private grade',verification='Every joint action independently replayed; exact full state/RNG and all8 terminals; exact projected combat frames at every round and finish; natural outcome',actionCount=responses,trajectorySha256=digest([a.tolist() for a in actions]),finalStateSha256=hashlib.sha256(final.encode()).hexdigest(),verifiedRounds=checked,candidateSeat=seat,recordingOnlyRule='record_log=true captures combat snapshots; gameplay rules unchanged',policyIdentitySha256=identities,profile=profile),races=[r for j,r in enumerate(RACES) if state['lobby_races']&(1<<j)],players=[dict(seat=p['id'],hero=name(p['hero']),policy=lineup[p['id']]) for p in state['players']],rounds=rounds,outcome=dict(kind='natural',winner=places.index(1),placements=places))
    write(out/(identifier+'.json'),movie)
    receipt=dict(status='passed',public_seed=seed,candidate=candidate,profile=profile,frames=len(actions),visible_action_responses=responses,rounds=checked,all8_natural=True,exact_native_replay=True,exact_projected_combat_replay=True,movie_sha256=sha(out/(identifier+'.json')),private_schedule_used=False)
    write(out/'verification.json',receipt);write(out/'index.json',[dict(gameId='tavern',id=identifier,title=title,subtitle=subtitle,moves=len(rounds),url=f'replays/tavern/strong-bots-20261109/{identifier}.json')]);return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--registry',type=Path,required=True);p.add_argument('--candidate',required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--index',type=int,default=0);p.add_argument('--seat',type=int,default=0);p.add_argument('--profile',default='incumbent-mixed');a=p.parse_args();print(json.dumps(record(a.registry,a.candidate,a.out,a.index,a.profile,a.seat)))
