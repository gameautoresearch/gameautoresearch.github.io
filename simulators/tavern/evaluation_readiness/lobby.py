"""Full-lobby mixed driver (every seat of the rules' `num_players`) with explicit fault accounting."""
import hashlib,json,time
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
import taverngym as tg

STOP={0:'ongoing',1:'natural',2:'no_progress',3:'driver_guard',4:'turn_limit',5:'combat_cutoff',6:'recruit_cutoff'}

class IllegalPolicyAction(ValueError): pass

def checked_actions(policy,obs,indices):
    out=policy.remote_act(obs,np.asarray(indices,np.int64))[0]
    a=np.asarray(out)
    if a.shape!=(len(indices),) or a.dtype.kind not in 'iu' or a.dtype.kind=='b':raise IllegalPolicyAction('Action shape/type mismatch')
    if ((a<0)|(a>=245)).any() or not obs['action_mask'][indices,a].all():raise IllegalPolicyAction('Policy action outside visible legal mask')
    return a.astype(np.int64,copy=False)

def run_lobby(policies,seed,rules,threads=1,max_frames=12000,seconds=120,trace=None):
    n=int(rules.get('num_players',8))
    if len(policies)!=n:raise ValueError('One explicit visible policy per seat required')
    env=tg.BatchEnv(1,seed=int(seed),controlled=range(n),config=rules,threads=threads)
    initial=env.save(0);finished=np.zeros(n,bool);terminal={};counts=Counter();latency=defaultdict(list);actions_trace=[];started=time.perf_counter();failure=None
    for frame in range(max_frames):
        if time.perf_counter()-started>=seconds:failure=dict(kind='administrative_cap',detail='Full-lobby wall deadline');break
        obs=env.observe()
        if not np.array_equal(obs['ready'],obs['action_mask'].any(-1)):failure=dict(kind='simulator_fault',detail='ready/mask interface mismatch');break
        ready=np.flatnonzero(obs['ready'] & ~finished);actions=np.zeros(n,np.int64);groups={}
        for seat in ready:groups.setdefault(id(policies[seat]),(policies[seat],[]))[1].append(int(seat))
        try:
            for policy,indices in groups.values():
                t=time.perf_counter();a=checked_actions(policy,obs,indices);elapsed=time.perf_counter()-t
                latency[getattr(policy,'policy_id',getattr(policy,'style',type(policy).__name__))].append(dict(seconds=elapsed,rows=len(indices)))
                actions[indices]=a
                for seat,action in zip(indices,a):
                    name=tg.action_name(int(action));category=name.split('(')[0].split(' {')[0];counts['action_'+category]+=1
                    pending=int(np.argmax(obs['features'][seat,59:68]));counts['pending_'+str(pending)]+=int(30<=action<46)
        except IllegalPolicyAction as exc:failure=dict(kind='illegal_action',detail=str(exc));break
        except TimeoutError as exc:failure=dict(kind='inference_deadline',detail=str(exc));break
        except Exception as exc:failure=dict(kind='infrastructure_fault',detail=type(exc).__name__+': '+str(exc));break
        actions_trace.append(actions.copy())
        try:reward,done,info=env.step(actions)
        except Exception as exc:failure=dict(kind='simulator_fault',detail=type(exc).__name__+': '+str(exc));break
        counts['frames']+=1;counts['rejected']+=int(info['rejected'].sum());counts['deferred']+=int(info['deferred'].sum());counts['applied']+=int(info['applied'].sum())
        if info['rejected'].any():failure=dict(kind='visible_engine_rejection',detail='A masked joint-frame response was rejected');break
        for seat in np.flatnonzero(done & ~finished):
            reason=int(info['stop_reason'][seat]);terminal[int(seat)]=dict(placement=int(info['placement'][seat]) if reason==1 else None,stop_reason=STOP[reason],natural=reason==1,reward=float(reward[seat]));finished[seat]=True
        fault=[int(x) for x in info['stop_reason'] if x>1]
        if fault:failure=dict(kind='simulator_fault',detail=STOP[fault[0]]);break
        if finished.all():break
    else:failure=dict(kind='administrative_cap',detail='External frame guard')
    valid=failure is None and len(terminal)==n and all(row['natural'] and 1<=row['placement']<=n for row in terminal.values())
    if valid and sorted(row['placement'] for row in terminal.values())!=list(range(1,n+1)):raise RuntimeError('Natural placement permutation invariant')
    final=env.save(0)
    result=dict(seed=int(seed),complete_valid_lobby=valid,terminals=terminal,score_eligible=valid,failure=failure,counts=dict(counts),latency=dict(latency),process_seconds=time.perf_counter()-started,native=tg.build_info()['source_hash'],initial_state_sha256=hashlib.sha256(initial.encode()).hexdigest(),final_state_sha256=hashlib.sha256(final.encode()).hexdigest())
    if trace:
        trace=Path(trace);trace.mkdir(parents=True,exist_ok=False);(trace/'initial.json').write_text(initial);(trace/'final.json').write_text(final);np.savez_compressed(trace/'actions.npz',actions=np.array(actions_trace,np.int64));(trace/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    return result
