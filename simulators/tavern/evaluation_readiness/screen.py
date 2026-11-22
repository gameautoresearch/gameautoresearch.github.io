"""Predeclared mixed-lobby public/fresh schedules; no policy repair or best-seed choice."""
import argparse,json,time,os
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
import torch
from .opponents import VisibleScript
from .policies import NeuralVisible,IsolatedVisible,admit_bundle
from .lobby import run_lobby


def summarize(rows,expected,players=8):
    points=lambda p:100*(players-p)/(players-1)
    valid=[r for r in rows if r['complete_valid_lobby']];complete=len(rows)==expected and len(valid)==expected
    placements=[r['candidate_placement'] for r in valid];groups=sorted({r['seed'] for r in rows});ci=None
    if complete and len(groups)>=2:
        means=np.array([np.mean([points(r['candidate_placement']) for r in rows if r['seed']==g]) for g in groups]);rng=np.random.default_rng(92317);ci=np.quantile(rng.choice(means,(4000,len(means))).mean(1),[.025,.975]).tolist()
    failures=Counter(r['failure']['kind'] for r in rows if r.get('failure'))
    extra=dict(top4_rate=float(np.mean([p<=4 for p in placements])) if complete else None) if players==8 else {}
    return dict(players=players,expected_cases=expected,observed_cases=len(rows),complete_valid_schedule=complete,natural_complete_lobbies=len(valid),completion_rate=len(valid)/expected,raw_score=float(np.mean([points(p) for p in placements])) if complete else None,mean_placement=float(np.mean(placements)) if complete else None,top_half_rate=float(np.mean([p<=players//2 for p in placements])) if complete else None,**extra,placement_histogram={str(p):placements.count(p) for p in range(1,players+1)},conditional_valid_mean_placement=float(np.mean(placements)) if placements else None,failures=dict(failures),confidence_interval_95=ci,interval_method=f'lobby-seed-group bootstrap, preserving profile and all {players} seats; {len(groups)} environment groups; small-panel uncertainty only',normalized_reference_score=None)


def screen(registry_root,panel,candidate_id,out,isolated=True,stage_seconds=500):
    root=Path(registry_root);registry=json.loads((root/'registry.json').read_text());panel=json.loads(Path(panel).read_text());out=Path(out);out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    items={r['policy_id']:r for r in registry['opponents']};cache={};candidate=None;rows=[];start=time.monotonic();log=(out/'candidate-stderr.log').open('w')
    def policy(identity,candidate=False):
        if candidate and identity==candidate_id:
            item=items[identity]
            if item['kind']=='script':obj=VisibleScript(item['style']);obj.policy_id=identity;return obj
            return IsolatedVisible(Path(os.environ.get('TAVERN_EVAL_CANDIDATE_BUNDLE',str(root/item['path']))),identity,log) if isolated else NeuralVisible(root/item['path'],identity)
        if identity not in cache:
            item=items[identity]
            if item['kind']=='script':obj=VisibleScript(item['style']);obj.policy_id=identity
            else:obj=NeuralVisible(root/item['path'],identity)
            cache[identity]=obj
        return cache[identity]
    rules=registry['binding']['rules'];n=int(rules.get('num_players',8))
    expected=len(panel['seeds'])*n*len(registry['profiles']);pending=[]
    try:
        candidate=policy(candidate_id,True);candidate.policy_id='candidate:'+candidate_id
        for profile,lineup in registry['profiles'].items():
            if len(lineup)!=n-1:raise ValueError('Full mixed roster (one reference per other seat) required')
            for seed in panel['seeds']:
                for seat in range(n):
                    if time.monotonic()-start>stage_seconds-130:pending.append(dict(profile=profile,seed=seed,seat=seat));continue
                    policies=[None]*n;policies[seat]=candidate
                    ids=[None]*n;ids[seat]=candidate_id
                    for offset,identity in enumerate(lineup,1):policies[(seat+offset)%n]=policy(identity);ids[(seat+offset)%n]=identity
                    row=run_lobby(policies,seed,rules,seconds=120,trace=out/'traces'/f'{profile}-{seed}-{seat}');row.update(profile=profile,seat=seat,candidate_id=candidate_id,lineup=ids,candidate_placement=row['terminals'].get(seat,{}).get('placement') if row['complete_valid_lobby'] else None)
                    rows.append(row);(out/'episodes.json').write_text(json.dumps(rows,indent=2)+'\n');(out/'progress.json').write_text(json.dumps(dict(completed_cases=len(rows),expected_cases=expected,current=dict(profile=profile,seed=seed,seat=seat)),indent=2)+'\n')
                    # Durable v2-volume writes through host disconnect, without guest credentials.
                    import subprocess;subprocess.run(['sync',str(out)],timeout=10,check=True)
    finally:
        if candidate and hasattr(candidate,'close'):candidate.close()
        log.close()
    result=summarize(rows,expected,n);result.update(candidate_id=candidate_id,panel_label=panel['label'],process_seconds=time.monotonic()-start,pending=pending,per_profile={p:summarize([r for r in rows if r['profile']==p],len(panel['seeds'])*n,n) for p in registry['profiles']},per_seat={str(s):summarize([r for r in rows if r['seat']==s],len(panel['seeds'])*len(registry['profiles']),n) for s in range(n)})
    pairwise={};latency=defaultdict(list);counts=Counter()
    for row in rows:
        counts.update(row['counts'])
        for name,values in row['latency'].items():latency[name].extend(values)
        if row['complete_valid_lobby']:
            for seat,name in enumerate(row['lineup']):
                if seat==row['seat']:continue
                p=pairwise.setdefault(name,dict(cases=0,candidate_ahead=0,opponent_placements=[]));p['cases']+=1;p['candidate_ahead']+=int(row['candidate_placement']<row['terminals'][seat]['placement']);p['opponent_placements'].append(row['terminals'][seat]['placement'])
    result['per_opponent']={name:dict(cases=d['cases'],candidate_ahead_fraction=d['candidate_ahead']/d['cases'],opponent_mean_placement=float(np.mean(d['opponent_placements'])),interpretation=f'Pairwise placement order within mixed {n}-seat lobbies, not a head-to-head win rate') for name,d in pairwise.items()}
    result['latency']={name:dict(calls=len(v),visible_rows=sum(r['rows'] for r in v),total_seconds=sum(r['seconds'] for r in v),call_p50_seconds=float(np.median([r['seconds'] for r in v])),call_p95_seconds=float(np.quantile([r['seconds'] for r in v],.95)),max_seconds=max(r['seconds'] for r in v)) for name,v in latency.items()};result['action_counts']=dict(counts)
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--registry',type=Path,required=True);p.add_argument('--panel',type=Path,required=True);p.add_argument('--candidate',required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--seconds',type=int,default=500);p.add_argument('--trusted-local',action='store_true');a=p.parse_args();print(json.dumps(screen(a.registry,a.panel,a.candidate,a.out,not a.trusted_local,a.seconds)))
