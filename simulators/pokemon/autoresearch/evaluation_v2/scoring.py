"""Completion is a gate. Natural engine ties count; unresolved cases only bound total points."""
import numpy as np


def summarize(rows,scheduled,schedule=None):
    keys=[(r['opponent'],r['seed_index'],r['seat']) for r in rows]
    if len(set(keys))!=len(keys) or len(rows)>scheduled:raise ValueError('Duplicate/oversized evaluation rows')
    if schedule is not None and (len(schedule)!=scheduled or not set(keys)<=set(schedule)):raise ValueError('Evaluation rows differ from committed schedule')
    natural=[r for r in rows if r.get('natural') and r.get('fault') is None]
    completed=len(natural);known=sum(r['points'] for r in natural)
    complete=len(rows)==scheduled==completed
    groups={}
    for key in sorted({(name,seat) for name,_,seat in schedule} if schedule is not None else {(r['opponent'],r['seat']) for r in rows}):
        part=[r for r in rows if (r['opponent'],r['seat'])==key]
        done=[r for r in part if r.get('natural') and r.get('fault') is None]
        expected=sum(name==key[0] and seat==key[1] for name,_,seat in schedule) if schedule is not None else len(part)
        groups[f'{key[0]}/foul-seat{key[1]}']={'scheduled':expected,'attempted':len(part),'natural':len(done),
            'raw_score_0_100':100*sum(r['points'] for r in done)/expected if expected>0 and len(done)==expected else None}
    interval=None
    if complete:
        indices=sorted({r['seed_index'] for r in rows});means=np.array([np.mean([r['points'] for r in rows if r['seed_index']==i]) for i in indices])
        rng=np.random.default_rng(20261109);boot=means[rng.integers(len(means),size=(4000,len(means)))].mean(1)*100
        interval=np.quantile(boot,[.025,.975]).tolist()
    return {'scheduled_games':scheduled,'attempted_games':len(rows),'unstarted_games':scheduled-len(rows),
        'natural_games':completed,'natural_wins':sum(r['points']==1 for r in natural),'natural_ties':sum(r['points']==.5 for r in natural),
        'natural_losses':sum(r['points']==0 for r in natural),'administrative_caps':sum(r.get('truncated',False) for r in rows),
        'faults':sum(r.get('fault') is not None for r in rows),'fault_categories':{kind:sum(r.get('fault_category')==kind for r in rows) for kind in sorted({r['fault_category'] for r in rows if r.get('fault_category')})},
        'visible_choice_rejections':sum(r.get('visible_choice_rejections',0) for r in rows),
        'illegal_action_faults':sum('illegal_action' in (r.get('fault_category') or '') for r in rows),
        'infrastructure_faults':sum(r.get('fault_category')=='infrastructure_failure' for r in rows),
        'simulator_faults':sum(r.get('fault_category') in ('simulator_stall','simulator_or_adapter_fault') for r in rows),
        'completion_rate':completed/scheduled,'raw_score_0_100':100*known/scheduled if complete else None,
        'total_point_bounds_0_100':[100*known/scheduled,100*(known+scheduled-completed)/scheduled],
        'seed_cluster_bootstrap_95_percent':interval,'by_opponent_and_seat':groups,
        'status':'complete' if complete else 'incomplete','normalized_reference_score':None,
        'harness_qualified':False,'learned_reference_qualified':False,'strong_panel_qualified':False,'superhuman_evidence':False,
        'note':'Natural engine outcomes only; incomplete panels have null primary. Bounds include caps, faults and unstarted cases. Seed bootstrap preserves both seats and opponent strata.'}
