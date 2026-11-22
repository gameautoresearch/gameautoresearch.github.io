"""Role-balanced paired scoring. Team-seat games are not treated as 1v1 Elo."""
import numpy as np
from .contract import DEFAULT,digest

ROLES=('landlord','landlord_down','landlord_up')

def by_deal(rows):
    if not rows:raise ValueError('No evaluation games')
    result={};keys=set();panels={r['opponent'] for r in rows};deals={r['deal_index'] for r in rows}
    for r in rows:
        key=(r['opponent'],r['deal_index'],r['role'])
        if key in keys or r['role'] not in ROLES or not r['natural'] or r['truncated'] or r['won'] not in (False,True):raise ValueError('Invalid or duplicate evaluation row')
        keys.add(key);result.setdefault(r['deal_index'],[]).append(float(r['won']))
    if len(keys)!=len(panels)*len(deals)*3:raise ValueError('Incomplete role/opponent/deal schedule')
    return {i:float(np.mean(v)) for i,v in result.items()}

def summary(rows,replicates=2000):
    values=by_deal(rows);v=np.array([values[i] for i in sorted(values)])
    rng=np.random.default_rng(83)
    boots=np.array([v[rng.integers(len(v),size=len(v))].mean() for _ in range(replicates)])
    return {'quality':float(v.mean()),'quality_ci95':np.quantile(boots,[.025,.975]).tolist(),
        'raw_score':float(v.mean()*100),'score_scale':[0,100],
        'per_opponent':{p:float(np.mean([r['won'] for r in rows if r['opponent']==p])) for p in sorted({r['opponent'] for r in rows})},
        'economic_score_by_role':{role:float(np.mean([r['economic_score'] for r in rows if r['role']==role])) for role in ROLES} if all('economic_score' in r for r in rows) else None,
        'natural_games':len(rows),'truncations':0,'illegal_actions':0,'simulator_faults':0,
        'raw_score_ci95':(np.quantile(boots,[.025,.975])*100).tolist(),
        'games':len(rows),'unique_deals':len(v),'roles':{role:float(np.mean([r['won'] for r in rows if r['role']==role])) for role in ROLES},
        'aggregation':'equal weight per role and panel opponent; paired bootstrap clustered by complete deal'}

def calibration_gate(starter_rows,reference_rows,minimum_gap=.05,replicates=2000):
    a=by_deal(starter_rows);b=by_deal(reference_rows)
    ka={(r['opponent'],r['deal_index'],r['role']) for r in starter_rows}
    kb={(r['opponent'],r['deal_index'],r['role']) for r in reference_rows}
    if ka!=kb:raise ValueError('Anchor schedules differ')
    delta=np.array([b[i]-a[i] for i in sorted(a)])
    rng=np.random.default_rng(73)
    boots=np.array([delta[rng.integers(len(delta),size=len(delta))].mean() for _ in range(replicates)])
    ci=np.quantile(boots,[.025,.975]).tolist();gap=float(delta.mean())
    return {'qualified':gap>=minimum_gap and ci[0]>0,'gap':gap,'gap_ci95':ci,
        'minimum_gap':minimum_gap,'starter_quality':float(np.mean(list(a.values()))),'reference_quality':float(np.mean(list(b.values())))}

def grade(rows,anchors,*,development=False):
    scores=summary(rows)
    if not anchors.get('qualified'):
        if not development:raise ValueError('Official grade blocked: full-budget reference anchors have not passed calibration')
        return {'status':'development_only','reward':None,'normalized_quality':None,**scores}
    gap=anchors['reference_quality']-anchors['starter_quality']
    if gap<anchors.get('minimum_gap',.05):raise ValueError('Invalid calibration gap')
    value=(scores['quality']-anchors['starter_quality'])/gap
    return {'status':'complete','reward':float(np.clip(value,0,1)),'normalized_quality':value,**scores,
        'anchor_commitment':digest(anchors)}


def multi_seed_gate(starter_parts,reference_parts,minimum_gap=.05,replicates=2000):
    if len(starter_parts)<3 or len(starter_parts)!=len(reference_parts):raise ValueError('At least three paired training seeds required')
    all_parts=starter_parts+reference_parts
    schedules=[{(r['opponent'],r['deal_index'],r['role']) for r in rows} for rows in all_parts]
    if any(s!=schedules[0] for s in schedules):raise ValueError('All anchor seeds need identical evaluation schedules')
    keyed=[by_deal(rows) for rows in all_parts]
    indices=sorted(keyed[0]);n=len(starter_parts)
    a=np.asarray([[keyed[s][i] for i in indices] for s in range(n)])
    b=np.asarray([[keyed[n+s][i] for i in indices] for s in range(n)])
    rng=np.random.default_rng(731);boots=[]
    for _ in range(replicates):
        seeds=rng.integers(n,size=n);deals=rng.integers(len(indices),size=len(indices))
        boots.append(float((b[seeds][:,deals]-a[seeds][:,deals]).mean()))
    ci=np.quantile(boots,[.025,.975]).tolist();gaps=(b-a).mean(axis=1)
    return {'qualified':float(gaps.mean())>=minimum_gap and ci[0]>0 and bool(np.all(gaps>0)),
        'starter_quality':float(a.mean()),'reference_quality':float(b.mean()),'gap':float(gaps.mean()),
        'gap_ci95':ci,'minimum_gap':minimum_gap,'paired_training_seeds':n,'seed_gaps':gaps.tolist(),
        'uncertainty':'paired training-seed resampling and complete-deal cluster bootstrap'}
