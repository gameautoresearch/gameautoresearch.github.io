"""Matched whole-seed-group comparisons; faults never disappear by filtering."""
import json,hashlib
from pathlib import Path
import numpy as np


def compare(baseline,challenger,expected_groups=None):
    baseline,challenger=map(Path,(baseline,challenger));raw=[p.read_bytes() for p in (baseline,challenger)];a,b=map(json.loads,raw);provenance=dict(input_sha256={str(p.resolve()):hashlib.sha256(value).hexdigest() for p,value in zip((baseline,challenger),raw)},helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    def key(r):return r['profile'],r['seed'],r['seat']
    aa={key(r):r for r in a};bb={key(r):r for r in b}
    if len(aa)!=len(a) or len(bb)!=len(b) or set(aa)!=set(bb):raise ValueError('Paired schedules differ or contain duplicate cells')
    if not all(r['complete_valid_lobby'] and r['score_eligible'] and r['failure'] is None for r in [*a,*b]):return dict(all_valid=False,raw_score_gain=None,paired_cases=len(a),reason='A required cell is faulted or incomplete; no valid-only selection',**provenance)
    if any(x['native']!=bb[k]['native'] or [p for i,p in enumerate(x['lineup']) if i!=x['seat']]!=[p for i,p in enumerate(bb[k]['lineup']) if i!=bb[k]['seat']] for k,x in aa.items()):raise ValueError('Native runtime/opponent assignment differs')
    groups=sorted({r['seed'] for r in a});profiles=sorted({r['profile'] for r in a});gain=lambda k:100*(aa[k]['candidate_placement']-bb[k]['candidate_placement'])/7
    if set(profiles)!={'scripts','incumbent-mixed','diverse-history'} or len(groups)<2 or expected_groups is not None and len(groups)!=expected_groups:raise ValueError('Required complete three-profile seed schedule missing')
    if any({(r['profile'],r['seat']) for r in a if r['seed']==s}!={(p,seat) for p in profiles for seat in range(8)} for s in groups):raise ValueError('Each group must preserve all8seats and fixed profiles')
    means=np.array([np.mean([gain(k) for k in aa if k[1]==s]) for s in groups]);rng=np.random.default_rng(92317);ci=np.quantile(rng.choice(means,(10000,len(means))).mean(1),[.025,.975]).tolist() if len(groups)>1 else None
    return dict(all_valid=True,paired_cases=len(a),seed_groups=len(groups),raw_score_gain=float(means.mean()),raw_score_gain_95=ci,per_profile_raw_gain={p:float(np.mean([gain(k) for k in aa if k[0]==p])) for p in profiles},method='Paired whole-lobby-seed-group bootstrap, preserving all8 seats and fixed profiles',interpretation='Positive favors challenger; environment uncertainty, not independent training-seed or human/expert evidence',**provenance)
