"""Paired public-screen comparison; whole seed groups stay together."""
import argparse,json
from pathlib import Path
import numpy as np
from .common import sha,write

def compare(baseline,challenger,out):
    baseline,challenger=Path(baseline),Path(challenger)
    commitments=[json.loads((p/'commitment.json').read_text()) for p in (baseline,challenger)]
    if any(commitments[0][k]!=commitments[1][k] for k in ('public_manifest_sha256','panel_sha256','full_seats','profiles')):
        raise ValueError('Public schedules/opponents differ')
    rows=[json.loads((p/'screen/episodes.json').read_text()) for p in (baseline,challenger)]
    keyed=[{(r['profile'],r['seed'],r['seat']):r for r in group} for group in rows]
    results=[json.loads((p/'screen/result.json').read_text()) for p in (baseline,challenger)]
    if any(not r['complete_valid_schedule'] for r in results):raise ValueError('Incomplete or invalid screen is not a score')
    if set(keyed[0])!=set(keyed[1]) or any(len(k)!=len(r) for k,r in zip(keyed,rows)):
        raise ValueError('Cases are not uniquely paired')
    cases=[dict(profile=k[0],seed=k[1],seat=k[2],placement_gain=keyed[0][k]['candidate_placement']-keyed[1][k]['candidate_placement']) for k in sorted(keyed[0])]
    seeds=sorted({r['seed'] for r in cases})
    groups=np.array([np.mean([r['placement_gain'] for r in cases if r['seed']==seed]) for seed in seeds])
    ci=(np.quantile(np.random.default_rng(92317).choice(groups,(10000,len(groups))).mean(1),[.025,.975])*100/7).tolist() if len(groups)>1 else None
    gain=float(np.mean(groups)*100/7)
    result=dict(baseline=str(baseline.resolve()),challenger=str(challenger.resolve()),baseline_identity=commitments[0]['candidate_identity_sha256'],challenger_identity=commitments[1]['candidate_identity_sha256'],raw_score_gain=gain,raw_score_gain_95=ci,paired_cases=len(cases),seed_groups=len(seeds),per_profile_raw_gain={p:float(np.mean([r['placement_gain'] for r in cases if r['profile']==p])*100/7) for p in commitments[0]['profiles']},baseline_score=results[0]['raw_score'],challenger_score=results[1]['raw_score'],all_valid=True,method='Paired whole-lobby-seed-group bootstrap, preserving all seats and profiles',interpretation='Positive is challenger improvement; development data and one pretrained lineage, not independent training-seed or human-strength evidence',input_sha256={str(p/'screen/episodes.json'):sha(p/'screen/episodes.json') for p in (baseline,challenger)},helper_sha256=sha(__file__))
    write(out,result);return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--baseline',type=Path,required=True);p.add_argument('--challenger',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();print(json.dumps(compare(a.baseline,a.challenger,a.out),indent=2))
