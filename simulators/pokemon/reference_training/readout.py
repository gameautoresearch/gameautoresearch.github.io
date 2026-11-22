"""Offline aggregation; retains all attempted workers, unresolved games and costs."""
from __future__ import annotations
import json,hashlib
from pathlib import Path
import numpy as np
from .remote import BASE,ROOT,budget,role_usage
from .trainer import atomic

def panel_summary(panel):
    games=panel['games'];scheduled=panel['scheduled_games'];natural=[x for x in games if x['terminated']];wins=sum(x['score']==1 for x in natural);ties=sum(x['score']==.5 for x in natural);losses=sum(x['score']==0 for x in natural)
    unresolved=scheduled-len(natural)
    seats={str(seat):{'natural':sum(x['terminated'] for x in games if x['learner_seat']==seat),'caps':sum(x['truncated'] for x in games if x['learner_seat']==seat),'wins':sum(x.get('score')==1 for x in games if x['learner_seat']==seat),'ties':sum(x.get('score')==.5 for x in games if x['learner_seat']==seat),'losses':sum(x.get('score')==0 for x in games if x['learner_seat']==seat)} for seat in (0,1)}
    return {'opponent':panel['opponent_name'],'scheduled':scheduled,'attempted':len(games),'wins':wins,'ties':ties,'losses':losses,'unresolved':unresolved,'caps':panel['truncations'],'visible_rejections':panel['rejected_visible_choices'],'score_0_100':panel['score']*100 if panel['score'] is not None else None,'possible_points_bounds_0_100':[100*(wins+.5*ties)/scheduled,100*(wins+.5*ties+unresolved)/scheduled],'by_seat':seats,'complete':panel['valid_complete_schedule']}

def paired(a,b):
    # Resample team/battle seeds, retaining every opponent and both seats in a cluster.
    left={(p['opponent_name'],g['seed'],g['learner_seat']):g['score'] for p in a for g in p['games']}
    right={(p['opponent_name'],g['seed'],g['learner_seat']):g['score'] for p in b for g in p['games']}
    if left.keys()!=right.keys() or any(left[k] is None or right[k] is None for k in left):return {'delta_0_100':None,'reason':'incomplete or unmatched panel; no cap imputation'}
    seeds=sorted({k[1] for k in left});d=np.asarray([np.mean([left[k]-right[k] for k in left if k[1]==s])*100 for s in seeds]);rng=np.random.default_rng(20261109);draw=rng.choice(d,(4000,len(d)),replace=True).mean(-1)
    return {'delta_0_100':float(d.mean()),'paired_seed_bootstrap_95':list(np.quantile(draw,[.025,.975])),'seed_clusters':len(seeds),'training_seeds':1,'interpretation':'Conditional on this single matched training seed and frozen development opponents; not broad strength calibration'}

def build():
    jobs=[]
    for target in sorted((BASE/'jobs').iterdir()):
        request=json.loads((target/'request.json').read_text());allocation=json.loads((target/'allocation.json').read_text()) if (target/'allocation.json').exists() else None
        result=json.loads((target/'collected/worker-result.json').read_text()) if (target/'collected/worker-result.json').exists() else None
        metrics=target/'collected/training/metrics.jsonl'
        if result and result.get('training') and metrics.exists():
            records=[json.loads(line) for line in metrics.read_text().splitlines()]
            result['training']['ppo_batches_updated']=sum(x['ppo']['samples']>0 for x in records)
            result['training']['optimizer_minibatch_steps']=sum(x['ppo'].get('minibatch_updates',0) for x in records)
            result['training']['discarded_allocation_batches']=sum(x['ppo'].get('skipped')=='allocation_expired' for x in records)
        jobs.append({'label':target.name,'request':request,'provider':allocation,'result':result,'local_checkpoint':str(target/'collected/training/checkpoint.pt') if (target/'collected/training/checkpoint.pt').exists() else None})
    assessment_attempts=[]
    for target in sorted((BASE/'assessments').glob('*')):
        request=json.loads((target/'request.json').read_text());seeds=request.pop('seeds');request.update(seed_count=len(seeds),seed_commitment_sha256=hashlib.sha256(json.dumps(seeds,separators=(',',':')).encode()).hexdigest())
        assessment_attempts.append({'label':target.name,'request':request,'provider':json.loads((target/'allocation.json').read_text()) if (target/'allocation.json').exists() else None,'status':json.loads((target/'collected/status.json').read_text()) if (target/'collected/status.json').exists() else None,'partial_panels':[str(p.relative_to(BASE)) for p in (target/'collected').glob('*/*.json') if p.name!='summary.json']})
    assessments=[];raw={}
    for path in sorted((BASE/'assessments').glob('*/collected/*/summary.json')):
        result=json.loads(path.read_text());label=path.parents[2].name;name=path.parent.name;raw[(label,name)]=result['panels']
        assessments.append({'assessment':label,'candidate':name,'source_admitted':result['source_admitted'],'panels':[panel_summary(x) for x in result['panels']]})
    effects=[]
    for label in sorted({x[0] for x in raw}):
        if (label,'selfplay-2718') in raw and (label,'control-2718') in raw:effects.append({'assessment':label,**paired(raw[(label,'selfplay-2718')],raw[(label,'control-2718')])})
    snapshot=budget().snapshot();own=[r for r in snapshot['operations'] if r['game']=='pokemon' and r['kind'].startswith('pokemon/rl/')]
    proof=any(x['result'] and x['result'].get('proof',{}).get('valid') for x in jobs)
    trained=any(x['result'] and x['result'].get('training',{}).get('trained_samples',0)>0 and x['result']['training'].get('source_admitted') for x in jobs)
    complete=any(len(x['panels'])>=3 and all(p['complete'] for p in x['panels']) and x['candidate']=='selfplay-2718' for x in assessments)
    result={'status':'baseline-complete' if effects else 'in-progress','jobs':jobs,'assessment_attempts':assessment_attempts,'assessments':assessments,'paired_effects':effects,'readiness':{'training_harness_qualified':proof and trained,'harness_qualified':False,'learned_reference_qualified':proof and trained and complete,'strong_panel_qualified':False,'superhuman_evidence':False,'harness_note':'Independent evaluator-v2 admission/qualification belongs to evaluator owner; own source-admitted development inference is reported separately'},'paired_initialization_audit':json.loads((BASE/'paired-initialization-audit.json').read_text()) if (BASE/'paired-initialization-audit.json').exists() else None,'normalized_reference_score':None,'learned_reference_qualification_definition':'Source-admitted learned neural baseline, exact resume and complete diverse both-seat diagnostic trajectories; independent strength calibration is separate','role_usage':role_usage(),'role_settled':{'gpu_seconds':sum(x['gpu_charged'] for x in own if x['status']=='settled'),'resource_usd':sum(x['usd_charged'] for x in own if x['status']=='settled')},'role_held':{'gpu_seconds':sum(x['gpu_hold'] for x in own if x['status']!='settled'),'resource_usd':sum(x['usd_hold'] for x in own if x['status']!='settled')},'model_usd':0,'all_owned_workers_settled':all(x['status']=='settled' for x in own),'provider_invoice_verified':False,'private_evidence_root':str(BASE),'initializer':'random only; teacher/imported weight contribution zero'}
    atomic(ROOT/'reports/pokemon/selfplay-reference-2026-11-09.json',result);return result
if __name__=='__main__':
    x=build();print(json.dumps({k:v for k,v in x.items() if k not in ('jobs','assessments')},indent=2))
