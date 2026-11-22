"""Offline full-lineage audit, without inference, training or a paid allocation."""
import argparse,collections,hashlib,json,math
from pathlib import Path
import torch

def digest(state,actor_only=False):
    h=hashlib.sha256()
    for k,v in sorted(state.items()):
        if actor_only and k.startswith('critic.'):continue
        h.update(k.encode());h.update(v.cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def audit(checkpoint,out):
    checkpoint,out=Path(checkpoint),Path(out)
    if out.exists():raise FileExistsError(out)
    ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if ck['schema']!='tavern-strong-population-v1':raise ValueError('Wrong schema')
    if not {'policy','optimizer','rng','population','history','env_states','next_cohort_seed','binding','config','lineage','update','architecture'}<=set(ck):raise ValueError('Full-state checkpoint fields missing')
    if set(ck['rng'])!={'torch','python','numpy','cuda'}:raise ValueError('RNG fields missing')
    if ck['update'] and len(ck['env_states'])!=ck['config']['n_games']:raise ValueError('Native state count differs from the complete cohort')
    if not {'parent_resume_sha256','input_manifest_sha256','parent_policy_sha256','not_an_exact_old_source_resume'}<=set(ck['lineage']):raise ValueError('Explicit parent provenance fields missing')
    history=ck['history'];counts=collections.Counter();seats=collections.Counter();faults=[]
    for row in history:
        if row['learning']['samples']!=row['counts']['accepted_learning_samples']:raise ValueError('Unexecuted or duplicate rows in PPO')
        if not all(math.isfinite(v) for v in row['learning'].values() if isinstance(v,(int,float))):raise ValueError('Nonfinite learner metrics')
        counts.update({k:v for k,v in row['counts'].items() if isinstance(v,(int,float))})
        for r in row['episodes']:
            if r['valid']:seats[str(r['seat'])]+=1
            else:faults.append(dict(update=row['update'],**r))
    expected=(ck['config']['seed']+ck['update']*1000003)%2**64
    if ck['next_cohort_seed']!=expected:raise ValueError('Stored next seed differs from the deterministic cursor')
    if not all(torch.isfinite(v).all() for v in ck['policy'].values()):raise ValueError('Nonfinite policy')
    for state in ck['optimizer']['state'].values():
        if not all(torch.isfinite(v).all() for v in state.values() if torch.is_tensor(v)):raise ValueError('Nonfinite optimizer')
    entries=ck['population']['entries'];samples=sum(h['learning']['samples'] for h in history)
    result=dict(schema='tavern-strong-checkpoint-audit-v1',checkpoint=str(checkpoint.resolve()),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),update=ck['update'],lineage_samples=samples,total_known_prior_and_current_samples=ck['lineage'].get('parent_total_known_samples',ck['lineage']['parent_samples'])+samples,counts=dict(counts),natural_learner_episodes_per_physical_seat=dict(seats),faults=faults,finite_policy_optimizer_metrics=True,only_executed_on_policy_rows=True,derived_next_seed_matches=True,parameters=sum(v.numel() for v in ck['policy'].values()),population_entries=len(entries),distinct_population_actor_forward_hashes=len({digest(e['policy'],True) for e in entries}),all_parent_and_environment_state_fields_retained=True,source_schema=ck['binding']['schema'],helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();r=audit(a.checkpoint,a.out);print(json.dumps({k:r[k] for k in ['update','lineage_samples','faults','natural_learner_episodes_per_physical_seat','distinct_population_actor_forward_hashes']}))
