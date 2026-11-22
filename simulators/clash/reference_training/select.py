"""Choose one public-development stage per family; preserve every paired seed."""
import json
from .launch import CAMPAIGN
from .evaluate import bundle
from simulators.clash.autoresearch.common import dump,sha

def select():
 root=CAMPAIGN/'clash/rl'
 if (root/'selection-close.json').exists():raise ValueError('Selection already closed')
 roster=[];choices={}
 variants=[('v3',[49192,69193],'')]
 if (root/'paired-v4-49192/collected/result.json').exists():variants.append(('v4',[49192],'spatial-'))
 for variant,seeds,prefix in variants:
  for family,case,alias in [('reference','teacher-pfsp','teacher'),('control','teacher-off-control','control'),('untrained','teacher-pfsp','untrained')]:
   stages=[('initial','untrained')] if family=='untrained' else [('post-teacher',alias+'-post'),('final',alias+'-final')]
   scores={}
   for stage,label in stages:
    values=[]
    for seed in seeds:
     folder=root/f'paired-{variant}-{seed}';path=folder/(label+'-development/result.json')
     if family=='untrained' and not path.exists():path=folder/'initial-development/result.json'
     if not path.exists():break
     value=json.loads(path.read_text())
     if not value['complete']:raise ValueError('Invalid development panel cannot select a candidate')
     values.append(value['mean_points'])
    if len(values)==len(seeds):scores[stage]={'mean_points':sum(values)/len(values),'by_training_seed':dict(zip(map(str,seeds),values)),'alias':label}
   if not scores:raise ValueError('Missing complete development stages')
   # Fixed tie-break: earlier predeclared stage. No per-seed stage selection.
   selected=max(scores,key=lambda x:scores[x]['mean_points']);choices[prefix+family]={'selected_stage':selected,'public_development':scores,'tie_break':'earlier predeclared stage'}
   for seed in seeds:
    attempt=root/f'paired-{variant}-{seed}';validation=json.loads((attempt/'artifact-validation.json').read_text())
    if not validation['cases'][case]['proof']['exact_next_update_resume']:raise ValueError('GPU resume proof missing')
    original=attempt/'collected'/case/selected;label=scores[selected]['alias'];target=attempt/(label+'-bundle')
    spec=bundle(original,target,{'track':'clash-reference-final-selection','variant':variant,'training_seed':seed,'stage':selected})
    lineage=json.loads((original/'lineage.json').read_text());full=attempt/'collected'/case/'checkpoint.pt'
    import torch
    from .league import tensor_identity
    checkpoint=torch.load(full,map_location='cpu',weights_only=True)
    import numpy as np
    with np.load(original/'weights.npz',allow_pickle=False) as weights:
     if tensor_identity({k:torch.from_numpy(weights[k]) for k in weights.files})!=lineage['policy_identity']:raise ValueError('Selected milestone tensor bytes differ from declared identity')
    if lineage['binding']!=checkpoint['binding'] or any(sha(original/n)!=checkpoint['binding']['source'][n] for n in ('policy.py','learner.py')):raise ValueError('Selected milestone source binding differs from complete checkpoint')
    identities=[tensor_identity(x) for x in checkpoint['league']]
    if lineage['policy_identity']!=tensor_identity(checkpoint['model']) and lineage['policy_identity'] not in identities:raise ValueError('Selected milestone weights absent from the validated complete checkpoint')
    roster.append({'name':f'{variant}-{family}-{seed}','family':prefix+family,'training_seed':seed,'stage':selected,'bundle':str(target.resolve()),'identity':spec['identity'],'parameters':spec['parameters'],'raw_policy_identity':lineage['policy_identity'],'source_archive_sha256':validation['source_archive_sha256'],'milestone_lineage':str((original/'lineage.json').resolve()),'full_final_resume_checkpoint':str(full.resolve()),'full_final_resume_checkpoint_sha256':sha(full),'selected_milestone_is_full_resume_checkpoint':selected=='final','selected_weights_verified_in_complete_checkpoint':True,'frozen_league_index':identities.index(lineage['policy_identity']) if lineage['policy_identity'] in identities else None})
 receipt={'selection_bank':'reused public development only','one_stage_per_family_across_all_training_seeds':True,'all_precommitted_v3_seeds_retained':True,'choices':choices,'roster':roster,'fresh_bank_not_yet_created':True}
 dump(root/'selection-development.json',receipt);dump(root/'candidates.json',roster);return receipt

if __name__=='__main__':print(json.dumps(select(),indent=2))
