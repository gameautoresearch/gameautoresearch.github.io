"""Close one chosen lineage before drawing a fresh assessment bank. No GPU calls."""
import hashlib,json,secrets,os,sys,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[4]
RL=ROOT/'evidence/reference-readiness-20261109-0251Z/tavern/rl'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def seal(p,x):
 if p.exists():raise FileExistsError('Immutable selection/bank already exists: '+str(p))
 t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');t.replace(p);p.chmod(0o444)
if __name__=='__main__':
 allocation=json.loads((RL/'attempt-03/allocation.json').read_text())
 if not allocation['provider'].get('finished_at'):raise ValueError('GPU worker still active')
 initial=RL/'attempt-02/export-initial';final=RL/'attempt-03/collected/export';checkpoint=RL/'attempt-03/collected/training/latest.pt'
 ck=torch.load(checkpoint,map_location='cpu',weights_only=False);parent=torch.load(RL/'attempt-02/collected/training/latest.pt',map_location='cpu',weights_only=False)
 if ck['binding']!=parent['binding'] or ck['config']!=parent['config'] or ck['update']<=parent['update']:raise ValueError('Continuation source/recipe/update lineage mismatch')
 from simulators.tavern.reference_training.assessment.run import verify
 runtime=RL/'assessment-runtime-v2'
 _,ii=verify(initial,runtime);_,fi=verify(final,runtime)
 if ii['reference_lineage']['reference_binding']!=fi['reference_lineage']['reference_binding'] or ii['config']!=fi['config']:raise ValueError('Initial/final is not matched')
 if fi['reference_lineage']['checkpoint_sha256']!=sha(checkpoint):raise ValueError('Final is not exact last checkpoint')
 if ii['reference_lineage']['update']!=0 or fi['reference_lineage']['update']!=ck['update']:raise ValueError('Wrong selected milestone')
 summary=json.loads((final/'summary.json').read_text())
 if not summary['finite_losses'] or not summary['changed_weights'] or summary['parameters']>2000000:raise ValueError('Invalid learned export')
 roster=dict(initial=dict(path=str(initial),identity_sha256=sha(initial/'identity.json'),weights_sha256=sha(initial/'weights.npz')),final=dict(path=str(final),identity_sha256=sha(final/'identity.json'),weights_sha256=sha(final/'weights.npz'),checkpoint_sha256=sha(checkpoint),update=ck['update']))
 selection=dict(closed=True,closed_unix=time.time(),policy_tuning_closed=True,more_training_planned=False,rule='Last complete update of predeclared corrected lineage after single continuation; no seed/checkpoint selection by strength.',training_seed=ck['config']['seed'],roster=roster,runtime_source=str(RL/'attempt-02/source'),native='217577b83fc4525e',admission='admission-v2')
 seal(RL/'selection-closed.json',selection)
 # Exclude all training lobby seeds after mapping each original game index to game0.
 mask=(1<<64)-1
 def mix(z):
  z=(z+0x9E3779B97F4A7C15)&mask;z=((z^(z>>30))*0xBF58476D1CE4E5B9)&mask;z=((z^(z>>27))*0x94D049BB133111EB)&mask;return z^(z>>31)
 used={1209101,1209102}
 for row in ck['history']:
  cohort=(ck['config']['seed']+(row['update']-1)*1000003)%2**64
  used.update((cohort^mix(g)^mix(0))&mask for g in range(ck['config']['n_games']))
 seeds=[]
 while len(seeds)<4:
  seed=secrets.randbits(63)
  if seed not in used and seed not in seeds:seeds.append(seed)
 scripts=RL/'assessment-interface-v2';pool=RL/'attempt-01a/collected/training/latest.pt'
 bank=dict(schema='tavern-reference-fresh-bank-v1',created_unix=time.time(),selection_sha256=sha(RL/'selection-closed.json'),seeds=seeds,seed_groups=4,candidate_seats=list(range(8)),profiles=['scripts-mixed','scripts-history-mixed','history-heavy'],policies=roster,required_full_lobbies=192,scripts_sha256=sha(scripts/'opponents.py'),lobby_sha256=sha(scripts/'lobby.py'),pool_checkpoint_sha256=sha(pool),pool_runtime=str(RL/'assessment-runtime-v1'),candidate_runtime=str(runtime),assessment_code_sha256=sha(Path(__file__).with_name('run.py')),neural_inference_cap_seconds=15,tuning_after_opening=False)
 seal(RL/'fresh-bank-commitment.json',bank)
 p=RL.parent/'coordination/rl.json';s=json.loads(p.read_text());s.update(phase='selection_closed_fresh_bank_precommitted',active_attempt=None,selection_closed=selection,private_final_bank=dict(path=str(RL/'fresh-bank-commitment.json'),sha256=sha(RL/'fresh-bank-commitment.json'),groups=4,required_lobbies=192,no_private_seeds_in_status=True),chosen_final_export=str(final),matched_initial_export_path=str(initial));s.setdefault('available_exports',[]).append(dict(path=str(final),role='selected-final-reference',summary=summary,runtime_source=str(RL/'attempt-02/source'),selection_closed=True));t=p.with_suffix('.tmp');t.write_text(json.dumps(s,indent=2)+'\n');t.replace(p)
 print(json.dumps(dict(selection_closed=True,final_update=ck['update'],final_samples=summary['samples'],parameters=summary['parameters'],fresh_bank_precommitted=True,required_lobbies=192),indent=2))
