"""Publish exact incumbent lineage and NEW public development cases only."""
import datetime,hashlib,json,shutil
from pathlib import Path
import numpy as np
import torch
from ..policies import admit_bundle
ROOT=Path(__file__).resolve().parents[4]
CAMPAIGN=ROOT/'evidence/strong-bots-20261109-0516Z'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,data):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');tmp.replace(p)

def prepare():
    root=CAMPAIGN/'tavern/evaluator';root.mkdir(parents=True,exist_ok=True)
    old=ROOT/'evidence/overnight-20261108-1533Z/tavern/researchers/deepseek';bundle=old/'submissions/r3';checkpoint=old/'runs/r3/artifact/resume.pt'
    ident,summary=admit_bundle(bundle);ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if ck['identity']!=ident:raise ValueError('Checkpoint/export identity differs')
    with np.load(bundle/'weights.npz',allow_pickle=False) as z:
        if set(z.files)!=set(ck['policy']) or any(not np.array_equal(z[k],ck['policy'][k].numpy()) for k in z.files):raise ValueError('Checkpoint/export weights differ')
    if any(sha(old/'runs/r3/source'/f)!=h for f,h in ident['source'].items()):raise ValueError('Original editable source drift')
    audit=dict(policy_id='deepseek-r3',inference_bundle=str(bundle),original_editable_source=str(old/'runs/r3/source'),original_recipe=str(old/'runs/r3/recipe.json'),full_checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),identity_sha256=sha(bundle/'identity.json'),weights_sha256=sha(bundle/'weights.npz'),update=ck['update'],samples=summary['samples'],full_checkpoint_fields=list(ck),optimizer_state_present=bool(ck['optimizer']['state']),RNG_state_present=all(k in ck for k in ['torch_rng','numpy_rng','python_rng','cuda_rng']),native_states=len(ck['env_states']),checkpoint_identity_equals_export=True,checkpoint_arrays_exact_export=True,canonical_reference_class_compatible=False,architecture_difference='Original two Tanh layers have body0/body2 weights; canonical reference includes LayerNorm and body3. No exact direct load or blind mapping.',exact_resume_scope='Original source/config/train_entry identity only. New training must declare imported parent lineage and prove exact original logits/value parity.',native_fingerprint=ident['binding']['native_source_hash'],training_recipe=ident['config'],source=ident['source'])
    write(root/'incumbent-audit.json',audit)
    parent=ROOT/'evidence/reference-readiness-20261109-0251Z/tavern/evaluator/registry-v5';release=root/'public-development-v1';release.mkdir(exist_ok=False)
    reg=json.loads((parent/'registry.json').read_text());keep={'starter-111901','starter-111902','starter-111903','recipe-111901','deepseek-r3','glm-run3','tempo','economy','synergy','selfplay-final'};reg['opponents']=[i for i in reg['opponents'] if i['policy_id'] in keep]
    for item in reg['opponents']:
        if item['kind']=='learned':
            dest=release/item['path'];dest.mkdir(parents=True)
            for f in ['policy.py','learner.py','train.py','weights.npz','identity.json','summary.json']:shutil.copy2(parent/item['path']/f,dest/f)
    reg['schema']='tavern-strong-campaign-public-v1'
    reg['profiles']={'scripts':['tempo','economy','synergy','tempo','synergy','economy','tempo'],'incumbent-mixed':['tempo','economy','synergy','deepseek-r3','starter-111901','deepseek-r3','starter-111902'],'diverse-history':['tempo','economy','synergy','deepseek-r3','glm-run3','recipe-111901','starter-111903']}
    reg['profile_assignment']='Seven slots clockwise from candidate seat; all8 candidate seats required';reg['difficulty_labels']='Neutral compositions, no assumed difficulty order';reg['readiness']={'harness_qualified':True,'strong_panel_qualified':False,'superhuman_evidence':False};reg['previous_reference_id']='selfplay-final is old194update diagnostic, not this campaign final'
    write(release/'registry.json',reg)
    for name in ['opponents.py','lobby.py']:shutil.copy2(Path(__file__).resolve().parents[1]/name,release/name)
    write(release/'panel.json',dict(label='strong-campaign-NEW-PUBLIC-development-v1',seeds=[50101,50102]));write(release/'panel-extended.json',dict(label='strong-campaign-NEW-PUBLIC-extended-v1',seeds=list(range(50111,50119))))
    (release/'README.md').write_text('Public development only. VisibleScript/ScriptMixture/act_visible use exact visible arrays and current static catalog; run_lobby requires8 explicit actors through all8 natural endings. NeuralVisible adapter is canonical evaluation_readiness/policies.py, loading each exact source/NPZ identity. DeepSeek-r3 is the incumbent; original full checkpoint audit is sibling incumbent-audit.json. Current LayerNorm class is NOT compatible; preserve original Policy or prove explicit adapter output parity. No private schedules included. Profile names describe composition, not assumed strength.\n')
    write(release/'manifest.json',{str(p.relative_to(release)):sha(p) for p in release.rglob('*') if p.is_file()})
    for p in release.rglob('*'):
        if p.is_file():p.chmod(0o444)
    for p in sorted([p for p in release.rglob('*') if p.is_dir()],key=lambda p:len(p.parts),reverse=True):p.chmod(0o555)
    release.chmod(0o555)
    p=CAMPAIGN/'tavern/coordination/evaluator.json';d=json.loads(p.read_text());d.update(phase='incumbent_full_checkpoint_audited_public_development_published',incumbent=audit,public_development=dict(path=str(release),manifest_sha256=sha(release/'manifest.json'),seeds=[50101,50102],extended_seeds=list(range(50111,50119)),profiles=reg['profiles'],cases_per_policy=48,neural_adapter='evaluation_readiness.policies.NeuralVisible(bundle,policy_id).remote_act(obs,indices)',scripts_API='opponents.py: VisibleScript, ScriptMixture, act_visible(features,cards,mask,style)',full_lobby_API='lobby.py: run_lobby(policies8,seed,rules,threads=1,seconds=120,trace=NEW_DIR)'),next_action='New OpenCreditBudget CPU launcher and new integrated benchmark workflow',updated_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    d['requested_changes']=['RL: exact full checkpoint/source/recipe in incumbent-audit.json; arrays exactly match r3 export. Original class has no LayerNorm; preserve exact class/forward for explicit warm start and record optimizer/RNG choices.','RL: NEW PUBLIC 50101/50102 and extended50111..50118 are available, 3 neutral mixed/script profiles and all8seats. Publish immutable stage exports, exact inference-source mapping, archived runtime source and full checkpoint; evaluator will verify full original lineage and arrays, without substituting the canonical LayerNorm class.']
    write(p,d);return dict(public_release=str(release),manifest_sha256=sha(release/'manifest.json'),incumbent_full_checkpoint=str(checkpoint))

if __name__=='__main__':print(json.dumps(prepare(),indent=2))
