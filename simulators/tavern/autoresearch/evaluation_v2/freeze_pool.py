"""Create a new immutable reference/profile version from public selection evidence."""
import json,shutil
from pathlib import Path
from .release import sha,write
from simulators.tavern.evaluation_readiness.policies import admit_bundle

# Reviewed exact two-Tanh/embedding Policy.forward, independently parity-tested
# against the original incumbent. No engine/state/search/service calls in forward.
REVIEWED_POLICY_SHA256={'6749d595352bdefd1df5ead52749f44792cbc6b3d4ab6d1fb0ea964e671e1eec'}


def freeze(parent,out,profiles,new_references,public_evidence):
    parent,out=Path(parent),Path(out);manifest=json.loads((parent/'manifest.json').read_text());reg=json.loads((parent/'registry.json').read_text());items={r['policy_id']:r for r in reg['opponents']}
    if any(sha(parent/f)!=h for f,h in manifest.items()):raise ValueError('Parent inference pool changed')
    if set(profiles)!={'scripts','incumbent-mixed','diverse-history'} or any(len(v)!=7 or set(v)-set(items) for v in profiles.values()):raise ValueError('Three explicit seven-reference profiles required')
    if not all({'tempo','economy','synergy'}<=set(v) for v in profiles.values()) or not all('deepseek-r3' in profiles[p] for p in ('incumbent-mixed','diverse-history')):raise ValueError('Retained incumbent and visible script controls required')
    promoted={}
    for identity,record in new_references.items():
        record=Path(record);audit=json.loads(record.read_text());item=items[identity];bundle=parent/item['path'];ident,_=admit_bundle(bundle)
        if item['kind']!='learned' or not audit.get('admission_qualified') or audit.get('export_identity_sha256')!=sha(bundle/'identity.json') or audit.get('weights_sha256')!=sha(bundle/'weights.npz') or sha(Path(audit['checkpoint']))!=audit['checkpoint_sha256']:raise ValueError('Promoted reference lacks exact admitted checkpoint/source/export')
        if sha(bundle/'policy.py') not in REVIEWED_POLICY_SHA256:raise ValueError('New host reference inference source requires explicit review')
        if not any(identity in v for v in profiles.values()):raise ValueError('Selected learned reference must actually appear in opposition')
        promoted[identity]=dict(admission_sha256=sha(record),identity_sha256=sha(bundle/'identity.json'),weights_sha256=sha(bundle/'weights.npz'),checkpoint_sha256=audit['checkpoint_sha256'],reviewed_policy_sha256=sha(bundle/'policy.py'))
    if not promoted:raise ValueError('At least one admitted improved learned reference required')
    evidence={str(Path(p)):sha(p) for p in public_evidence}
    if not evidence:raise ValueError('Public selection evidence required')
    out.mkdir(parents=True,exist_ok=False)
    for rel in manifest:
        target=out/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(parent/rel,target);target.chmod(0o644)
    reg['profiles']=profiles;reg['profile_selection']='Frozen new reference version chosen on PUBLIC data before private grading; labels do not assert difficulty ordering'
    for item in reg['opponents']:
        if item['policy_id'] in promoted:item.update(role='selected-learned-reference',admission='Exact full checkpoint/source/recursive lineage audited; exact finite feedforward source reviewed; visible tensors only')
    write(out/'registry.json',reg);write(out/'pool-selection.json',dict(parent_manifest_sha256=sha(parent/'manifest.json'),promoted=promoted,profiles=profiles,public_evidence=evidence,private_final_feedback_used=False,old_incumbent_retained=True,strong_human_expert_claim=False))
    write(out/'manifest.json',{str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file() and p!=out/'manifest.json'})
    for p in out.rglob('*'):
        if p.is_file():p.chmod(0o444)
    for p in sorted([p for p in out.rglob('*') if p.is_dir()],key=lambda p:len(p.parts),reverse=True):p.chmod(0o555)
    out.chmod(0o555);return out
