"""Mixed-lobby release identity (four-player small form): shared paths, hashing and fail-closed verification.

The constructor is ../rules_enabled/release.py.
"""
import hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
AUTO=HERE.parent;GAME=AUTO.parent;ROOT=GAME.parents[1]
RUNTIME=('opponents.py','lobby.py','policies.py','screen.py','guest.py')
# The evaluation identity of the four-player small form. The eight-player mixed-v2 form is at tag full-size-20261113.
EVALUATOR_ID='tavern-four-small-v1'
RECEIPT='selected_registry-four-small-v1.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,data):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');tmp.replace(p)

def verify(release):
    release=Path(release).resolve();manifest=json.loads((release/'release.json').read_text())
    if manifest.get('evaluator_id')!=EVALUATOR_ID:raise ValueError('Explicit '+EVALUATOR_ID+' release required')
    frozen=release/'frozen'
    if any(not (frozen/f).is_file() or sha(frozen/f)!=h for f,h in manifest['frozen_files'].items()):raise ValueError('Frozen runtime source changed')
    private=release/'private';registry=private/'registry';entries=json.loads((registry/'manifest.json').read_text())
    if sha(registry/'manifest.json')!=manifest['registry_manifest_sha256'] or any(sha(registry/f)!=h for f,h in entries.items()):raise ValueError('Mixed reference registry mutated')
    if sha(release/'binding.json')!=manifest['binding_sha256']:raise ValueError('Task source binding changed')
    if any(not (release/'public'/f).is_file() or sha(release/'public'/f)!=h for f,h in manifest.get('public_files',{}).items()):raise ValueError('Frozen public starter/instruction/declaration changed')
    if manifest.get('qualification_panel_sha256') and sha(private/'panel-qualification.json')!=manifest['qualification_panel_sha256']:raise ValueError('Public qualification schedule changed')
    for name,h in manifest['decision_source'].items():
        if sha(GAME/'evaluation_readiness'/name)!=h:raise ValueError('Canonical evaluator changed since release: '+name)
    reg=json.loads((registry/'registry.json').read_text());seats=int(reg['binding']['rules']['num_players'])
    if reg['binding']!=json.loads((release/'binding.json').read_text()) or len(reg['profiles'])!=3 or any(len(x)!=seats-1 for x in reg['profiles'].values()):raise ValueError('Mixed profile contract differs')
    kinds={r['policy_id']:r['kind'] for r in reg['opponents']}
    if any(i not in kinds for x in reg['profiles'].values() for i in x):raise ValueError('Profile names an undeclared reference')
    if not any(kinds[i]=='learned' for x in reg['profiles'].values() for i in x):raise ValueError('No learned reference in the opposition')
    return manifest
