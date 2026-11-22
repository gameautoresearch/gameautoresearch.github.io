"""Public-development registry creation; no final seeds or strength assertions."""
import json,shutil,sys,hashlib,time
from pathlib import Path
from .policies import admit_bundle
from .budget import ROOT,CAMPAIGN
sys.path.insert(0,str(ROOT/'simulators/tavern/autoresearch'))
from contract import binding,sha,write,EDITABLE

def prepare(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False);old=ROOT/'evidence/overnight-20261108-1533Z/tavern'
    sources={
      'starter-111901':old/'stage-b2/starter-111901/artifact',
      'starter-111902':old/'stage-b2/starter-111902/artifact',
      'starter-111903':old/'stage-b2/starter-111903/artifact',
      'recipe-111901':old/'stage-b2/agent-111901/artifact',
      'deepseek-r3':old/'researchers/deepseek/submissions/r3',
      'glm-run3':old/'researchers/glm/submissions/run3',
    };items=[]
    for name,path in sources.items():
        ident,summary=admit_bundle(path);dest=out/'learned'/name;dest.mkdir(parents=True)
        for f in [*EDITABLE,'weights.npz','identity.json','summary.json']:shutil.copy2(path/f,dest/f);(dest/f).chmod(0o444)
        items.append(dict(policy_id=name,kind='learned',path='learned/'+name,source_origin=str(path),files={f.name:sha(f) for f in dest.iterdir()},admission_qualified=True,source_visibility_audit='Exact retained visible-only feedforward policy modules; no search, state access or external weights. Strict NPZ/architecture/source binding checks.',strong_qualified=False,reference_lineage=ident.get('reference_lineage'),historical_experiment=True))
    for style in ('tempo','economy','synergy'):items.append(dict(policy_id=style,kind='script',style=style,source_sha256=sha(Path(__file__).parent/'opponents.py'),admission_qualified=True,strong_qualified=False))
    profiles={
      'easy':['tempo','economy','synergy','recipe-111901','glm-run3','starter-111901','starter-111901'],
      'competent-mixed':['tempo','economy','synergy','starter-111902','starter-111903','deepseek-r3','glm-run3'],
      'harder-mixed':['tempo','economy','synergy','starter-111902','starter-111903','deepseek-r3','deepseek-r3'],
    }
    registry=dict(schema='tavern-mixed-visible-v1',binding=binding(),opponents=items,profiles=profiles,profile_assignment='Seven slots go clockwise from candidate seat; all8 candidate seats required.',difficulty_labels='Precommitted hypotheses; empirical ordering and strength remain unqualified.',candidate_constraints=dict(max_parameters=2000000,max_weights_bytes=32000000,forward_deadline_seconds=15,visibility='candidate exact numeric/cards/mask'),reference_constraints='Host-owned statically reviewed finite feedforward modules; same visible tensors only,CPU1thread; no engine/search/LLM or state access. Scripts pure numeric/card/mask/static catalog.',readiness=dict(harness_qualified=False,learned_reference_qualified=False,strong_panel_qualified=False,superhuman_evidence=False),created_at_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    write(out/'registry.json',registry);write(out/'manifest.json',{str(f.relative_to(out)):sha(f) for f in out.rglob('*') if f.is_file()})
    return out

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();print(prepare(a.out))
