"""Archive an explicitly selected own checkpoint for a new training domain."""
import argparse,json,shutil
from pathlib import Path
import torch
from .common import ROOT,sha,write
from .lineage import import_parent

def prepare(checkpoint,bundle,parent_release,out):
    checkpoint,bundle,parent_release,out=map(Path,(checkpoint,bundle,parent_release,out))
    ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if ck.get('schema')!='tavern-strong-population-v1':raise ValueError('Only a retained strong full-state checkpoint is accepted')
    release_files=json.loads((parent_release.parent/'source-files.json').read_text())
    if any(sha(parent_release/f)!=h for f,h in release_files.items()):raise ValueError('Parent immutable runtime release mutated')
    out.mkdir(parents=True,exist_ok=False)
    (out/'incumbent').mkdir()
    for f in ('policy.py','learner.py','train.py','identity.json','summary.json','weights.npz'):shutil.copy2(bundle/f,out/'incumbent'/f)
    shutil.copy2(checkpoint,out/'parent-resume.pt')
    scripts=ROOT/'evidence/strong-bots-20261109-0516Z/tavern/rl/inputs-v1/scripts'
    (out/'scripts').mkdir();shutil.copy2(scripts/'opponents.py',out/'scripts/opponents.py')
    for f,h in ck['binding']['files'].items():
        source=parent_release/f
        if not source.exists():source=ROOT/f
        if sha(source)!=h:raise ValueError('Exact parent-bound source unavailable: '+f)
        dest=out/'parent-source'/f;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
    shutil.copy2(__file__,out/'prepare-helper.py')
    write(out/'fork-request.json',dict(kind='explicit-strong-parent-import-v1',parent_checkpoint_sha256=sha(checkpoint),parent_identity_sha256=sha(bundle/'identity.json'),parent_runtime_manifest_sha256=sha(parent_release.parent/'source-files.json'),parent_runtime=str(parent_release.resolve()),copied_state=['policy','Adam moments','all RNG','historical population'],new_domain='Logical update0 and new cohort seeds/config/source; old environment states remain archived in the exact full parent checkpoint',ordinary_exact_resume_bypass=False,private_final_feedback_used=False))
    if any(p.suffix=='.pyc' or '__pycache__' in p.parts for p in out.rglob('*')):raise ValueError('Derived bytecode is not a frozen training input')
    write(out/'manifest.json',{str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file()})
    model,parent,pop,receipt=import_parent(out,torch.device('cpu'))
    # A receipt beside the inputs does not alter the frozen input manifest.
    write(out.parent/(out.name+'-verified.json'),dict(parent_update=parent['update'],parameters=sum(p.numel() for p in model.parameters()),population=len(pop.entries),input_manifest_sha256=sha(out/'manifest.json'),lineage=receipt))
    print(str(out.resolve()),flush=True);return out

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--bundle',type=Path,required=True);p.add_argument('--parent-release',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();prepare(a.checkpoint,a.bundle,a.parent_release,a.out)
