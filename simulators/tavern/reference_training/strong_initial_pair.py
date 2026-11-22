"""Collect immutable update-zero artifacts and verify a matched treatment pair."""
import argparse,hashlib,json,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
RL=ROOT/'evidence/strong-bots-20261109-0516Z/tavern/rl'
def collect(name):
    os.environ.setdefault('MODAL_DISABLE_API_PROXY','1')
    import modal
    stage=RL/name;p=json.loads((stage/'provider.json').read_text());out=stage/'initial-audit';out.mkdir(exist_ok=False)
    volume=modal.Volume.from_name(p['volume_name'],environment_name='staging');temp=out/'initial.pt.download'
    with temp.open('wb') as f:
        for chunk in volume.read_file('/'+name+'/training/initial.pt'):f.write(chunk)
    temp.replace(out/'initial.pt');(out/'collection.json').write_text(json.dumps(dict(provider=p,remote='/'+name+'/training/initial.pt',sha256=hashlib.sha256((out/'initial.pt').read_bytes()).hexdigest(),immutable='Trainer writes initial.pt once before its first cohort; atomic replacement, no subsequent rewrites',new_paid_allocation=False),indent=2)+'\n');print(name,'initial state collected',flush=True)
def audit(names,out):
    import torch
    from .validate import equal
    out=Path(out)
    if out.exists():raise FileExistsError(out)
    a,b=[torch.load(RL/n/'initial-audit/initial.pt',map_location='cpu',weights_only=False) for n in names]
    flags={k:equal(a[k],b[k]) for k in ('policy','optimizer','rng','population','binding','architecture','lineage','update','history','env_states','next_cohort_seed')}
    difference={k:[a['config'].get(k),b['config'].get(k)] for k in set(a['config'])|set(b['config']) if a['config'].get(k)!=b['config'].get(k)}
    flags['only_declared_treatment_differs']=difference=={'anchor_kl_coef':[0.0,0.2]}
    flags['fresh_logical_domain']=a['update']==b['update']==0 and not a['history'] and not b['history'] and not a['env_states'] and not b['env_states']
    result=dict(names=names,checks=flags,config_differences=difference,all_passed=all(flags.values()),parent_sha256=a['lineage']['parent_resume_sha256'],new_cohort_seed=a['next_cohort_seed'],helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    out.write_text(json.dumps(result,indent=2)+'\n')
    if not result['all_passed']:raise ValueError('Actual initial states do not form a matched pair')
    print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['collect','audit']);p.add_argument('--names',nargs='+',required=True);p.add_argument('--out',type=Path);a=p.parse_args()
    if a.mode=='collect':
        for name in a.names:collect(name)
    else:
        if len(a.names)!=2 or a.out is None:raise ValueError('Exactly two ordered arms and an output required')
        audit(a.names,a.out)
