"""Exact source/parent/checkpoint admission for the explicitly imported r3 lineage."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
import hashlib
from ..policies import admit_bundle
from .budget import ROOT
from .prepare import sha,write

MAPS={
 'tavern-strong-population-v1':{f: 'simulators/tavern/reference_training/strong/'+f for f in ('policy.py','learner.py','train.py')},
 'tavern-reference-population-v1':{'policy.py':'simulators/tavern/model.py','learner.py':'simulators/tavern/run.py','train.py':'simulators/tavern/reference_training/train.py'},
}
def weight_hash(state):
    h=hashlib.sha256()
    for name,value in sorted(state.items()):h.update(name.encode());h.update(value.cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def verify_training_summary(ck,summary,schema):
    history=ck['history'];samples=sum(row['learning']['samples'] for row in history)
    finite=all(np.isfinite(row['learning']['loss']) for row in history)
    changed=any(row['learning']['parameter_delta_l2']>0 for row in history)
    if ck.get('schema')!=schema or summary['updates']!=ck['update'] or summary['samples']!=samples or summary['finite_losses']!=finite or summary['changed_weights']!=changed:raise ValueError('Checkpoint training history differs from export summary')
    if not finite:raise ValueError('Nonfinite checkpoint training history')
    return samples

def verify_full_resume(ck):
    if ck['schema']!='tavern-strong-population-v1':return
    c=ck['config'];update=ck['update']
    if ck['architecture']!=c['network'] or ck['population']['architecture']!=c['network'] or not ck['population']['entries'] or not ck['optimizer']['state'] or not ck['optimizer']['param_groups']:raise ValueError('Full optimizer/population/architecture state missing')
    if any(weight_hash(entry['policy'])!=entry['sha256'] or any(not torch.isfinite(value).all() for value in entry['policy'].values()) for entry in ck['population']['entries']):raise ValueError('Historical population snapshot changed or became nonfinite')
    if set(ck['rng'])!={'torch','python','numpy','cuda'} or ck['rng']['torch'] is None or ck['rng']['python'] is None or ck['rng']['numpy'] is None or c['device'].startswith('cuda') and not ck['rng']['cuda']:raise ValueError('Complete RNG state missing')
    if ck['next_cohort_seed']!=(c['seed']+update*1000003)%2**64 or [r['update'] for r in ck['history']]!=list(range(1,update+1)):raise ValueError('Logical update/cohort history differs')
    if update and len(ck['env_states'])!=c['n_games']:raise ValueError('Native cohort state count differs')
    for serialized in ck['env_states']:
        state=json.loads(serialized)
        # The frozen wrapper forbids experimental_mode; native RulesConfig has
        # no such field. Every actual native rule must still match exactly.
        rules={k:v for k,v in c['rules'].items() if k!='experimental_mode'}
        if c['rules'].get('experimental_mode') is not False or any(state['game']['cfg'].get(k)!=v for k,v in rules.items()) or len(state['game']['players'])!=8:raise ValueError('Saved native cohort rules differ')

def verify_source(pin,runtime,task):
    if pin['native_source_hash']!='217577b83fc4525e' or pin.get('admission',task)!=task:raise ValueError('Parent native/task identity differs')
    for name,digest in pin['files'].items():
        path=ROOT/name if name.startswith('vendor/') else Path(runtime)/name
        if not path.is_file() or sha(path)!=digest:raise ValueError('Parent archived source changed: '+name)

def parent_chain(node,task,index,depth=0):
    if depth>8:raise ValueError('Unexpectedly deep/cyclic checkpoint ancestry')
    digest=node['parent_resume_sha256']
    if node['not_an_exact_old_source_resume'] is not True:raise ValueError('Warm-start was relabeled as exact resume')
    if node['kind']=='explicit-new-campaign-warmstart':
        old=ROOT/'evidence/overnight-20261108-1533Z/tavern/researchers/deepseek';path=old/'runs/r3/artifact/resume.pt';bundle=old/'submissions/r3'
        if digest!=sha(path):raise ValueError('Original incumbent ancestor differs')
        ck=torch.load(path,map_location='cpu',weights_only=False);ident,_=admit_bundle(bundle)
        if ck['identity']!=ident or sha(bundle/'identity.json')!=node['parent_identity_sha256'] or weight_hash(ck['policy'])!=node['parent_policy_sha256'] or ck['update']!=node['parent_update'] or sum(r['learning']['samples'] for r in ck['history'])!=node['parent_samples']:raise ValueError('Original incumbent ancestry mismatch')
        return [dict(kind=node['kind'],checkpoint_sha256=digest,original_update=ck['update'])]
    if node['kind']!='explicit-strong-parent-warmstart' or digest not in index:raise ValueError('Strong parent needs an admitted full checkpoint/source/export index: '+digest)
    record=index[digest];path=Path(record['checkpoint']);bundle=Path(record['bundle'])
    if sha(path)!=digest:raise ValueError('Ancestor checkpoint mutated')
    ck=torch.load(path,map_location='cpu',weights_only=False);ident,summary=admit_bundle(bundle);verify_source(ck['binding'],record['runtime_source'],task);verify_full_resume(ck)
    samples=verify_training_summary(ck,summary,ident['reference_lineage']['schema'])
    prior=ck['lineage'].get('parent_total_known_samples',ck['lineage']['parent_samples'])
    if ck['binding']!=node['parent_source_binding'] or ck['lineage']!=node['parent_lineage'] or ck['update']!=node['parent_update'] or weight_hash(ck['policy'])!=node['parent_policy_sha256']:raise ValueError('Strong parent binding/lineage/state mismatch')
    if sha(bundle/'identity.json')!=node['parent_identity_sha256'] or ident['reference_lineage']['checkpoint_sha256']!=digest or ident['reference_lineage']['reference_binding']!=ck['binding'] or ident['config']!=ck['config'] or ident['reference_lineage']['update']!=ck['update'] or ident['reference_lineage']['parent']!=ck['lineage'] or node['parent_samples']!=samples or node['parent_total_known_samples']!=prior+samples:raise ValueError('Strong parent export identity mismatch')
    mapping=MAPS[ident['reference_lineage']['schema']]
    if any(sha(bundle/f)!=ck['binding']['files'][source] for f,source in mapping.items()):raise ValueError('Ancestor inference source substituted')
    with np.load(bundle/'weights.npz',allow_pickle=False) as z:
        if set(z.files)!=set(ck['policy']) or any(not np.array_equal(z[k],ck['policy'][k].numpy()) for k in z.files):raise ValueError('Ancestor NPZ differs')
    return [dict(kind=node['kind'],checkpoint_sha256=digest,logical_update=ck['update'],runtime_source=record['runtime_source'])]+parent_chain(ck['lineage'],task,index,depth+1)
def audit(bundle,runtime_source,out,checkpoint):
    bundle,runtime,checkpoint=map(lambda p:Path(p).resolve(),(bundle,runtime_source,checkpoint));ident,summary=admit_bundle(bundle);lineage=ident['reference_lineage'];pin=lineage['reference_binding'];mapping=MAPS[lineage['schema']]
    task=pin.get('admission',ident['binding'])
    if pin['native_source_hash']!='217577b83fc4525e' or task!=ident['binding'] or task['observations']!=[3227,124,245]:raise ValueError('Native/public interface changed')
    for name,digest in task['files'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Native/task source changed: '+name)
    for name,digest in pin['files'].items():
        path=ROOT/name if name.startswith('vendor/') else runtime/name
        if not path.is_file() or sha(path)!=digest:raise ValueError('Archived training source changed: '+name)
    for name,source in mapping.items():
        if sha(bundle/name)!=pin['files'][source]:raise ValueError('Export substituted inference/training source: '+name)
    if sha(checkpoint)!=lineage['checkpoint_sha256']:raise ValueError('Checkpoint digest differs')
    ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
    verify_full_resume(ck)
    verify_training_summary(ck,summary,lineage['schema'])
    if ck['binding']!=pin or ck['update']!=lineage['update'] or ck['config']!=ident['config']:raise ValueError('Checkpoint source/config/update mismatch')
    with np.load(bundle/'weights.npz',allow_pickle=False) as z:
        if set(z.files)!=set(ck['policy']) or any(not np.array_equal(z[k],ck['policy'][k].numpy()) for k in z.files):raise ValueError('Export changed selected checkpoint arrays')
    parent=lineage.get('parent');chain=[]
    if lineage['schema']=='tavern-strong-population-v1':
        if parent!=ck['lineage'] or parent['not_an_exact_old_source_resume'] is not True:raise ValueError('Imported parent provenance missing')
        index={}
        for file in Path(out).parent.glob('admission-*.json'):
            prior=json.loads(file.read_text())
            if prior.get('admission_qualified') and prior.get('checkpoint_sha256') and all(k in prior for k in ('bundle','runtime_source','checkpoint')):index[prior['checkpoint_sha256']]=prior
        chain=parent_chain(parent,task,index)
    result=dict(admission_qualified=True,source_map=mapping,source_compatible_task_binding=True,full_training_source_archive_verified=True,exact_checkpoint_arrays=True,full_optimizer_RNG_population_native_state_present=True,export_identity_sha256=sha(bundle/'identity.json'),weights_sha256=sha(bundle/'weights.npz'),checkpoint_sha256=sha(checkpoint),checkpoint=str(checkpoint),bundle=str(bundle),runtime_source=str(runtime),update=ck['update'],samples=summary['samples'],total_known_lineage_samples=summary['samples']+(parent.get('parent_total_known_samples',parent['parent_samples']) if parent else 0),parameters=summary['parameters'],finite_losses=summary['finite_losses'],changed_weights=summary['changed_weights'],explicit_imported_parent=parent,recursive_parent_chain=chain,no_old_identity_relabeling=True,strong_panel_qualified=False,superhuman_evidence=False)
    write(out,result);return result

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('bundle','runtime-source','out','checkpoint'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();print(json.dumps(audit(a.bundle,a.runtime_source,a.out,a.checkpoint),indent=2))
