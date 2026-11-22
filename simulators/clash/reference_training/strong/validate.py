"""Independent downloaded checkpoint/source/world/export validation."""
from pathlib import Path
import argparse,json
import numpy as np,torch
from .provider import OWN
from simulators.clash.reference_training.league import tensor_identity
from simulators.clash.autoresearch.common import dump,sha,engine_binding,setup_native,load_module

def validate(name,export=None):
 root=OWN/name;p=json.loads((root/'provider.json').read_text());out=root/'collected/output';out=out/'training' if (out/'training/result.json').exists() else out
 result=json.loads((out/'result.json').read_text());state=torch.load(out/'checkpoint.pt',map_location='cpu',weights_only=True);pin=state['binding']
 if sha(root/'strong-source.tar.gz')!=p['source_archive_sha256'] or pin!=result['binding'] or pin['engine']!=engine_binding():raise ValueError('Source archive/result/checkpoint/engine identity mismatch')
 source=root/'source/simulators/clash/reference_training'/p['variant']
 if any(sha(source/n)!=v for n,v in pin['source'].items()):raise ValueError('Checkpoint source differs from immutable snapshot')
 export=Path(export) if export else out/'final';export.relative_to(root)
 lineage=json.loads((export/'lineage.json').read_text())
 if lineage['binding']!=pin or lineage['updates']!=state['updates'] or json.loads((export/'config.json').read_text())!=pin['config'] or any(sha(export/n)!=pin['source'][n] for n in ('policy.py','learner.py')):raise ValueError('Final inference source/config/counters differ from complete checkpoint')
 with np.load(export/'weights.npz',allow_pickle=False) as weights:
  if set(weights.files)!=set(state['model']) or any(not torch.equal(torch.from_numpy(weights[k]),state['model'][k]) for k in weights.files):raise ValueError('Final tensor bytes differ from complete checkpoint')
 if [tensor_identity(x) for x in state['league']]!=state['identities']:raise ValueError('Frozen neural history differs')
 setup_native();from snapshots import decode_engine_state
 for key in ('engine','demo_engine'):
  if state.get(key) is not None:
   decoded=decode_engine_state(state[key],as_tensor=False,expected_count=pin['config']['envs'],expected_world_bytes=4195384,max_total_bytes=16*1024**3,workers=1);del decoded
 if not result['proof'].get('exact_next_update_resume') or not result['proof'].get('deterministic_legal_inference'):raise ValueError('Actual GPU proof incomplete')
 if not (result.get('finite_actual_learning') or result.get('finite_losses')) or result['initial_policy_identity']==result['final_policy_identity']:raise ValueError('Finite actual learning/changed weights not established')
 module=load_module(source/'policy.py','validated_reference_policy');model=module.create(pin['config']);parameters=sum(x.numel() for x in model.parameters())
 value={'name':name,'variant':p['variant'],'checkpoint':str(out/'checkpoint.pt'),'checkpoint_sha256':sha(out/'checkpoint.pt'),'inference_export':str(export),'source_archive_sha256':p['source_archive_sha256'],'source_bound':True,'full_native_world_payload_integrity':True,'frozen_history_identity':True,'final_export_tensor_bytes_match':True,'actual_gpu_exact_resume':True,'greedy_deterministic_legal':True,'policy_identity':tensor_identity(state['model']),'updates':state['updates'],'parameters':parameters,'persistent_tensor_elements':sum(x.numel() for x in state['model'].values()),'parent_checkpoint_sha256':p['parent_checkpoint_sha256']};dump(root/'artifact-validation.json',value);return value

def validate_milestone(name,stage):
 import re
 if not re.fullmatch(r'milestone-\d{6}',stage):raise ValueError('Closed milestone name required')
 root=OWN/name;export=root/'exports'/stage;p=json.loads((root/'provider.json').read_text());receipt=json.loads((export/'fetch-receipt.json').read_text());lineage=json.loads((export/'lineage.json').read_text())
 checkpoint=Path(receipt['full_checkpoint'])
 if checkpoint.relative_to(root/'collected').as_posix()!=receipt['remote_checkpoint'].lstrip('/') or sha(checkpoint)!=receipt['checkpoint_sha256']:raise ValueError('Complete milestone checkpoint receipt differs')
 state=torch.load(checkpoint,map_location='cpu',weights_only=True);pin=state['binding'];source=root/'source/simulators/clash/reference_training'/p['variant']
 if pin!=lineage['binding'] or pin['engine']!=engine_binding() or state['updates']!=lineage['updates'] or sha(root/'strong-source.tar.gz')!=p['source_archive_sha256']:raise ValueError('Milestone binding/counters/source differs')
 if any(sha(source/n)!=v for n,v in pin['source'].items()):raise ValueError('Milestone trainer differs from frozen source')
 with np.load(export/'weights.npz',allow_pickle=False) as w:
  if set(w.files)!=set(state['model']) or any(not torch.equal(torch.from_numpy(w[k]),state['model'][k]) for k in w.files):raise ValueError('Milestone inference and full checkpoint tensors differ')
 if tensor_identity(state['model'])!=lineage['policy_identity'] or [tensor_identity(x) for x in state['league']]!=state['identities'] or state['identities']!=lineage['frozen_league_identities']:raise ValueError('Milestone model/history identity differs')
 proof_path=root/'collected/output/proof.json'
 proof=json.loads(proof_path.read_text()) if proof_path.exists() else json.loads((root/'live-status.json').read_text())['proof']
 if not proof.get('exact_next_update_resume') or not proof.get('parent_source_config_build_runtime_exact') or proof.get('parent_checkpoint_sha256')!=p['parent_checkpoint_sha256']:raise ValueError('Matching actual GPU stage resume proof required')
 setup_native();from snapshots import decode_engine_state
 for key in ('engine','demo_engine'):
  if state.get(key) is not None:
   decoded=decode_engine_state(state[key],as_tensor=False,expected_count=pin['config']['envs'],expected_world_bytes=4195384,max_total_bytes=16*1024**3,workers=1);del decoded
 torch.set_num_threads(1);module=load_module(source/'policy.py','validated_milestone_policy');model=module.create(pin['config']);model.load_state_dict(state['model']);model.eval();o=state['outputs']['observations'].reshape(-1,7024);mask=state['outputs']['masks'].reshape(-1,2307)
 with torch.no_grad():a=module.act(model,o,mask);b=module.act(model,o,mask)
 if not torch.equal(a,b) or not mask[torch.arange(len(a)),a].all():raise ValueError('Milestone deterministic legal inference failed')
 value={'name':name,'stage':stage,'checkpoint':str(checkpoint),'checkpoint_sha256':sha(checkpoint),'source_archive_sha256':p['source_archive_sha256'],'source_bound':True,'full_native_world_payload_integrity':True,'frozen_history_identity':True,'inference_tensor_bytes_match':True,'actual_gpu_stage_resume_proof':True,'greedy_deterministic_legal':True,'policy_identity':lineage['policy_identity'],'updates':state['updates'],'parameters':sum(x.numel() for x in model.parameters()),'persistent_tensor_elements':sum(x.numel() for x in state['model'].values()),'no_later_optimizer_or_world_state_substitution':True};dump(export/'full-artifact-validation.json',value);return value

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--stage');p.add_argument('--export');a=p.parse_args();print(json.dumps(validate_milestone(a.name,a.stage) if a.stage else validate(a.name,a.export),indent=2))
