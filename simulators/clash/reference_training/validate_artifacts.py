"""Validate the declared source/checkpoint/export chain without relaxing admission."""
from pathlib import Path
import argparse,json,math
import numpy as np
import torch
from .league import tensor_identity
from simulators.clash.autoresearch.common import dump,sha,engine_binding

def validate(attempt):
 root=Path(attempt);collected=root/'collected';top=json.loads((collected/'result.json').read_text());provider=json.loads((root/'provider.json').read_text())
 if sha(root/'source.tar.gz')!=provider['source_archive_sha256']:raise ValueError('run source archive changed')
 variant=provider.get('variant',top.get('variant','v2'));source=root/'source/simulators/clash/reference_training'/variant;results={}
 for name,case in top['cases'].items():
  result=case['result'];folder=collected/name;saved=torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=True)
  if saved['binding']!=result['binding'] or saved['binding']['engine']!=engine_binding():raise ValueError('checkpoint/result/engine binding differs')
  if any(sha(source/n)!=v for n,v in saved['binding']['source'].items()):raise ValueError('checkpoint source differs from immutable run snapshot')
  if [tensor_identity(s) for s in saved['league']]!=saved['identities']:raise ValueError('frozen league identity differs')
  with np.load(folder/'final/weights.npz',allow_pickle=False) as weights:
   if set(weights.files)!=set(saved['model']) or any(not torch.equal(torch.from_numpy(weights[k]),saved['model'][k]) for k in weights.files):raise ValueError('inference export tensors differ from checkpoint')
  c=saved['binding']['config']
  from simulators.clash.autoresearch.common import setup_native
  setup_native();from snapshots import decode_engine_state
  for key in ('engine','demo_engine'):
   if saved.get(key) is not None:
    decoded=decode_engine_state(saved[key],as_tensor=False,expected_count=c['envs'],expected_world_bytes=4195384,max_total_bytes=16*1024**3,workers=1);del decoded
  proof=result['proof'];extra=c['envs']*c['rollout_steps'] if proof.get('exact_next_update_resume') else 0
  results[name]={'source_bound_checkpoint':True,'full_world_payload_integrity_verified':True,'final_export_tensor_bytes_match':True,'frozen_history_hashes_match':True,'checkpoint_sha256':sha(folder/'checkpoint.pt'),'logical_trajectory_updates':result['updates'],'logical_learner_rows':result['samples'],'logical_optimizer_steps':result['optimizer_steps'],'resume_audit_extra_discarded_updates':int(bool(extra)),'resume_audit_extra_learner_rows':extra,'resume_audit_extra_optimizer_steps':c['epochs']*math.ceil(extra/c['minibatch']),'actual_executed_learner_rows_including_resume_audit':result['samples']+extra,'initial_policy_identity':result['initial_policy_identity'],'final_policy_identity':result['final_policy_identity'],'proof':proof,'case_process_seconds':case['case_process_seconds']}
 if len({r['initial_policy_identity'] for r in results.values()})!=1:raise ValueError('paired initial weights differ')
 receipt={'variant':variant,'seed':top['seed'],'source_archive_sha256':sha(root/'source.tar.gz'),'paired_initial_weights_identical':True,'cases':results,'full_allocation_charged_including_discarded_resume_audit':True};dump(root/'artifact-validation.json',receipt);return receipt

def main():
 p=argparse.ArgumentParser();p.add_argument('--attempt',required=True);a=p.parse_args();print(json.dumps(validate(a.attempt),indent=2))
if __name__=='__main__':main()
