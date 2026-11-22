"""Evaluate strict exported reference admission and fail-closed tampering under v2."""
import argparse,json,shutil
from pathlib import Path
import numpy as np
import torch
from .reference_inference import load
from .env import EvaluationVector
from ...network import batch,scripted
from ..contract import write


def main():
 p=argparse.ArgumentParser();p.add_argument('--request',required=True);p.add_argument('--output',required=True);a=p.parse_args()
 spec=json.loads(Path(a.request).read_text());out=Path(a.output);torch.set_num_threads(1)
 model,meta=load(spec['bundle'],spec['manifest_sha256']);rows=[]
 for index,seed in enumerate([900000171,900000172]):
  for seat in (0,1):
   with EvaluationVector(1) as v:
    states=v.reset([seed],4096);rng=np.random.default_rng(seed)
    while not(states[0]['terminated'] or states[0]['truncated']):
     s=states[0];actions=[None,None]
     for side,ready in enumerate(s['ready']):
      if not ready:continue
      if side==seat:
       with torch.no_grad():logits,_=model(batch([s['observations'][side]],'cpu'));actions[side]=int(logits.argmax(-1).item())
      else:actions[side]=scripted(s['observations'][side],rng,'power')
     states=v.step([actions])
    s=states[0];rows.append({'seed_index':index,'seat':seat,'natural':s['terminated'],'truncated':s['truncated'],
      'points':(s['returns'][seat]+1)/2 if s['terminated'] else None,'terminal_reason':s['terminal_reason'],
      'engine_turns':s['engine_turns'],'adapter_phases':s['adapter_phases'],'visible_rejections':sum(s['rejections'])})
 forged=out/'tamper-fixture';shutil.copytree(spec['bundle'],forged)
 for p in forged.iterdir():p.chmod(0o600)
 weights=forged/'weights.pt';raw=bytearray(weights.read_bytes());raw[-1]^=1;weights.write_bytes(raw)
 try:load(forged,spec['manifest_sha256'])
 except ValueError:tamper_rejected=True
 else:tamper_rejected=False
 assert tamper_rejected
 try:load(spec['bundle'],'wrong-attestation')
 except ValueError:attestation_rejected=True
 else:attestation_rejected=False
 assert attestation_rejected
 write(out/'summary.json',{'passed':all(r['natural'] for r in rows),'rows':rows,'weights_tampering_rejected':tamper_rejected,
  'wrong_export_attestation_rejected':attestation_rejected,'inference_only':True,'resume_not_claimed':True,
  'manifest_sha256':meta['manifest_sha256'],'model_sha256':meta['model_sha256'],'original_source_and_engine_bindings_preserved':True,
  'strength_qualified':False,'note':'Proof1729 source-admission fixture, not selected matched2718 final quality assessment.'})
if __name__=='__main__':main()
