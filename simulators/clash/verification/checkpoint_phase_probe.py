"""Paired durable checkpoint timings on one paused actual 1024-world trainer.

No training or simulator advance occurs. Both strategies serialize identical
complete state with identical entrant metadata. The baseline is the retained
native-save/load/save sequence, with end-of-phase fsync so both timings cover
file durability. Every output is new; existing checkpoints are read-only.
"""
from pathlib import Path
import argparse,json,os,sys,time

def main():
 p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--out',required=True);a=p.parse_args()
 from simulators.clash.autoresearch.common import setup_native,DATA,EDITABLE,sha,dump,load_module
 from simulators.clash.autoresearch.fast_entry import runtime_binding
 setup_native();os.environ.setdefault('TRITON_CACHE_DIR',f'/tmp/clash-triton-{os.getuid()}')
 import torch
 from simulators.clash.native.train import Trainer as NativeTrainer
 from simulators.clash.native.test_training import equal
 source=Path(a.source);out=Path(a.out);out.mkdir(parents=True,exist_ok=False);c=json.loads((source/'config.json').read_text())
 origin=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
 expected={'source':{n:sha(source/n) for n in (*EDITABLE,'config.json')},'trusted_runtime':runtime_binding()}
 if origin.get('entrant_binding')!=expected:raise ValueError('entrant resume binding differs')
 history=origin['entrant_state'];del origin
 sys.path.insert(0,str(source))
 for n in ('policy','learner'):sys.modules.pop(n,None)
 trainer=load_module(source/'train.py','probe_trainer').Trainer(c,DATA,a.checkpoint)
 rows=[];reference=None
 try:
  for i,kind in enumerate(('old-double','new-single','new-single','old-double')):
   path=out/(str(i)+'-'+kind+'.pt');torch.cuda.synchronize();started=time.perf_counter()
   if kind=='old-double':
    NativeTrainer.checkpoint(trainer,path,0.)
    payload=torch.load(path,map_location='cpu',weights_only=False);payload.update(entrant_binding=expected,entrant_state=history)
    torch.save(payload,path);del payload
    descriptor=os.open(path,os.O_RDONLY)
    try:os.fsync(descriptor)
    finally:os.close(descriptor)
    descriptor=os.open(path.parent,os.O_RDONLY)
    try:os.fsync(descriptor)
    finally:os.close(descriptor)
    passes=2
   else:
    diagnostic=trainer.checkpoint(path,0.,entrant_binding=expected,entrant_state=history);passes=diagnostic['serialization_passes']
   elapsed=time.perf_counter()-started
   # Read-back validation is outside the timed phase for both strategies.
   payload=torch.load(path,map_location='cpu',weights_only=False)
   if reference is None:reference=payload
   else:
    for key in reference:equal(reference[key],payload[key],key)
    assert reference.keys()==payload.keys()
   rows.append({'strategy':kind,'seconds':elapsed,'serialization_passes':passes,'same_complete_payload':True,'file_bytes':path.stat().st_size});dump(out/'progress.json',rows)
  grouped={kind:sum(r['seconds'] for r in rows if r['strategy']==kind)/2 for kind in ('old-double','new-single')}
  dump(out/'result.json',{'passed':True,'new_learner_decisions':0,'envs':c['envs'],'same_paused_trainer':True,'same_complete_payload':True,'both_strategies_include_file_fsync':True,'order':['old','new','new','old'],'rows':rows,'mean_seconds':grouped,'phase_reduction_fraction':1-grouped['new-single']/grouped['old-double']})
 finally:trainer.close()
if __name__=='__main__':main()
