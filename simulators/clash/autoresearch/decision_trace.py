"""Passive public-decision tape and exact frozen-engine replay qualification."""
import argparse,fcntl,gzip,hashlib,json
from pathlib import Path
import numpy as np
from .common import ROOT,dump,digest,engine_binding,setup_native,sha

def run(specs,out):
 from .evaluation_v2 import screen,PUBLIC_SEEDS
 setup_native()
 import cpu_batch
 original=cpu_batch.NativeCPUBatch;out=Path(out);out.mkdir(parents=True,exist_ok=False);receipts=[]
 def public_hash(env):
  return hashlib.sha256(b''.join(getattr(env,n).tobytes() for n in ('observations','masks','infos','rewards'))).hexdigest()
 class Traced:
  def __init__(self,seeds,threads=4,form='full'):
   self.form=form;self.inner=original(seeds,threads=threads,form=form);self.seeds=list(seeds);self.initial=public_hash(self.inner);self.frames=[];self.index=len(receipts)
  def __getattr__(self,n):return getattr(self.inner,n)
  def step(self,actions,*args):
   assert all(self.inner.masks[i,s,a] for i,row in enumerate(actions) for s,a in enumerate(row))
   value=self.inner.step(actions,*args);self.frames.append({'actions':actions.tolist(),'public_sha256':public_hash(self.inner)});return value
  def close(self):
   self.inner.close();path=out/'traces'/('batch-'+str(self.index)+'.json.gz');path.parent.mkdir(exist_ok=True)
   payload={'public_seeds':self.seeds,'initial_public_sha256':self.initial,'frames':self.frames};path.write_bytes(gzip.compress(json.dumps(payload,separators=(',',':')).encode(),mtime=0));path.chmod(0o444)
   replay=original(self.seeds,threads=4,form=self.form)
   try:
    assert public_hash(replay)==self.initial
    for frame in self.frames:
     actions=np.array(frame['actions'],np.int32);assert all(replay.masks[i,s,a] for i,row in enumerate(actions) for s,a in enumerate(row));replay.step(actions);assert public_hash(replay)==frame['public_sha256']
   finally:replay.close()
   receipts.append({'trace':str(path),'trace_sha256':sha(path),'frames':len(self.frames),'worlds':len(self.seeds),'every_action_legal':True,'all_public_output_bytes_replayed_exactly':True})
 cpu_batch.NativeCPUBatch=Traced
 try:result=screen(specs,out/'evaluation',seeds=PUBLIC_SEEDS[:2],development=True,max_seconds=240)
 finally:cpu_batch.NativeCPUBatch=original
 value={'complete':result['complete'],'natural_games':result['natural_games'],'batches':receipts,'engine_binding_sha256':digest(engine_binding()),'purpose':'Public development replay conformance, not an independent strength replicate','private_state_recorded':False};dump(out/'result.json',value);return value

def main():
 p=argparse.ArgumentParser();p.add_argument('--specs',required=True);p.add_argument('--out',required=True);a=p.parse_args()
 with (ROOT/'evidence/local-performance.lock').open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);v=run(json.loads(Path(a.specs).read_text()),a.out)
 print(json.dumps(v))
if __name__=='__main__':main()
