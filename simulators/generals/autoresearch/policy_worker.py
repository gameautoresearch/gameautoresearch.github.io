"""Unprivileged inference. Exact submitted code receives only visible x and legal mask."""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

def limit():
 """No new privileges, no core dumps, few files and processes: applied in the child before it loads a policy."""
 import ctypes,resource
 if ctypes.CDLL(None).prctl(38,1,0,0,0)!=0:raise RuntimeError('Could not enforce no-new-privileges')
 resource.setrlimit(resource.RLIMIT_CORE,(0,0))
 resource.setrlimit(resource.RLIMIT_NOFILE,(64,64))
 resource.setrlimit(resource.RLIMIT_NPROC,(64,64))

def main():
 protocol=sys.stdout.buffer
 sys.stdout=sys.stderr
 from .wire import read_frame,decode,reply
 from .bundle import load_weights
 import jax
 import numpy as np
 p=argparse.ArgumentParser();p.add_argument('--bundle',required=True);p.add_argument('--tensor-cap',type=int);a=p.parse_args();root=Path(a.bundle)
 config=json.loads((root/'model.json').read_text());sys.path.insert(0,str(root))
 spec=importlib.util.spec_from_file_location('entrant_policy',root/'policy.py');module=importlib.util.module_from_spec(spec)
 sys.modules[spec.name]=module;spec.loader.exec_module(module)
 model=module.create(config);params=load_weights(root/'weights.npz',a.tensor_cap)
 # A submitted policy must fit the compute-per-move cap; a sealed panel opponent (given its own tensor cap) is the
 # evaluator's choice and is not measured.
 probe=None
 if a.tensor_cap is None:
  from .flops import check
  check(model,params)
  # ... and must be learned: its moves have to come from its weights (learned.py), checked on the boards it plays.
  from .learned import Probe
  probe=Probe(model,params).warm()
 @jax.jit
 def act(x,mask):
  def one(a,m):return model.apply({'params':params},a[None],m[None])[0][0]
  logits=jax.vmap(one)(x,mask)
  return logits,jax.numpy.argmax(jax.numpy.where(mask,logits,-jax.numpy.inf),axis=-1)
 # Ready: loading the policy and the host's own checks above are not on the policy's clock (arena waits for this
 # empty frame, with its own generous limit, before the first request and its deadline).
 protocol.write(reply([]));protocol.flush()
 while (raw:=read_frame(sys.stdin.buffer)) is not None:
  x,mask=decode(raw);logits,ids=act(x,mask);logits=np.asarray(logits);ids=np.asarray(ids)
  if logits.shape!=mask.shape or not np.isfinite(logits[mask]).all():raise ValueError('Invalid policy logits')
  if ids.shape!=(len(x),) or not mask[np.arange(len(x)),ids].all():raise ValueError('Illegal policy response')
  protocol.write(reply(ids));protocol.flush()
  if probe is not None:probe.see(x,mask,ids)  # after the answer is out: the probe's time is the host's

if __name__=='__main__':main()
