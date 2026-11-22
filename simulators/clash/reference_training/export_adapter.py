"""Compile factorized greedy decoding into a stateless flat-argmax neural head.

Weights and training checkpoint are unchanged. An explicit derived inference
source and provenance receipt preserve the exact trained-source binding.
"""
from pathlib import Path
import ast,json,shutil
from simulators.clash.autoresearch.common import dump,sha,load_module

HEAD='''
class GreedyHead(Policy):
 def forward(self,observation,mask):
  logits,value=super().forward(observation,mask)
  tiles=logits[:,1:2305].reshape(-1,4,576)
  marginal=tiles.logsumexp(-1)
  # Within each legal card, the best tile has its card marginal as score.
  # Flat argmax therefore equals card-marginal argmax then tile argmax.
  scores=tiles-tiles.amax(-1,keepdim=True)+marginal[:,:,None]
  result=torch.cat((logits[:,:1],scores.flatten(1),logits[:,-2:]),1)
  return result.masked_fill(~mask.bool(),-1e9),value
def create(config):return GreedyHead(config)
def act(model,observation,mask):return model(observation,mask)[0].argmax(-1)
'''

def derive(original,destination):
 import torch,numpy as np
 torch.set_num_threads(1)
 from simulators.clash.autoresearch.common import setup_native
 setup_native()
 from cpu_batch import NativeCPUBatch
 from .league import visible_teacher
 original=Path(original);destination=Path(destination);destination.mkdir(parents=True,exist_ok=False)
 for name in ('weights.npz','config.json','learner.py','train.py'):shutil.copyfile(original/name,destination/name)
 tree=ast.parse((original/'policy.py').read_text());tree.body=[n for n in tree.body if not (isinstance(n,ast.FunctionDef) and n.name in ('create','act'))]
 (destination/'policy.py').write_text(ast.unparse(tree)+'\n'+HEAD)
 c=json.loads((original/'config.json').read_text());source=load_module(original/'policy.py','source_reference');exported=load_module(destination/'policy.py','export_reference')
 a=source.create(c);b=exported.create(c)
 with np.load(original/'weights.npz',allow_pickle=False) as weights:state={k:torch.from_numpy(weights[k].copy()) for k in weights.files}
 a.load_state_dict(state);b.load_state_dict(state);a.eval();b.eval();env=NativeCPUBatch([1901,1902,1903,1904],threads=1);rows=0
 try:
  with torch.inference_mode():
   for tick in range(64):
    o=torch.from_numpy(env.observations.copy()).reshape(-1,7024);m=torch.from_numpy(env.masks.copy()).reshape(-1,2307)
    x=source.act(a,o,m);y=exported.act(b,o,m)
    if not torch.equal(x,y) or not m[torch.arange(len(y)),y].all():raise ValueError('derived neural head changed original hierarchical action')
    rows+=len(x);actions=visible_teacher(o,m,True).reshape(-1,2).numpy();env.step(actions)
 finally:env.close()
 receipt={'format':'clash-reference-greedy-head-v1','training_policy_sha256':sha(original/'policy.py'),'export_policy_sha256':sha(destination/'policy.py'),'weights_sha256':sha(original/'weights.npz'),'weights_unchanged':sha(original/'weights.npz')==sha(destination/'weights.npz'),'decoder':'exact card-marginal then tile-max represented as a stateless NN head and direct flat argmax','public_fixture_rows_exact':rows,'inference_distribution_note':'Exported scores preserve greedy actions; their softmax is NOT the PPO sampling distribution.','training_lineage':json.loads((original/'lineage.json').read_text())}
 dump(destination/'lineage.json',receipt);return receipt
