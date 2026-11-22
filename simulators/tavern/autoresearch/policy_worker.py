"""Unprivileged candidate forward. Only visible arrays and masks cross this pipe."""
import sys,json,base64,io,importlib.util,resource
from pathlib import Path
import numpy as np
import torch
resource.setrlimit(resource.RLIMIT_NOFILE,(64,64));torch.set_num_threads(1)
protocol=sys.stdout;sys.stdout=sys.stderr
bundle=Path(sys.argv[1]);c=json.loads((bundle/'identity.json').read_text())['config']
spec=importlib.util.spec_from_file_location('candidate_policy',bundle/'policy.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);policy=m.Policy(**c['network'])
with np.load(bundle/'weights.npz',allow_pickle=False) as z:
 state={k:torch.from_numpy(z[k].copy()) for k in z.files}
 if not all(torch.isfinite(v).all() for v in state.values()):raise ValueError('Nonfinite checkpoint')
 policy.load_state_dict(state,strict=True)
policy.eval()
for line in sys.stdin:
 if len(line)>32000000:raise ValueError('Oversized input')
 with np.load(io.BytesIO(base64.b64decode(line)),allow_pickle=False) as z:
  f,cards,mask=[torch.from_numpy(z[k].copy()) for k in ('features','cards','mask')]
 with torch.no_grad():logits,_=policy(f,cards)
 if logits.shape!=mask.shape or not torch.isfinite(logits[mask]).all():raise ValueError('Invalid logits')
 actions=logits.masked_fill(~mask,-torch.inf).argmax(-1).tolist()
 protocol.write(json.dumps(actions)+'\n');protocol.flush()
