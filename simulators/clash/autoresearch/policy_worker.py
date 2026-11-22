"""Unprivileged, credential-free public-only inference process."""
from pathlib import Path
import argparse,json,sys,struct
from .common import load_module
from .wire import read_frame,decode,reply

def main():
 import torch,numpy as np
 p=argparse.ArgumentParser();p.add_argument('--bundle',required=True);p.add_argument('--diagnostics',action='store_true');a=p.parse_args();root=Path(a.bundle)
 torch.set_num_threads(1);torch.manual_seed(0)
 module=load_module(root/'policy.py','submitted_policy');model=module.create(json.loads((root/'config.json').read_text()))
 if sum(v.numel() for v in model.state_dict().values())>1000000:raise ValueError('policy tensor cap')
 with np.load(root/'weights.npz',allow_pickle=False) as data:state={k:torch.from_numpy(data[k].copy()) for k in data.files}
 model.load_state_dict(state);model.eval()
 while (raw:=read_frame(sys.stdin.buffer)) is not None:
  x,mask=decode(raw)
  with torch.inference_mode():indices=module.act(model,torch.from_numpy(x),torch.from_numpy(mask))
  values=indices.tolist()
  if len(values)!=len(x) or any(type(v)!=int or v<0 or v>=2307 or not mask[i,v] for i,v in enumerate(values)):raise ValueError('illegal policy action')
  if a.diagnostics:
   with torch.inference_mode():
    logits=model(torch.from_numpy(x),torch.from_numpy(mask))[0];probability=logits.softmax(-1)
   payload=json.dumps({'actions':values,'wait_probability':probability[:,0].tolist(),'aggregate_nonwait_probability':(1-probability[:,0]).tolist(),'flat_map_wait':(logits.argmax(-1)==0).tolist()},allow_nan=False).encode()
   sys.stdout.buffer.write(struct.pack('>I',len(payload))+payload)
  else:sys.stdout.buffer.write(reply(values))
  sys.stdout.buffer.flush()
if __name__=='__main__':main()
