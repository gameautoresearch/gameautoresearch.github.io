"""Fetch only a closed, lineage-bound milestone from an owned durable Volume."""
from pathlib import Path
import argparse,hashlib,json,os
os.environ.setdefault('MODAL_DISABLE_API_PROXY','1')
import numpy as np
from .launch import CAMPAIGN
from simulators.clash.autoresearch.common import dump,sha

def fetch(attempt,case,stage,alias):
 import modal
 from simulators.generals.autoresearch.provider import configure_sandbox_waits
 configure_sandbox_waits();root=CAMPAIGN/'clash/rl'/attempt;p=json.loads((root/'provider.json').read_text());volume=modal.Volume.from_name(p['volume_name'],environment_name='staging');prefix='/'+case+'/'+stage
 try:lineage=json.loads(b''.join(volume.read_file(prefix+'/lineage.json')))
 except FileNotFoundError:return {'ready':False}
 target=root/'early-exports'/alias;target.mkdir(parents=True,exist_ok=True)
 for name in ('policy.py','learner.py','train.py','config.json','weights.npz'):
  data=b''.join(volume.read_file(prefix+'/'+name));tmp=target/(name+'.part');tmp.write_bytes(data);tmp.replace(target/name)
 dump(target/'lineage.json',lineage)
 if sha(target/'policy.py')!=lineage['binding']['source']['policy.py'] or sha(target/'learner.py')!=lineage['binding']['source']['learner.py']:raise ValueError('milestone source differs from run binding')
 if json.loads((target/'config.json').read_text())!=lineage['binding']['config']:raise ValueError('milestone config differs')
 h=hashlib.sha256()
 with np.load(target/'weights.npz',allow_pickle=False) as data:
  for k in sorted(data.files):h.update(k.encode());h.update(data[k].tobytes())
 if h.hexdigest()!=lineage['policy_identity']:raise ValueError('milestone weight identity differs')
 receipt={'ready':True,'attempt':attempt,'case':case,'stage':stage,'volume_name':p['volume_name'],'sandbox_id':p['sandbox_id'],'source_archive_sha256':p['source_archive_sha256'],'policy_identity':h.hexdigest(),'updates':lineage['updates'],'artifact':str(target)};dump(target/'fetch-receipt.json',receipt);return receipt

def main():
 p=argparse.ArgumentParser();p.add_argument('--attempt',required=True);p.add_argument('--case',required=True);p.add_argument('--stage',required=True);p.add_argument('--alias',required=True);a=p.parse_args();print(json.dumps(fetch(a.attempt,a.case,a.stage,a.alias),indent=2))
if __name__=='__main__':main()
