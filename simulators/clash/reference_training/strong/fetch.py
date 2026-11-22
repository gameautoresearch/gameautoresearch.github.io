"""Fetch a closed exported milestone, with exact source/config/tensor verification."""
from pathlib import Path
import argparse,json,hashlib,os
os.environ.setdefault('MODAL_DISABLE_API_PROXY','1')
from .provider import OWN
from simulators.clash.autoresearch.common import dump,sha,engine_binding

def fetch(name,stage,checkpoint=False):
 import modal,numpy as np
 from simulators.generals.autoresearch.provider import configure_sandbox_waits
 configure_sandbox_waits();root=OWN/name;p=json.loads((root/'provider.json').read_text());v=modal.Volume.from_name(p['volume_name'],environment_name='staging');lineage=None
 for prefix in ('/output/training/','/output/'):
  try:lineage=json.loads(b''.join(v.read_file(prefix+stage+'/lineage.json')));break
  except FileNotFoundError:pass
  except (ValueError,UnicodeError):
   from simulators.generals.autoresearch.provider import allocation_info
   receipt=allocation_info(p['app_id'],p['sandbox_id'])
   if not receipt or not receipt.get('finished_at'):return {'ready':False,'reason':'active Volume export metadata not yet durably readable'}
   raise
 if lineage is None:return {'ready':False}
 if lineage['binding']['engine']!=engine_binding() or sha(root/'strong-source.tar.gz')!=p['source_archive_sha256']:raise ValueError('Milestone engine or archived source identity differs')
 source=root/'source/simulators/clash/reference_training'/p['variant']
 if any(sha(source/n)!=v for n,v in lineage['binding']['source'].items()):raise ValueError('Milestone training source differs from immutable run snapshot')
 target=root/'exports'/stage;target.mkdir(parents=True,exist_ok=True)
 for n in ('policy.py','learner.py','train.py','config.json','weights.npz'):
  part=target/(n+'.fetch-part-'+str(os.getpid()));part.write_bytes(b''.join(v.read_file(prefix+stage+'/'+n)));part.replace(target/n)
 dump(target/'lineage.json',lineage)
 for n in ('policy.py','learner.py'):
  if sha(target/n)!=lineage['binding']['source'][n]:raise ValueError('Milestone source changed')
 if json.loads((target/'config.json').read_text())!=lineage['binding']['config']:raise ValueError('Milestone config changed')
 h=hashlib.sha256()
 with np.load(target/'weights.npz',allow_pickle=False) as weights:
  for k in sorted(weights.files):h.update(k.encode());h.update(weights[k].tobytes())
 if h.hexdigest()!=lineage['policy_identity']:raise ValueError('Milestone tensor identity differs')
 value={'ready':True,'name':name,'stage':stage,'artifact':str(target),'updates':lineage['updates'],'policy_identity':h.hexdigest(),'source_archive_sha256':p['source_archive_sha256']}
 if checkpoint:
  receipt=json.loads(b''.join(v.read_file(prefix+stage+'/checkpoint-receipt.json')))
  if receipt['updates']!=lineage['updates']:raise ValueError('Resume checkpoint and inference milestone updates differ')
  remote_path=prefix+stage+'/checkpoint.pt';out=root/'collected'/remote_path.lstrip('/');out.parent.mkdir(parents=True,exist_ok=True)
  if not out.exists() or sha(out)!=receipt['checkpoint_sha256']:
   part=out.with_name(out.name+'.fetch-part-'+str(os.getpid()))
   with part.open('wb') as stream:
    for block in v.read_file(remote_path):stream.write(block)
   if sha(part)!=receipt['checkpoint_sha256']:raise ValueError('Complete milestone checkpoint transfer failed its declared checksum')
   part.replace(out)
  value.update(full_checkpoint=str(out),remote_checkpoint=remote_path,checkpoint_sha256=receipt['checkpoint_sha256'])
 dump(target/'fetch-receipt.json',value);return value

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--stage',default='final');p.add_argument('--checkpoint',action='store_true');a=p.parse_args();print(json.dumps(fetch(a.name,a.stage,a.checkpoint),indent=2))
