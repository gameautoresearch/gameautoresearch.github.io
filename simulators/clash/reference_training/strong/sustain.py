"""Chain only the frozen v11 trajectory, with final handoff time reserved."""
import argparse,json,math,time
from pathlib import Path
from .provider import OWN,launch
from simulators.clash.autoresearch.common import GAME,sha,dump,engine_binding
from tools.overnight_budget import stamp

def continuation(parent_name,next_name,execute=False):
 contract=json.loads((OWN/'sustained-continuation-contract.json').read_text())
 root=OWN/parent_name;validation=root/'artifact-validation.json'
 if not validation.exists():return {'ready':False,'reason':'complete parent artifact validation pending','parent':parent_name}
 v=json.loads(validation.read_text());p=json.loads((root/'provider.json').read_text())
 checks=('source_bound','full_native_world_payload_integrity','frozen_history_identity','final_export_tensor_bytes_match','actual_gpu_exact_resume','greedy_deterministic_legal')
 if p['variant']!='v11' or v['name']!=parent_name or not all(v[k] for k in checks):raise ValueError('Qualified fixedv11 parent required')
 checkpoint=Path(v['checkpoint']);relative=str(checkpoint.relative_to(root/'collected'))
 if sha(checkpoint)!=v['checkpoint_sha256']:raise ValueError('Complete parent checkpoint changed')
 result=json.loads((checkpoint.parent/'result.json').read_text());pin=result['binding']
 if pin['source']!=contract['source_files'] or pin['config']!=contract['fixed_recipe'] or pin['engine']!=engine_binding():raise ValueError('Frozen sustained source/recipe/engine differs')
 if any(sha(GAME/'reference_training/v11'/n)!=h for n,h in contract['source_files'].items()):raise ValueError('Current trainer changed from sustained contract')
 seconds=min(contract['max_next_stage_seconds'],math.floor(stamp(contract['last_training_target_utc'])-time.time()-contract['reserved_tail_seconds']))
 if seconds<600:raise ValueError('Insufficient sustained training time before final artifact handoff')
 value={'ready':True,'parent':parent_name,'next_stage':next_name,'parent_checkpoint':str(checkpoint),'parent_sha256':v['checkpoint_sha256'],'seconds':seconds,'mode':'exact','variant':'v11','recipe_changed':False,'target_training_end_utc':contract['last_training_target_utc']}
 if execute:
  dump(OWN/(next_name+'-prelaunch-plan.json'),value)
  value['provider']=launch(next_name,root/'provider.json',relative,v['checkpoint_sha256'],seconds=seconds,variant='v11',mode='exact')
 return value

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--parent',required=True);p.add_argument('--next',required=True);p.add_argument('--execute',action='store_true');a=p.parse_args();print(json.dumps(continuation(a.parent,a.next,a.execute),indent=2))
