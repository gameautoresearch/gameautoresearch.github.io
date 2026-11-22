"""Independent processes: actual starter GPU backward/resume/export/source checks."""
import argparse,copy,json,shutil,subprocess,sys,time
from pathlib import Path
from .common import dump,load_module,DATA,NATIVE

def main():
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);started=time.perf_counter()
 import numpy as np,torch
 from simulators.clash.native.test_training import equal,state_images
 source=a.out/'source';shutil.copytree(a.source,source)
 c=json.loads((source/'config.json').read_text());c.update(envs=32,updates=2,rollout_steps=32,minibatch=1024,threads=4);dump(source/'config.json',c)
 def command(name,resume=None,updates=2):
  cmd=[sys.executable,'-m','simulators.clash.autoresearch.train_entry','--source',str(source),'--out',str(a.out/name),'--seconds','180','--updates',str(updates)]
  if resume:cmd+=['--resume',str(resume)]
  r=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=180);(a.out/(name+'.log')).write_text(r.stdout);return r
 for name,updates,resume in [('full',2,None),('prefix',1,None),('resumed',2,a.out/'prefix/checkpoint.pt')]:
  r=command(name,resume,updates)
  if r.returncode:raise RuntimeError(name+' failed: '+r.stdout[-3500:])
 full=torch.load(a.out/'full/checkpoint.pt',map_location='cpu',weights_only=False);resumed=torch.load(a.out/'resumed/checkpoint.pt',map_location='cpu',weights_only=False)
 keys=['format','config','binding','backend_build','auxiliary_builds','execution_runtime','metadata','model','optimizer','engine_outputs','update','samples','counters','bucket_usage','python_rng','numpy_rng','torch_rng','cuda_rng','entrant_binding','entrant_state']
 for key in keys:equal(full[key],resumed[key],key)
 # Read layout independently from C++ ABI; only these four constant pointer
 # addresses may differ between processes, every other world byte must match.
 cpp=a.out/'layout.cpp';binary=a.out/'layout'
 cpp.write_text('#include <cstddef>\n#include <cstdio>\n#include "'+str(NATIVE/'world.hpp')+'"\nint main(){using namespace clash_native;std::printf("%zu %zu %zu %zu %zu %zu",sizeof(NativeWorld),sizeof(void*),offsetof(NativeWorld,spec),offsetof(NativeWorld,arenaGrid)+offsetof(GridView,tiles),offsetof(NativeWorld,arenaGrid)+offsetof(GridView,current),offsetof(NativeWorld,arenaGrid)+offsetof(GridView,previous));}\n')
 subprocess.run([shutil.which('g++') or shutil.which('clang++'),'-std=c++20',str(cpp),'-o',str(binary)],check=True)
 world,pointer,*offsets=map(int,subprocess.check_output([str(binary)],text=True).split())
 for x,y in zip(state_images(full['engine_state']),state_images(resumed['engine_state'])):
  x=bytearray(x);y=bytearray(y)
  for offset in offsets:x[offset:offset+pointer]=b'\0'*pointer;y[offset:offset+pointer]=b'\0'*pointer
  assert x==y
 from .common import setup_native
 setup_native()
 module=load_module(source/'policy.py','qualified_export_policy');m=module.create(c).cuda();m.load_state_dict(full['model']);n=module.create(c).cuda()
 with np.load(a.out/'full/weights.npz',allow_pickle=False) as data:n.load_state_dict({k:torch.from_numpy(data[k]).cuda() for k in data.files})
 obs=full['engine_outputs']['observations'].cuda().flatten(0,1);mask=full['engine_outputs']['masks'].cuda().flatten(0,1)
 with torch.no_grad():
  for x,y in zip(m(obs,mask),n(obs,mask)):assert torch.equal(x,y)
 from simulators.clash.model import Policy
 from simulators.clash.native.train import schema
 from gpu_features import DeviceBatchEncoder
 data=json.loads(DATA.read_text());native=Policy(schema(data),len(data['vocabulary']),len(data['buff_names']),c['width'],c['embedding']).cuda();native.load_state_dict(full['model'])
 with torch.no_grad():
  for x,y in zip(m(obs,mask),native(DeviceBatchEncoder(native.s,copy=False)(obs,mask))):assert torch.equal(x,y)
 rejected=[];base=torch.load(a.out/'prefix/checkpoint.pt',map_location='cpu',weights_only=False)
 for kind in ('entrant','source','data','rules','recipe','backend','runtime','auxiliary'):
  saved=copy.deepcopy(base)
  if kind=='entrant':saved['entrant_binding']['source']['train.py']='0'*64
  elif kind=='source':saved['binding']['source']['native/train.py']='0'*64
  elif kind=='data':saved['binding']['task_data_sha256']='0'*64
  elif kind=='rules':saved['binding']['rules_lock_sha256']='0'*64
  elif kind=='recipe':saved['config']['gamma']=.7
  elif kind=='backend':saved['backend_build']['library_sha256']='0'*64
  elif kind=='runtime':saved['execution_runtime']['torch']='incompatible'
  else:saved['auxiliary_builds']={'incompatible':'test'}
  path=a.out/('bad-'+kind+'.pt');torch.save(saved,path);r=command('reject-'+kind,path)
  assert r.returncode and ('binding differs' in r.stdout or 'recipe differs' in r.stdout),(kind,r.stdout[-1000:]);rejected.append(kind)
 result={'passed':True,'cuda_executed':True,'exact_fields':keys,'complete_world_bytes_equal_except_four_pointers':True,'pointer_offsets':offsets,'world_bytes':world,'export_import_logits_values_exact':True,'native_adapter_logits_values_exact':True,'tamper_rejections':rejected,'samples':full['samples'],'finite_gradients':json.loads((a.out/'full/result.json').read_text())['finite_gradients'],'elapsed_seconds':time.perf_counter()-started,'timing_claim':False}
 dump(a.out/'result.json',result)
if __name__=='__main__':main()
