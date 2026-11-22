"""Training entry: save initial policy for an independently checked learning delta."""
import json,os,sys,time
from pathlib import Path
os.environ.setdefault('PC_CUDA_WARP_TICK','1');os.environ.setdefault('PC_CUDA_STACK_KB','64')
sys.path.insert(0,'/workspace/src');sys.path.insert(0,'/workspace/experiments')
from polytopia_bot.rl.gpu_resident import GpuResidentTrainer
original=GpuResidentTrainer.run
original_update=GpuResidentTrainer._update
clock_origin=float(os.environ.get('POLYTOPIA_ALLOCATION_STARTED',time.time()))
def update(self):
 result=original_update(self)
 for minute in (5,10):
  if time.time()-clock_origin>=minute*60 and not (self.root/('clock-'+str(minute)+'m.npz')).exists():
   self._write('clock-'+str(minute)+'m',self._host_state())
   (self.root/('clock-'+str(minute)+'m-receipt.json')).write_text(json.dumps({'allocation_elapsed_seconds':time.time()-clock_origin,'update':self.version,'requested_minutes':minute}))
 return result
GpuResidentTrainer._update=update

def run(self):
 from polytopia_bot.rl.runner import save_checkpoint
 save_checkpoint(self.root/'initial.npz',self.params,self.run_config,stage='initial')
 return original(self)
GpuResidentTrainer.run=run
import gpu_resident_train
gpu_resident_train.main()
