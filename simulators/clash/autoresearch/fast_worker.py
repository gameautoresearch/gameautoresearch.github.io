"""Finite verification worker, durable outputs, no retries or model API calls."""
import argparse,json,os,subprocess,sys,time,traceback
from pathlib import Path
from .common import dump,sha

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--deadline-at',type=float,required=True);p.add_argument('--mode',default='train',choices=['train','historical','resume-check']);a=p.parse_args();root=Path(a.root);started=time.perf_counter()
 os.environ.setdefault('CLASH_CUDA_LAUNCH_THREADS','1');os.environ.setdefault('CLASH_CUDA_RENDER_MODE','4')
 output=root/'output';output.mkdir(exist_ok=False)
 dump(output/'hardware.json',{'cgroup':{n:(Path('/sys/fs/cgroup/'+n).read_text() if Path('/sys/fs/cgroup/'+n).exists() else None) for n in ('cpu.max','memory.max')},'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,driver_version,memory.total','--format=csv,noheader'],text=True),'started_unix':time.time(),'task_id':os.getenv('MODAL_TASK_ID')})
 def command(label,args):
  mark=time.perf_counter()
  # faulthandler: a native crash (e.g. inside the GPU driver) leaves its Python stack in the log instead of nothing.
  with (output/(label+'.log')).open('w') as log:r=subprocess.run([sys.executable,'-X','faulthandler','-u',*args],stdout=log,stderr=subprocess.STDOUT,timeout=max(1,a.deadline_at-time.time()-8))
  dump(output/(label+'-process.json'),{'returncode':r.returncode,'complete_process_seconds':time.perf_counter()-mark,'command':[sys.executable,*args]})
  if r.returncode:raise RuntimeError(label+' failed: '+(output/(label+'.log')).read_text()[-3000:])
 try:
  if a.mode=='train':
   root.chmod(0o755);output.chmod(0o755);os.chown(output,65534,65534)
   code='import os;os.setgroups([]);os.setgid(65534);os.setuid(65534);from simulators.clash.autoresearch.train_entry import main;main()'
   command('starter',['-c',code,'--source',str(root/'inputs/source'),'--out',str(output/'artifact'),'--seconds',str(max(1,a.deadline_at-time.time()-20)),'--deadline-at',str(a.deadline_at-20)])
  elif a.mode=='historical':
   from .common import NATIVE
   command('cpu-cuda-qualification',[str(NATIVE/'qualify_cuda.py'),str(output/'cpu-cuda-qualification')])
   for trial in (1,2):
    command('historical-'+str(trial),[str(NATIVE/'train.py'),'--config',str(root/'inputs'/('historical-'+str(trial)+'.json')),'--out',str(output/('historical-'+str(trial)))])
  else:
   command('resume-check',['-m','simulators.clash.autoresearch.fast_qualification','--source',str(root/'inputs/source'),'--out',str(output/'resume-check')])
  dump(output/'worker-result.json',{'complete':True,'mode':a.mode,'body_seconds':time.perf_counter()-started})
 except BaseException:
  (output/'worker-error.txt').write_text(traceback.format_exc());dump(output/'worker-result.json',{'complete':False,'mode':a.mode,'body_seconds':time.perf_counter()-started});raise
 finally:
  dump(root/'output-manifest.json',{'complete':(output/'worker-result.json').exists() and json.loads((output/'worker-result.json').read_text())['complete'],'files':{str(p.relative_to(root)):sha(p) for p in output.rglob('*') if p.is_file()}})
  subprocess.run(['sync',str(root)],check=True,timeout=15)
if __name__=='__main__':main()
