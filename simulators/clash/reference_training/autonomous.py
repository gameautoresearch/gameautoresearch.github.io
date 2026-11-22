"""Remote finite supervisor; durable checkpoints and lifecycle survive disconnect."""
from pathlib import Path
import argparse,json,os,subprocess,sys,time,traceback
from simulators.clash.autoresearch.common import dump

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--seconds',type=float,required=True);p.add_argument('--volume',required=True);p.add_argument('--resume');p.add_argument('--proof',action='store_true');a=p.parse_args()
 root=Path(a.root);root.mkdir(parents=True,exist_ok=True);start=time.time();sync=lambda:subprocess.run(['sync',str(root)],check=True,timeout=20)
 dump(root/'lifecycle.json',{'phase':'starting','started_unix':start,'deadline_unix':start+a.seconds,'automatic_retries':0})
 args=[sys.executable,'-u','-m','simulators.clash.reference_training.run','--config','/opt/clash/simulators/clash/reference_training/recipe.json','--out',str(root/'output'),'--deadline',str(start+a.seconds-35)]
 if a.proof:args+=['--proof']
 if a.resume:args+=['--resume',a.resume]
 with (root/'stdout.log').open('w') as out,(root/'stderr.log').open('w') as err:
  proc=subprocess.Popen(args,stdout=out,stderr=err,start_new_session=True)
  while proc.poll() is None and time.time()<start+a.seconds-10:
   dump(root/'heartbeat.json',{'checked_unix':time.time(),'child_pid':proc.pid,'elapsed':time.time()-start});sync();time.sleep(5)
  if proc.poll() is None:
   import signal
   os.killpg(proc.pid,signal.SIGKILL);proc.wait()
  dump(root/'lifecycle.json',{'phase':'completed' if proc.returncode==0 else 'failed','started_unix':start,'finished_unix':time.time(),'trainer_returncode':proc.returncode,'result_present':(root/'output/result.json').exists(),'automatic_retries':0});sync()
 if proc.returncode:raise RuntimeError('reference trainer failed; inspect durable stderr')
if __name__=='__main__':main()
