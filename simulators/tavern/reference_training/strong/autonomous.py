"""Finite detached stage; source/checkpoints/diagnostics persist in a v2 Volume."""
import time
ENTRY_STARTED=time.time()
import argparse,json,os,signal,shutil,subprocess,sys,traceback
os.environ['PYTHONDONTWRITEBYTECODE']='1'
sys.dont_write_bytecode=True
from pathlib import Path
from .common import HERE,write

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--config',type=Path,required=True);p.add_argument('--cap',type=int,required=True);p.add_argument('--selection-close',type=float,required=True);p.add_argument('--proof',action='store_true');p.add_argument('--resume',type=Path);a=p.parse_args()
 a.root.mkdir(parents=True,exist_ok=True);(a.root/'entry-claim').mkdir(exist_ok=False);hard=min(ENTRY_STARTED+a.cap-35,a.selection_close-35)
 shutil.copytree('/reference-source',a.root/'source');shutil.copytree('/inputs',a.root/'inputs');shutil.copy2(a.config,a.root/'config.json')
 def sync():subprocess.run(['sync',str(a.root)],check=True,timeout=10)
 write(a.root/'lifecycle.json',dict(status='started',entry_unix=ENTRY_STARTED,hard_stop_unix=hard,selection_close=a.selection_close,task_id=os.environ.get('MODAL_TASK_ID'),automatic_GPU_retries=0));sync()
 def command(args,deadline):
  with (a.root/'trainer.log').open('a',buffering=1) as log:
   child=subprocess.Popen([sys.executable,'-u','-m',*args],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   try:code=child.wait(timeout=max(1,deadline-time.time()))
   except subprocess.TimeoutExpired:
    os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=10);raise
   if code:raise RuntimeError('Stage subprocess failed '+str(code)+': '+str(args))
 try:
  if a.proof:
   command(['simulators.tavern.reference_training.strong.run_proof','--config',str(a.config),'--inputs','/inputs','--out',str(a.root/'proof')],min(hard-100,time.time()+180))
  seconds=max(0,hard-time.time()-70)
  if seconds<10:raise RuntimeError('No training interval remains before absolute close')
  args=['simulators.tavern.reference_training.strong.train','--config',str(a.config),'--inputs','/inputs','--out',str(a.root/'training'),'--seconds',str(seconds)]
  if a.resume:args+=['--resume',str(a.resume)]
  command(args,hard-35)
  command(['simulators.tavern.reference_training.strong.export','--checkpoint',str(a.root/'training/latest.pt'),'--out',str(a.root/'export')],hard-10)
  write(a.root/'lifecycle.json',dict(status='completed',entry_unix=ENTRY_STARTED,finished_unix=time.time(),automatic_GPU_retries=0))
 except BaseException:
  (a.root/'failure.txt').write_text(traceback.format_exc());write(a.root/'lifecycle.json',dict(status='failed',entry_unix=ENTRY_STARTED,finished_unix=time.time(),automatic_GPU_retries=0));raise
 finally:sync()
if __name__=='__main__':main()
