"""Bounded GPU proof/training entrypoint. Version2 volume is durable through sync."""
import argparse,json,os,shutil,signal,subprocess,sys,time,traceback
from pathlib import Path
from .bootstrap import HERE,write

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--seconds',type=int,required=True);p.add_argument('--proof',action='store_true');p.add_argument('--resume',type=Path);a=p.parse_args()
 a.root.mkdir(parents=True,exist_ok=True);(a.root/'entry-claim').mkdir(exist_ok=False);started=time.time();hard=started+a.seconds
 write(a.root/'lifecycle.json',dict(status='started',started_unix=started,hard_deadline_unix=hard,task_id=os.environ.get('MODAL_TASK_ID'),automatic_retries=0))
 shutil.copytree('/reference-source',a.root/'source');shutil.copy2(HERE/'recipe.json',a.root/'recipe.json')
 def command(args,until):
  with (a.root/'trainer.log').open('a',buffering=1) as f:
   child=subprocess.Popen([sys.executable,'-u','-m',*args],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
   try:code=child.wait(timeout=max(1,until-time.time()))
   except subprocess.TimeoutExpired:
    os.killpg(child.pid,signal.SIGKILL);child.wait();raise
   if code:raise RuntimeError(f'Child failed with {code}: {args}')
 try:
  if a.proof:
   c=json.loads((HERE/'recipe.json').read_text());c.update(n_games=2,network=dict(hidden=16,embedding=2),population_snapshot_every=1)
   write(a.root/'proof-config.json',c)
   command(['simulators.tavern.reference_training.validate','--config',str(a.root/'proof-config.json'),'--out',str(a.root/'proof')],min(hard-90,time.time()+130))
  remaining=max(1,hard-time.time()-65)
  args=['simulators.tavern.reference_training.train','--config',str(HERE/'recipe.json'),'--out',str(a.root/'training'),'--seconds',str(remaining)]
  if a.resume:args+=['--resume',str(a.resume)]
  command(args,hard-35)
  command(['simulators.tavern.reference_training.export','--checkpoint',str(a.root/'training/latest.pt'),'--out',str(a.root/'export')],hard-10)
  write(a.root/'lifecycle.json',dict(status='completed',started_unix=started,finished_unix=time.time(),automatic_retries=0,export_present=True))
 except BaseException:
  (a.root/'failure.txt').write_text(traceback.format_exc());write(a.root/'lifecycle.json',dict(status='failed',started_unix=started,finished_unix=time.time(),automatic_retries=0));raise
 finally:subprocess.run(['sync',str(a.root)],timeout=9,check=True)
if __name__=='__main__':main()
