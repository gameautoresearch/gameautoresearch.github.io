"""Finite CPU-only qualification/screen, durable v2 volume, no host dependency."""
import argparse,json,os,signal,subprocess,sys,time,traceback,shutil
from pathlib import Path


def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--candidate',required=True);p.add_argument('--seconds',type=int,required=True);p.add_argument('--proof',action='store_true');a=p.parse_args();a.root.mkdir(parents=True,exist_ok=True);a.root.chmod(0o700);(a.root/'entry-claim').mkdir(exist_ok=False);start=time.time();hard=start+a.seconds
 def save(name,obj):(a.root/name).write_text(json.dumps(obj,indent=2)+'\n');subprocess.run(['sync',str(a.root)],timeout=10,check=True)
 def run(args,deadline,log):
  with (a.root/log).open('a',buffering=1) as f:
   process=subprocess.Popen(args,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
   try:code=process.wait(timeout=max(1,deadline-time.time()))
   except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait();raise
   if code:raise RuntimeError('Child failed: '+str(args))
 save('lifecycle.json',dict(status='started',task_id=os.environ.get('MODAL_TASK_ID'),started_unix=start,hard_deadline_unix=hard,automatic_retries=0))
 try:
  if a.proof:
   run([sys.executable,'-m','pytest','-q','simulators/tavern/tests/test_evaluation_readiness.py','simulators/tavern/tests/test_admission_credit.py','simulators/tavern/tests/test_contract.py','simulators/tavern/autoresearch/tests/test_harness.py','--junitxml='+str(a.root/'qualification.xml')],min(hard-180,time.time()+150),'qualification.log')
   # Official root-container checkpoint gate must reject the old source identity.
   old=a.root/'old-bundle';old.mkdir();identity=json.loads(Path('/opt/evaluator-assets/learned/starter-111901/identity.json').read_text());identity['binding']['native_source_hash']='unqualified-old-native';(old/'identity.json').write_text(json.dumps(identity));panel=a.root/'unused-panel.json';panel.write_text(json.dumps(dict(seeds=[1],games_per_seat=1)))
   check=subprocess.run([sys.executable,'simulators/tavern/autoresearch/grade.py','--bundle',str(old),'--panel',str(panel),'--out',str(a.root/'must-reject')],capture_output=True,text=True,timeout=20)
   if not check.returncode or 'Source/rules-bound checkpoint mismatch' not in check.stderr:raise RuntimeError('Old source identity gate did not reject')
   save('qualification.json',dict(passed=True,old_source_identity_rejected=True,admission_v2=True,CPU_only=True))
  # Candidate gets only its own readable inference bundle; registry/final panel stay root-private.
  candidate=Path('/opt/evaluator-assets/learned')/a.candidate
  if candidate.exists():
   dest=Path('/tmp/evaluator-candidates')/a.candidate;dest.mkdir(parents=True,exist_ok=True)
   for name in ('policy.py','learner.py','train.py','weights.npz','identity.json','summary.json'):
    shutil.copy2(candidate/name,dest/name);(dest/name).chmod(0o444)
   # Registry paths for host-owned references remain private. Candidate path override is explicit.
   os.environ['TAVERN_EVAL_CANDIDATE_BUNDLE']=str(dest)
  from .screen import screen
  result=screen('/opt/evaluator-assets','/opt/evaluator-panel.json',a.candidate,a.root/'screen',stage_seconds=max(140,hard-time.time()-20))
  save('lifecycle.json',dict(status='completed',started_unix=start,finished_unix=time.time(),complete_valid_schedule=result['complete_valid_schedule'],automatic_retries=0))
 except BaseException:
  (a.root/'failure.txt').write_text(traceback.format_exc());save('lifecycle.json',dict(status='failed',started_unix=start,finished_unix=time.time(),automatic_retries=0));raise
 finally:subprocess.run(['sync',str(a.root)],timeout=10,check=True)
if __name__=='__main__':main()
