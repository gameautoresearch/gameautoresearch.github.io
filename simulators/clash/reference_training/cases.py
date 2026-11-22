"""Finite paired fresh-seed teacher/PFSP and teacher-off control on one allocation."""
from pathlib import Path
import argparse,json,subprocess,sys,time
from simulators.clash.autoresearch.common import dump

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--seconds',type=float,required=True);p.add_argument('--seed',type=int,default=27191);p.add_argument('--variant',choices=('v2','v3'),default='v2');a=p.parse_args()
 root=Path(a.root);root.mkdir(parents=True,exist_ok=True);start=time.time()
 # All planning and both fresh random-weight controls are fixed before case one.
 case_seconds=(a.seconds-75)/2
 c=json.loads((Path(__file__).parent/a.variant/'recipe.json').read_text());c['seed']=a.seed
 cases=[('teacher-pfsp',1.0),('teacher-off-control',0.0)]
 if a.seed%2==0:cases.reverse() # Precommitted counterbalancing, never score-dependent.
 dump(root/'case-plan.json',{'seed':a.seed,'variant':a.variant,'cases':cases,'seconds_each':case_seconds,'shared_full_allocation_seconds':a.seconds,'equal_case_process_time_cap':True,'same_hardware':True,'all_costs_in_shared_ledger':True})
 results={}
 for label,coefficient in cases:
  config=dict(c,teacher_coef=coefficient);path=root/(label+'-recipe.json');dump(path,config)
  began=time.time();deadline=min(start+a.seconds-20,began+case_seconds)
  command=[sys.executable,'-u','-m','simulators.clash.reference_training.'+a.variant+'.run','--config',str(path),'--out',str(root/label),'--deadline',str(deadline),'--proof']
  with (root/(label+'-stdout.log')).open('w') as out,(root/(label+'-stderr.log')).open('w') as err:
   proc=subprocess.Popen(command,stdout=out,stderr=err,start_new_session=True)
   try:proc.wait(timeout=max(1,deadline-time.time()+15))
   except subprocess.TimeoutExpired:
    import os,signal
    os.killpg(proc.pid,signal.SIGKILL);proc.wait();raise RuntimeError('case deadline exceeded')
  if proc.returncode:raise RuntimeError(label+' failed: '+(root/(label+'-stderr.log')).read_text()[-2000:])
  results[label]={'case_process_seconds':time.time()-began,'result':json.loads((root/label/'result.json').read_text())}
  dump(root/'case-status.json',{'completed':list(results),'elapsed':time.time()-start});subprocess.run(['sync',str(root)],check=True,timeout=20)
 dump(root/'result.json',{'format':'clash-paired-reference-'+a.variant,'variant':a.variant,'seed':a.seed,'case_time_cap_seconds':case_seconds,'cases':results,'elapsed_seconds':time.time()-start})
 subprocess.run(['sync',str(root)],check=True,timeout=20)
if __name__=='__main__':main()
