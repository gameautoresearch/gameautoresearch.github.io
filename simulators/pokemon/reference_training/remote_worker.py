"""Runs as Sandbox primary process; finite provider lifetime, durable private files."""
import copy,json,os,subprocess,sys,time,traceback
from pathlib import Path
START=time.monotonic()

def main():
    from .trainer import train,resume_proof,atomic,assess,HERE
    from .. import cli
    import torch
    request_path=Path(sys.argv[1]);request=json.loads(request_path.read_text());out=request_path.parent
    def persist():subprocess.run(['sync','/runs'],check=True,timeout=10)
    result={'status':'running','request':request,'started_at_unix':time.time()}
    atomic(out/'worker-result.json',result);persist()
    try:
        if not torch.cuda.is_available():raise RuntimeError('L4 required')
        cfg=json.loads((HERE/'recipes'/(request['recipe']+'.json')).read_text());cfg['seed']=request['seed']
        # Use a new nonoverlapping training domain for the second matched training seed.
        cfg['training_seed_start']+=0 if request['seed']==1729 else 10000000
        torch.set_num_threads(cfg['hardware']['cpu_threads']);torch.cuda.init()
        if request['mode']=='proof':
            proof=copy.deepcopy(cfg);proof.update(iterations=3,allocation_seconds=40,max_decision_phases=4096,snapshot_every=1)
            proof['hardware'].update(workers=1,environments_per_worker=2)
            result['proof']=resume_proof(proof,out/'resume-proof',persist=persist)
            atomic(out/'worker-result.json',result);persist()
        available=request['seconds']-100-(time.monotonic()-START)
        if available<30:raise RuntimeError('Proof/setup left insufficient bounded training interval')
        cfg['allocation_seconds']=available
        resume=Path(request['resume']) if request.get('resume') else None
        result['training']=train(cfg,out/'training',resume,persist=persist)
        atomic(out/'executed-recipe.json',cfg)
        result['checkpoint']='/runs/jobs/'+request['label']+'/training/checkpoint.pt'
        result['status']='complete';result['elapsed_body_seconds']=time.monotonic()-START
    except BaseException as e:
        result.update(status='failed',error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc());traceback.print_exc()
    atomic(out/'worker-result.json',result);persist()
if __name__=='__main__':main()
