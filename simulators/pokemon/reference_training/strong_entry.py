"""Standalone detached supervisor of an exact historical-source continuation."""
import hashlib,json,os,signal,subprocess,sys,tarfile,time,traceback
from pathlib import Path
START=time.monotonic()
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def atomic(p,v):
 p=Path(p);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def main():
 path=Path(sys.argv[1]);request=json.loads(path.read_text());job=path.parent;result={'status':'running','request':request,'started_at_unix':time.time()};atomic(job/'worker-result.json',result)
 try:
  if sha(job/'input/parent.pt')!=request['parent_sha256'] or sha(job/'input/original-source.tar')!=request['original_source_tar_sha256']:raise ValueError('Frozen input identity mismatch')
  dest=Path('/tmp/exact-reference');dest.mkdir()
  with tarfile.open(job/'input/original-source.tar') as tar:tar.extractall(dest,filter='data')
  os.symlink('/opt/pk/candidates',dest/'source/candidates')
  available=min(request['max_seconds']-100-(time.monotonic()-START),request['selection_close_unix']-time.time()-75)
  if available<20:raise RuntimeError('Absolute selection deadline leaves no training time')
  code='''import json,sys,subprocess,time;from pathlib import Path;sys.path.insert(0,sys.argv[1]);import torch;from integrations.pokemon.reference_training import trainer;t=Path(sys.argv[2]);p=t/'input/parent.pt';cp=torch.load(p,map_location='cpu',weights_only=False);cfg=cp['config'];torch.set_num_threads(cfg['hardware']['cpu_threads']);trainer.admit(p,cfg);cfg['allocation_seconds']=float(sys.argv[3]);result=trainer.train(cfg,t/'training',p,persist=lambda:subprocess.run(['sync','/runs'],check=True,timeout=10));(t/'training-result.json').write_text(json.dumps(result,indent=2));'''
  proc=subprocess.Popen(['python','-c',code,str(dest/'source'),str(job),str(available)],stdout=(job/'trainer-stdout.log').open('w'),stderr=(job/'trainer-stderr.log').open('w'),start_new_session=True)
  try:rc=proc.wait(timeout=available+35)
  except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait();raise TimeoutError('Finite training body deadline')
  if rc:raise RuntimeError('Original trainer returned '+str(rc))
  result.update(status='complete',training=json.loads((job/'training-result.json').read_text()),checkpoint=str(job/'training/checkpoint.pt'),body_seconds=time.monotonic()-START)
 except BaseException as exc:result.update(status='failed',error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc());traceback.print_exc()
 atomic(job/'worker-result.json',result);subprocess.run(['sync','/runs'],timeout=15,check=True)
if __name__=='__main__':main()
