"""Finite primary process. Source/checkpoints survive creator disconnect on private Volume."""
import hashlib,json,os,pathlib,subprocess,sys,tarfile,time,traceback,signal
START=time.monotonic()
def atomic(p,value):
 p=pathlib.Path(p);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def main():
 path=pathlib.Path(sys.argv[1]);spec=json.loads(path.read_text());job=path.parent;job.chmod(0o700)
 watchdog_code="import os,signal,sys,time;time.sleep(max(0,float(sys.argv[1])-time.time()));os.kill(int(sys.argv[2]),signal.SIGKILL)"
 watchdog=subprocess.Popen([sys.executable,'-c',watchdog_code,str(min(spec['absolute_cutoff_unix']-12,time.time()+spec['max_seconds']-35)),str(os.getpid())],stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
 out={'status':'running','started_at_unix':time.time(),'mode':spec['mode'],'release':spec['release']};atomic(job/'worker-result.json',out)
 try:
  if spec.get('entrypoint_sha256') and hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()!=spec['entrypoint_sha256']:raise ValueError('Frozen worker entrypoint identity changed')
  archive=pathlib.Path('/runs/releases')/spec['release']/'source.tar'
  if hashlib.sha256(archive.read_bytes()).hexdigest()!=spec['runtime_archive_sha256']:raise ValueError('Runtime archive identity changed')
  root=pathlib.Path('/tmp/pokemon-strong-runtime');root.mkdir()
  with tarfile.open(archive) as tar:tar.extractall(root,filter='data')
  os.symlink('/opt/pk/candidates',root/'source/candidates');sys.path.insert(0,str(root/'source'));os.environ['PYTHONPATH']=str(root/'source');os.umask(0o022)
  # This deadline is checked independently of any host connection or queue delay.
  spec['deadline_unix']=min(spec['absolute_cutoff_unix']-35,time.time()+spec['max_seconds']-100-(time.monotonic()-START))
  if time.time()>=spec['deadline_unix']-20:raise TimeoutError('Absolute stage window expired before execution')
  if spec['mode']=='teacher':
   from integrations.pokemon.reference_training.strong_teacher import run
  elif spec['mode'] in ('train','proof'):
   from integrations.pokemon.reference_training.strong_runtime import run
  elif spec['mode']=='transport-proof':
   from integrations.pokemon.reference_training.strong_transport_proof import run
  elif spec['mode']=='export':
   from integrations.pokemon.reference_training.strong_export import run
  elif spec['mode']=='public-screen':
   from integrations.pokemon.reference_training.strong_public import run
  else:raise ValueError('Unknown finite worker mode')
  result=run(spec,job)
  if spec['mode']=='proof':
   import copy
   from integrations.pokemon.reference_training import trainer
   import torch
   parent=job/'training/checkpoint.pt';parent_sha=hashlib.sha256(parent.read_bytes()).hexdigest();resume_spec=copy.deepcopy(spec);resume_spec.update(mode='train',parent_kind='reference-full',parent_checkpoint=str(parent),parent_sha256=parent_sha,parent_source_archive=str(archive),parent_source_archive_sha256=spec['runtime_archive_sha256'],resume_strict=True,migrate_strong=False);resume_spec['config']['iterations']=result['iterations']+1
   paths=[]
   for label in ('resume-a','resume-b'):
    sub=job/label;sub.mkdir();atomic(sub/'request.json',resume_spec)
    code="import json,sys;from pathlib import Path;sys.path.insert(0,sys.argv[1]);from integrations.pokemon.reference_training.strong_runtime import run;s=json.loads(Path(sys.argv[2]).read_text());run(s,Path(sys.argv[2]).parent)"
    subprocess.run(['python','-c',code,str(root/'source'),str(sub/'request.json')],check=True,timeout=min(120,int(spec['deadline_unix']-time.time())-10));paths.append(sub/'training/checkpoint.pt')
   a,b=[torch.load(p,map_location='cpu',weights_only=False) for p in paths];keys=['model','optimizer','numpy_rng','torch_rng','python_rng','cuda_rng','history','population_lineage','iteration','next_battle_index','bc_rng','bc_anchor'];checks={k:trainer.equal(a[k],b[k]) for k in keys}
   if not all(checks.values()):raise AssertionError(checks)
   out['proof']={'exact_resume':checks,'valid':True}
   if spec.get('untrained_control'):
    if spec.get('parent_kind')!='random-canonical':raise ValueError('Untrained control requires fresh canonical initialization')
    sub=job/'untrained-control';sub.mkdir();control=copy.deepcopy(spec);control.update(mode='train',resume_strict=False,migrate_strong=False);control.pop('teacher_data',None);control['config'].update(iterations=0,bc_epochs=0,aux_bc_coef=0.);atomic(sub/'request.json',control)
    subprocess.run(['python','-c',code,str(root/'source'),str(sub/'request.json')],check=True,timeout=min(90,int(spec['deadline_unix']-time.time())-10))
    initial=torch.load(sub/'training/checkpoint.pt',map_location='cpu',weights_only=False)
    if initial['model_sha256']!=result['initial_parent_model_sha256']:raise AssertionError('Fresh untrained control tensor differs from learned branch initial parent')
    out['proof']['untrained_control_initial_tensor_match']=True
   atomic(job/'proof.json',out['proof'])
  out.update(status='complete',result=result,body_seconds=time.monotonic()-START)
 except BaseException as exc:out.update(status='failed',error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc());traceback.print_exc()
 atomic(job/'worker-result.json',out);subprocess.run(['sync','/runs'],timeout=15,check=True);watchdog.terminate();watchdog.wait(timeout=5)
if __name__=='__main__':main()
