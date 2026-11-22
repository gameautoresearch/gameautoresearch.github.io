"""Host-owned evaluation; neither submitted Python nor panel Python runs in verifier globals."""
from pathlib import Path
import ctypes,json,os,resource,select,shutil,signal,subprocess,sys,tempfile,time,struct
import numpy as np
from .common import ROOT,DATA,MATCH_FORM,setup_native,dump,digest
from .bundle import verify,FILES
from .wire import encode

def drop(uid):
 os.setgroups([]);os.setgid(uid);os.setuid(uid)
 if ctypes.CDLL(None).prctl(38,1,0,0,0)!=0:raise RuntimeError('no_new_privs')
 resource.setrlimit(resource.RLIMIT_CORE,(0,0));resource.setrlimit(resource.RLIMIT_NOFILE,(64,64));resource.setrlimit(resource.RLIMIT_NPROC,(128,128))

class PolicyProcess:
 def __init__(self,bundle,development=False,uid=65534,diagnostics=False):
  verify(bundle)
  if not development and (sys.platform!='linux' or os.geteuid()!=0):raise RuntimeError('official grading requires root Linux supervisor and distinct unprivileged children')
  self.tmp=tempfile.TemporaryDirectory(prefix='clash-policy-');root=Path(self.tmp.name);root.chmod(0o700)
  target=root/'bundle';target.mkdir()
  for n in FILES:shutil.copyfile(Path(bundle)/n,target/n);(target/n).chmod(0o444)
  self.err=(root/'stderr.log').open('wb')
  if not development:os.chown(root,uid,uid)
  code=f'import sys;sys.path.insert(0,{str(ROOT)!r});from simulators.clash.autoresearch.policy_worker import main;main()'
  env={'PATH':'/usr/local/bin:/usr/bin:/bin','HOME':str(root),'TMPDIR':str(root),'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','CUDA_VISIBLE_DEVICES':''}
  self.diagnostics=diagnostics;self.last_diagnostics=None
  args=[sys.executable,'-I','-c',code,'--bundle',str(target)]+(['--diagnostics'] if diagnostics else [])
  self.process=subprocess.Popen(args,cwd=root,env=env,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.err,start_new_session=True,preexec_fn=None if development else lambda:drop(uid))
  self.boundary={'development_only':development,'pid':self.process.pid,'uid':None,'no_new_privileges':False}
  if not development:
   status=Path('/proc')/str(self.process.pid)/'status';fields={line.split(':',1)[0]:line.split(':',1)[1].strip() for line in status.read_text().splitlines() if ':' in line}
   actual=[int(v) for v in fields['Uid'].split()]
   if actual!=[uid]*4 or fields.get('NoNewPrivs')!='1':self.close();raise ValueError('policy privilege boundary failed')
   self.boundary.update(uid=uid,no_new_privileges=True)
 def read(self,n,deadline):
  chunks=[]
  while n:
   ready,_,_=select.select([self.process.stdout],[],[],max(0,deadline-time.monotonic()))
   if not ready:raise RuntimeError('policy inference read deadline')
   chunk=os.read(self.process.stdout.fileno(),n)
   if not chunk:raise RuntimeError('policy inference process exited')
   chunks.append(chunk);n-=len(chunk)
  return b''.join(chunks)
 def choose(self,x,mask):
  raw=encode(x,mask);deadline=time.monotonic()+15;fd=self.process.stdin.fileno();os.set_blocking(fd,False);view=memoryview(raw)
  while view:
   _,ready,_=select.select([],[fd],[],max(0,deadline-time.monotonic()))
   if not ready:raise RuntimeError('policy request write deadline')
   sent=os.write(fd,view);view=view[sent:]
  size=struct.unpack('>I',self.read(4,deadline))[0]
  if size>65536:raise ValueError('policy response cap')
  values=json.loads(self.read(size,deadline))
  if self.diagnostics:
   self.last_diagnostics={k:v for k,v in values.items() if k!='actions'};values=values['actions']
   for key in ('wait_probability','aggregate_nonwait_probability'):
    numbers=self.last_diagnostics[key]
    if len(numbers)!=len(x) or any(not np.isfinite(v) or not 0<=v<=1 for v in numbers):raise ValueError('invalid public probability diagnostic')
  if len(values)!=len(x) or any(type(v)!=int or not 0<=v<2307 or not mask[i,v] for i,v in enumerate(values)):raise ValueError('illegal policy response')
  return values
 def close(self):
  if self.process.poll() is None:
   os.killpg(self.process.pid,signal.SIGTERM)
   try:self.process.wait(timeout=2)
   except subprocess.TimeoutExpired:os.killpg(self.process.pid,signal.SIGKILL);self.process.wait()
  self.err.close();self.tmp.cleanup()

def summarize(rows):
 if not rows or any(not r['natural'] or any(r.get(k,False) for k in ('truncated','illegal','fault','entity_overflow')) for r in rows):raise ValueError('incomplete/invalid evaluation has no official score')
 seeds=sorted(set(r['seed_index'] for r in rows));per_seed=[np.mean([r['points'] for r in rows if r['seed_index']==seed]) for seed in seeds]
 rng=np.random.default_rng(73241);bootstrap=[np.mean(rng.choice(per_seed,len(per_seed),replace=True))*100 for _ in range(2000)]
 return {'games':len(rows),'raw_score_0_100':float(np.mean([r['points'] for r in rows])*100),'normalized_reference_score':None,'interval_95_percent':list(np.quantile(bootstrap,[.025,.975])),'interval_method':'paired seed-cluster percentile bootstrap; smoke intervals are descriptive only','by_opponent':{p:float(np.mean([r['points'] for r in rows if r['opponent']==p])*100) for p in sorted({r['opponent'] for r in rows})},'by_seat':{str(s):float(np.mean([r['points'] for r in rows if r['seat']==s])*100) for s in (0,1)},'natural_games':len(rows),'truncations':0,'illegal_actions':0,'simulator_faults':0}

def evaluate(bundle,schedule,out,*,development=False,split='private',diagnostics=False):
 setup_native()
 from cpu_batch import NativeCPUBatch
 from evaluate import StaticSchema,PublicActor
 data=json.loads(DATA.read_text());schema=StaticSchema(data);schedule=json.loads(Path(schedule).read_text());rows=[];started=time.monotonic()
 panel=schedule['opponents'];seeds=schedule['engine_seeds']
 if len(set(seeds))!=len(seeds) or not seeds or not panel or len({p['name'] for p in panel})!=len(panel):raise ValueError('invalid panel')
 if any(p['kind'] not in ('scripted','random','wait','checkpoint','teacher','intermediate') for p in panel):raise ValueError('unsupported panel')
 processes={};candidate=None
 try:
  candidate=PolicyProcess(bundle,development,diagnostics=diagnostics)
  for i,p in enumerate(panel):
   if p['kind']=='checkpoint':
    verify(p['bundle'],p.get('identity'));processes[p['name']]=PolicyProcess(p['bundle'],development,65533-i)
  jobs=[(p,i,seed,side) for p in panel for i,seed in enumerate(seeds) for side in (0,1)]
  for offset in range(0,len(jobs),16):
   part=jobs[offset:offset+16];env=NativeCPUBatch([v[2] for v in part],threads=4,form=MATCH_FORM);active=set(range(len(part)));counts=[0]*len(part)
   behavior=[{'decisions':0,'forced_wait':0,'voluntary_wait':0,'deployments':0,'deployment_row_sum':0,'wait_elixir_sum':0,'cards':{}} for _ in part]
   unique_actions=[set() for _ in part]
   from .visible_teacher import PublicTeacher
   bots={i:(PublicTeacher(p['kind'],data=data) if p['kind'] in ('teacher','intermediate') else PublicActor(p['kind'],int(digest([schedule['policy_seed'],idx,p['name'],side])[:16],16),data=data)) for i,(p,idx,_,side) in enumerate(part) if p['kind']!='checkpoint'}
   for i,actor in bots.items():
    options=part[i][0].get('options',{})
    if options and part[i][0]['kind']!='scripted':raise ValueError('only scripted opponents admit frozen options')
    for key,value in options.items():
     if key not in ('attack_elixir','support_elixir','back_row','spell_targets','spell_worth','support_gap'):raise ValueError('unsupported scripted opponent parameter')
     if isinstance(value,bool) or not isinstance(value,(int,float)) or not np.isfinite(value) or value<0:raise ValueError('invalid scripted parameter')
     setattr(actor.bot,key,value)
   try:
    while active:
     actions=np.zeros((len(part),2),np.int32);ids=sorted(active)
     x=np.stack([env.observations[i,part[i][3]] for i in ids]);mask=np.stack([env.masks[i,part[i][3]] for i in ids]);chosen=candidate.choose(x,mask)
     for j,(i,a) in enumerate(zip(ids,chosen)):
      actions[i,part[i][3]]=a;stats=behavior[i];stats['decisions']+=1
      unique_actions[i].add(a);stats['unique_selected_legal_actions']=len(unique_actions[i])
      if diagnostics:
       diag=candidate.last_diagnostics
       stats['stochastic_aggregate_nonwait_probability_sum']=stats.get('stochastic_aggregate_nonwait_probability_sum',0)+diag['aggregate_nonwait_probability'][j]
       stats['flat_map_wait_decisions']=stats.get('flat_map_wait_decisions',0)+int(diag['flat_map_wait'][j])
      if np.count_nonzero(mask[j])==1:stats['forced_wait']+=1
      elif a==0:stats['voluntary_wait']+=1
      if a==0:stats['wait_elixir_sum']+=int(x[j,schema.header['elixir'][0]])
      elif a<2305:
       stats['deployments']+=1;stats['deployment_row_sum']+=((a-1)%576)//18
       card=int(x[j,schema.header['hand_card'][(a-1)//576]]);name=data['vocabulary'][card]
       stats['cards'][name]=stats['cards'].get(name,0)+1
     for p in panel:
      group=[i for i in ids if part[i][0]['name']==p['name']]
      if not group:continue
      if p['kind']=='checkpoint':
       values=processes[p['name']].choose(np.stack([env.observations[i,1-part[i][3]] for i in group]),np.stack([env.masks[i,1-part[i][3]] for i in group]))
      else:values=[bots[i](env.observations[i,1-part[i][3]],env.masks[i,1-part[i][3]]) for i in group]
      for i,a in zip(group,values):actions[i,1-part[i][3]]=a
     env.step(actions)
     for i in ids:
      counts[i]+=1;info=env.infos[i]
      if env.observations[i,:,49].any():raise ValueError('evaluation entity observation overflow')
      if any(info[[5,7,8,9,11]]):raise ValueError('evaluation invalid: fault/illegal/truncation/unsupported')
      if counts[i]>3000:raise ValueError('natural horizon watchdog; no partial score')
      if info[2]>=0:
       p,index,_,side=part[i];winner=int(info[2]);rows.append({'opponent':p['name'],'seed_index':index,'seat':side,'winner':winner,'points':.5 if winner==2 else float(winner==side),'decisions':counts[i],'natural':True,'truncated':False,'illegal':False,'fault':False,'behavior':behavior[i]});active.remove(i)
   finally:env.close()
  if len(rows)!=len(jobs):raise ValueError('incomplete schedule')
  result={**summarize(rows),'split':split,'seconds':time.monotonic()-started,'development_only':development,'submission_identity':verify(bundle)['identity'],'schedule_commitment':digest(schedule),'rows':rows,'inference_boundary':{'candidate':candidate.boundary,'checkpoint_opponents':{name:proc.boundary for name,proc in processes.items()},'only_public_arrays_on_wire':True}}
  dump(out,result);return result
 except Exception as exc:
  # Never create a cacheable successful-grade path for an invalid evaluation.
  # The sidecar retains natural completed rows separately from the failure.
  from .evaluation_v2 import classify
  try:identity=verify(bundle)['identity']
  except (ValueError,OSError,KeyError):identity=None
  failure={'valid':False,'raw_score_0_100':None,'failure_kind':classify(exc),'error':type(exc).__name__+': '+str(exc),'completed_natural_games':len(rows),'rows':rows,'submission_identity':identity,'schedule_commitment':digest(schedule),'development_only':development}
  dump(Path(out).with_suffix('.failure.json'),failure);raise
 finally:
  if candidate:candidate.close()
  for p in processes.values():p.close()
