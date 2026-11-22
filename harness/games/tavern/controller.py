"""Host session authority for Tavern: source snapshots, the worker-time ledger, the job queue,
immutable submissions and private final grading.

Every cloud call goes through `provider` (provider.ModalProvider); nothing here imports Modal.
A request is validated and its source snapshotted when it arrives; `tick` then starts it on a worker thread.
Worker time is reserved in the ledger immediately before a sandbox is created and settled only from the
provider's own receipt.

The session's budget mode (harness/budget.py, recorded in session.json and in the ledger) decides how jobs run.
`seconds`: one worker thread runs one job at a time, training, practice and reconciliation alike, within a
number of training-worker seconds (the native ledger: an unknown end blocks every launch). `usd`: every admitted
job gets its own worker thread at once, and admission keeps the active jobs (queued, running, and allocations
whose end is unknown) within the mode's limits; each training run holds its whole dollar grant from admission
until its receipt settles it, so holds and charges never pass the budget (ledger.DollarLedger). Each job keeps
its own worker, sandbox record, hand-off job and ledger row; a reconciliation runs on a thread of its own.

Results leave a worker through a private hand-off Volume (harness.handoff), not over the host's link while
the worker is paid for: the worker publishes its files there as root before it ends, the host ends the
worker and then collects them from the Volume with retries and no deadline. Only when nothing was published
and the worker is still up does the host read the worker directly, as before; the job records which way its
results came (`handoff`: 'volume' or 'live') and why a fallback happened.
"""
from harness.policy import ENVIRONMENT
from harness import budget as budgets, handoff, hardware, workers
from pathlib import Path
import copy,json,shutil,tarfile,threading,time,uuid
from simulators.tavern.autoresearch.contract import EDITABLE,write,sha,validate
from .ledger import DollarLedger, Ledger, recorded_mode
REMOTE='/opt/task/simulators/tavern/autoresearch'
OVERHEAD=150  # worker seconds reserved beyond training for setup, export and teardown
TRAIN_GRACE=65  # seconds the training command may run beyond its training time (start-up, last update, export)
# A worker's own lifetime ends inside its reservation, and leaves time after the training command's limit to
# package what an overrunning trainer last exported.
LIFETIME_MARGIN=OVERHEAD-10
FINALIZE_SECONDS=30  # the trusted packaging of a trainer that ended early (finalize_entry.py)
STOP_GRACE=90  # after a stop request, how long packaging may take before the worker itself is ended
MIN_TRAIN=10
SMOKE_TRAIN=180
SMOKE_RUNS=2
ARTIFACT=['weights.npz','identity.json','metrics.json','summary.json','resume.pt']
LIFETIME=dict(practice=900,grade=1800)  # CPU evaluation sandboxes; not GPU worker time
LIVE=('queued','reserved','running')
CORES=4  # CPU cores of every Tavern worker (training and evaluation alike; provider.ModalProvider.create)
# The in-datacenter publish of a finished job's files to the hand-off Volume. A finished trainer's publish runs
# inside its own training command (limit + PUBLISH_SECONDS); a packaged early end publishes after finalize_entry.py,
# both inside LIFETIME_MARGIN. A cut-off run needs worker setup + TRAIN_GRACE + FINALIZE_SECONDS + PUBLISH_SECONDS
# to fit in LIFETIME_MARGIN (140): 20 s leaves 25 s for setup. The publish copies a few MB inside the datacenter.
PUBLISH_SECONDS=20
COLLECT_ROUNDS=12  # host-side collection rounds (each retried by harness.handoff); no worker is waiting on them
TRAINER_LOG='/job/trainer-stdout.log'
EVALUATION_ROOT='/private/evaluation'
EVALUATION_FILES={'lifecycle.json':EVALUATION_ROOT+'/lifecycle.json','qualification-privacy.json':EVALUATION_ROOT+'/qualification-privacy.json',
 'qualification-terminal-replay.json':EVALUATION_ROOT+'/qualification-terminal-replay.json',
 'result.json':EVALUATION_ROOT+'/screen/result.json','episodes.json':EVALUATION_ROOT+'/screen/episodes.json'}
# Evaluation workers hand off through their own Volume: researcher code runs as root in a training worker, so it
# must never mount the Volume that holds evaluation recordings.
EVALUATION_HANDOFF='evaluation-handoff'
PUBLISHED,UNPUBLISHED='\x1ehandoff-published ','\x1ehandoff-failed '

# Runs in a worker as root: python -c PACK SPEC. Tars the requested folders (plain files only), then runs the
# shared publish script (harness.handoff) on the tars and on whichever listed files exist. Prints the manifest.
PACK=r'''import json,os,subprocess,sys,tarfile
spec=json.loads(sys.argv[1]);pairs=[]
for name,t in sorted(spec.get('tars',{}).items()):
 with tarfile.open(t['path'],'w') as archive:
  for top in t.get('include') or ['.']:
   base=os.path.normpath(os.path.join(t['root'],top))
   if not os.path.isdir(base):continue
   for folder,dirs,names in os.walk(base):
    dirs.sort()
    for n in sorted(names):
     path=os.path.join(folder,n)
     if os.path.isfile(path) and not os.path.islink(path):archive.add(path,arcname=os.path.relpath(path,t['root']),recursive=False)
 pairs.append(name+'='+t['path'])
for name,path in sorted(spec.get('files',{}).items()):
 if name in spec.get('optional',()) and not os.path.isfile(path):continue
 pairs.append(name+'='+path)
sys.exit(subprocess.run(spec['publish']+pairs).returncode)'''

# Runs in a training worker as root: python -c RUN_AND_PUBLISH SPEC. Runs the trusted trainer (passing its output
# through and into a log), ends it at its time limit, and when it finished by itself publishes its export to the
# hand-off Volume at once, before the worker idles: a host that is unreachable at that moment loses nothing.
# The last line it prints says whether it published. The spec is one argument, so KILL_TRAINER never matches it.
RUN_AND_PUBLISH=r'''import json,subprocess,sys,threading
spec=json.loads(sys.argv[1]);out=sys.stdout.buffer;expired=[]
log=open(spec['log'],'wb');trainer=subprocess.Popen(spec['train'],stdout=subprocess.PIPE)
def pump():
 for line in trainer.stdout:out.write(line);out.flush();log.write(line);log.flush()
def expire():expired.append(1);trainer.kill()
reader=threading.Thread(target=pump,daemon=True);reader.start()
timer=threading.Timer(spec['limit'],expire);timer.daemon=True;timer.start()
code=trainer.wait();timer.cancel();reader.join(10);log.flush()
if code:
 sys.stderr.write('trainer ended with %d%s\n'%(code,' at its time limit' if expired else ''));sys.stderr.flush()
 sys.exit(code if code>0 else 128-code)
done=subprocess.run([sys.executable,'-c',spec['pack_script'],json.dumps(spec['pack'])],capture_output=True)
lines=done.stdout.decode(errors='replace').strip().splitlines()
published,unpublished=spec['marks']
if done.returncode==0 and lines:out.write(b'\n'+(published+lines[-1]).encode()+b'\n')
else:out.write(b'\n'+(unpublished+json.dumps((done.stdout+done.stderr).decode(errors='replace')[-1500:])).encode()+b'\n')
out.flush()'''

def practice_folder(name,attempt):return 'practice-'+name+('' if attempt==1 else '-attempt'+str(attempt))

def split_publication(text):
 """The runner's own last line says whether it published; what comes before it is the trainer's output.
 Returns (trainer output, manifest or None, why nothing was published)."""
 head,_,last=text.rstrip('\n').rpartition('\n')
 if last.startswith(PUBLISHED):
  try:return head,json.loads(last[len(PUBLISHED):]),None
  except ValueError:return text,None,'the worker printed an unreadable manifest'
 if last.startswith(UNPUBLISHED):return head,None,'publish failed in the worker: '+last[len(UNPUBLISHED):][-600:]
 return text,None,'the worker reported no publication'

def unpack(archive,dest):
 """Extract a worker's tar of plain files; nothing may land outside `dest`."""
 dest=Path(dest)
 with tarfile.open(archive) as tar:
  for member in tar:
   relative=Path(member.name)
   if not member.isfile() or relative.is_absolute() or '..' in relative.parts:raise ValueError('Invalid retained path')
   target=dest/relative;target.parent.mkdir(parents=True,exist_ok=True)
   with tar.extractfile(member) as source,target.open('wb') as stream:shutil.copyfileobj(source,stream)

def pack_spec(job,files=None,optional=(),tars=None):
 return dict(publish=handoff.publish_command(job,{},python='python'),files=files or {},optional=list(optional),tars=tars or {})

def require_gpu_device(recipe):
 """A training run always holds a GPU worker; a config that trains elsewhere would idle it for the whole run."""
 device=recipe.get('device','cuda')
 if not isinstance(device,str) or not (device=='cuda' or device.startswith('cuda:')):
  raise ValueError(f'Training runs on an {hardware.GPU} GPU worker, so the config\'s "device" must be "cuda" '
                   f'(got {device!r}); a CPU-only config would hold the GPU idle for the whole run')

class Controller:
 def __init__(self,release,out,provider):
  self.release=Path(release);self.out=Path(out);self.provider=provider
  settings=self.out/'session.json'
  self.settings=json.loads(settings.read_text()) if settings.exists() else {}
  # The budget mode (harness/budget.py). In usd mode `cap` is the training-worker seconds the dollar budget buys.
  self.mode=self.settings.get('budget_mode',budgets.SECONDS);self.parallel=self.mode==budgets.USD
  self.limits=self.settings.get('budget_limits') or budgets.limits(self.mode)
  self.rate=self.settings.get('usd_per_second');self.cap_usd=self.settings.get('budget_usd')
  self.cap=self.cap_usd/self.rate if self.parallel else self.settings.get('gpu_limit_seconds',600)
  self.smoke=self.settings.get('mode','smoke' if self.cap==600 else 'research')=='smoke'
  self.workspace=self.out/'workspace'
  # `thread`: seconds mode's one worker thread (its job is `current`), or usd mode's reconciliation thread;
  # `threads`: usd mode's job threads by (kind, name).
  self.lock=threading.RLock();self.local=threading.local();self.thread=None;self.current=None;self.threads={}
  self.stops=set();self.halts=set();self.training=set();self.sandboxes={};self.recover_at=0.;self.recovering=threading.Lock()
  self.start_poll_seconds=2.0;self.abandon_poll_seconds=1.0  # provider-receipt polling (harness.workers)
  self.volumes={};self.collect_wait=5.0  # hand-off Volumes by kind, opened on first use; seconds between collection reads
  recorded=recorded_mode(self.out/'worker-ledger.sqlite')
  if recorded not in (None,self.mode):raise ValueError(f'This session was created in {recorded} mode; it cannot be reopened in {self.mode} mode')
  self.ledger  # the first connection fixes the session cap (and in usd mode its price); a different one is refused
  self.state=json.loads((self.out/'state.json').read_text()) if (self.out/'state.json').exists() else dict(runs={},practice={},queue=[],submission=None,finalized=False)
  if not (self.out/'state.json').exists():self.save()
 @property
 def ledger(self):
  # A SQLite connection stays on the thread that opened it; writers serialize on BEGIN IMMEDIATE.
  if not hasattr(self.local,'ledger'):
   path=self.out/'worker-ledger.sqlite'
   self.local.ledger=DollarLedger(path,self.cap_usd,self.rate) if self.parallel else Ledger(path,self.cap)
  return self.local.ledger
 def save(self):
  with self.lock:write(self.out/'state.json',self.state)
 def mark(self,row,**values):
  with self.lock:row.update(values);self.save()
 def snapshot(self):
  with self.lock:return copy.deepcopy(self.state)
 def budget(self):return self.ledger.snapshot()
 def status(self):return dict(state=self.snapshot(),budget=self.budget())
 def busy(self):
  """A worker thread (a job's, or a reconciliation) is alive."""
  return (self.thread is not None and self.thread.is_alive()) or any(t.is_alive() for t in list(self.threads.values()))
 def running(self):
  """The (kind, name) of every job a worker thread is working on now."""
  if not self.parallel:
   current=self.current
   return [current] if current and self.busy() else []
  return [key for key,thread in list(self.threads.items()) if thread.is_alive()]
 def reconciling(self):
  """The thread of a reconciliation under way, or None."""
  thread=self.thread
  return thread if thread is not None and thread.is_alive() and self.current is None else None
 def join(self,timeout):
  """Wait at most `timeout` seconds in all for every worker thread to end."""
  deadline=time.monotonic()+timeout
  for thread in [self.thread]+list(self.threads.values()):
   if thread is not None:thread.join(timeout=max(0.,deadline-time.monotonic()))
 def unresolved(self):
  """Reservations without a provider end time that no running job owns."""
  # usd mode: the ledger is read before the owners, so a job that reserves in between is never taken for an orphan.
  rows=self.budget()['allocations'] if self.parallel else None
  owners={name for kind,name in self.running() if kind=='train'}
  return [row['name'] for row in (self.budget()['allocations'] if rows is None else rows) if row['actual'] is None and row['name'] not in owners]
 def queued_seconds(self):
  """Reserved seconds (training plus OVERHEAD) of admitted runs the ledger does not hold yet."""
  held={row['name'] for row in self.budget()['allocations']}
  with self.lock:return sum(r['seconds']+OVERHEAD for n,r in self.state['runs'].items() if r['status']=='queued' and n not in held)
 def active_jobs(self):
  """Every job holding one of usd mode's slots: queued, reserved or running, and every allocation whose end is unknown."""
  with self.lock:
   jobs={('train',n) for n,r in self.state['runs'].items() if r['status'] in LIVE}
   jobs|={('practice',n) for n,r in self.state['practice'].items() if r['status'] in ('queued','running')}
  return jobs|{('train',n) for n in self.unresolved()}
 def require_slot(self,kind):
  """usd mode: refuse a job the mode's concurrency limits have no room for (harness.budget.PARALLEL)."""
  active=self.active_jobs();limits=self.limits;jobs=limits['max_concurrent_jobs']
  if len(active)>=jobs:raise ValueError(f'At most {jobs} jobs at a time; wait for or stop one (an allocation whose end is unknown counts)')
  gpus=sum(k=='train' for k,_ in active)+(kind=='train')
  if gpus>limits['max_concurrent_gpus']:raise ValueError(f'At most {limits["max_concurrent_gpus"]} GPUs at a time: each training run holds one; wait for or stop one')
  cores=limits.get('max_concurrent_cpu_cores')
  if cores is not None and (len(active)+1)*CORES>cores:raise ValueError(f'At most {cores} CPU cores at a time across your active jobs')

 # Requests: validate and snapshot now; the worker thread runs them later.
 def admit(self,name,seconds,resume_from=None):
  """Snapshot the native trainer tree as run `name` and queue it. Nothing is reserved yet."""
  with self.lock:
   if self.state['finalized']:raise ValueError('Finalized')
   if not isinstance(name,str) or not name.isascii() or not name.isalnum() or name in self.state['runs']:raise ValueError('Fresh alphanumeric run name required')
   if self.smoke and sum(r['status']!='cancelled' for r in self.state['runs'].values())>=SMOKE_RUNS:raise ValueError('Smoke at most two attempts; failed attempts count')
   live=[n for n,r in self.state['runs'].items() if r['status'] in LIVE]
   if self.parallel:self.require_slot('train')
   elif live:raise ValueError('One training run at a time; wait for or stop '+live[0])
   seconds=min(float(seconds),SMOKE_TRAIN if self.smoke else self.cap-OVERHEAD)
   if not seconds>=MIN_TRAIN:raise ValueError('At least10 seconds')
   budget=self.budget()
   if budget['closed']:raise ValueError('Session finalized')
   if self.parallel:  # every queued run holds its grant too
    if (seconds+OVERHEAD+self.queued_seconds())*self.rate>budget['available_usd']+1e-9:raise ValueError('Allocation exceeds the remaining session budget')
   else:
    if budget['unresolved_seconds']:raise ValueError('Unresolved allocation blocks launch')
    if seconds+OVERHEAD>budget['available_seconds']:raise ValueError('GPU ceiling')
   recipe=json.loads((self.workspace/('recipes/smoke.json' if self.smoke else 'recipes/starter.json')).read_text());validate(recipe);require_gpu_device(recipe)
   if resume_from is not None and self.state['runs'].get(resume_from,{}).get('status')!='completed':raise ValueError('Resume requires registered completed artifact')
   target=self.out/'runs'/name;target.mkdir(parents=True,exist_ok=False);source=target/'source';source.mkdir()
   for f in EDITABLE:shutil.copy2(self.workspace/'starter'/f,source/f)
   write(target/'recipe.json',recipe)
   self.state['runs'][name]=dict(status='queued',seconds=seconds,resume_from=resume_from,source={f:sha(source/f) for f in EDITABLE},recipe_sha256=sha(target/'recipe.json'),submission_at_launch=self.state['submission'])
   self.state['queue'].append(['train',name]);self.save()
   return dict(queued=name,train_seconds=seconds,reserved_seconds=seconds+OVERHEAD)
 def practice(self,name):
  """Queue the fixed public practice schedule for a completed run."""
  with self.lock:
   if self.state['finalized']:raise ValueError('Finalized')
   if self.state['runs'].get(name,{}).get('status')!='completed':raise ValueError('Completed registered artifact required')
   row=self.state['practice'].get(name)
   if row and row['status'] in ('queued','running'):raise ValueError('Practice for this run is already active')
   if row and row['status']=='completed':raise ValueError('Public practice for this run is complete; its schedule is fixed, read the result')
   if self.parallel:self.require_slot('practice')
   attempt=1
   while (self.out/practice_folder(name,attempt)).exists():attempt+=1  # an earlier failed attempt keeps its folder
   self.state['practice'][name]=dict(status='queued',attempt=attempt)
   self.state['queue'].append(['practice',name]);self.save()
   return dict(queued='practice-'+name,attempt=attempt)
 def stop(self,kind,name,now=False):
  """End a queued, running or unresolved job. A training run the researcher stops keeps the network it last
  exported; `now` (the session is closing: nothing unsubmitted is kept) ends its worker at once instead."""
  with self.lock:
   row=self.state['runs' if kind=='train' else 'practice'].get(name)
   if row is None:raise ValueError('Unknown run')
   if row['status']=='queued' and [kind,name] in self.state['queue']:
    self.state['queue'].remove([kind,name]);row['status']='cancelled';self.save()
    return dict(stopped=name,charged=False)
   uncertain=kind=='train' and name in self.unresolved()  # its worker's end is unknown: resolve it now
   if not uncertain and row['status'] not in LIVE:raise ValueError('Run is not active')
   key=(kind,name);self.stops.add(key);sandbox=self.sandboxes.get(key)
  if uncertain:return self.resolve(name)
  if sandbox is None:return dict(stopping=name)
  # The worker thread is blocked on this sandbox's training command. Ending only the trainer process releases it
  # with the worker still up, and `execute` then packages the last export. If no trainer was ended (the worker is
  # still queued or being set up, or the request failed), or for any other job, ending the worker itself is what
  # releases the job.
  kept=False
  if kind=='train' and not now and key in self.training and hasattr(self.provider,'kill_trainer'):
   try:kept=self.provider.kill_trainer(sandbox)>0
   except Exception:kept=False
  if not kept:
   self.end_worker(key,sandbox)
   return dict(stopping=name)
  timer=threading.Timer(STOP_GRACE,self.end_worker,args=(key,sandbox));timer.daemon=True;timer.start()
  return dict(stopping=name,note='The trainer was ended. If it had exported a network, the run finishes as complete '
              'with stopped_early and keeps that network; otherwise it finishes as stopped. Check ar status.')
 def end_worker(self,key,sandbox):
  """End a stopped job's worker, unless its job already released it. Nothing it exported is packaged."""
  with self.lock:
   if self.sandboxes.get(key) is not sandbox:return
   self.halts.add(key)
  self.provider.terminate(sandbox)
 def overran(self,began,limit):
  """True when the training command ended because it reached its own time limit."""
  return time.monotonic()-began>=limit-2
 def package_last_export(self,sb,failure):
  """Have the worker write the summary for what its ended trainer last exported; without a complete export the
  trainer's own failure stands."""
  try:
   if hasattr(self.provider,'kill_trainer'):self.provider.kill_trainer(sb)  # nothing may still be writing
   text=self.provider.command(sb,['python',REMOTE+'/finalize_entry.py','--out','/job/train'],timeout=FINALIZE_SECONDS)
  except Exception:raise failure from None
  return 'The trainer ended early: '+str(failure)[-4000:]+'\nPackaged its last export: '+text

 # The hand-off Volume: workers publish there before they end; the host collects afterwards.
 def handoff_volume(self,kind='train'):
  """This root's private hand-off Volume for training workers, or the separate one its evaluation workers mount.
  Opened (and created) on first use, before any reservation."""
  with self.lock:
   if kind not in self.volumes:self.volumes[kind]=self.provider.volume(self.out if kind=='train' else self.out/EVALUATION_HANDOFF)
   return self.volumes[kind]
 def handoff_job(self,kind,name):
  """A fresh job name nothing that ran earlier could have known."""
  return kind+'-'+name+'-'+uuid.uuid4().hex[:12]
 def published(self,volume,job):
  """The job's manifest if the Volume already holds it; one short read."""
  try:return volume.manifest(job)
  except Exception:return None
 def publish(self,sb,volume,job,pack):
  """Have the worker publish `pack` to the hand-off Volume (a short in-datacenter exec, as root). Returns the
  manifest, or None and why nothing was published."""
  try:text=self.provider.command(sb,['python','-c',PACK,json.dumps(pack)],timeout=PUBLISH_SECONDS)
  except Exception as exc:
   manifest=None if str(exc).startswith('Remote command failed') else self.published(volume,job)  # only the reply may be lost
   return (manifest,None) if manifest else (None,'publish failed: '+str(exc)[-600:])
  try:return json.loads(text.strip().splitlines()[-1]),None
  except (ValueError,IndexError):return None,'publish returned no manifest'
 def collect(self,volume,job,dest,names=None):
  """Fetch a published job's files after its worker has ended: every read is retried, none has a worker deadline,
  and every byte is checked against the job's manifest (harness.handoff)."""
  for attempt in range(COLLECT_ROUNDS):
   try:return volume.collect(job,dest,names,wait=self.collect_wait)
   except (ValueError,handoff.NotPublished):raise
   except Exception:
    if attempt+1==COLLECT_ROUNDS:raise
    time.sleep(min(60.,self.collect_wait*(attempt+1)))
 def end_published(self,key,sb):
  """End a worker whose results are in the hand-off Volume: nothing more is read from it."""
  with self.lock:
   if self.sandboxes.get(key) is sb:self.sandboxes.pop(key)
  self.provider.terminate(sb)
 def training_pack(self,job,diagnostics):
  files={f:'/job/train/'+f for f in ARTIFACT};files['trainer-stdout.log']=TRAINER_LOG
  return pack_spec(job,files,['trainer-stdout.log'],self.diagnostics_tar() if diagnostics else None)
 def diagnostics_tar(self):
  return {'diagnostics.tar':dict(root='/job/train',include=['faults','action-traces'],path='/job/diagnostics.tar')}
 def keep_diagnostics(self,key,sb,volume,job,target,published,released):
  """Keep the run's fault records and action traces (record_actions) through the Volume, or live from a worker that
  is still up when nothing could be published. Returns True when it ended the worker."""
  try:
   if not (published and 'diagnostics.tar' in published):
    if released:raise ValueError('The ended worker published no diagnostics')
    job=job+'-diagnostics';published,reason=self.publish(sb,volume,job,pack_spec(job,tars=self.diagnostics_tar()))
    if not published:
     self.live_diagnostics(sb,target);write(target/'diagnostics-handoff.json',dict(handoff='live',handoff_fallback=reason))
     return False
    self.end_published(key,sb);released=True
   got=self.collect(volume,job,target/'handoff',['diagnostics.tar'])
   unpack(got['diagnostics.tar']['path'],target/'diagnostics')
   write(target/'diagnostics-handoff.json',dict(handoff='volume',job=job,archive=got['diagnostics.tar']))
   Path(got['diagnostics.tar']['path']).unlink()
  except Exception as exc:write(target/'diagnostic-download-failure.json',dict(error=str(exc),evidence_incomplete=True))
  return released
 def live_diagnostics(self,sb,target):
  listing=self.provider.command(sb,['python','-c',"import pathlib,json; r=pathlib.Path('/job/train'); print(json.dumps([str(p.relative_to(r)) for p in r.rglob('*') if p.is_file() and ('faults' in p.parts or 'action-traces' in p.parts)]))"],timeout=10)
  for relative in json.loads(listing):
   if '..' in Path(relative).parts or Path(relative).is_absolute():raise ValueError('Invalid diagnostic path')
   dest=target/'diagnostics'/relative;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(self.provider.read(sb,'/job/train/'+relative))
 def collect_run(self,volume,job,target,published):
  """The run's export into artifact/, the trainer's own output into handoff/. Returns that output, if published."""
  self.collect(volume,job,target/'artifact',ARTIFACT)
  if 'trainer-stdout.log' not in published:return None
  return Path(self.collect(volume,job,target/'handoff',['trainer-stdout.log'])['trainer-stdout.log']['path']).read_text(errors='replace')
 def register(self,name):
  """Check a collected export's learning and identity, and add the source it was trained from."""
  target=self.out/'runs'/name;row=self.state['runs'][name]
  summary=json.loads((target/'artifact/summary.json').read_text())
  if not summary['finite_losses'] or not summary['changed_weights']:raise ValueError('Learning verification failed')
  ident=json.loads((target/'artifact/identity.json').read_text())
  if ident['source']!=row['source'] or ident['binding']!=json.loads((self.release/'binding.json').read_text()):raise ValueError('Artifact identity mismatch')
  for f in EDITABLE:shutil.copy2(target/'source'/f,target/'artifact'/f)
  return summary
 def recover_published(self,name,job):
  """A run whose host was interrupted may still have published its export before its worker ended: collect it."""
  target=self.out/'runs'/name
  try:
   volume=self.handoff_volume();manifest=self.published(volume,job)
   if not manifest:return None
   text=self.collect_run(volume,job,target,manifest)
   if text is not None and not (target/'train.log').exists():(target/'train.log').write_text(text)
   with self.lock:summary=self.register(name);self.verify_artifact(name)
  except Exception:return None
  early=(dict(stopped_early=True) if ('train',name) in self.stops else dict(cut_short=True)) if summary.get('ended_early') else {}
  return dict(summary=summary,handoff='volume',**early)
 def resolve(self,name):
  """Stop an uncertain run now: terminate its recorded worker and settle it from the provider's record."""
  reconciliation=self.reconciling()
  if reconciliation is not None:reconciliation.join(timeout=120)  # a reconciliation already under way
  self.recover([name])
  row=self.snapshot()['runs'][name];held=name in self.unresolved()
  reply=dict(stopped=name,resolved=not held,status=row['status'],charged_seconds=next(
   (r['actual'] for r in self.budget()['allocations'] if r['name']==name),None))
  if held:reply['reason']='The provider has not reported this worker\'s end; its hold stays until it does'
  # The settled run's own failure is `job_error`: a top-level `error` would report the stop as failed.
  if row.get('error'):reply['job_error']=row['error'][-600:]
  return reply
 def tick(self):
  """Start the next queued job, or reconcile a recorded allocation whose end is still unknown."""
  if self.parallel:return self.tick_parallel()
  if self.busy():return
  self.thread=None
  if time.time()>=self.recover_at and self.unresolved():
   self.recover_at=time.time()+20;target=self.recover
  else:
   with self.lock:
    if not self.state['queue']:return
    kind,name=self.state['queue'].pop(0);self.current=(kind,name);self.save()
   target=lambda:self.work(kind,name)
  self.thread=threading.Thread(target=target,daemon=True);self.thread.start()
 def tick_parallel(self):
  """usd mode: start every admitted job on its own worker thread (admission kept them within the limits), and
  reconcile allocations whose end is unknown on a thread of their own."""
  with self.lock:
   for key in [key for key,thread in self.threads.items() if not thread.is_alive()]:del self.threads[key]
   started=bool(self.state['queue'])
   while self.state['queue']:
    kind,name=self.state['queue'].pop(0);thread=threading.Thread(target=self.work,args=(kind,name),daemon=True)
    self.threads[(kind,name)]=thread;thread.start()
   if started:self.save()
  if self.reconciling() is None and time.time()>=self.recover_at and self.unresolved():
   self.recover_at=time.time()+20;self.thread=threading.Thread(target=self.recover,daemon=True);self.thread.start()
 def work(self,kind,name):
  try:
   if kind=='train':self.execute(name)
   else:self.execute_practice(name)
  except BaseException as exc:
   record=self.out/'runs'/name/'host-error.json' if kind=='train' else self.out/('practice-'+name+'-host-error.json')
   write(record,dict(type=type(exc).__name__,message=str(exc)[-6000:]))
  finally:self.current=None

 def receipt(self,sb,gpu,name,terminate=True):
  if terminate:self.provider.terminate(sb)  # a worker whose results went through the Volume was already ended
  receipt=None
  for _ in range(100):
   receipt=self.provider.allocation(self.provider.app_id,self.provider.identity(sb))
   if receipt and receipt.get('finished_at',0)>0:break
   time.sleep(.2)
  write(self.out/(name+'-provider.json'),receipt)
  if gpu and receipt and receipt.get('finished_at',0)>0:self.ledger.settle(name,receipt)
  return receipt
 def recover(self,names=None):
  """Reconcile recorded provider identities; never repeat a paid create/run.

  A recorded worker the provider never started is terminated and settled at 0. One that started and was
  abandoned by its job (or is being stopped) is terminated now and charged from its receipt. One still running
  for a job that is itself still live (the host was interrupted) keeps its hold until the provider ends it.
  """
  with self.recovering:
   repairs=[]
   for name in self.unresolved():
    if names is not None and name not in names:continue
    if self.parallel and ('train',name) in self.running():continue  # a job thread owns it now
    target=self.out/'runs'/name;identity=target/'sandbox.json'
    if not identity.exists():
     repairs.append(dict(name=name,status='unknown-create; hold retained'));continue
    info=json.loads(identity.read_text());read=lambda:self.provider.allocation(info['app_id'],info['id'])
    receipt=read() or {}
    with self.lock:row=self.state['runs'][name];live=row['status'] in LIVE and ('train',name) not in self.stops
    recorded=receipt.get('provider_id')==info['id']
    if not receipt.get('finished_at',0) and recorded and (workers.never_started(receipt) or not live):
     receipt=workers.abandon(lambda:self._terminate_recorded(name,info['id']),read,poll=self.abandon_poll_seconds)
     if receipt.get('provider_id')==info['id'] and workers.never_started(receipt):
      stopped=('train',name) in self.stops
      self.ledger.settle_never_started(name,receipt,'stopped' if stopped else 'failed');write(target/'never-started.json',receipt)
      with self.lock:
       row.update(status='stopped' if stopped else 'failed',never_started=True,charged_seconds=0,
                  error=(row.get('error')+' | ' if row.get('error') else '')+workers.NEVER_STARTED)
       self.save()
      repairs.append(dict(name=name,status=row['status'],never_started=True,charged_seconds=0,provider_receipt=receipt));continue
    if not receipt or not receipt.get('finished_at',0):
     repairs.append(dict(name=name,status='provider-live-or-unknown; hold retained'));continue
    self.ledger.settle(name,receipt);write(target/'recovery-provider.json',receipt)
    with self.lock:job=row.get('handoff_job') if row['status'] in LIVE and ('train',name) not in self.stops else None
    handed=self.recover_published(name,job) if job else None  # outside the lock: collection may be slow
    with self.lock:
     if row['status'] in LIVE and handed:row.update(status='completed',recovered=True,**handed)
     elif row['status'] in LIVE:  # the host stopped while this allocation was live
      try:row.update(status='completed',summary=self.verify_artifact(name),recovered=True)
      except (ValueError,KeyError,FileNotFoundError,json.JSONDecodeError):row.update(status='stopped' if ('train',name) in self.stops else 'failed',error='Interrupted allocation ended without complete verified artifact; paid time retained')
     self.save()
    repairs.append(dict(name=name,status=self.state['runs'][name]['status'],provider_receipt=receipt))
   return dict(reconciliation=repairs,budget=self.budget())
 def _terminate_recorded(self,name,sandbox_id):
  sandbox=self.sandboxes.get(('train',name))
  if sandbox is not None:self.provider.terminate(sandbox)
  else:self.provider.terminate_id(sandbox_id)  # after a host restart only the recorded identity is left
 def verify_artifact(self,name):
  """The registered artifact is still what this run exported for the source it snapshotted."""
  artifact=self.out/'runs'/name/'artifact';row=self.state['runs'][name]
  summary=json.loads((artifact/'summary.json').read_text());ident=json.loads((artifact/'identity.json').read_text())
  if sha(artifact/'weights.npz')!=summary['weights_sha256'] or sha(artifact/'identity.json')!=summary['identity_sha256'] or ident['source']!=row['source']:raise ValueError('Incomplete/drifted artifact')
  return summary

 def run(self,name,seconds=90,resume_from=None):
  """Synchronous training run. `resume_from` continues a completed run of the exact same source and recipe."""
  self.admit(name,seconds,resume_from)
  with self.lock:self.state['queue'].remove(['train',name]);self.save()
  return self.execute(name)
 def execute(self,name):
  key=('train',name);row=self.state['runs'][name];seconds=row['seconds'];target=self.out/'runs'/name;source=target/'source';sb=None;never=None
  record=self.settings.get('record_actions',False);published=None;released=False;volume=job=None;route={}
  try:
   if key in self.stops:raise RuntimeError('Stopped before a worker was allocated')
   self.provider.ready()  # the image build is not worker time and must not strand a reservation
   volume=self.handoff_volume()  # created before the reservation too
   self.ledger.reserve(name,seconds+OVERHEAD)
  except BaseException as exc:
   self.mark(row,status='stopped' if key in self.stops else 'failed',error=str(exc));raise
  job=self.handoff_job('train',name);self.mark(row,status='reserved',handoff_job=job,handoff_volume=volume.name)
  try:
   sb=self.provider.create(True,int(seconds+LIFETIME_MARGIN),volume);self.sandboxes[key]=sb
   write(target/'sandbox.json',dict(id=self.provider.identity(sb),app_id=self.provider.app_id,gpu=hardware.GPU,timeout_seconds=seconds+LIFETIME_MARGIN))
   self.mark(row,status='running')
   # A created worker can wait in the provider's queue for a GPU for many minutes. Wait for it (queued time is
   # never charged); if it never starts, terminate it and settle the allocation at 0 from the provider's record.
   identity=self.provider.identity(sb);info=lambda:self.provider.allocation(self.provider.app_id,identity)
   if workers.await_start(info,stop=lambda:key in self.stops,poll=self.start_poll_seconds) is None:
    final=workers.abandon(lambda:self.provider.terminate(sb),info,poll=self.abandon_poll_seconds)
    if final.get('provider_id')==identity and workers.never_started(final):
     stopped=key in self.stops
     reason=('Stopped by request while queued; ' if stopped else '')+workers.NEVER_STARTED
     self.ledger.settle_never_started(name,final,'stopped' if stopped else 'failed');write(target/'never-started.json',final)
     never=workers.NeverStarted(final,reason);raise never
    raise RuntimeError('The worker started only after it was abandoned; charged from its provider receipt')
   if key in self.stops:raise RuntimeError('Stopped by request')
   self.provider.command(sb,handoff.prepare_command('python'),timeout=20)  # /handoff is root-only before any researcher code
   self.provider.upload(sb,source,'/job/source');self.provider.copy(sb,target/'recipe.json','/job/recipe.json')
   resume_args=[]
   if row.get('resume_from'):
    self.provider.copy(sb,self.out/'runs'/row['resume_from']/'artifact/resume.pt','/job/resume.pt');resume_args=['--resume','/job/resume.pt']
   trace_args=['--record-actions'] if record else []
   train=['python',REMOTE+'/train_entry.py','--source','/job/source','--recipe','/job/recipe.json','--out','/job/train','--seconds',str(seconds),'--device','cuda:0']+resume_args+trace_args
   pack=self.training_pack(job,record);reason=None
   limit=int(seconds+TRAIN_GRACE);began=time.monotonic();early={};self.training.add(key)
   try:
    # The worker ends the trainer at `limit` itself and, if it finished, publishes its export before replying.
    text=self.provider.command(sb,['python','-c',RUN_AND_PUBLISH,json.dumps(dict(train=train,limit=limit,log=TRAINER_LOG,pack=pack,pack_script=PACK,marks=[PUBLISHED,UNPUBLISHED]))],timeout=limit+PUBLISH_SECONDS)
    text,published,reason=split_publication(text)
   except Exception as failure:
    # A trainer the researcher stopped, or one cut off at its time limit, keeps the network it last exported, as
    # Generals and Dou Dizhu package a cut-off run. A trainer that failed by itself, and an ended worker, stay failures.
    stopped=key in self.stops and key not in self.halts
    if not stopped and not self.overran(began,limit):
     # Only the reply may have been lost: a trainer that finished has already published its export.
     published=None if str(failure).startswith('Remote command failed') else self.published(volume,job)
     if not published:raise
     text=None
    else:
     text=self.package_last_export(sb,failure);early=dict(stopped_early=True) if stopped else dict(cut_short=True)
     published,reason=self.publish(sb,volume,job,pack)
   finally:self.training.discard(key)
   if text is not None:(target/'train.log').write_text(text)
   if published:
    self.end_published(key,sb);released=True  # the results are in the Volume: the worker's paid time ends here
    route=dict(handoff='volume')
    try:logged=self.collect_run(volume,job,target,published)
    except (ValueError,KeyboardInterrupt):raise
    except Exception as exc:raise RuntimeError(f'Published to hand-off Volume {volume.name} as {job}, which keeps it, but not collected: {exc}') from exc
    if text is None:(target/'train.log').write_text(logged or '')
   else:  # nothing was published and the worker is still up: read it directly, as before the Volume
    route=dict(handoff='live',handoff_fallback=reason)
    self.provider.download(sb,'/job/train',target/'artifact',ARTIFACT)
   summary=self.register(name)
   self.mark(row,status='completed',summary=summary,**early,**route)
  except BaseException as exc:
   self.mark(row,status='stopped' if key in self.stops else 'failed',error=str(exc),**route,**(dict(never_started=True,charged_seconds=0) if exc is never else {}));raise
  finally:
   try:
    if sb and never is None and record:released=self.keep_diagnostics(key,sb,volume,job,target,published,released)
    if sb and never is None:self.receipt(sb,True,name,terminate=not released)  # a never-started worker is already terminated and settled
   finally:self.sandboxes.pop(key,None)
  return row
 def execute_practice(self,name):
  key=('practice',name);row=self.state['practice'][name]
  try:
   if key in self.stops:raise RuntimeError('Stopped before a worker was allocated')
   self.mark(row,status='running')
   result=self.evaluate(name);self.mark(row,status='completed')
   return result
  except BaseException as exc:
   self.mark(row,status='stopped' if key in self.stops else 'failed',error=str(exc));raise
  finally:self.stops.discard(key)
 def evaluate(self,name,private=False):
  """The mixed-v2 evaluator with its terminal and privacy proof: public practice or the private final."""
  from simulators.tavern.autoresearch.rules_enabled.release import verify
  from simulators.tavern.autoresearch.evaluation_v2.registry import append
  verify(self.release)
  if self.state['runs'].get(name,{}).get('status')!='completed':raise ValueError('Completed registered artifact required')
  if private and not self.state['finalized']:raise ValueError('Finalize before private grading')
  if private:
   from .final_bank import panel_for
   panel=panel_for(self.release,self.out);tag='grade';folder='private-grade'
  else:
   folder=practice_folder(name,self.state['practice'].get(name,{}).get('attempt',1))
   panel=self.release/'public/practice.json';tag=folder.replace('practice-','practice',1)
  dest=self.out/folder
  if dest.exists():raise ValueError('Evaluation name already exists; recover the same worker before retry')
  volume=self.handoff_volume('evaluation')  # opened before anything is written for this evaluation
  dest.mkdir(parents=True);bundle=self.out/'submissions'/name if private else self.out/'runs'/name/'artifact'
  registry=append(self.release/'private/registry',bundle,dest/'registry','entrant-'+name)
  sb=None;key=('grade' if private else 'practice',name);lifetime=LIFETIME[key[0]];failure=None;published=None;released=False
  job=self.handoff_job(tag,name)
  try:
   sb=self.provider.create(False,lifetime,volume);self.sandboxes[key]=sb
   write(dest/'provider.json',dict(app_id=self.provider.app_id,sandbox_id=self.provider.identity(sb),environment=ENVIRONMENT))
   if key in self.stops:raise RuntimeError('Stopped by request')
   self.provider.command(sb,handoff.prepare_command('python'),timeout=20)  # /handoff is root-only before the guest runs a candidate
   self.provider.upload(sb,registry,'/opt/evaluator-assets');self.provider.copy(sb,panel,'/opt/evaluator-panel.json')
   args=['python','-m','simulators.tavern.evaluation_readiness.five_hour.guest','--root',EVALUATION_ROOT,'--candidate','entrant-'+name,'--seconds',str(lifetime-60),'--absolute-stop',str(time.time()+lifetime-20)]
   self.provider.command(sb,args,timeout=lifetime-30)
  except BaseException as exc:failure=exc
  if sb:
   route=dict(job=job,volume=volume.name)
   try:
    # Preserve all native recordings, including failed/incomplete cases, before teardown: the worker publishes the
    # scored results and one tar of everything it recorded to the hand-off Volume, and the host collects them after.
    published,reason=self.publish(sb,volume,job,pack_spec(job,EVALUATION_FILES,EVALUATION_FILES,
     {'retained.tar':dict(root=EVALUATION_ROOT,path='/private/handoff-retained.tar')}))
    if published:self.end_published(key,sb);released=True;route.update(handoff='volume')
    else:  # nothing was published and the worker is still up: read it directly, as before the Volume
     route.update(handoff='live',handoff_fallback=reason)
     if failure is None:
      try:
       self.provider.download(sb,EVALUATION_ROOT,dest,['lifecycle.json','qualification-privacy.json','qualification-terminal-replay.json'])
       self.provider.download(sb,EVALUATION_ROOT+'/screen',dest,['result.json','episodes.json'])
      except BaseException as exc:failure=exc
     self.retain_live(sb,dest)
   finally:
    try:self.receipt(sb,False,tag,terminate=not released)
    finally:self.sandboxes.pop(key,None)
   if published:
    if failure is None:
     try:self.collect(volume,job,dest,[f for f in EVALUATION_FILES if f in published])
     except BaseException as exc:failure=exc
    try:
     got=self.collect(volume,job,dest/'handoff',['retained.tar'])['retained.tar']
     unpack(got['path'],dest/'retained');route['retained']=dict(bytes=got['bytes'],sha256=got['sha256']);Path(got['path']).unlink()
    except Exception as exc:write(dest/'retention-blocker.json',dict(error=str(exc),archive_complete=False))
   write(dest/'handoff.json',route)
   if not private and name in self.state['practice']:self.mark(self.state['practice'][name],**{k:route[k] for k in ('handoff','handoff_fallback') if k in route})
  if failure is not None:raise failure
  life=json.loads((dest/'lifecycle.json').read_text());privacy=json.loads((dest/'qualification-privacy.json').read_text());terminal=json.loads((dest/'qualification-terminal-replay.json').read_text());result=json.loads((dest/'result.json').read_text())
  if life['status']!='completed' or not result['complete_valid_schedule'] or privacy.get('uid')!=65534 or privacy.get('private_paths_denied')!=3 or not privacy.get('credential_env_absent') or not terminal.get('all_lobby_actual_terminals_retained') or terminal.get('recorded_lobbies')!=result['expected_cases']:raise ValueError('Mixed evaluator/terminal/privacy qualification failed')
  result.update(benchmark_id=json.loads((self.release/'release.json').read_text())['evaluator_id'],normalized_reference_score=None)
  write(dest/'result.json',result)
  if not private:self.mark(self.state['runs'][name],practice=result)
  return result
 def retain_live(self,sb,dest):
  """Copy every native recording straight out of a worker that published nothing (one file at a time)."""
  try:
   listing=json.loads(self.provider.command(sb,['python','-c',"import pathlib,json;r=pathlib.Path('/private/evaluation');print(json.dumps([str(p.relative_to(r))for p in r.rglob('*')if p.is_file()]))"],timeout=15))
   for rel in listing:
    if Path(rel).is_absolute() or '..'in Path(rel).parts:raise ValueError('Invalid retained path')
    target=dest/'retained'/rel;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(self.provider.read(sb,EVALUATION_ROOT+'/'+rel))
  except Exception as exc:write(dest/'retention-blocker.json',dict(error=str(exc),archive_complete=False))

 def submit(self,name,require_practice=False):
  """Copy a completed run's verified artifact into a new read-only submission and select it."""
  with self.lock:
   if self.state['finalized']:raise ValueError('Finalized')
   row=self.state['runs'].get(name)
   if not row or row['status']!='completed':raise ValueError('Completed registered run required')
   if require_practice and not row.get('practice',{}).get('complete_valid_schedule'):raise ValueError('Completed public practice required')
   src=self.out/'runs'/name/'artifact';dest=self.out/'submissions'/name
   if dest.exists():raise ValueError('Immutable submission already exists')
   self.verify_artifact(name)
   if any(sha(src/f)!=row['source'][f] for f in EDITABLE):raise ValueError('Incomplete/drifted artifact')
   shutil.copytree(src,dest,ignore=shutil.ignore_patterns('*.pt'))
   hashes={p.name:sha(p) for p in dest.iterdir() if p.is_file()};write(dest/'submission.json',dict(name=name,files=hashes,created=time.time()))
   for p in dest.iterdir():p.chmod(0o444)
   dest.chmod(0o555);self.state['submission']=name
   self.save();return dict(submission=name,files=hashes)
 def select(self,name):
  """Select an already immutable submission without changing its files."""
  with self.lock:
   if self.state['finalized']:raise ValueError('Finalized')
   if not name.isalnum() or name not in self.state['runs']:raise ValueError('Registered submission name required')
   bundle=self.out/'submissions'/name;manifest=json.loads((bundle/'submission.json').read_text())
   if any(sha(bundle/f)!=h for f,h in manifest['files'].items()):raise ValueError('Submission mutated')
   self.state['submission']=name;self.state.setdefault('selection_events',[]).append(dict(name=name,at=time.time(),gpu_seconds=self.budget()['allocated_seconds']))
   self.save();return dict(selected=name,immutable_manifest_sha256=sha(bundle/'submission.json'))
 def settling(self):
  """Every worker thread is only settling a job that already ended (diagnostics, provider receipt), or reconciling."""
  with self.lock:
   if not self.busy() or self.state['queue']:return False
   for kind,name in self.running():
    row=self.state['runs' if kind=='train' else 'practice'].get(name)
    if row is None or row['status'] in LIVE:return False
   return True
 def finalize(self,timeout=300):
  # A just-completed job is not active work: wait for its worker thread to finish settling it.
  if self.settling():self.join(timeout)
  with self.lock:
   if not self.state['submission']:raise ValueError('No submission')
   if self.state['queue'] or self.busy():raise ValueError('Resolve active work before finalizing')
   self.ledger.close();self.state['finalized']=True;self.state['final_submission']=self.state['submission'];self.save();return self.status()
 def grade(self):
  if not self.state['finalized']:raise ValueError('Finalize before private grading')
  name=self.state['submission'];p=self.out/'submissions'/name;manifest=json.loads((p/'submission.json').read_text())
  if any(sha(p/f)!=h for f,h in manifest['files'].items()):raise ValueError('Submission mutated')
  return self.evaluate(name,True)
