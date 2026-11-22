"""Persistent host-owned allocation authority, including unknown failed attempts."""
from pathlib import Path
import fcntl,json,time,math
from .common import dump
class Ledger:
 def __init__(self,path,limit=600):
  self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True);self.lock=self.path.with_suffix('.lock').open('a+')
  fcntl.flock(self.lock,fcntl.LOCK_EX)
  try:
   if not self.path.exists():dump(self.path,{'limit_seconds':limit,'allocations':[],'submissions':[],'milestones':{},'finalized':False})
   if self.read()['limit_seconds']!=limit:raise ValueError('existing allocation ceiling cannot be reset')
  finally:fcntl.flock(self.lock,fcntl.LOCK_UN)
 def read(self):return json.loads(self.path.read_text())
 def used(self,state=None):return sum(r.get('charged_seconds',r['reserved_seconds']) for r in (state or self.read())['allocations'])
 def mutate(self,fn):
  fcntl.flock(self.lock,fcntl.LOCK_EX)
  try:s=self.read();result=fn(s);dump(self.path,s);return result
  finally:fcntl.flock(self.lock,fcntl.LOCK_UN)
 def reserve(self,name,seconds):
  def go(s):
   if s['finalized'] or any(r['status'] in ('reserved','running','uncertain') for r in s['allocations']):raise ValueError('closed session or unresolved active allocation')
   if any(r['name']==name for r in s['allocations']):raise ValueError('run name already reserved; historical attempts remain immutable')
   if not math.isfinite(seconds) or seconds<=0:raise ValueError('positive finite reservation required')
   remaining=s['limit_seconds']-self.used(s)
   if seconds>remaining:raise ValueError(f'allocation needs {seconds:.2f} seconds including setup/cleanup reserve, but only {remaining:.2f} remain; no GPU was launched')
   s['allocations'].append({'name':name,'reserved_seconds':seconds,'status':'reserved','at':time.time()})
  return self.mutate(go)
 def record(self,name,**values):
  return self.mutate(lambda s:next(r for r in s['allocations'] if r['name']==name).update(values))
 @staticmethod
 def _tick_milestones(state,allocated_seconds):
  for fraction in (1/3,2/3):
   if allocated_seconds>=state['limit_seconds']*fraction:state['milestones'].setdefault(str(fraction),state['submissions'][-1] if state['submissions'] else None)
 def settle(self,name,receipt,*,status='complete',error=None):
  """Charge provider start/end time and capture teardown crossings atomically."""
  start=receipt.get('started_at');end=receipt.get('finished_at')
  if status not in ('complete','failed') or any(type(v) not in (int,float) or not math.isfinite(v) for v in (start,end)) or start<=0 or end<start or not receipt.get('provider_id'):raise ValueError('completed authoritative provider receipt required')
  def go(s):
   row=next(r for r in s['allocations'] if r['name']==name)
   if 'charged_seconds' in row:raise ValueError('allocation already settled')
   if row.get('sandbox_id',receipt['provider_id'])!=receipt['provider_id']:raise ValueError('provider receipt belongs to another allocation')
   row.update(status=status,charged_seconds=end-start,receipt=receipt,error=error)
   # Reservations are liabilities, not elapsed time. Settlement runs before a
   # later submission can acquire this lock, so a crossed empty point stays null.
   self._tick_milestones(s,sum(r.get('charged_seconds',0) for r in s['allocations']))
  return self.mutate(go)
 def settle_never_started(self,name,receipt,*,status='failed',error=None):
  """Close a reserved/uncertain allocation whose worker the provider never started: charged 0, receipt kept.

  The receipt must be the provider's own record of this allocation's sandbox and show no start.
  """
  if status not in ('failed','stopped') or not isinstance(receipt,dict) or not receipt.get('provider_id') or receipt.get('started_at'):raise ValueError('provider receipt of a never-started worker required')
  def go(s):
   row=next(r for r in s['allocations'] if r['name']==name)
   if 'charged_seconds' in row or row['status'] not in ('reserved','running','uncertain'):raise ValueError('only an unresolved allocation can be settled as never started')
   if row.get('sandbox_id')!=receipt['provider_id']:raise ValueError('provider receipt belongs to another allocation')
   row.update(status=status,charged_seconds=0,receipt=receipt,never_started=True,error=error or 'the provider never started this worker; terminated and charged 0 seconds')
  return self.mutate(go)
 def tick(self,allocated_seconds):
  return self.mutate(lambda s:self._tick_milestones(s,allocated_seconds))
 def settle_no_allocation(self,name,proof_path):
  """Release only a proven pre-create reservation; preserve the failed attempt."""
  proof=json.loads(Path(proof_path).read_text())
  if proof.get('provider_inventory_checked') is not True or proof.get('worker_count')!=0 or proof.get('worker_create_never_attempted') is not True:raise ValueError('verified pre-create provider inventory required')
  def go(s):
   row=next(r for r in s['allocations'] if r['name']==name)
   if row.get('sandbox_id') or 'charged_seconds' in row or row.get('strong_job')!=proof.get('stage'):raise ValueError('cannot clear a known, settled, or different worker')
   row.update(status='failed',charged_seconds=0,no_allocation_proof=str(proof_path),no_allocation_receipt=proof,error='worker create never attempted; separate build/storage/observer costs remain counted')
  return self.mutate(go)
 def submit(self,name,identity):
  def go(s):
   if s['finalized']:raise ValueError('session finalized')
   row={'name':name,'identity':identity,'at':time.time(),'charged_or_reserved_seconds':self.used(s)};s['submissions'].append(row)
   return row
  return self.mutate(go)
 def finalize(self):
  def go(s):
   if any(r['status'] in ('reserved','running','uncertain') for r in s['allocations']):raise ValueError('unresolved allocation prevents finalization')
   if not s['submissions']:raise ValueError('cannot finalize without submission')
   s['finalized']=True;s['milestones']['final']=s['submissions'][-1];return s
  return self.mutate(go)
