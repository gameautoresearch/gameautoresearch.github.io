"""Host-only persistent reservations: unknown ends block further GPU work."""
import sqlite3,json,time,math
from pathlib import Path
class Ledger:
 def __init__(self,path,cap=600):
  self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True);self.db=sqlite3.connect(self.path,isolation_level=None)
  self.db.executescript('CREATE TABLE IF NOT EXISTS settings(cap REAL,closed INTEGER); CREATE TABLE IF NOT EXISTS allocations(name TEXT PRIMARY KEY,reserved REAL,actual REAL,receipt TEXT,created REAL);')
  row=self.db.execute('SELECT cap FROM settings').fetchone()
  if row is None:self.db.execute('INSERT INTO settings VALUES(?,0)',(cap,))
  elif row[0]!=cap:raise ValueError('Cannot reset budget')
 def reserve(self,name,seconds):
  self.db.execute('BEGIN IMMEDIATE')
  try:
   if self.db.execute('SELECT closed FROM settings').fetchone()[0]:raise ValueError('Session finalized')
   rows=self.db.execute('SELECT reserved,actual FROM allocations').fetchall()
   if any(a is None for _,a in rows):raise ValueError('Unresolved allocation blocks launch')
   if not math.isfinite(seconds) or seconds<=0 or sum(a for _,a in rows)+seconds>self.snapshot()['cap_seconds']:raise ValueError('GPU ceiling')
   self.db.execute('INSERT INTO allocations VALUES(?,?,NULL,NULL,?)',(name,seconds,time.time()));self.db.execute('COMMIT')
  except BaseException:self.db.execute('ROLLBACK');raise
 def settle(self,name,receipt):
  start,end=receipt.get('started_at',0),receipt.get('finished_at',0)
  if not receipt.get('provider_id') or not 0<start<=end:raise ValueError('Authoritative completed provider receipt required')
  self.db.execute('UPDATE allocations SET actual=?,receipt=? WHERE name=? AND actual IS NULL',(end-start,json.dumps(receipt),name))
 def settle_never_started(self,name,receipt,status='failed'):
  """Close an unresolved allocation whose worker the provider never started: charged 0, its receipt kept."""
  if status not in ('failed','stopped') or not isinstance(receipt,dict) or not receipt.get('provider_id') or receipt.get('started_at'):raise ValueError('Provider receipt of a never-started worker required')
  record=dict(receipt,never_started=True,charged_seconds=0,outcome=status)
  if self.db.execute('UPDATE allocations SET actual=0,receipt=? WHERE name=? AND actual IS NULL',(json.dumps(record),name)).rowcount!=1:raise ValueError('Only an unresolved allocation can be settled as never started')
 def snapshot(self):
  rows=[dict(zip(('name','reserved','actual','receipt','created'),r)) for r in self.db.execute('SELECT * FROM allocations')];cap,closed=self.db.execute('SELECT cap,closed FROM settings').fetchone()
  actual=sum(r['actual'] or 0 for r in rows);hold=sum(r['reserved'] for r in rows if r['actual'] is None)
  return dict(cap_seconds=cap,allocated_seconds=actual,unresolved_seconds=hold,available_seconds=max(0,cap-actual-hold),closed=bool(closed),allocations=rows)
 def close(self):
  if self.snapshot()['unresolved_seconds']:raise ValueError('Cannot finalize unresolved allocations')
  self.db.execute('UPDATE settings SET closed=1')
