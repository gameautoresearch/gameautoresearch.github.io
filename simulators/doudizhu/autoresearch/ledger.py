"""Host-only transactional authority. No guest-supplied timing or refunds."""
from contextlib import contextmanager
from pathlib import Path
import json
import sqlite3
import time
import re
import math

from .contract import digest,validate_session,write_json

ACTIVE=('queued','running','uncertain')
MODES=('seconds','usd')
# A session that records no concurrency limits (seconds mode, every ledger opened before the dollar mode) runs
# one job at a time. Every Dou Dizhu worker, training or practice, has one L40S and four CPU cores.
LINEAR={'max_concurrent_jobs':1,'max_concurrent_gpus':1,'max_concurrent_cpu_cores':None}


def check_mode(config):
    """The research budget mode a new session records. `seconds`: one job at a time, the allowance is
    `gpu_minutes` of the worker. `usd`: parallel jobs up to the recorded limits, the allowance is `budget_usd`
    of worker time at `usd_per_second` (so `gpu_minutes` is the worker time the dollars buy), and a queued or
    running job holds its whole grant until it ends."""
    mode=config.get('budget_mode','seconds')
    if mode not in MODES:raise ValueError('budget_mode is seconds or usd')
    if mode=='usd':
        usd,rate=config.get('budget_usd'),config.get('usd_per_second')
        if not all(isinstance(v,(int,float)) and math.isfinite(v) and v>0 for v in (usd,rate)):
            raise ValueError('A dollar-mode session needs a positive budget_usd and usd_per_second')
        if abs(config['gpu_minutes']*60*rate-usd)>1e-6:raise ValueError('gpu_minutes must be the worker time budget_usd buys')
    for key in LINEAR:
        value=config.get(key,LINEAR[key])
        if value is not None and (not isinstance(value,int) or value<1):raise ValueError('Invalid concurrency limit '+key)


class Ledger:
    def __init__(self,root,config=None):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
        self.path=self.root/'ledger.sqlite3'
        fresh=not self.path.exists()
        self.db=sqlite3.connect(self.path,timeout=30,isolation_level=None,check_same_thread=False)
        self.db.row_factory=sqlite3.Row
        self.db.executescript('''PRAGMA journal_mode=WAL;
          CREATE TABLE IF NOT EXISTS settings(id INTEGER PRIMARY KEY,config TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS jobs(name TEXT PRIMARY KEY,status TEXT,grant REAL,started REAL,ended REAL,charged REAL DEFAULT 0,
             payload TEXT,artifact TEXT,artifact_sha TEXT,provider_id TEXT,error TEXT);
          CREATE TABLE IF NOT EXISTS submissions(id INTEGER PRIMARY KEY AUTOINCREMENT,job TEXT,artifact_sha TEXT,at REAL);
          CREATE TABLE IF NOT EXISTS points(fraction TEXT PRIMARY KEY,submission INTEGER,charged REAL,at REAL);
          CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY AUTOINCREMENT,at REAL,kind TEXT,data TEXT);
        ''')
        if fresh:
            if config is None:raise ValueError('A new ledger needs an explicit session contract')
            check_mode(validate_session(config))
            self.db.execute('INSERT INTO settings VALUES(1,?)',(json.dumps(validate_session(config)),))
        self.config=json.loads(self.db.execute('SELECT config FROM settings WHERE id=1').fetchone()[0])
        if config is not None and validate_session(config)!=self.config:raise ValueError('Existing session budget is immutable')

    @contextmanager
    def transaction(self):
        self.db.execute('BEGIN IMMEDIATE')
        try:yield;self.db.execute('COMMIT')
        except BaseException:self.db.execute('ROLLBACK');raise

    def event(self,kind,data):
        self.db.execute('INSERT INTO events(at,kind,data) VALUES(?,?,?)',(time.time(),kind,json.dumps(data)))

    def jobs(self):return [dict(r) for r in self.db.execute('SELECT * FROM jobs ORDER BY rowid')]
    def mode(self):return self.config.get('budget_mode','seconds')
    def limit(self,key):return self.config.get(key,LINEAR[key])
    def rate(self):
        """US dollars per worker second (dollar mode only)."""
        return float(self.config['usd_per_second'])

    def progress(self,now=None):
        """What milestones measure (and, in seconds mode, what is held): a running worker's time so far, the
        grant of queued or unresolved work, the charge of ended work. Worker seconds."""
        now=time.time() if now is None else now
        total=0.
        for r in self.jobs():
            if r['status']=='running':total+=max(0.,now-r['started'])
            elif r['status'] in ('queued','uncertain'):total+=r['grant']
            else:total+=r['charged']
        return total

    def used(self,now=None):
        """Worker seconds charged or held. In dollar mode a running worker holds its whole grant, not its time so
        far: with several jobs at once anything less would let a new grant be queued against money a running
        one is still going to spend."""
        if self.mode()!='usd':return self.progress(now)
        return sum(r['grant'] if r['status'] in ACTIVE else r['charged'] for r in self.jobs())

    def public(self):
        jobs=[{k:r[k] for k in ('name','status','grant','started','ended','charged','artifact_sha','error')} for r in self.jobs()]
        used=self.used();budget=self.config['gpu_minutes']*60;reserve=self.config.get('teardown_reserve_seconds',0)
        state={'config':self.config,'charged_or_reserved_seconds':used,
            'remaining_seconds':max(0.,budget-used),
            'allocatable_seconds':max(0.,budget-used-reserve),'jobs':jobs,
            'submissions':[dict(r) for r in self.db.execute('SELECT * FROM submissions')],
            'points':[dict(r) for r in self.db.execute('SELECT * FROM points')]}
        if self.mode()=='usd':
            rate=self.rate();usd=float(self.config['budget_usd'])
            for job in jobs:job['charged_usd']=job['charged']*rate
            state.update(charged_or_reserved_usd=used*rate,remaining_usd=max(0.,usd-used*rate),
                allocatable_usd=max(0.,usd-(used+reserve)*rate),usd_per_second=rate)
        return state

    def queue(self,name,seconds,payload,minimum_seconds=None):
        if not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_-]{0,63}',name):raise ValueError('Invalid run name')
        minimum=self.config['min_run_seconds'] if minimum_seconds is None else minimum_seconds
        if not math.isfinite(seconds) or seconds<minimum:raise ValueError('Run shorter than minimum or non-finite allocation')
        with self.transaction():
            held=self.db.execute("SELECT COUNT(*) FROM jobs WHERE status IN ('queued','running','uncertain')").fetchone()[0]
            jobs=self.limit('max_concurrent_jobs')
            if held>=jobs:
                raise ValueError('One job at a time; an unresolved provider lease blocks new allocations' if jobs==1 else
                    f'At most {jobs} jobs at a time; stop one or wait (an unresolved provider lease counts)')
            # Every worker, practice included, has one GPU and the contract's CPU cores.
            for key,each,what in (('max_concurrent_gpus',1,'GPUs'),('max_concurrent_cpu_cores',self.config['cpu_cores'],'CPU cores')):
                if self.limit(key) is not None and (held+1)*each>self.limit(key):
                    raise ValueError(f'At most {self.limit(key)} {what} at a time across your active jobs '
                        f'(every worker, practice included, has one GPU and {self.config["cpu_cores"]} CPU cores)')
            if seconds>self.config['gpu_minutes']*60-self.used()-self.config.get('teardown_reserve_seconds',0)+1e-9:raise ValueError('Allocation exceeds remaining session budget')
            self.db.execute('INSERT INTO jobs(name,status,grant,payload) VALUES(?,?,?,?)',(name,'queued',seconds,json.dumps(payload)))
            self.event('queued',dict({'name':name,'payload_sha256':digest(payload),'reserved_seconds':seconds},
                **({'reserved_usd':seconds*self.rate(),'usd_per_second':self.rate()} if self.mode()=='usd' else {})))

    def start(self,name,provider_id,allocated_start):
        if allocated_start<=0:raise ValueError('Provider allocation timestamp required')
        with self.transaction():
            cur=self.db.execute("UPDATE jobs SET status='running',provider_id=?,started=? WHERE name=? AND status='queued'",(provider_id,allocated_start,name))
            if cur.rowcount!=1:raise ValueError('Job was not queued')
            self.event('allocated',{'name':name,'provider_id':provider_id,'started':allocated_start})

    def milestones(self,now=None,final=False):
        used=self.progress(now)
        latest=self.db.execute('SELECT id FROM submissions ORDER BY id DESC LIMIT 1').fetchone()
        sub=latest[0] if latest else None
        for frac in self.config['grading_points']:
            if frac==1 and not final:continue
            if final and frac==1:
                self.db.execute('INSERT OR REPLACE INTO points VALUES(?,?,?,?)',(str(frac),sub,used,time.time()))
            elif used>=frac*self.config['gpu_minutes']*60:
                self.db.execute('INSERT OR IGNORE INTO points VALUES(?,?,?,?)',(str(frac),sub,used,time.time()))

    def finish(self,name,allocated_end,status='complete',artifact=None,artifact_sha=None,error=None):
        if status not in ('complete','stopped','failed'):raise ValueError('Invalid terminal status')
        with self.transaction():
            r=self.db.execute('SELECT * FROM jobs WHERE name=?',(name,)).fetchone()
            if not r or r['status']!='running':raise ValueError('Only a running provider allocation can finish')
            if allocated_end<r['started']:raise ValueError('Provider end precedes start')
            self.db.execute('UPDATE jobs SET status=?,ended=?,charged=?,artifact=?,artifact_sha=?,error=? WHERE name=?',
                (status,allocated_end,allocated_end-r['started'],str(artifact) if artifact else None,artifact_sha,error,name))
            self.milestones();self.event('finished',{'name':name,'charged':allocated_end-r['started'],'status':status})

    def uncertain(self,name,error):
        with self.transaction():
            self.db.execute("UPDATE jobs SET status='uncertain',error=? WHERE name=? AND status IN ('queued','running')",(error,name))
            self.event('provider_uncertain',{'name':name,'error':error})

    def reconcile(self,name,provider_id,started,ended,error):
        if not 0<started<=ended or not all(math.isfinite(v) for v in (started,ended)):
            raise ValueError('Valid external provider timestamps required')
        with self.transaction():
            r=self.db.execute('SELECT status FROM jobs WHERE name=?',(name,)).fetchone()
            if not r or r['status'] not in ('queued','running','uncertain'):raise ValueError('Only unresolved leases may be reconciled')
            self.db.execute("UPDATE jobs SET status='failed',provider_id=?,started=?,ended=?,charged=?,error=? WHERE name=?",(provider_id,started,ended,ended-started,error,name))
            self.milestones();self.event('provider_reconciled',{'name':name,'charged':ended-started,'provider_id':provider_id})

    def abandon_unstarted(self,name,receipt,reason,stopped=False):
        """Settle a job whose provider worker was created but never started: no allocation, 0 seconds charged.

        Only an unstarted queued (including claimed) or uncertain row qualifies, and only on the provider's
        own record showing no start. Queued waiting time is never charged; a started worker settles from its
        start/finish receipt through finish/reconcile instead.
        """
        receipt=dict(receipt or {})
        if not receipt.get('provider_id'):raise ValueError('Provider record of the abandoned worker required')
        if receipt.get('started_at'):raise ValueError('A started allocation is charged from its receipt')
        status='stopped' if stopped else 'failed'
        with self.transaction():
            r=self.db.execute('SELECT * FROM jobs WHERE name=?',(name,)).fetchone()
            if not r or r['status'] not in ('queued','uncertain') or r['started']:
                raise ValueError('Only an unstarted queued or uncertain job can be abandoned uncharged')
            if r['provider_id'] and r['provider_id']!=receipt['provider_id']:raise ValueError('Abandoned provider identity differs')
            self.db.execute('UPDATE jobs SET status=?,provider_id=?,charged=0,artifact=NULL,artifact_sha=NULL,error=? WHERE name=?',
                (status,receipt['provider_id'],reason,name))
            self.event('abandoned_unstarted',{'name':name,'provider_id':receipt['provider_id'],'charged':0,'status':status,
                'reason':reason,'receipt':receipt})

    def cancel_queued(self,name):
        with self.transaction():
            cur=self.db.execute("UPDATE jobs SET status='cancelled',charged=0 WHERE name=? AND status='queued'",(name,))
            if cur.rowcount!=1:raise ValueError('Allocated jobs must be stopped and charged through the provider')

    def submit(self,name):
        with self.transaction():
            r=self.db.execute('SELECT * FROM jobs WHERE name=?',(name,)).fetchone()
            if not r or r['status'] not in ('complete','stopped') or not r['artifact_sha']:raise ValueError('Submission requires a validated completed allocation artifact')
            self.milestones()  # Freeze any crossed point before accepting this later policy.
            cur=self.db.execute('INSERT INTO submissions(job,artifact_sha,at) VALUES(?,?,?)',(name,r['artifact_sha'],time.time()))
            self.event('submitted',{'name':name,'id':cur.lastrowid,'artifact_sha':r['artifact_sha']})
            return cur.lastrowid

    def finalize(self):
        with self.transaction():
            if any(x['status'] in ACTIVE for x in self.jobs()):raise ValueError('Finish or reconcile every active allocation before finalizing')
            self.milestones(final=True)
            self.event('finalized',{})
        return self.public()
