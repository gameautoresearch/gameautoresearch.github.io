"""Host-only transactional authority. No guest-supplied timing or refunds."""
from contextlib import contextmanager
from pathlib import Path
import json
import sqlite3
import time
import re
import math

from .contract import TRAIN_SHAPE,digest,rate as shape_rate,validate_session,write_json

ACTIVE=('queued','running','uncertain')


def terms(row):
    """A job's worker shape and charging rate (US dollars per wall second of the worker)."""
    payload=json.loads(row['payload']) if row['payload'] else {}
    shape=payload.get('shape') or dict(TRAIN_SHAPE)
    return shape,float(payload.get('rate',shape_rate(shape)))

# How a complete training run can have ended before its own end: the researcher stopped it, or its time
# limit cut it off. Either way it is packaged from the network the trainer last saved.
EARLY=('stopped_early','cut_short')


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
          CREATE TABLE IF NOT EXISTS lifecycle(id INTEGER PRIMARY KEY,state TEXT);
          INSERT OR IGNORE INTO lifecycle VALUES(1,'active');
        ''')
        if fresh:
            if config is None:raise ValueError('A new ledger needs an explicit session contract')
            self.db.execute('INSERT INTO settings VALUES(1,?)',(json.dumps(validate_session(config)),))
        self.config=json.loads(self.db.execute('SELECT config FROM settings WHERE id=1').fetchone()[0])
        if config is not None:
            asked=validate_session(config)
            if asked['budget_mode']!=self.mode:raise ValueError(f'This session was opened in {self.mode} budget mode; it cannot continue in {asked["budget_mode"]} mode')
            if asked!=self.config:raise ValueError('Existing session budget is immutable')

    @contextmanager
    def transaction(self):
        self.db.execute('BEGIN IMMEDIATE')
        try:yield;self.db.execute('COMMIT')
        except BaseException:self.db.execute('ROLLBACK');raise

    def event(self,kind,data):
        self.db.execute('INSERT INTO events(at,kind,data) VALUES(?,?,?)',(time.time(),kind,json.dumps(data)))

    def jobs(self):return [dict(r) for r in self.db.execute('SELECT * FROM jobs ORDER BY rowid')]
    def used(self):
        """Dollars charged or held: the whole grant of queued, running and unresolved work, the charge of ended work.

        A running worker holds its whole grant, not its time so far: with several jobs at once, anything less
        would let a second grant be queued against money the first is still going to spend.
        """
        return sum(r['grant']*terms(r)[1] if r['status'] in ACTIVE else r['charged'] for r in self.jobs())

    def budget(self):return float(self.config['budget_usd'])

    @property
    def mode(self):
        """'usd' (parallel jobs within dollars) or 'seconds' (one job at a time within training-worker seconds)."""
        return self.config.get('budget_mode','usd')

    def seconds(self,usd):
        """Dollars as seconds of the training worker: the unit a seconds-mode session reports in."""
        return usd/shape_rate(TRAIN_SHAPE)

    def limit(self,key):
        """A concurrency limit of this session; None is no limit."""
        return self.config.get(key)

    def early(self):
        """{run: 'stopped_early' or 'cut_short'} for complete runs that did not reach their own end."""
        found={}
        for row in self.db.execute("SELECT data FROM events WHERE kind IN ('finished','reconciled') ORDER BY id"):
            data=json.loads(row[0])
            if data.get('early') in EARLY:found[data['name']]=data['early']
        return found

    def public(self):
        jobs=[dict({k:r[k] for k in ('name','status','grant','started','ended','charged','artifact_sha','error')},rate=terms(r)[1]) for r in self.jobs()]
        early=self.early()
        for job in jobs:
            if job['status']=='complete' and job['name'] in early:job[early[job['name']]]=True
        used=self.used();remaining=max(0.,self.budget()-used)
        state={'config':self.config,'budget_mode':self.mode,'charged_or_reserved_usd':used,
            'lifecycle':self.db.execute('SELECT state FROM lifecycle WHERE id=1').fetchone()[0],
            'remaining_usd':remaining,'allocatable_usd':remaining,'jobs':jobs,
            'submissions':[dict(r) for r in self.db.execute('SELECT * FROM submissions')],
            'points':[dict(r) for r in self.db.execute('SELECT * FROM points')]}
        if self.mode=='seconds':
            # The same ledger in training-worker seconds (rounded to the microsecond against float noise).
            state.update(budget_seconds=float(self.config['budget_seconds']),
                charged_or_reserved_seconds=round(self.seconds(used),6),remaining_seconds=round(self.seconds(remaining),6),
                allocatable_seconds=round(self.seconds(remaining),6))
            for job in jobs:job.update(charged_seconds=round(self.seconds(job['charged']),6),seconds_weight=self.seconds(job['rate']))
        return state

    def allocated_elapsed(self,now=None):
        """Dollars of worker time actually allocated so far. Milestones use this, never queued/unknown holds."""
        now=time.time() if now is None else now
        total=0.
        for row in self.jobs():
            if row['status']=='running':total+=max(0.,now-row['started'])*terms(row)[1]
            elif row['status'] not in ('queued','uncertain'):total+=row['charged']
        return total

    def queue(self,name,seconds,payload):
        if not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_-]{0,63}',name):raise ValueError('Invalid run name')
        if not math.isfinite(seconds) or seconds<self.config['min_run_seconds']:raise ValueError('Run shorter than minimum or non-finite allocation')
        shape,rate=terms({'payload':json.dumps(payload)})
        with self.transaction():
            if self.db.execute('SELECT state FROM lifecycle WHERE id=1').fetchone()[0]!='active':raise ValueError('Session finalized; no new allocations')
            held=[terms(r)[0] for r in self.jobs() if r['status'] in ACTIVE]
            jobs=self.limit('max_concurrent_jobs')
            if len(held)>=jobs:
                raise ValueError('One job at a time: a run, practice or lease is already queued or running (an unresolved provider allocation counts); wait for it or stop it' if jobs==1 else
                    f'At most {jobs} jobs at a time; stop one or wait (an unresolved provider lease counts)')
            for key,field,what in (('max_concurrent_gpus','gpu','GPUs'),('max_concurrent_cpu_cores','cpu_cores','CPU cores')):
                if self.limit(key) is not None and sum(h[field] for h in held)+shape[field]>self.limit(key):
                    raise ValueError(f'At most {self.limit(key)} {what} at a time across your active jobs')
            if seconds*rate>self.budget()-self.used()+1e-9:raise ValueError('Allocation exceeds remaining session budget')
            self.db.execute('INSERT INTO jobs(name,status,grant,payload) VALUES(?,?,?,?)',(name,'queued',seconds,json.dumps(payload)))
            self.event('queued',{'name':name,'payload_sha256':digest(payload),'reserved_usd':seconds*rate,
                'worker_seconds':seconds,'rate':rate,'shape':shape})

    def start(self,name,provider_id,allocated_start):
        if allocated_start<=0:raise ValueError('Provider allocation timestamp required')
        with self.transaction():
            cur=self.db.execute("UPDATE jobs SET status='running',provider_id=?,started=? WHERE name=? AND status='queued'",(provider_id,allocated_start,name))
            if cur.rowcount!=1:raise ValueError('Job was not queued')
            self.event('allocated',{'name':name,'provider_id':provider_id,'started':allocated_start})

    def milestones(self,now=None,final=False):
        used=self.allocated_elapsed(now)
        latest=self.db.execute('SELECT id FROM submissions ORDER BY id DESC LIMIT 1').fetchone()
        sub=latest[0] if latest else None
        for frac in self.config['grading_points']:
            if frac==1 and not final:continue
            if final and frac==1:
                self.db.execute('INSERT OR REPLACE INTO points VALUES(?,?,?,?)',(str(frac),sub,used,time.time()))
            elif used>=frac*self.budget():
                self.db.execute('INSERT OR IGNORE INTO points VALUES(?,?,?,?)',(str(frac),sub,used,time.time()))

    def finish(self,name,allocated_end,status='complete',artifact=None,artifact_sha=None,error=None,early=None):
        if status not in ('complete','stopped','failed'):raise ValueError('Invalid terminal status')
        if early is not None and (early not in EARLY or status!='complete' or not artifact_sha):raise ValueError('Only a complete packaged run can be marked as ended early')
        with self.transaction():
            r=self.db.execute('SELECT * FROM jobs WHERE name=?',(name,)).fetchone()
            if not r or r['status']!='running':raise ValueError('Only a running provider allocation can finish')
            if allocated_end<r['started']:raise ValueError('Provider end precedes start')
            charged=(allocated_end-r['started'])*terms(r)[1]
            self.db.execute('UPDATE jobs SET status=?,ended=?,charged=?,artifact=?,artifact_sha=?,error=? WHERE name=?',
                (status,allocated_end,charged,str(artifact) if artifact else None,artifact_sha,error,name))
            self.milestones();self.event('finished',{'name':name,'charged':charged,'worker_seconds':allocated_end-r['started'],'status':status,**({'early':early} if early else {})})

    def uncertain(self,name,error):
        with self.transaction():
            self.db.execute("UPDATE jobs SET status='uncertain',error=? WHERE name=? AND status IN ('queued','running')",(error,name))
            self.event('provider_uncertain',{'name':name,'error':error})

    def reconcile(self,name,receipt,status='failed',artifact=None,artifact_sha=None,error=None,early=None):
        """Host-only recovery from the SAME provider allocation, never a rerun."""
        start,end=receipt.get('started_at',0),receipt.get('finished_at',0)
        if not receipt.get('provider_id') or not all(math.isfinite(v) for v in (start,end)) or start<=0 or end<start:
            raise ValueError('Completed authoritative provider receipt required')
        if status not in ('complete','stopped','failed'):raise ValueError('Invalid recovery status')
        if early is not None and (early not in EARLY or status!='complete' or not artifact_sha):raise ValueError('Only a complete packaged run can be marked as ended early')
        with self.transaction():
            r=self.db.execute('SELECT * FROM jobs WHERE name=?',(name,)).fetchone()
            if not r or r['status'] not in ('queued','running','uncertain'):raise ValueError('Only an unresolved allocation can reconcile')
            if r['provider_id'] and r['provider_id']!=receipt['provider_id']:raise ValueError('Recovery provider identity differs')
            if r['started'] and abs(r['started']-start)>1e-6:raise ValueError('Recovery allocation start differs')
            charged=(end-start)*terms(r)[1]
            self.db.execute('UPDATE jobs SET status=?,provider_id=?,started=?,ended=?,charged=?,artifact=?,artifact_sha=?,error=? WHERE name=?',
                (status,receipt['provider_id'],start,end,charged,str(artifact) if artifact else None,artifact_sha,error,name))
            self.milestones();self.event('reconciled',{'name':name,'provider_id':receipt['provider_id'],'charged':charged,
                'worker_seconds':end-start,'status':status,**({'early':early} if early else {})})

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

    def register(self,name,artifact,artifact_sha,payload):
        """Record a policy package exported from a running lease. It holds no worker and costs nothing itself:
        the lease that produced it is charged."""
        if not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_-]{0,63}',name):raise ValueError('Invalid run name')
        now=time.time()
        with self.transaction():
            if self.db.execute('SELECT state FROM lifecycle WHERE id=1').fetchone()[0]!='active':raise ValueError('Session finalized; no new networks')
            self.db.execute('INSERT INTO jobs(name,status,grant,started,ended,charged,payload,artifact,artifact_sha) VALUES(?,?,0,?,?,0,?,?,?)',
                (name,'complete',now,now,json.dumps(dict(payload,rate=0.)),str(artifact),artifact_sha))
            self.event('packaged',{'name':name,'artifact_sha':artifact_sha,'payload_sha256':digest(payload)})

    def submit(self,name):
        with self.transaction():
            if self.db.execute('SELECT state FROM lifecycle WHERE id=1').fetchone()[0]!='active':raise ValueError('Session finalized; submission is immutable')
            if self.db.execute("SELECT 1 FROM jobs WHERE status='uncertain'").fetchone():raise ValueError('Reconcile unknown provider allocations before changing submissions')
            r=self.db.execute('SELECT * FROM jobs WHERE name=?',(name,)).fetchone()
            if not r or r['status'] not in ('complete','stopped') or not r['artifact_sha']:raise ValueError('Submission requires a validated completed allocation artifact')
            # A controller outage must not let a submission made after a crossed
            # threshold retroactively become that milestone's policy.
            self.milestones()
            cur=self.db.execute('INSERT INTO submissions(job,artifact_sha,at) VALUES(?,?,?)',(name,r['artifact_sha'],time.time()))
            self.event('submitted',{'name':name,'id':cur.lastrowid,'artifact_sha':r['artifact_sha']})
            return cur.lastrowid

    def finalize(self):
        with self.transaction():
            if any(x['status'] in ('queued','running','uncertain') for x in self.jobs()):raise ValueError('Finish or reconcile every active allocation before finalizing')
            if self.db.execute('SELECT state FROM lifecycle WHERE id=1').fetchone()[0]=='finalized':return self.public()
            self.milestones(final=True)
            self.db.execute("UPDATE lifecycle SET state='finalized' WHERE id=1")
            self.event('finalized',{})
        return self.public()
