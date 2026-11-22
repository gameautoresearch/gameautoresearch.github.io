"""Host-only transactional authority of a Polytopia session. No guest-supplied timing or refunds.

One SQLite ledger holds every job (training run, lease, packaged network, practice), the submissions and the
milestone points. A charge is provider start-to-finish seconds times the job's rate, in the session's unit
(`budget_mode` of its config, contract.py): US dollars at the price per second of the job's worker shape, or
seconds of the training worker (rate 1 for the training worker, a lease's price relative to it). Practice rows
carry rate 0: their CPU workers are outside the budget. A ledger without `budget_mode` is a dollar session.
"""
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3
import time

from simulators.polytopia.autoresearch.contract import TRAIN_SHAPE, rate as shape_rate, validate_session

ACTIVE = ('queued', 'running', 'uncertain')
NAME = '[A-Za-z0-9][A-Za-z0-9_-]{0,63}'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def terms(row):
    """A job's worker shape and charging rate (the session's unit per wall second of the worker; a row without
    one is a dollar session's, at its shape's list price)."""
    payload = json.loads(row['payload']) if row['payload'] else {}
    shape = payload.get('shape') or dict(TRAIN_SHAPE)
    return shape, float(payload.get('rate', shape_rate(shape)))


class Ledger:
    def __init__(self, root, config=None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / 'ledger.sqlite3'
        fresh = not self.path.exists()
        if fresh and config is None:
            raise ValueError('A new ledger needs an explicit session contract')
        self.db = sqlite3.connect(self.path, timeout=30, isolation_level=None, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript('''PRAGMA journal_mode=WAL;
          CREATE TABLE IF NOT EXISTS settings(id INTEGER PRIMARY KEY,config TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS jobs(name TEXT PRIMARY KEY,status TEXT,grant REAL,started REAL,ended REAL,
             charged REAL DEFAULT 0,payload TEXT,artifact TEXT,artifact_sha TEXT,provider_id TEXT,error TEXT);
          CREATE TABLE IF NOT EXISTS submissions(id INTEGER PRIMARY KEY AUTOINCREMENT,job TEXT,artifact_sha TEXT,at REAL);
          CREATE TABLE IF NOT EXISTS points(fraction TEXT PRIMARY KEY,submission INTEGER,charged REAL,at REAL);
          CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY AUTOINCREMENT,at REAL,kind TEXT,data TEXT);
          CREATE TABLE IF NOT EXISTS lifecycle(id INTEGER PRIMARY KEY,state TEXT);
          INSERT OR IGNORE INTO lifecycle VALUES(1,'active');
        ''')
        if fresh:
            self.db.execute('INSERT INTO settings VALUES(1,?)', (json.dumps(validate_session(config)),))
        self.config = json.loads(self.db.execute('SELECT config FROM settings WHERE id=1').fetchone()[0])
        if config is not None and validate_session(config) != self.config:
            if validate_session(config).get('budget_mode', 'usd') != self.mode:
                raise ValueError(f'This session was opened in {self.mode} budget mode; it cannot continue in another')
            raise ValueError('Existing session budget is immutable')

    @property
    def mode(self):
        """'usd' or 'seconds' (harness/budget.py): the unit of every grant, charge and remaining figure."""
        return self.config.get('budget_mode', 'usd')

    @property
    def unit(self):
        return 'usd' if self.mode == 'usd' else 'seconds'

    def amount(self, value):
        """A figure in the session's unit, for a message."""
        return f'${value:.2f}' if self.mode == 'usd' else f'{value:.0f} s'

    @contextmanager
    def transaction(self):
        self.db.execute('BEGIN IMMEDIATE')
        try:
            yield
            self.db.execute('COMMIT')
        except BaseException:
            self.db.execute('ROLLBACK')
            raise

    def event(self, kind, data):
        self.db.execute('INSERT INTO events(at,kind,data) VALUES(?,?,?)', (time.time(), kind, json.dumps(data)))

    def jobs(self):
        return [dict(r) for r in self.db.execute('SELECT * FROM jobs ORDER BY rowid')]

    def job(self, name):
        row = self.db.execute('SELECT * FROM jobs WHERE name=?', (name,)).fetchone()
        return dict(row) if row else None

    def lifecycle(self):
        return self.db.execute('SELECT state FROM lifecycle WHERE id=1').fetchone()[0]

    def submissions(self):
        return [dict(r) for r in self.db.execute('SELECT * FROM submissions ORDER BY id')]

    def used(self):
        """Charged or held, in the session's unit: the whole grant of queued, running and unresolved work, the charge of ended work.

        A running worker holds its whole grant, not its time so far: with several jobs at once, anything less
        would let a second grant be queued against money the first is still going to spend.
        """
        return sum(r['grant'] * terms(r)[1] if r['status'] in ACTIVE else r['charged'] for r in self.jobs())

    def budget(self):
        return float(self.config['budget_' + self.unit])

    def limit(self, key):
        """A concurrency limit of this session; None is no limit."""
        return self.config[key]

    def counted(self, payload):
        """Whether a job counts against the concurrency limits. In seconds mode practice does not: it keeps its own
        one-at-a-time rule beside the one training run or lease. A dollar session counts every job."""
        return not (self.mode == 'seconds' and payload.get('kind') == 'practice')

    def public(self):
        used = self.used()
        remaining = max(0., self.budget() - used)
        unit = self.unit
        return {'config': self.config, 'budget_mode': self.mode, 'charged_or_reserved_' + unit: used,
                'lifecycle': self.lifecycle(), 'remaining_' + unit: remaining, 'allocatable_' + unit: remaining,
                'submissions': self.submissions(), 'points': [dict(r) for r in self.db.execute('SELECT * FROM points')]}

    def allocated_elapsed(self, now=None):
        """Worker time (in the session's unit) actually allocated so far. Milestones use this, never queued or unknown holds."""
        now = time.time() if now is None else now
        total = 0.
        for row in self.jobs():
            if row['status'] == 'running':
                total += max(0., now - row['started']) * terms(row)[1]
            elif row['status'] not in ('queued', 'uncertain'):
                total += row['charged']
        return total

    def queue(self, name, seconds, payload, minimum=0.):
        if not re.fullmatch(NAME, name):
            raise ValueError('Invalid run name')
        if not math.isfinite(seconds) or seconds < minimum:
            raise ValueError(f'Allocation must be at least {minimum:g} seconds')
        shape, rate = terms({'payload': json.dumps(payload)})
        with self.transaction():
            if self.lifecycle() != 'active':
                raise ValueError('Session finalized; no new allocations')
            if self.job(name):
                raise ValueError('Fresh run name required; failed and stopped runs keep their names')
            if self.counted(payload):
                rows = [r for r in self.jobs() if r['status'] in ACTIVE and self.counted(json.loads(r['payload']))]
                held = [terms(r)[0] for r in rows]
                jobs = self.limit('max_concurrent_jobs')
                if len(held) >= jobs and jobs == 1:
                    raise ValueError('One training run or lease at a time in this session; wait for or stop: '
                                     + ', '.join(r['name'] for r in rows) + ' (an unresolved worker counts)')
                if len(held) >= jobs:
                    raise ValueError(f'At most {jobs} jobs at a time; stop one or wait (an unresolved worker counts)')
                for key, field, what in (('max_concurrent_gpus', 'gpu', 'GPUs'),
                                         ('max_concurrent_cpu_cores', 'cpu_cores', 'CPU cores')):
                    if self.limit(key) is not None and sum(h[field] for h in held) + shape[field] > self.limit(key):
                        raise ValueError(f'At most {self.limit(key)} {what} at a time across your active jobs')
            remaining = self.budget() - self.used()
            if seconds * rate > remaining + 1e-9:
                raise ValueError(f'Allocation of {self.amount(seconds * rate)} exceeds the remaining '
                                 f'{self.amount(max(0., remaining))}')
            self.db.execute('INSERT INTO jobs(name,status,grant,payload) VALUES(?,?,?,?)',
                            (name, 'queued', seconds, json.dumps(payload)))
            self.event('queued', {'name': name, 'payload_sha256': digest(payload), 'reserved_' + self.unit: seconds * rate,
                                  'worker_seconds': seconds, 'rate': rate, 'shape': shape})

    def start(self, name, provider_id, allocated_start):
        if allocated_start <= 0:
            raise ValueError('Provider allocation timestamp required')
        with self.transaction():
            cur = self.db.execute("UPDATE jobs SET status='running',provider_id=?,started=? WHERE name=? AND status='queued'",
                                  (provider_id, allocated_start, name))
            if cur.rowcount != 1:
                raise ValueError('Job was not queued')
            self.event('allocated', {'name': name, 'provider_id': provider_id, 'started': allocated_start})

    def milestones(self, now=None, final=False):
        used = self.allocated_elapsed(now)
        latest = self.db.execute('SELECT id FROM submissions ORDER BY id DESC LIMIT 1').fetchone()
        sub = latest[0] if latest else None
        for frac in self.config['grading_points']:
            if frac == 1 and not final:
                continue
            if final and frac == 1:
                self.db.execute('INSERT OR REPLACE INTO points VALUES(?,?,?,?)', (str(frac), sub, used, time.time()))
            elif used >= frac * self.budget():
                self.db.execute('INSERT OR IGNORE INTO points VALUES(?,?,?,?)', (str(frac), sub, used, time.time()))

    def settle(self, name, receipt, status='complete', error=None):
        """End a job on its worker's own start/finish record: the charge is that lifetime at the job's rate.

        The same record settles a job whose start was already written (`start`) and one recovered after a host
        interruption; a recorded start or worker must match it. Never a rerun and never an adopted artifact.
        """
        receipt = dict(receipt or {})
        start, end = receipt.get('started_at') or 0, receipt.get('finished_at') or 0
        if not all(math.isfinite(v) for v in (start, end)) or start <= 0 or end < start:
            raise ValueError('Completed authoritative provider receipt required')
        if status not in ('complete', 'stopped', 'failed'):
            raise ValueError('Invalid terminal status')
        with self.transaction():
            r = self.job(name)
            if not r or r['status'] not in ACTIVE:
                raise ValueError('Allocation already settled')
            provider_id = receipt.get('provider_id') or r['provider_id']
            if r['provider_id'] and provider_id != r['provider_id']:
                raise ValueError('Settlement provider identity differs')
            if r['started'] and abs(r['started'] - start) > 1e-6:
                raise ValueError('Settlement allocation start differs')
            charged = (end - start) * terms(r)[1]
            self.db.execute('UPDATE jobs SET status=?,provider_id=?,started=?,ended=?,charged=?,error=? WHERE name=?',
                            (status, provider_id, start, end, charged, error, name))
            self.milestones()
            self.event('settled', {'name': name, 'provider_id': provider_id, 'charged': charged,
                                   'worker_seconds': end - start, 'status': status, 'receipt': receipt})

    def uncertain(self, name, error):
        """The worker's end is unknown: the whole grant stays held until its provider record settles it."""
        with self.transaction():
            self.db.execute("UPDATE jobs SET status='uncertain',error=? WHERE name=? AND status IN ('queued','running')",
                            (error, name))
            self.event('provider_uncertain', {'name': name, 'error': error})

    def attach(self, name, artifact, artifact_sha):
        """Bind the sealed bundle a completed run left to its row."""
        with self.transaction():
            cur = self.db.execute("UPDATE jobs SET artifact=?,artifact_sha=? WHERE name=? AND status='complete' "
                                  'AND artifact IS NULL', (str(artifact), artifact_sha, name))
            if cur.rowcount != 1:
                raise ValueError('Only a completed run without a bundle takes one')
            self.event('artifact', {'name': name, 'artifact_sha': artifact_sha})

    def annotate(self, name, error, stopped=False):
        """Say why a settled job ended without a network; one the researcher stopped is `stopped`, not `failed`."""
        with self.transaction():
            r = self.job(name)
            if not r or r['status'] not in ('failed', 'stopped'):
                raise ValueError('Only a failed or stopped job is explained')
            status = 'stopped' if stopped else r['status']
            self.db.execute('UPDATE jobs SET status=?,error=? WHERE name=?', (status, error, name))
            self.event('explained', {'name': name, 'status': status, 'error': error})

    def abandon_unstarted(self, name, receipt, reason, stopped=False):
        """Settle a job whose provider worker was created but never started: no allocation, nothing charged.

        Only an unstarted queued (including claimed) or uncertain row qualifies, and only on the provider's
        own record showing no start. Queued waiting time is never charged; a started worker settles from its
        start/finish receipt through finish/reconcile instead.
        """
        receipt = dict(receipt or {})
        if not receipt.get('provider_id'):
            raise ValueError('Provider record of the abandoned worker required')
        if receipt.get('started_at'):
            raise ValueError('A started allocation is charged from its receipt')
        status = 'stopped' if stopped else 'failed'
        with self.transaction():
            r = self.job(name)
            if not r or r['status'] not in ('queued', 'uncertain') or r['started']:
                raise ValueError('Only an unstarted queued or uncertain job can be abandoned uncharged')
            if r['provider_id'] and r['provider_id'] != receipt['provider_id']:
                raise ValueError('Abandoned provider identity differs')
            self.db.execute('UPDATE jobs SET status=?,provider_id=?,charged=0,artifact=NULL,artifact_sha=NULL,error=? '
                            'WHERE name=?', (status, receipt['provider_id'], reason, name))
            self.event('abandoned_unstarted', {'name': name, 'provider_id': receipt['provider_id'], 'charged': 0,
                                               'status': status, 'reason': reason, 'receipt': receipt})

    def cancel_queued(self, name, error=None, status='stopped'):
        """End a queued job no worker was ever created for: nothing allocated, nothing charged."""
        if status not in ('stopped', 'failed'):
            raise ValueError('Invalid terminal status')
        with self.transaction():
            cur = self.db.execute("UPDATE jobs SET status=?,charged=0,error=? WHERE name=? AND status='queued'",
                                  (status, error, name))
            if cur.rowcount != 1:
                raise ValueError('Allocated jobs must be stopped and charged through the provider')
            self.event('cancelled', {'name': name, 'status': status, 'error': error})

    def register(self, name, artifact, artifact_sha, payload):
        """Record a bundle sealed from a running lease. It holds no worker and costs nothing itself: the lease
        that produced it is charged."""
        if not re.fullmatch(NAME, name):
            raise ValueError('Invalid run name')
        now = time.time()
        with self.transaction():
            if self.lifecycle() != 'active':
                raise ValueError('Session finalized; no new networks')
            self.db.execute('INSERT INTO jobs(name,status,grant,started,ended,charged,payload,artifact,artifact_sha) '
                            'VALUES(?,?,0,?,?,0,?,?,?)',
                            (name, 'complete', now, now, json.dumps(dict(payload, rate=0.)), str(artifact), artifact_sha))
            self.event('packaged', {'name': name, 'artifact_sha': artifact_sha, 'payload_sha256': digest(payload)})

    def submit(self, name):
        with self.transaction():
            if self.lifecycle() != 'active':
                raise ValueError('Session finalized; submission is immutable')
            if self.db.execute("SELECT 1 FROM jobs WHERE status='uncertain'").fetchone():
                raise ValueError('Reconcile unknown provider allocations before changing submissions')
            r = self.job(name)
            if not r or r['status'] != 'complete' or not r['artifact_sha']:
                raise ValueError('Only a finished run can be submitted')
            # A controller outage must not let a submission made after a crossed
            # threshold retroactively become that milestone's policy.
            self.milestones()
            at = time.time()
            cur = self.db.execute('INSERT INTO submissions(job,artifact_sha,at) VALUES(?,?,?)', (name, r['artifact_sha'], at))
            self.event('submitted', {'name': name, 'id': cur.lastrowid, 'artifact_sha': r['artifact_sha']})
            return {'id': cur.lastrowid, 'job': name, 'artifact_sha': r['artifact_sha'], 'at': at}

    def finalize(self):
        with self.transaction():
            if any(x['status'] in ACTIVE for x in self.jobs()):
                raise ValueError('Finish or reconcile every active allocation before finalizing')
            if self.lifecycle() != 'finalized':
                self.milestones(final=True)
                self.db.execute("UPDATE lifecycle SET state='finalized' WHERE id=1")
                self.event('finalized', {})
        return self.public()
