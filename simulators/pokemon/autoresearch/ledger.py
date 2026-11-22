"""One session's allocation ledger; every full provider allocation remains chargeable.

A reservation for the whole lease precedes any provider call. It settles only on
a verified provider stop, at the provider's own interval. A reservation that
provably never reached the provider settles at zero; anything uncertain stays
held and blocks further dispatch.

The research allocation is in one of two modes (harness/budget.py), bound with
the session's limits: `seconds`, one worker at a time within a number of
allocated worker seconds, or `usd`, several workers at once within a dollar
budget, each lease held at `usd_per_second` until it settles. In dollar mode an
uncertain allocation keeps its whole hold and its place among the jobs at once
instead of blocking every other dispatch.
"""
from contextlib import contextmanager
import json
import math
from pathlib import Path
import secrets
import sqlite3
import time
import uuid

from .contract import identity

RATE_GPU = .000542 + 4 * .00003942 + 16 * .00000667   # L40S, 4 CPU, 16 GiB
RATE_CPU = 8 * .00003942 + 32 * .00000667             # conservative quote retained for CPU graders
KINDS = ('research', 'final')                         # GPU training/practice leases; CPU private grading
MIN_LEASE_SECONDS = 60
SCHEMA = '''
CREATE TABLE IF NOT EXISTS metadata(name TEXT PRIMARY KEY,value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY,name TEXT UNIQUE NOT NULL,kind TEXT,seconds REAL,cash REAL,
    status TEXT,provider TEXT,actual_seconds REAL,actual_cash REAL,receipt TEXT,created REAL);
CREATE TABLE IF NOT EXISTS submissions(id INTEGER PRIMARY KEY,name TEXT,bundle TEXT,bundle_sha256 TEXT,
    at_allocated_seconds REAL,created REAL);
'''
LIMITS = ('research_seconds', 'max_lease_seconds', 'grading_seconds')
USD_LIMITS = ('research_usd', 'usd_per_second', 'max_lease_seconds', 'grading_seconds')
MODES = ('seconds', 'usd')
CONCURRENCY = ('max_concurrent_jobs', 'max_concurrent_gpus', 'max_concurrent_cpu_cores')


def rate(kind):
    return RATE_CPU if kind == 'final' else RATE_GPU


class Ledger:
    def __init__(self, path, limits=None, fresh_final=False):
        """`limits` binds a new session; None reopens an existing ledger unchanged.

        `fresh_final` marks a new session as the one fresh final training of a
        submitted recipe. It is part of the session's own immutable state.
        """
        self.path = Path(path)
        if limits is None and not self.path.exists():
            raise ValueError('No existing session ledger')
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.transaction() as db:
            for statement in SCHEMA.split(';'):
                db.execute(statement)
            if limits is not None:
                self._bind(db, 'limits', json.dumps(self.bound(limits), sort_keys=True))
                self._bind(db, 'fresh_final', 'true' if fresh_final else 'false')
            saved = self._get(db, 'limits')
            if saved is None:
                raise ValueError('No existing session ledger')
            self.limits = json.loads(saved)
            self.fresh_final = self._get(db, 'fresh_final') == 'true'
        # A ledger bound before the dollar mode existed is a seconds-mode ledger.
        self.mode = self.limits.get('budget_mode', 'seconds')
        self.capacity = self.jobs_at_once(self.limits)

    @staticmethod
    def bound(limits):
        """The immutable limits of a new session: its mode's allowances, concurrency and (dollars) prices."""
        mode = limits.get('budget_mode', 'seconds')
        if mode not in MODES:
            raise ValueError('budget_mode is seconds or usd')
        value = {key: float(limits[key]) for key in (USD_LIMITS if mode == 'usd' else LIMITS)}
        if any(not math.isfinite(number) or number <= 0 for number in value.values()):
            raise ValueError('Positive finite session limits required')
        if 'budget_mode' not in limits:
            return value  # the original seconds-mode binding
        value['budget_mode'] = mode
        for key in CONCURRENCY:
            if limits.get(key) is not None:
                value[key] = int(limits[key])
        for key in ('research_shape', 'prices'):  # how a dollar-mode lease's price was derived
            if limits.get(key) is not None:
                value[key] = dict(limits[key])
        if Ledger.jobs_at_once(value) < 1:
            raise ValueError('At least one job at a time required')
        return value

    @staticmethod
    def jobs_at_once(limits):
        """How many research workers may hold an allocation at once: every one has the research shape."""
        if 'budget_mode' not in limits:
            return 1
        shape = limits.get('research_shape') or {}
        caps = [limits.get('max_concurrent_jobs')]
        if limits.get('max_concurrent_gpus') is not None and shape.get('gpu'):
            caps.append(limits['max_concurrent_gpus'] // shape['gpu'])
        if limits.get('max_concurrent_cpu_cores') is not None and shape.get('cpu_cores'):
            caps.append(limits['max_concurrent_cpu_cores'] // shape['cpu_cores'])
        caps = [int(cap) for cap in caps if cap is not None]
        return min(caps) if caps else 1

    @contextmanager
    def transaction(self):
        # A connection per transaction: the request thread and a worker's own thread never share one.
        db = sqlite3.connect(self.path, timeout=60, isolation_level=None)
        try:
            db.execute('BEGIN IMMEDIATE')
            yield db
            db.execute('COMMIT')
        except BaseException:
            if db.in_transaction:
                db.execute('ROLLBACK')
            raise
        finally:
            db.close()

    @staticmethod
    def _get(db, name):
        row = db.execute('SELECT value FROM metadata WHERE name=?', (name,)).fetchone()
        return row[0] if row else None

    def _bind(self, db, name, value):
        old = self._get(db, name)
        if old is not None and old != value:
            raise ValueError('Immutable binding changed: ' + name)
        db.execute('INSERT OR IGNORE INTO metadata VALUES (?,?)', (name, value))

    @staticmethod
    def _usage(db, kind):
        return db.execute('SELECT COALESCE(SUM(COALESCE(actual_seconds,seconds)),0) FROM jobs WHERE kind=?',
                          (kind,)).fetchone()[0]

    def reserve(self, name, kind, seconds, now=None):
        now = time.time() if now is None else now
        if kind not in KINDS or type(seconds) not in (float, int) or not math.isfinite(seconds) \
                or seconds < MIN_LEASE_SECONDS:
            raise ValueError('Finite lease of at least 60 seconds required')
        with self.transaction() as db:
            if self._get(db, 'allocation_quote_breached'):
                raise ValueError('A worker exceeded its reservation; no further launches in this session')
            if self.mode == 'usd' and kind == 'research':
                # Parallel: every unsettled research allocation (queued, running or uncertain) holds its place.
                held = db.execute("SELECT COUNT(*) FROM jobs WHERE kind='research' AND status!='settled'").fetchone()[0]
                if held >= self.capacity:
                    raise ValueError(f'At most {self.capacity} jobs at once; {held} still hold their allocation')
            else:
                if db.execute("SELECT 1 FROM jobs WHERE status IN ('unknown','reserved') LIMIT 1").fetchone():
                    raise ValueError('Unattached or unknown allocation blocks dispatch')
                if db.execute("SELECT 1 FROM jobs WHERE status!='settled' LIMIT 1").fetchone():
                    raise ValueError('One worker at a time')
            if db.execute('SELECT 1 FROM jobs WHERE name=?', (name,)).fetchone():
                raise ValueError('Use a new job name; failed attempts are retained')
            locked = self._get(db, 'final_locked')
            if kind == 'research':
                if locked:
                    raise ValueError('Session finalized')
                if self.fresh_final and db.execute("SELECT 1 FROM jobs WHERE kind='research'").fetchone():
                    raise ValueError('One fresh final attempt only')
                maximum = self.limits['max_lease_seconds']
                if self.mode == 'usd':
                    # Dollars: a lease holds its whole price until it settles, so grants never pass the budget.
                    price = self.limits['usd_per_second']
                    cap, used, need = self.limits['research_usd'], self._usage(db, kind) * price, seconds * price
                else:
                    cap, used, need = self.limits['research_seconds'], self._usage(db, kind), seconds
            else:
                if not locked:
                    raise ValueError('The final policy must lock before private evaluation')
                cap = maximum = self.limits['grading_seconds']
                used, need = self._usage(db, kind), seconds
            if seconds > maximum:
                raise ValueError(f'Lease exceeds the {maximum:g}-second limit for one worker')
            if round(used + need, 9) > cap:
                raise ValueError('Lease exceeds the remaining allocation')
            ident = uuid.uuid4().hex
            db.execute('INSERT INTO jobs VALUES(?,?,?,?,?,?,?,?,?,?,?)',
                       (ident, name, kind, seconds, seconds * rate(kind) * 1.05, 'reserved', None, None, None, '{}', now))
            return ident

    def cancel_unattached(self, ident, reason):
        """Record a known pre-provider failure; never erase a row.

        Call only before entering Sandbox.create. Once creation was attempted,
        ownership is uncertain and must use unknown()/provider reconciliation.
        """
        with self.transaction() as db:
            row = db.execute('SELECT status,provider FROM jobs WHERE id=?', (ident,)).fetchone()
            if row != ('reserved', None):
                raise ValueError('Only an unattempted unattached reservation can settle at zero')
            receipt = dict(provider_created=False, provider_creation_not_attempted=True, reason=str(reason),
                           reservation_retained=True)
            db.execute("UPDATE jobs SET status='settled',actual_seconds=0,actual_cash=0,receipt=? WHERE id=?",
                       (json.dumps(receipt), ident))
        return receipt

    def attach(self, ident, provider):
        with self.transaction() as db:
            if db.execute('SELECT status,provider FROM jobs WHERE id=?', (ident,)).fetchone() != ('reserved', None):
                raise ValueError('Provider already attached')
            db.execute("UPDATE jobs SET provider=?,status='active' WHERE id=?", (provider, ident))

    def settle(self, ident, seconds, receipt):
        if not math.isfinite(seconds) or seconds < 0 or not receipt.get('provider_verified_stopped'):
            raise ValueError('Complete provider stop receipt required')
        with self.transaction() as db:
            row = db.execute('SELECT seconds,status,kind,actual_seconds,receipt FROM jobs WHERE id=?',
                             (ident,)).fetchone()
            if not row:
                raise ValueError('Unknown allocation')
            if row[1] == 'settled':
                if row[3] != seconds or json.loads(row[4]) != receipt:
                    raise ValueError('Settled allocation receipt cannot change')
                return
            state = 'settled' if seconds <= row[0] else 'unknown'
            db.execute('UPDATE jobs SET actual_seconds=?,actual_cash=?,status=?,receipt=? WHERE id=?',
                       (seconds, seconds * rate(row[2]) * 1.05, state, json.dumps(receipt), ident))
            if state == 'unknown':
                self._bind(db, 'allocation_quote_breached', 'true')
        if state == 'unknown':
            raise RuntimeError('Provider exceeded reservation; no more launches')

    def settle_unstarted(self, ident, receipt, reason=None):
        """Settle a created worker the provider never started at 0 seconds, keeping the provider's receipt.

        Only an unmeasured reserved, active or unknown row qualifies, and only on the provider's own record of
        that exact worker showing no start (harness.workers: queued time is never charged). A started worker
        always settles from its start/finish receipt instead.
        """
        receipt = dict(receipt or {})
        if receipt.get('started_at') or not receipt.get('provider_id'):
            raise ValueError("Only the provider's record of an unstarted worker settles at zero")
        with self.transaction() as db:
            row = db.execute('SELECT status,provider,actual_seconds,receipt FROM jobs WHERE id=?', (ident,)).fetchone()
            if not row:
                raise ValueError('Unknown allocation')
            if row[0] == 'settled':
                saved = json.loads(row[3])
                if saved.get('provider_never_started') and row[2] == 0 and saved.get('provider_id') == receipt['provider_id']:
                    return saved
                raise ValueError('Allocation already settled')
            if row[0] not in ('reserved', 'active', 'unknown') or row[2] is not None:
                raise ValueError('Only an unmeasured reserved, active or unknown allocation can settle as never started')
            if row[1] is not None and row[1] != receipt['provider_id']:
                raise ValueError('Receipt is for a different worker')
            value = dict(receipt, provider_never_started=True, charged_seconds=0,
                         reason=str(reason or 'The provider never started this worker; it was terminated'))
            db.execute("UPDATE jobs SET status='settled',provider=COALESCE(provider,?),actual_seconds=0,actual_cash=0,"
                       "receipt=? WHERE id=?", (receipt['provider_id'], json.dumps(value), ident))
        return value

    def unknown(self, ident, detail):
        with self.transaction() as db:
            db.execute("UPDATE jobs SET status='unknown',receipt=? WHERE id=? AND status!='settled'",
                       (json.dumps(detail), ident))

    def job(self, ident):
        return next((job for job in self.snapshot()['jobs'] if job['id'] == ident), None)

    def submit(self, name, bundle, digest):
        """Register the immutable published policy package of one of this session's own settled runs."""
        with self.transaction() as db:
            if self._get(db, 'final_locked'):
                raise ValueError('Session finalized')
            row = db.execute("SELECT status FROM jobs WHERE name=? AND kind='research'", (name,)).fetchone()
            if row is None or row[0] != 'settled':
                raise ValueError('Only a completed registered job can be submitted')
            used = self._usage(db, 'research')
            cursor = db.execute('INSERT INTO submissions(name,bundle,bundle_sha256,at_allocated_seconds,created) '
                                'VALUES(?,?,?,?,?)', (name, str(bundle), digest, used, time.time()))
            return dict(id=cursor.lastrowid, job=name, bundle=str(bundle), bundle_sha256=digest,
                        at_allocated_seconds=used)

    def finalize(self):
        """Lock the latest explicit submission as the session's final identity."""
        with self.transaction() as db:
            if db.execute("SELECT 1 FROM jobs WHERE status!='settled'").fetchone():
                raise ValueError('Unsettled allocations prevent finalization')
            row = db.execute('SELECT id,name,bundle,bundle_sha256 FROM submissions ORDER BY id DESC LIMIT 1').fetchone()
            if row is None:
                raise ValueError('No valid explicit submission')
            selected = dict(zip(('id', 'job', 'bundle', 'bundle_sha256'), row))
            self._bind(db, 'final_locked', json.dumps(selected, sort_keys=True))
            return selected

    def final(self):
        with self.transaction() as db:
            value = self._get(db, 'final_locked')
        return json.loads(value) if value else None

    def private_bank(self, clusters):
        """Host-only seeds, created exactly once and only after the final identity locked."""
        with self.transaction() as db:
            final = self._get(db, 'final_locked')
            if not final:
                raise ValueError('The final policy must lock before private seeds exist')
            existing = self._get(db, 'private_bank')
            if existing:
                return json.loads(existing)
            seeds = secrets.SystemRandom().sample(range(1_900_000_000, 2_000_000_000), clusters)
            value = dict(primary=seeds, final_sha256=identity(json.loads(final)))
            self._bind(db, 'private_bank', json.dumps(value, sort_keys=True))
            return value

    def snapshot(self):
        with self.transaction() as db:
            cols = [r[1] for r in db.execute('PRAGMA table_info(jobs)')]
            jobs = [dict(zip(cols, r)) for r in db.execute('SELECT * FROM jobs ORDER BY created')]
            metadata = dict(db.execute('SELECT name,value FROM metadata'))
            names = [r[1] for r in db.execute('PRAGMA table_info(submissions)')]
            submissions = [dict(zip(names, r)) for r in db.execute('SELECT * FROM submissions ORDER BY id')]
            used = self._usage(db, 'research')
        if 'private_bank' in metadata:
            metadata['private_bank'] = 'host-private commitment:' + identity(json.loads(metadata['private_bank']))
        unresolved = sum(j['seconds'] for j in jobs if j['status'] != 'settled')
        value = dict(limits=self.limits, metadata=metadata, jobs=jobs, submissions=submissions,
                     budget_mode=self.mode, jobs_at_once=self.capacity,
                     research_seconds_used_or_reserved=used, unresolved_seconds=unresolved,
                     estimated_compute_usd=sum(j['actual_cash'] if j['actual_cash'] is not None else j['cash']
                                               for j in jobs),
                     cash_admission_enforced=self.mode == 'usd', finalized='final_locked' in metadata)
        if self.mode == 'usd':
            price = self.limits['usd_per_second']
            value.update(research_usd_used_or_reserved=used * price,
                         research_usd_remaining=max(0.0, self.limits['research_usd'] - used * price),
                         unresolved_usd=sum(j['seconds'] * price for j in jobs
                                            if j['status'] != 'settled' and j['kind'] == 'research'))
        else:
            value['research_seconds_remaining'] = max(0.0, self.limits['research_seconds'] - used)
        return value
