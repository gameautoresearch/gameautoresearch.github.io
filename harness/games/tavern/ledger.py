"""Tavern's dollar-mode worker ledger (harness/budget.py `usd`), beside the native seconds ledger.

The seconds mode keeps the native ledger (simulators/tavern/autoresearch/ledger.py) unchanged: one hold at a time,
and an unknown end blocks every launch. The dollar mode allows several holds at once: each training run holds its
whole grant until the provider's receipt settles it, and holds plus charges never pass the budget. Rows stay in
seconds of the training worker, the only shape on this ledger (CPU practice and grading are not charged); the cap
is US dollars and every second is priced at the session's recorded `usd_per_second`.

The ledger records its mode: a session is never reopened in the other one.
"""
import math
from pathlib import Path
import sqlite3
import time

from simulators.tavern.autoresearch.ledger import Ledger


def recorded_mode(path):
    """'usd' or 'seconds' for an existing ledger, None for one not created yet. Creates nothing."""
    path = Path(path)
    if not path.exists():
        return None
    db = sqlite3.connect(path)
    try:
        tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if 'budget' in tables and db.execute('SELECT mode FROM budget').fetchone():
            return db.execute('SELECT mode FROM budget').fetchone()[0]
        if 'settings' in tables and db.execute('SELECT cap FROM settings').fetchone():
            return 'seconds'
        return None
    finally:
        db.close()


class DollarLedger(Ledger):
    """Concurrent holds within a dollar cap. `cap` is US dollars; `usd_per_second` prices the training worker."""

    def __init__(self, path, cap, usd_per_second):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path, isolation_level=None)
        self.rate = float(usd_per_second)
        self.db.executescript('CREATE TABLE IF NOT EXISTS settings(cap REAL,closed INTEGER); '
                              'CREATE TABLE IF NOT EXISTS allocations(name TEXT PRIMARY KEY,reserved REAL,actual REAL,'
                              'receipt TEXT,created REAL); '
                              'CREATE TABLE IF NOT EXISTS budget(mode TEXT,usd_per_second REAL);')
        self.db.execute('BEGIN IMMEDIATE')
        try:
            settings = self.db.execute('SELECT cap FROM settings').fetchone()
            budget = self.db.execute('SELECT mode,usd_per_second FROM budget').fetchone()
            if settings is None and budget is None:
                self.db.execute('INSERT INTO settings VALUES(?,0)', (cap,))
                self.db.execute("INSERT INTO budget VALUES('usd',?)", (self.rate,))
            elif budget is None:
                raise ValueError('This session was created in seconds mode; it cannot be reopened in usd mode')
            elif settings[0] != cap or budget[1] != self.rate:
                raise ValueError('Cannot reset budget')
            self.db.execute('COMMIT')
        except BaseException:
            self.db.execute('ROLLBACK')
            raise

    def reserve(self, name, seconds):
        """Hold `seconds` of the training worker; refused when holds and charges would pass the dollar cap."""
        self.db.execute('BEGIN IMMEDIATE')
        try:
            if self.db.execute('SELECT closed FROM settings').fetchone()[0]:
                raise ValueError('Session finalized')
            rows = self.db.execute('SELECT reserved,actual FROM allocations').fetchall()
            held = sum(reserved if actual is None else actual for reserved, actual in rows)
            cap = self.db.execute('SELECT cap FROM settings').fetchone()[0]
            if not math.isfinite(seconds) or seconds <= 0 or (held + seconds) * self.rate > cap + 1e-9:
                raise ValueError('Allocation exceeds the remaining session budget')
            self.db.execute('INSERT INTO allocations VALUES(?,?,NULL,NULL,?)', (name, seconds, time.time()))
            self.db.execute('COMMIT')
        except BaseException:
            self.db.execute('ROLLBACK')
            raise

    def snapshot(self):
        rows = [dict(zip(('name', 'reserved', 'actual', 'receipt', 'created'), r))
                for r in self.db.execute('SELECT * FROM allocations')]
        cap, closed = self.db.execute('SELECT cap,closed FROM settings').fetchone()
        actual = sum(r['actual'] or 0 for r in rows)
        hold = sum(r['reserved'] for r in rows if r['actual'] is None)
        return dict(budget_mode='usd', usd_per_second=self.rate, cap_usd=cap, charged_usd=actual * self.rate,
                    unresolved_usd=hold * self.rate, available_usd=max(0., cap - (actual + hold) * self.rate),
                    cap_seconds=cap / self.rate, allocated_seconds=actual, unresolved_seconds=hold,
                    available_seconds=max(0., cap / self.rate - actual - hold), closed=bool(closed), allocations=rows)
