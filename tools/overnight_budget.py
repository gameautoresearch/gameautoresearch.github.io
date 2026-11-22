#!/usr/bin/env python3
"""Shared host-side reservations for the authorized eight-hour campaign.

No provider calls or secrets. Reserve before creating a worker/model session;
settle from completed provider metadata. Unknown starts/ends retain full holds.
"""
import argparse
from contextlib import closing, contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / 'evidence/overnight-20261108-1533Z/campaign.sqlite3'


def finite(value, *, positive=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < 0 or (positive and value == 0)):
        raise ValueError('Finite nonnegative amount required')
    return float(value)


def stamp(value):
    return datetime.fromisoformat(value.replace('Z', '+00:00')).timestamp()


class CampaignBudget:
    def __init__(self, game, path=DEFAULT, authorization=None):
        self.path = Path(path).resolve()
        self.authorization_path = Path(authorization or self.path.parent / 'authorization.json')
        raw = self.authorization_path.read_bytes()
        self.auth = json.loads(raw)
        self.auth_sha = hashlib.sha256(raw).hexdigest()
        self.game = game
        if game not in self.auth['initial_grants']:
            raise ValueError('Unknown game')
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self.connect()) as db:
            db.executescript('''
              CREATE TABLE IF NOT EXISTS settings(id INTEGER PRIMARY KEY, auth_sha TEXT);
              CREATE TABLE IF NOT EXISTS grants(game TEXT PRIMARY KEY,gpu_seconds REAL,
                resource_usd REAL,model_usd REAL);
              CREATE TABLE IF NOT EXISTS operations(id TEXT PRIMARY KEY,game TEXT,kind TEXT,
                bucket TEXT,gpu_hold REAL,usd_hold REAL,status TEXT,gpu_charged REAL,
                usd_charged REAL,created REAL,provider_id TEXT,spec TEXT,receipt TEXT);
              CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY,at REAL,game TEXT,
                action TEXT,operation TEXT,data TEXT);
            ''')
        with self.transaction() as db:
            row = db.execute('SELECT auth_sha FROM settings WHERE id=1').fetchone()
            if row is None:
                db.execute('INSERT INTO settings VALUES(1,?)', (self.auth_sha,))
                for game, grant in self.auth['initial_grants'].items():
                    db.execute('INSERT INTO grants VALUES(?,?,?,?)',
                               (game, grant['gpu_seconds'], grant['resource_usd'], grant['model_usd']))
            elif row['auth_sha'] != self.auth_sha:
                raise ValueError('Authorization changed; existing campaign cannot be reset')

    def connect(self):
        db = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        db.row_factory = sqlite3.Row
        db.execute('PRAGMA busy_timeout=30000')
        return db

    @contextmanager
    def transaction(self):
        db = self.connect()
        db.execute('BEGIN IMMEDIATE')
        try:
            yield db
            db.execute('COMMIT')
        except BaseException:
            db.execute('ROLLBACK')
            raise
        finally:
            db.close()

    def _event(self, db, action, operation=None, data=None):
        db.execute('INSERT INTO events(at,game,action,operation,data) VALUES(?,?,?,?,?)',
                   (time.time(), self.game, action, operation, json.dumps(data or {}, allow_nan=False)))

    @staticmethod
    def _usage(db, game=None):
        query = 'SELECT * FROM operations' + (' WHERE game=?' if game else '')
        rows = list(db.execute(query, (game,) if game else ()))
        gpu = sum(r['gpu_charged'] if r['status'] == 'settled' else r['gpu_hold'] for r in rows)
        resource = sum((r['usd_charged'] if r['status'] == 'settled' else r['usd_hold'])
                       for r in rows if r['bucket'] == 'resource')
        model = sum((r['usd_charged'] if r['status'] == 'settled' else r['usd_hold'])
                    for r in rows if r['bucket'] == 'model')
        return dict(gpu_seconds=gpu, resource_usd=resource, model_usd=model)

    def snapshot(self):
        with closing(self.connect()) as db:
            grants = {r['game']: dict(r) for r in db.execute('SELECT * FROM grants')}
            operations = [dict(r) for r in db.execute('SELECT * FROM operations ORDER BY created')]
            def split_usage(rows, settled):
                rows = [r for r in rows if (r['status'] == 'settled') == settled]
                gpu = 'gpu_charged' if settled else 'gpu_hold'
                usd = 'usd_charged' if settled else 'usd_hold'
                return dict(gpu_seconds=sum(r[gpu] for r in rows),
                            resource_usd=sum(r[usd] for r in rows if r['bucket'] == 'resource'),
                            model_usd=sum(r[usd] for r in rows if r['bucket'] == 'model'))
            own = [r for r in operations if r['game'] == self.game]
            return {'campaign_id': self.auth['campaign_id'], 'global_limits': self.auth['global_limits'],
                    'global_usage': self._usage(db), 'grants': grants,
                    'global_settled_usage': split_usage(operations, True),
                    'global_held_usage': split_usage(operations, False),
                    'game': self.game, 'game_usage': self._usage(db, self.game),
                    'game_settled_usage': split_usage(own, True),
                    'game_held_usage': split_usage(own, False),
                    'operations': operations,
                    'unresolved_gpu_seconds': sum(r['gpu_hold'] for r in operations if r['status'] != 'settled')}

    def _reserve(self, kind, bucket, gpu_hold, usd_hold, spec):
        gpu_hold, usd_hold = finite(gpu_hold), finite(usd_hold)
        if not isinstance(kind, str) or not kind or len(kind) > 160:
            raise ValueError('Concise operation kind required')
        now = time.time()
        if now >= stamp(self.auth['deadline_utc']):
            raise ValueError('Campaign deadline reached; no new paid work')
        if gpu_hold and now + gpu_hold > stamp(self.auth['worker_stop_utc']):
            raise ValueError('GPU reservation would cross worker-stop deadline')
        identity = uuid.uuid4().hex
        with self.transaction() as db:
            if gpu_hold:
                active = list(db.execute("SELECT game FROM operations WHERE gpu_hold>0 AND status!='settled'"))
                if any(r['game'] == self.game for r in active):
                    raise ValueError('Active/unknown GPU allocation blocks another game worker')
                if len(active) >= self.auth['global_limits']['max_concurrent_gpu_workers']:
                    raise ValueError('Global GPU concurrency limit reached')
            local = self._usage(db, self.game)
            total = self._usage(db)
            grant = dict(db.execute('SELECT * FROM grants WHERE game=?', (self.game,)).fetchone())
            limit = self.auth['global_limits']
            amounts = {'gpu_seconds': gpu_hold, 'resource_usd': usd_hold if bucket == 'resource' else 0,
                       'model_usd': usd_hold if bucket == 'model' else 0}
            global_keys = {'gpu_seconds': 'gpu_allocated_seconds',
                           'resource_usd': 'resource_charged_or_reserved_usd',
                           'model_usd': 'model_charged_or_reserved_usd'}
            for key, amount in amounts.items():
                if local[key] + amount > grant[key] + 1e-8:
                    raise ValueError(f'{self.game} {key} grant exhausted; ask parent for reserve transfer')
                if total[key] + amount > limit[global_keys[key]] + 1e-8:
                    raise ValueError(f'Global {key} ceiling exhausted')
            db.execute('INSERT INTO operations VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)',
                       (identity, self.game, kind, bucket, gpu_hold, usd_hold, 'held', None, None,
                        now, None, json.dumps(spec, allow_nan=False), None))
            self._event(db, 'reserve', identity, spec)
        return identity

    def reserve_worker(self, kind, max_seconds, gpu=None, cpu_cores=4, memory_gib=16,
                       sandbox=True, extra_usd=0):
        seconds = finite(max_seconds, positive=True)
        if time.time() + seconds > stamp(self.auth['deadline_utc']):
            raise ValueError('Worker reservation would cross the campaign deadline')
        cores, memory, extra = finite(cpu_cores, positive=True), finite(memory_gib, positive=True), finite(extra_usd)
        if gpu not in (None, 'H100', 'H100!', 'L4'):
            raise ValueError('Only pinned H100 or L4 GPU supported')
        rates = self.auth['pricing']
        flavor = 'sandbox' if sandbox else 'function'
        rate = cores * rates[f'{flavor}_cpu_per_core_second'] + memory * rates[f'{flavor}_memory_per_gib_second']
        if gpu:
            rate += rates['gpu_per_second'][gpu.rstrip('!')]
        quote = (rate * seconds + extra) * rates['reservation_margin']
        spec = dict(max_seconds=seconds, gpu=gpu, cpu_cores=cores, memory_gib=memory,
                    sandbox=bool(sandbox), quoted_rate_usd_per_second=rate, extra_usd=extra,
                    reservation_margin=rates['reservation_margin'])
        return self._reserve(kind, 'resource', seconds if gpu else 0, quote, spec)

    def reserve_model(self, kind, max_usd, model):
        allowed = self.auth.get('researcher_models', {})
        names = [*allowed.get('core', []), *allowed.get('optional', []), allowed.get('control_previous')]
        if model not in names:
            raise ValueError('Model absent from live-catalog campaign allowlist')
        return self._reserve(kind, 'model', 0, finite(max_usd, positive=True), {'model': model})

    def attach(self, operation, provider_id):
        with self.transaction() as db:
            row = self._operation(db, operation)
            if row['status'] == 'settled' or (row['provider_id'] and row['provider_id'] != provider_id):
                raise ValueError('Cannot replace provider identity or attach a settled operation')
            db.execute('UPDATE operations SET provider_id=? WHERE id=?', (provider_id, operation))
            self._event(db, 'attach', operation, {'provider_id': provider_id})

    def _operation(self, db, operation):
        row = db.execute('SELECT * FROM operations WHERE id=? AND game=?', (operation, self.game)).fetchone()
        if row is None:
            raise ValueError('Unknown operation or wrong game owner')
        return row

    def settle_worker(self, operation, receipt):
        start, end = receipt.get('started_at'), receipt.get('finished_at')
        finite(start, positive=True); finite(end, positive=True)
        if end < start or not receipt.get('provider_id'):
            raise ValueError('Completed authoritative provider interval required')
        with self.transaction() as db:
            row = self._operation(db, operation)
            if row['status'] == 'settled' or row['bucket'] != 'resource':
                raise ValueError('Wrong or already settled operation')
            if row['provider_id'] and row['provider_id'] != receipt['provider_id']:
                raise ValueError('Provider receipt does not match attached allocation')
            spec = json.loads(row['spec'])
            actual_gpu = str(receipt.get('gpu_type', '')).upper()
            if spec['gpu'] and spec['gpu'].rstrip('!') not in actual_gpu:
                raise ValueError('Actual GPU differs from pinned SKU; retain hold and audit with parent')
            if not spec['gpu'] and actual_gpu not in ('', 'CPU', 'NONE'):
                raise ValueError('CPU-only reservation received GPU; retain hold')
            seconds = end - start
            usd = (seconds * spec['quoted_rate_usd_per_second'] + spec['extra_usd']) * spec['reservation_margin']
            # Charge overrun in full, even beyond the reservation/allowance.
            db.execute("UPDATE operations SET status='settled',gpu_charged=?,usd_charged=?,provider_id=?,receipt=? WHERE id=?",
                       (seconds if spec['gpu'] else 0, usd, receipt['provider_id'], json.dumps(receipt, allow_nan=False), operation))
            self._event(db, 'settle_worker', operation, {'seconds': seconds, 'conservative_usd': usd})

    def settle_model(self, operation, liability_usd, proof):
        amount = finite(liability_usd)
        if proof.get('closed') is not True or not proof.get('ledger_path') or not proof.get('ledger_sha256'):
            raise ValueError('Closed persistent model ledger with hash/path required')
        path = Path(proof['ledger_path'])
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != proof['ledger_sha256']:
            raise ValueError('Model ledger proof mismatch')
        if json.loads(path.read_text()).get('closed') is not True:
            raise ValueError('Underlying model-call ledger is not closed')
        with self.transaction() as db:
            row = self._operation(db, operation)
            if row['status'] == 'settled' or row['bucket'] != 'model':
                raise ValueError('Wrong or already settled operation')
            # Caller reports actual priced calls plus every unknown call hold;
            # parent independently reviews the hashed underlying ledger.
            db.execute("UPDATE operations SET status='settled',gpu_charged=0,usd_charged=?,receipt=? WHERE id=?",
                       (amount, json.dumps(proof, allow_nan=False), operation))
            self._event(db, 'settle_model', operation, {'charged_plus_unknown_holds_usd': amount})

    def no_allocation(self, operation, proof):
        if proof.get('provider_inventory_checked') is not True or proof.get('worker_count') != 0 or not proof.get('receipt_path'):
            raise ValueError('Host provider-inventory proof of no allocation required')
        if not Path(proof['receipt_path']).is_file():
            raise ValueError('Missing no-allocation receipt')
        with self.transaction() as db:
            row = self._operation(db, operation)
            if row['status'] == 'settled' or row['bucket'] != 'resource' or row['provider_id']:
                raise ValueError('Cannot clear known or already settled worker')
            db.execute("UPDATE operations SET status='settled',gpu_charged=0,usd_charged=0,receipt=? WHERE id=?",
                       (json.dumps(proof, allow_nan=False), operation))
            self._event(db, 'proven_no_allocation', operation, proof)

    def parent_grant(self, target, gpu_seconds=0, resource_usd=0, model_usd=0, reason=''):
        """Parent only: allocate withheld reserve, with immutable event history."""
        values = [finite(v) for v in (gpu_seconds, resource_usd, model_usd)]
        with self.transaction() as db:
            rows = [dict(r) for r in db.execute('SELECT * FROM grants')]
            keys = ('gpu_seconds', 'resource_usd', 'model_usd')
            limits = self.auth['global_limits']
            ceiling = (limits['gpu_allocated_seconds'], limits['resource_charged_or_reserved_usd'], limits['model_charged_or_reserved_usd'])
            if target not in {r['game'] for r in rows}:
                raise ValueError('Unknown grant target')
            if any(sum(r[k] for r in rows) + v > cap + 1e-8 for k, v, cap in zip(keys, values, ceiling)):
                raise ValueError('Parent reserve exhausted; no allowance reset permitted')
            db.execute('UPDATE grants SET gpu_seconds=gpu_seconds+?,resource_usd=resource_usd+?,model_usd=model_usd+? WHERE game=?',
                       (*values, target))
            self._event(db, 'parent_reserve_grant', data=dict(target=target, gpu_seconds=values[0], resource_usd=values[1], model_usd=values[2], reason=reason))

    def parent_transfer(self, source, target, gpu_seconds=0, resource_usd=0, model_usd=0, reason=''):
        """Parent only: transfer genuinely uncommitted allowance; keep total fixed."""
        amounts = dict(zip(('gpu_seconds', 'resource_usd', 'model_usd'),
                           (finite(v) for v in (gpu_seconds, resource_usd, model_usd))))
        if source == target:
            raise ValueError('Different source and target required')
        with self.transaction() as db:
            rows = {r['game']: dict(r) for r in db.execute('SELECT * FROM grants')}
            if source not in rows or target not in rows:
                raise ValueError('Unknown grant owner')
            used = self._usage(db, source)
            if any(amount > rows[source][key]-used[key]+1e-8 for key, amount in amounts.items()):
                raise ValueError('Cannot transfer spent or reserved allowance')
            values = tuple(amounts[k] for k in ('gpu_seconds','resource_usd','model_usd'))
            db.execute('UPDATE grants SET gpu_seconds=gpu_seconds-?,resource_usd=resource_usd-?,model_usd=model_usd-? WHERE game=?', (*values,source))
            db.execute('UPDATE grants SET gpu_seconds=gpu_seconds+?,resource_usd=resource_usd+?,model_usd=model_usd+? WHERE game=?', (*values,target))
            self._event(db, 'parent_unused_transfer', data=dict(source=source,target=target,reason=reason,**amounts))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--game', required=True)
    p.add_argument('--db', type=Path, default=DEFAULT)
    p.add_argument('--authorization', type=Path)
    p.add_argument('command', choices=('status', 'reserve-worker', 'reserve-model', 'attach', 'settle-worker', 'settle-model', 'no-allocation'))
    p.add_argument('--kind'); p.add_argument('--seconds', type=float); p.add_argument('--gpu', choices=('H100', 'H100!', 'L4'))
    p.add_argument('--cpu', type=float, default=4); p.add_argument('--memory', type=float, default=16)
    p.add_argument('--function', action='store_true'); p.add_argument('--usd', type=float)
    p.add_argument('--model'); p.add_argument('--operation'); p.add_argument('--provider-id'); p.add_argument('--receipt', type=Path)
    a = p.parse_args()
    b = CampaignBudget(a.game, a.db, a.authorization)
    result = None
    if a.command == 'status': result = b.snapshot()
    elif a.command == 'reserve-worker': result = {'operation': b.reserve_worker(a.kind, a.seconds, a.gpu, a.cpu, a.memory, not a.function)}
    elif a.command == 'reserve-model': result = {'operation': b.reserve_model(a.kind, a.usd, a.model)}
    elif a.command == 'attach': b.attach(a.operation, a.provider_id)
    elif a.command == 'settle-worker': b.settle_worker(a.operation, json.loads(a.receipt.read_text()))
    elif a.command == 'settle-model': b.settle_model(a.operation, a.usd, json.loads(a.receipt.read_text()))
    elif a.command == 'no-allocation': b.no_allocation(a.operation, json.loads(a.receipt.read_text()))
    print(json.dumps(result or {'status': 'recorded'}, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
