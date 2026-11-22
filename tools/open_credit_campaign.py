"""Time-bounded, explicitly uncapped Modal accounting for reference research.

Historical capped campaigns keep using overnight_budget.CampaignBudget unchanged.
This subclass requires a separate authorization and an explicit new ledger path.
"""
import json
from pathlib import Path
import time
import uuid

from tools.overnight_budget import CampaignBudget, finite, stamp


class OpenCreditBudget(CampaignBudget):
    def __init__(self, game, path, authorization=None):
        path = Path(path).resolve()
        auth_path = Path(authorization or path.parent / 'authorization.json')
        auth = json.loads(auth_path.read_text())
        if auth.get('budget_mode') != 'explicit-user-uncapped-modal-time-bounded':
            raise ValueError('Explicit uncapped Modal authorization required')
        limits = auth['global_limits']
        if limits['resource_charged_or_reserved_usd'] is not None:
            raise ValueError('Use the historical guard for a capped campaign')
        if stamp(auth['deadline_utc']) <= stamp(auth['started_at_utc']):
            raise ValueError('Finite positive campaign window required')
        for key in ('max_concurrent_gpu_workers', 'max_concurrent_gpu_workers_per_game'):
            if not isinstance(limits[key], int) or isinstance(limits[key], bool) or limits[key] < 1:
                raise ValueError('Positive bounded concurrency required')
        if game not in auth.get('worker_profiles', {}):
            raise ValueError('Game profile required')
        super().__init__(game, path, auth_path)

    def _reserve(self, kind, bucket, gpu_hold, usd_hold, spec):
        gpu_hold, usd_hold = finite(gpu_hold), finite(usd_hold)
        if not isinstance(kind, str) or not kind or len(kind) > 160:
            raise ValueError('Concise operation kind required')
        identity = uuid.uuid4().hex
        with self.transaction() as db:
            now = time.time()
            if now >= stamp(self.auth['deadline_utc']):
                raise ValueError('Campaign deadline reached')
            if now + spec.get('max_seconds', 0) > stamp(self.auth['worker_stop_utc']):
                raise ValueError('Reservation crosses absolute worker stop')
            if spec.get('purpose') == 'training' and now + spec['max_seconds'] > stamp(self.auth['selection_close_utc']):
                raise ValueError('Training would cross selection closure')
            limits = self.auth['global_limits']
            if gpu_hold:
                active = list(db.execute("SELECT game FROM operations WHERE gpu_hold>0 AND status!='settled'"))
                if sum(r['game'] == self.game for r in active) >= limits['max_concurrent_gpu_workers_per_game']:
                    raise ValueError('Game GPU concurrency/unknown holds exhausted')
                if len(active) >= limits['max_concurrent_gpu_workers']:
                    raise ValueError('Global GPU concurrency/unknown holds exhausted')
            local, total = self._usage(db, self.game), self._usage(db)
            grant = dict(db.execute('SELECT * FROM grants WHERE game=?', (self.game,)).fetchone())
            amounts = {'gpu_seconds': gpu_hold, 'resource_usd': usd_hold if bucket == 'resource' else 0,
                       'model_usd': usd_hold if bucket == 'model' else 0}
            global_keys = {'gpu_seconds': 'gpu_allocated_seconds', 'resource_usd': 'resource_charged_or_reserved_usd',
                           'model_usd': 'model_charged_or_reserved_usd'}
            for key, amount in amounts.items():
                for used, cap in ((local[key], grant[key]), (total[key], limits[global_keys[key]])):
                    if cap is not None and used + amount > finite(cap) + 1e-8:
                        raise ValueError(f'{key} ceiling exhausted')
            db.execute('INSERT INTO operations VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)',
                       (identity, self.game, kind, bucket, gpu_hold, usd_hold, 'held', None, None,
                        now, None, json.dumps(spec, allow_nan=False), None))
            self._event(db, 'reserve', identity, spec)
        return identity

    def reserve_worker(self, kind, max_seconds, gpu=None, cpu_cores=4, memory_gib=16,
                       sandbox=True, extra_usd=0, purpose='training'):
        seconds = finite(max_seconds, positive=True)
        if purpose not in ('training', 'development', 'qualification', 'final', 'build', 'observer'):
            raise ValueError('Explicit supported worker purpose required')
        if seconds > self.auth['max_worker_seconds']:
            raise ValueError('Split work into bounded resumable stages')
        cores, memory, extra = finite(cpu_cores, positive=True), finite(memory_gib, positive=True), finite(extra_usd)
        profile = self.auth['worker_profiles'][self.game]
        if gpu is not None:
            if gpu not in ('H100', 'H100!', 'L4') or gpu.rstrip('!') != profile['gpu'].rstrip('!'):
                raise ValueError('Wrong pinned GPU')
            if cores != profile['cpu_cores'] or memory != profile['memory_gib']:
                raise ValueError('Wrong attached CPU/RAM')
        rates = self.auth['pricing']
        flavor = 'sandbox' if sandbox else 'function'
        rate = cores * rates[f'{flavor}_cpu_per_core_second'] + memory * rates[f'{flavor}_memory_per_gib_second']
        if gpu:
            rate += rates['gpu_per_second'][gpu.rstrip('!')]
        spec = dict(max_seconds=seconds, gpu=gpu, cpu_cores=cores, memory_gib=memory,
                    sandbox=bool(sandbox), quoted_rate_usd_per_second=rate, extra_usd=extra,
                    reservation_margin=rates['reservation_margin'], purpose=purpose)
        return self._reserve(kind, 'resource', seconds if gpu else 0,
                             (rate * seconds + extra) * rates['reservation_margin'], spec)

    def parent_grant(self, *args, **kwargs):
        raise ValueError('Uncapped Modal grants require no transfers; preserve authorization')

    def parent_transfer(self, *args, **kwargs):
        raise ValueError('Uncapped Modal grants require no transfers; preserve authorization')
