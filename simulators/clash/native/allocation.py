"""Cumulative allocation elapsed-time sidecar, using only the standard library.

One exclusive sibling ``.lock`` owns an account for an entire process attempt.
Every attempt and failed/abandoned attempt is retained. Atomic fsync/replace
writes preserve the last complete account on interruption. Checkpoint elapsed
time is a LOWER BOUND: the final save/cleanup tail belongs in this sidecar.

Only an allocated GPU (hardware.gpu_count == 1) and equal hostname + boot id +
perf_counter domains permit charging an idle gap between attempts. CPU-only
accounts charge each process runtime without charging command gaps. Across
clock domains such a gap is unknowable and is explicitly recorded without
inventing time. The external provider's complete
allocation receipt remains authoritative; this is no hard-cutoff mechanism.
"""
from __future__ import annotations

from pathlib import Path
import ctypes
import fcntl
import json
import math
import os
import socket
import sys
import tempfile
import time

FORMAT = 'clash-native-allocation-v1'
FINAL_STATUSES = frozenset(('completed', 'budget_exhausted', 'failed', 'interrupted'))


def _seconds(value, name, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError(name + ' must be ' + ('positive ' if positive else 'nonnegative ') + 'finite seconds')
    return float(value)


def _json_copy(value, name):
    try:
        return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError(name + ' must be finite JSON data') from exc


def _hardware(value):
    result = _json_copy(value, 'hardware')
    if not isinstance(result, dict) or type(result.get('gpu_count')) is not int or result['gpu_count'] not in (0, 1):
        raise ValueError('hardware identity requires integer gpu_count 0 or 1')
    return result


def clock_domain():
    """Actual boot identity; unknown boots deliberately never match processes."""
    boot = None
    try:
        boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip() or None
    except OSError:
        pass
    if boot is None and sys.platform == 'darwin':
        # Pure-stdlib sysctl call; do not invoke a shell or consume randomness.
        class Timeval(ctypes.Structure):
            _fields_ = [('seconds', ctypes.c_long), ('microseconds', ctypes.c_int)]
        try:
            libc = ctypes.CDLL(None, use_errno=True)
            result = Timeval(); size = ctypes.c_size_t(ctypes.sizeof(result))
            fn = libc.sysctlbyname
            fn.argtypes = [ctypes.c_char_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t), ctypes.c_void_p, ctypes.c_size_t]
            fn.restype = ctypes.c_int
            if fn(b'kern.boottime', ctypes.byref(result), ctypes.byref(size), None, 0) == 0:
                boot = 'darwin-boottime:' + str(result.seconds) + ':' + str(result.microseconds)
        except (AttributeError, OSError):
            pass
    return {'hostname': socket.gethostname(), 'boot_id': boot, 'clock': 'perf_counter'}


def _domain(value):
    if not isinstance(value, dict) or not {'hostname', 'boot_id'} <= value.keys() or set(value) - {'hostname', 'boot_id', 'clock'}:
        raise ValueError('clock domain requires hostname and boot_id')
    result = dict(value); result.setdefault('clock', 'perf_counter')
    if not isinstance(result['hostname'], str) or not result['hostname'] or result['clock'] != 'perf_counter':
        raise ValueError('clock domain hostname/clock is invalid')
    if result['boot_id'] is not None and (not isinstance(result['boot_id'], str) or not result['boot_id']):
        raise ValueError('clock domain boot_id is invalid')
    return result


def _same_domain(a, b):
    return a is not None and b is not None and a.get('boot_id') is not None and a == b


def _atomic_write(path, value):
    data = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()
    descriptor, temporary = tempfile.mkstemp(prefix='.' + path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'wb') as handle:
            handle.write(data); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _validate(value):
    """Fail closed on corruption instead of resetting an account to zero."""
    required = {'format', 'budget_seconds', 'hardware', 'cumulative_seconds', 'origin_floor_seconds',
                'idle_gap_seconds', 'reconciliations', 'attempts', 'last_clock_domain',
                'last_accounted_time', 'clock_discontinuities', 'floor_transfer', 'authority'}
    if not isinstance(value, dict) or set(value) != required or value['format'] != FORMAT:
        raise ValueError('allocation ledger format/fields differ')
    _json_copy(value, 'allocation ledger')
    _seconds(value['budget_seconds'], 'ledger budget', positive=True)
    for key in ('cumulative_seconds', 'origin_floor_seconds', 'idle_gap_seconds'):
        _seconds(value[key], key)
    if _hardware(value['hardware']) != value['hardware']:
        raise ValueError('allocation hardware identity is invalid')
    if not isinstance(value['attempts'], list) or not isinstance(value['reconciliations'], list) or not isinstance(value['clock_discontinuities'], list):
        raise ValueError('allocation history is invalid')
    total = value['origin_floor_seconds'] + value['idle_gap_seconds']
    for index, attempt in enumerate(value['attempts']):
        if not isinstance(attempt, dict) or set(attempt) != {'id', 'pid', 'clock_domain', 'process_start', 'recorded_start', 'runtime_seconds', 'prior_seconds', 'status', 'ended_at', 'details', 'unknown_previous_tail'}:
            raise ValueError('allocation attempt fields differ')
        if type(attempt['id']) is not int or attempt['id'] != index + 1 or type(attempt['pid']) is not int:
            raise ValueError('allocation attempt sequence is corrupt')
        _domain(attempt['clock_domain'])
        for key in ('process_start', 'recorded_start', 'runtime_seconds', 'prior_seconds'):
            _seconds(attempt[key], 'attempt ' + key)
        if attempt['recorded_start'] < attempt['process_start']:
            raise ValueError('allocation attempt starts before process')
        if attempt['status'] == 'running':
            if index != len(value['attempts']) - 1 or attempt['ended_at'] is not None:
                raise ValueError('allocation active attempt history is corrupt')
        elif attempt['status'] in FINAL_STATUSES or attempt['status'] == 'abandoned':
            if attempt['status'] != 'abandoned' and attempt['ended_at'] is None:
                raise ValueError('final allocation attempt has no end time')
            if attempt['ended_at'] is not None:
                _seconds(attempt['ended_at'], 'attempt end')
                if attempt['ended_at'] < attempt['process_start']:
                    raise ValueError('allocation attempt end precedes process')
        else:
            raise ValueError('allocation attempt status is invalid')
        if type(attempt['unknown_previous_tail']) is not bool or not isinstance(attempt['details'], dict):
            raise ValueError('allocation attempt details are invalid')
        total += attempt['runtime_seconds']
    for reconciliation in value['reconciliations']:
        if not isinstance(reconciliation, dict) or set(reconciliation) != {'reason', 'added_seconds', 'checkpoint_lower_bound', 'before_attempt'} or reconciliation['reason'] != 'checkpoint_lower_bound':
            raise ValueError('allocation reconciliation history is invalid')
        total += _seconds(reconciliation['added_seconds'], 'reconciled seconds')
        _seconds(reconciliation['checkpoint_lower_bound'], 'reconciled checkpoint floor')
    if value['last_clock_domain'] is None:
        if value['last_accounted_time'] is not None or value['attempts']:
            raise ValueError('allocation clock history is missing')
    else:
        _domain(value['last_clock_domain']); _seconds(value['last_accounted_time'], 'last accounted time')
    if not math.isclose(total, value['cumulative_seconds'], rel_tol=1e-10, abs_tol=1e-6):
        raise ValueError('allocation cumulative seconds disagree with preserved history')
    if value['authority'] != 'external provider full allocation receipt is authoritative':
        raise ValueError('allocation authority marker differs')
    if value['floor_transfer'] is not None:
        transfer = value['floor_transfer']
        if not isinstance(transfer, dict) or transfer.get('kind') != 'explicit_moved_checkpoint_floor_only' or transfer.get('unknown_prior_tail') is not True:
            raise ValueError('allocation moved-checkpoint floor provenance is invalid')
    return value


class AllocationLedger:
    """Exclusive cumulative account for one process attempt.

    ``prior_seconds`` excludes THIS process, so a trainer can use it with its
    already-running perf_counter timer. ``elapsed()`` includes setup since the
    supplied process_start, updating only memory. ``heartbeat()`` durably saves
    progress; the caller may rate-limit it between completed updates. ``record``
    always durably finalizes after checkpoint/cleanup and releases the lock.
    A context records exceptions as failed automatically.
    """

    @classmethod
    def start(cls, path=None, *, budget_seconds, hardware, process_start,
              resume_lower_bound=0., inherited_path=None, moved_checkpoint_floor=False,
              now=time.perf_counter, clock_domain=None):
        budget = _seconds(budget_seconds, 'budget_seconds', positive=True)
        lower_bound = _seconds(resume_lower_bound, 'resume_lower_bound')
        process_start = _seconds(process_start, 'process_start')
        if type(moved_checkpoint_floor) is not bool:
            raise ValueError('moved_checkpoint_floor must be boolean')
        identity = _hardware(hardware)
        inherited = None if inherited_path is None else Path(inherited_path).expanduser().resolve()
        requested = None if path is None else Path(path).expanduser().resolve()
        if inherited is not None and inherited.exists():
            if requested is not None and requested != inherited:
                raise ValueError('existing inherited allocation ledger owns the account; a different charge-to path would fork history')
            requested = inherited
        elif inherited is not None:
            if not moved_checkpoint_floor:
                raise ValueError('inherited allocation ledger is missing; explicit moved_checkpoint_floor is required')
            if requested is None or requested == inherited:
                raise ValueError('moved checkpoint floor requires an explicit new ledger path')
        if requested is None:
            raise ValueError('an allocation ledger path is required')
        domain = _domain(globals()['clock_domain']() if clock_domain is None else clock_domain)
        current = _seconds(now(), 'current perf_counter')
        if process_start > current:
            raise ValueError('process_start exceeds current perf_counter')
        requested.parent.mkdir(parents=True, exist_ok=True)
        self = cls()
        self.path, self._now, self._domain = requested, now, domain
        self._closed = False
        self._lock = requested.with_name(requested.name + '.lock').open('a+b')
        try:
            try:
                fcntl.flock(self._lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError('allocation ledger is already owned by an active attempt: ' + str(requested)) from exc
            if requested.exists():
                try:
                    value = _validate(json.loads(requested.read_text()))
                except (OSError, ValueError, TypeError, KeyError) as exc:
                    raise ValueError('allocation ledger is corrupt; refusing to reset history: ' + str(requested)) from exc
                if value['hardware'] != identity:
                    raise ValueError('allocation hardware identity differs from inherited account')
                if value['budget_seconds'] != budget:
                    raise ValueError('allocation budget differs; migration/resume cannot renew the allowance')
                if inherited is not None and not inherited.exists():
                    transfer = value['floor_transfer']
                    if transfer is None or transfer.get('inherited_path') != str(inherited):
                        raise ValueError('missing inherited ledger cannot be merged into an unrelated existing account')
                # A running record whose lock is no longer held is abandoned.
                # An allocated GPU and equal domains measure its missing tail
                # plus allocated gap up to this process. CPU-only accounts or
                # other clock domains must label the process tail unknowable.
                previous = value['attempts'][-1] if value['attempts'] else None
                last = value['last_accounted_time']
                comparable = _same_domain(value['last_clock_domain'], domain)
                if comparable and process_start < last:
                    raise ValueError('allocation perf_counter regressed or process attempts overlap')
                if previous is not None and previous['status'] == 'running':
                    previous['status'] = 'abandoned'
                    previous['details']['recovery'] = 'exclusive lock released without final attempt status'
                    if comparable and identity['gpu_count'] == 1:
                        missing = process_start - last
                        previous['runtime_seconds'] += missing
                        value['cumulative_seconds'] += missing
                        previous['ended_at'] = process_start
                    else:
                        previous['unknown_previous_tail'] = True
                        if comparable:
                            previous['details']['unknown_tail_reason'] = 'CPU-only missing process tail cannot be separated from the unallocated command gap'
                elif comparable and identity['gpu_count'] == 1:
                    gap = process_start - last
                    value['idle_gap_seconds'] += gap
                    value['cumulative_seconds'] += gap
                if last is not None and not comparable:
                    value['clock_discontinuities'].append({'from_domain': value['last_clock_domain'],
                                                          'to_domain': domain, 'before_attempt': len(value['attempts']) + 1,
                                                          'gap_charged': False, 'reason': 'different or unavailable boot/hostname clock domain'})
                if value['cumulative_seconds'] < lower_bound:
                    added = lower_bound - value['cumulative_seconds']
                    value['reconciliations'].append({'reason': 'checkpoint_lower_bound', 'added_seconds': added,
                                                     'checkpoint_lower_bound': lower_bound, 'before_attempt': len(value['attempts']) + 1})
                    value['cumulative_seconds'] += added
            else:
                if lower_bound > 0 and not moved_checkpoint_floor:
                    raise ValueError('resume lower bound without its ledger requires explicit moved_checkpoint_floor')
                transfer = None
                if moved_checkpoint_floor:
                    transfer = {'kind': 'explicit_moved_checkpoint_floor_only',
                                'inherited_path': None if inherited is None else str(inherited),
                                'checkpoint_lower_bound': lower_bound, 'unknown_prior_tail': True,
                                'scope': 'only the checkpoint floor is known; missing historical attempts/tail are not reconstructed'}
                value = {'format': FORMAT, 'budget_seconds': budget, 'hardware': identity,
                         'cumulative_seconds': lower_bound, 'origin_floor_seconds': lower_bound,
                         'idle_gap_seconds': 0., 'reconciliations': [], 'attempts': [],
                         'last_clock_domain': None, 'last_accounted_time': None,
                         'clock_discontinuities': [], 'floor_transfer': transfer,
                         'authority': 'external provider full allocation receipt is authoritative'}
            self.prior_seconds = value['cumulative_seconds']
            attempt = {'id': len(value['attempts']) + 1, 'pid': os.getpid(), 'clock_domain': domain,
                       'process_start': process_start, 'recorded_start': current,
                       'runtime_seconds': current - process_start, 'prior_seconds': self.prior_seconds,
                       'status': 'running', 'ended_at': None, 'details': {}, 'unknown_previous_tail': False}
            value['attempts'].append(attempt)
            value['cumulative_seconds'] += current - process_start
            value['last_clock_domain'], value['last_accounted_time'] = domain, current
            self._value = value
            self._save()
            return self
        except BaseException:
            self._release()
            raise

    @property
    def budget_seconds(self):
        return self._value['budget_seconds']

    @property
    def attempt_id(self):
        return self._value['attempts'][-1]['id']

    def _save(self):
        _validate(self._value)
        _atomic_write(self.path, self._value)

    def _advance(self):
        if self._closed:
            return
        current = _seconds(self._now(), 'current perf_counter')
        previous = self._value['last_accounted_time']
        if current < previous:
            raise ValueError('allocation perf_counter regressed during an attempt')
        elapsed = current - previous
        self._value['attempts'][-1]['runtime_seconds'] += elapsed
        self._value['cumulative_seconds'] += elapsed
        self._value['last_accounted_time'] = current

    def elapsed(self):
        if not self._closed:
            self._advance()
        return self._value['cumulative_seconds']

    def heartbeat(self):
        """Durably record current progress; no implicit rate-limit or RNG use.

        An allocated GPU recovered on the same boot/host can measure time even
        from a stale heartbeat. An abrupt kill followed by a clock-domain change
        still leaves an unknown tail; the provider receipt is authoritative.
        """
        if self._closed:
            raise RuntimeError('allocation attempt is already finalized')
        self._advance(); self._save()
        return self._value['cumulative_seconds']

    def exhausted(self):
        return self.elapsed() >= self.budget_seconds

    def record(self, status, *, details=None):
        if status not in FINAL_STATUSES:
            raise ValueError('final allocation status is invalid')
        if self._closed:
            raise RuntimeError('allocation attempt is already finalized')
        details = {} if details is None else _json_copy(details, 'attempt details')
        if not isinstance(details, dict):
            raise ValueError('attempt details must be a mapping')
        self._advance()
        attempt = self._value['attempts'][-1]
        attempt['status'], attempt['ended_at'], attempt['details'] = status, self._value['last_accounted_time'], details
        try:
            self._save()
        finally:
            self._release()
        return self._value['cumulative_seconds']

    def _release(self):
        self._closed = True
        if getattr(self, '_lock', None) is not None:
            try:
                fcntl.flock(self._lock.fileno(), fcntl.LOCK_UN)
            finally:
                self._lock.close(); self._lock = None

    def __enter__(self):
        return self

    def __exit__(self, kind, value, traceback):
        if not self._closed:
            self.record('completed' if kind is None else 'failed',
                        details={} if kind is None else {'exception_type': kind.__name__, 'exception': str(value)})
        return False
