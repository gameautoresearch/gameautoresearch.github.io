"""Deterministic allocation-ledger tests: injected clocks, real process locks.

No timing benchmark, GPU job, game transition, RNG or external command runs.
"""
from pathlib import Path
import argparse
import copy
import json
import multiprocessing
import sys

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

HARDWARE = {'gpu': 'test GPU', 'gpu_count': 1, 'cpu_cores': 16, 'memory_gib': 64}
CPU_HARDWARE = {'gpu': None, 'gpu_count': 0, 'cpu_cores': 16, 'memory_gib': 64}
DOMAIN = {'hostname': 'worker-a', 'boot_id': 'boot-a', 'clock': 'perf_counter'}


class Clock:
    def __init__(self, value): self.value = value
    def __call__(self): return self.value


def _lock_probe(path, channel):
    from allocation import AllocationLedger
    try:
        account = AllocationLedger.start(path, budget_seconds=100., hardware=HARDWARE,
                                         process_start=1., now=Clock(1.), clock_domain=DOMAIN)
    except RuntimeError as exc:
        channel.send(('blocked', str(exc)))
    except BaseException as exc:
        channel.send(('unexpected', type(exc).__name__, str(exc)))
    else:
        account.record('completed'); channel.send(('unexpected', 'lock was acquired'))
    finally:
        channel.close()


def qualify(out):
    from allocation import AllocationLedger, clock_domain
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    checks = []
    def reject(label, function, expected=ValueError):
        try:
            function()
        except expected:
            checks.append(label)
        else:
            raise AssertionError('invalid allocation accepted: ' + label)
    def start(path, clock, process_start, **kwargs):
        return AllocationLedger.start(path, budget_seconds=kwargs.pop('budget_seconds', 100.),
                                      hardware=kwargs.pop('hardware', HARDWARE),
                                      process_start=process_start, now=clock, clock_domain=kwargs.pop('clock_domain', DOMAIN), **kwargs)
    path = out / 'account.json'
    clock = Clock(12.)
    account = start(path, clock, 10.)
    assert account.prior_seconds == 0. and account.elapsed() == 2.
    assert account.path == path.resolve() and account.attempt_id == 1
    # A checkpoint is saved before torch.save/cleanup; its lower bound must
    # never replace the larger final account on resume.
    clock.value = 20.
    checkpoint_floor = account.elapsed()
    assert checkpoint_floor == 10.
    clock.value = 25.
    assert account.record('completed', details={'checkpoint_floor': checkpoint_floor, 'tail': 'save and cleanup'}) == 15.
    frozen = json.loads(path.read_text())
    assert frozen['attempts'][0]['runtime_seconds'] == 15.
    # Cheap polling must not fsync or rewrite the durable sidecar. An explicit
    # heartbeat persists progress, while final record preserves the last tail.
    heartbeat_path = out / 'heartbeat.json'; clock.value = 12.
    heartbeat = start(heartbeat_path, clock, 10.)
    before_poll = heartbeat_path.read_bytes()
    clock.value = 14.; assert heartbeat.elapsed() == 4.
    assert not heartbeat.exhausted() and heartbeat_path.read_bytes() == before_poll
    assert json.loads(heartbeat_path.read_text())['cumulative_seconds'] == 2.
    clock.value = 16.; assert heartbeat.heartbeat() == 6.
    assert json.loads(heartbeat_path.read_text())['cumulative_seconds'] == 6.
    clock.value = 19.; assert heartbeat.record('completed') == 9.
    assert json.loads(heartbeat_path.read_text())['cumulative_seconds'] == 9.
    reject('heartbeat_after_finalize_rejected', lambda: heartbeat.heartbeat(), RuntimeError)
    checks.append('elapsed_exhausted_memory_only_heartbeat_durable_final_tail_preserved')
    clock.value = 32.
    resumed = AllocationLedger.start(None, budget_seconds=100., hardware=HARDWARE, process_start=30.,
                                     resume_lower_bound=checkpoint_floor, inherited_path=path,
                                     now=clock, clock_domain=DOMAIN)
    # 15 previous runtime +5 allocated idle gap, excluding current setup2.
    assert resumed.prior_seconds == 20. and resumed.elapsed() == 22.
    assert resumed._value['idle_gap_seconds'] == 5.
    clock.value = 35.; resumed.record('failed', details={'exception': 'injected simulator failure'})
    assert json.loads(path.read_text())['cumulative_seconds'] == 25.
    checks.append('resume_preserves_save_cleanup_and_same_boot_idle_gap')
    # CPU-only development commands do not retain an allocated GPU between
    # processes: preserve measured save/cleanup tails, exclude command gaps.
    cpu_path = out / 'cpu-only.json'; clock.value = 12.
    cpu_first = start(cpu_path, clock, 10., hardware=CPU_HARDWARE)
    clock.value = 20.; cpu_checkpoint_floor = cpu_first.elapsed()
    clock.value = 25.; cpu_first.record('completed')
    clock.value = 1002.
    cpu_second = start(cpu_path, clock, 1000., hardware=CPU_HARDWARE, resume_lower_bound=cpu_checkpoint_floor)
    assert cpu_second.prior_seconds == 15. and cpu_second.elapsed() == 17.
    assert cpu_second._value['idle_gap_seconds'] == 0.
    clock.value = 1005.; cpu_second.record('completed')
    assert json.loads(cpu_path.read_text())['cumulative_seconds'] == 20.
    checks.append('cpu_only_measured_tail_preserved_command_gap_not_charged')
    # With an abandoned CPU process there is no evidence of when the process
    # died. Even on the same boot, never charge its whole inter-command gap.
    cpu_abandoned_path = out / 'cpu-abandoned.json'; clock.value = 3.
    cpu_lost = start(cpu_abandoned_path, clock, 1., hardware=CPU_HARDWARE); cpu_lost._release()
    clock.value = 1001.
    cpu_recovered = start(cpu_abandoned_path, clock, 1000., hardware=CPU_HARDWARE)
    assert cpu_recovered.prior_seconds == 2. and cpu_recovered.elapsed() == 3.
    previous = cpu_recovered._value['attempts'][0]
    assert previous['status'] == 'abandoned' and previous['unknown_previous_tail'] and previous['ended_at'] is None
    assert previous['runtime_seconds'] == 2. and cpu_recovered._value['idle_gap_seconds'] == 0.
    cpu_recovered.record('completed')
    checks.append('cpu_abandoned_unknown_tail_not_invented_from_command_gap')
    # A different boot AND host has no comparable monotonic idle duration.
    other = {'hostname': 'worker-b', 'boot_id': 'boot-b', 'clock': 'perf_counter'}
    clock.value = 103.
    cross = start(path, clock, 100., clock_domain=other, resume_lower_bound=10.)
    assert cross.prior_seconds == 25. and cross.elapsed() == 28.
    assert len(cross._value['clock_discontinuities']) == 1
    clock.value = 105.; cross.record('completed')
    checks.append('cross_boot_host_gap_not_invented')
    # A lower-bound checkpoint ahead of the last durable ledger update adds
    # only the missing floor, with a preserved reconciliation event.
    clock.value = 108.
    floor = start(path, clock, 108., clock_domain=other, resume_lower_bound=50.)
    assert floor.prior_seconds == 50. and floor.elapsed() == 50.
    assert floor._value['reconciliations'][-1]['added_seconds'] == 17.
    clock.value = 109.; floor.record('completed')
    checks.append('checkpoint_floor_reconciled_monotonically')
    # An explicit exception keeps its failed attempt and full elapsed tail.
    failed_path = out / 'failed-context.json'; clock.value = 5.
    try:
        with start(failed_path, clock, 2.):
            clock.value = 9.
            raise RuntimeError('deliberate failure')
    except RuntimeError:
        pass
    failed = json.loads(failed_path.read_text())
    assert failed['cumulative_seconds'] == 7. and failed['attempts'][0]['status'] == 'failed'
    assert failed['attempts'][0]['details']['exception_type'] == 'RuntimeError'
    checks.append('failed_attempt_retained')
    # Simulate abrupt death: a durable running entry remains, but no process
    # lock does. Recovery charges a same-domain missing tail exactly once.
    abandoned_path = out / 'abandoned.json'; clock.value = 3.
    lost = start(abandoned_path, clock, 1.)
    clock.value = 7.; assert lost.elapsed() == 6.
    assert json.loads(abandoned_path.read_text())['cumulative_seconds'] == 2.
    lost._release()
    clock.value = 11.
    recovered = start(abandoned_path, clock, 10.)
    assert recovered.prior_seconds == 9. and recovered.elapsed() == 10.
    assert recovered._value['attempts'][0]['status'] == 'abandoned'
    assert recovered._value['attempts'][0]['runtime_seconds'] == 9.
    recovered.record('completed')
    checks.append('same_domain_abandoned_attempt_recovers_from_stale_heartbeat')
    # Cross-domain abrupt failure has an explicitly unknown tail, not a fake
    # runtime computed by subtracting unrelated clocks.
    unknown_path = out / 'unknown-tail.json'; clock.value = 3.
    lost = start(unknown_path, clock, 1.)
    clock.value = 9.; assert lost.elapsed() == 8.
    lost._release()
    clock.value = 5.
    recovered = start(unknown_path, clock, 4., clock_domain=other)
    assert recovered.prior_seconds == 2. and recovered.elapsed() == 3.
    assert recovered._value['attempts'][0]['unknown_previous_tail']
    assert recovered._value['attempts'][0]['ended_at'] is None
    recovered.record('completed')
    checks.append('cross_domain_abandoned_tail_explicitly_unknown')
    missing = out / 'missing-history.json'
    clock.value = 2.
    reject('missing_inherited_ledger_rejected', lambda: start(out / 'forbidden-new.json', clock, 1., inherited_path=missing, resume_lower_bound=90.))
    reject('nonzero_floor_cannot_silently_start_new_account', lambda: start(out / 'forbidden-floor.json', clock, 1., resume_lower_bound=90.))
    moved_path = out / 'explicit-floor-transfer.json'
    moved = start(moved_path, clock, 1., inherited_path=missing, resume_lower_bound=90., moved_checkpoint_floor=True)
    assert moved.prior_seconds == 90. and moved.elapsed() == 91.
    assert moved._value['floor_transfer']['kind'] == 'explicit_moved_checkpoint_floor_only'
    clock.value = 12.; assert moved.exhausted(); moved.record('budget_exhausted')
    assert json.loads(moved_path.read_text())['cumulative_seconds'] == 101.
    checks.append('explicit_moved_checkpoint_floor_labels_unknown_prior_tail')
    # Never fork an extant inherited account into a new charge-to path, even
    # when a caller enables the missing-ledger floor option.
    reject('inherited_account_path_cannot_fork', lambda: start(out / 'fork.json', clock, 12., inherited_path=path, moved_checkpoint_floor=True))
    unchanged = path.read_bytes()
    reject('hardware_mismatch_rejected', lambda: start(path, clock, 12., hardware=HARDWARE | {'cpu_cores': 32}))
    reject('budget_renewal_rejected', lambda: start(path, clock, 12., budget_seconds=200.))
    assert path.read_bytes() == unchanged
    for label, hardware in [('missing', {'cpu_cores': 16}), ('boolean', HARDWARE | {'gpu_count': True}), ('negative', HARDWARE | {'gpu_count': -1}), ('multiple', HARDWARE | {'gpu_count': 2}), ('fractional', HARDWARE | {'gpu_count': .5})]:
        reject('invalid_gpu_count_' + label, lambda hardware=hardware: start(out / ('bad-hardware-' + label + '.json'), clock, 1., hardware=hardware))
    for value in (0., -1., float('nan'), float('inf'), True):
        reject('invalid_budget_' + str(value), lambda value=value: start(out / ('bad-budget-' + str(value) + '.json'), clock, 1., budget_seconds=value))
    reject('future_process_start_rejected', lambda: start(out / 'future.json', clock, 13.))
    # Exclusive ownership is tested in a separately spawned interpreter, not
    # two threads sharing an inherited POSIX flock file description.
    locked_path = out / 'exclusive.json'; clock.value = 1.
    locked = start(locked_path, clock, 1.)
    context = multiprocessing.get_context('spawn')
    receive, send = context.Pipe(duplex=False)
    process = context.Process(target=_lock_probe, args=(locked_path, send)); process.start(); send.close()
    assert receive.poll(20), 'allocation lock probe did not complete'
    message = receive.recv(); process.join(20)
    assert not process.is_alive() and process.exitcode == 0 and message[0] == 'blocked', message
    receive.close(); clock.value = 2.; locked.record('completed')
    clock.value = 3.
    next_attempt = start(locked_path, clock, 3.); assert next_attempt.attempt_id == 2; next_attempt.record('completed')
    checks.append('exclusive_process_lock_rejects_overlap_and_releases_after_finalize')
    # Corruption cannot erase an account or normalize malformed history.
    valid = json.loads(path.read_text())
    mutations = {
        'cumulative': lambda v: v.__setitem__('cumulative_seconds', 0.),
        'attempt_sequence': lambda v: v['attempts'][0].__setitem__('id', 999),
        'nonfinite': lambda v: v.__setitem__('cumulative_seconds', float('nan')),
        'missing_history': lambda v: v.pop('attempts'),
        'status': lambda v: v['attempts'][0].__setitem__('status', 'unknown'),
        'final_end': lambda v: v['attempts'][0].__setitem__('ended_at', None),
    }
    for label, change in mutations.items():
        bad = copy.deepcopy(valid); change(bad)
        target = out / ('corrupt-' + label + '.json'); target.write_text(json.dumps(bad))
        before = target.read_bytes()
        reject('corruption_' + label, lambda target=target: start(target, clock, 3.))
        assert target.read_bytes() == before
    target = out / 'corrupt-json.json'; target.write_text('{ unfinished')
    reject('truncated_json_rejected', lambda: start(target, clock, 3.))
    # Unknown boot IDs do not accidentally make cross-process clocks equal.
    unknown = {'hostname': 'worker-a', 'boot_id': None}
    clock.value = 2.
    a = start(out / 'unknown-domain.json', clock, 1., clock_domain=unknown); a.record('completed')
    clock.value = 101.
    b = start(out / 'unknown-domain.json', clock, 100., clock_domain=unknown)
    assert b.prior_seconds == 1. and b.elapsed() == 2.; b.record('completed')
    checks.append('unavailable_boot_id_prevents_idle_gap_charge')
    # Same boot but a different hostname is still not a common clock domain.
    clock.value = 2.
    a = start(out / 'different-host.json', clock, 1.); a.record('completed')
    clock.value = 51.
    b = start(out / 'different-host.json', clock, 50., clock_domain=DOMAIN | {'hostname': 'worker-c'})
    assert b.prior_seconds == 1.; b.record('completed')
    checks.append('matching_boot_alone_does_not_match_clock_domain')
    result = {'status': 'passed', 'checks': checks, 'check_count': len(checks),
              'actual_clock_domain': clock_domain(), 'external_receipt_authoritative': True,
              'phase_timing_comparison': False, 'history_account': json.loads(path.read_text())}
    (out / 'summary.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = qualify(args.out)
    print(json.dumps({k: result[k] for k in ('status', 'check_count', 'actual_clock_domain')}, indent=2))
