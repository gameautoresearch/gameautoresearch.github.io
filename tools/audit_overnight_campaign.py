#!/usr/bin/env python3
"""Read-only reconciliation of campaign receipts, liabilities and GPU overlap.

No provider requests. Unknown allocations and unpriced model requests remain
liabilities; this is an evidence audit, not a replacement for provider invoices.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time

from tools.overnight_budget import CampaignBudget, DEFAULT


def audit(budget, now=None):
    now = time.time() if now is None else now
    snapshot = budget.snapshot()
    errors, warnings, intervals = [], [], []
    model_priced = model_unknown = 0.0
    active_gpu = defaultdict(list)
    model_calls = {}
    for operation in snapshot['operations']:
        name = f"{operation['game']}/{operation['kind']}"
        spec = json.loads(operation['spec'])
        if operation['status'] != 'settled':
            if operation['gpu_hold']:
                active_gpu[operation['game']].append(name)
                if now > operation['created'] + spec['max_seconds'] + 90:
                    warnings.append(f'{name}: overdue unresolved GPU receipt')
                if not operation['provider_id']:
                    warnings.append(f'{name}: GPU hold has no attached provider identity')
            continue
        receipt = json.loads(operation['receipt'])
        if operation['bucket'] == 'model':
            path = Path(receipt['ledger_path'])
            if not path.is_file():
                errors.append(f'{name}: missing closed model ledger')
                continue
            raw = path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != receipt['ledger_sha256']:
                errors.append(f'{name}: model ledger hash changed after settlement')
            ledger = json.loads(raw)
            if ledger.get('closed') is not True:
                errors.append(f'{name}: settled model ledger is open')
            priced = unknown = 0.0
            for call in ledger.get('calls', []):
                identity = call.get('id')
                if identity:
                    if identity in model_calls:
                        errors.append(f'{name}: model call also accounted in {model_calls[identity]}')
                    model_calls[identity] = name
                cost = call.get('usage', {}).get('cost', call.get('cost_usd'))
                complete = call.get('status') in ('complete', 'priced')
                if (complete and isinstance(cost, (int, float))
                        and not isinstance(cost, bool) and math.isfinite(cost) and cost >= 0):
                    priced += cost
                else:
                    unknown += call.get('reserved_usd', 0)
            if not math.isclose(priced + unknown, operation['usd_charged'], abs_tol=1e-8):
                errors.append(f'{name}: model liability differs from per-call evidence')
            model_priced += priced
            model_unknown += unknown
        elif receipt.get('provider_id'):
            start, end = receipt['started_at'], receipt['finished_at']
            if end < start or (operation['provider_id'] and operation['provider_id'] != receipt['provider_id']):
                errors.append(f'{name}: provider receipt interval/identity mismatch')
            seconds = end - start
            expected = (seconds * spec['quoted_rate_usd_per_second'] + spec['extra_usd']) * spec['reservation_margin']
            if not math.isclose(expected, operation['usd_charged'], abs_tol=1e-8):
                errors.append(f'{name}: resource amount differs from receipt/rate')
            if operation['gpu_charged']:
                if not math.isclose(seconds, operation['gpu_charged'], abs_tol=1e-6):
                    errors.append(f'{name}: allocated GPU interval differs from charged seconds')
                intervals.append((start, end, operation['game'], name))
        elif not (receipt.get('provider_inventory_checked') is True
                  and receipt.get('worker_count') == 0
                  and Path(receipt.get('receipt_path', '')).is_file()):
            errors.append(f'{name}: zero-allocation settlement lacks evidence')
    for game, entries in active_gpu.items():
        if len(entries) > 1:
            errors.append(f'{game}: multiple active GPU holds')
    if sum(map(len, active_gpu.values())) > snapshot['global_limits']['max_concurrent_gpu_workers']:
        errors.append('Global active GPU concurrency exceeds authorization')
    for game in snapshot['grants']:
        completed = sorted(i for i in intervals if i[2] == game)
        for left, right in zip(completed, completed[1:]):
            if right[0] < left[1] - 0.01:
                errors.append(f'{game}: completed GPU allocation intervals overlap')
    events = sorted((t, change) for start, end, _, _ in intervals
                    for t, change in ((start, 1), (end, -1)))
    concurrency = peak = 0
    for _, change in events:
        concurrency += change
        peak = max(peak, concurrency)
    if peak > snapshot['global_limits']['max_concurrent_gpu_workers']:
        errors.append('Completed GPU allocation history exceeds concurrency authorization')
    limit_keys = {'gpu_seconds': 'gpu_allocated_seconds',
                  'resource_usd': 'resource_charged_or_reserved_usd',
                  'model_usd': 'model_charged_or_reserved_usd'}
    for key, limit_key in limit_keys.items():
        if snapshot['global_usage'][key] > snapshot['global_limits'][limit_key] + 1e-8:
            errors.append(f'Global charged-plus-held {key} exceeds authorization')
    per_game = {}
    for game in snapshot['grants']:
        operations = [o for o in snapshot['operations'] if o['game'] == game]
        per_game[game] = dict(
            settled_gpu_seconds=sum(o['gpu_charged'] for o in operations if o['status'] == 'settled'),
            held_gpu_seconds=sum(o['gpu_hold'] for o in operations if o['status'] != 'settled'),
            resource_charged_or_held_usd=sum(o['usd_charged'] if o['status'] == 'settled' else o['usd_hold']
                                            for o in operations if o['bucket'] == 'resource'))
        usage = dict(gpu_seconds=per_game[game]['settled_gpu_seconds'] + per_game[game]['held_gpu_seconds'],
                     resource_usd=per_game[game]['resource_charged_or_held_usd'],
                     model_usd=sum(o['usd_charged'] if o['status'] == 'settled' else o['usd_hold']
                                   for o in operations if o['bucket'] == 'model'))
        for key, amount in usage.items():
            if amount > snapshot['grants'][game][key] + 1e-8:
                errors.append(f'{game}: charged-plus-held {key} exceeds current grant')
    return dict(at_utc=datetime.fromtimestamp(now, timezone.utc).isoformat(),
                audit_passed=not errors, errors=errors, warnings=warnings,
                settled=snapshot['global_settled_usage'], held=snapshot['global_held_usage'],
                settled_model_priced_usd=model_priced,
                settled_model_unknown_liability_usd=model_unknown,
                active_gpu_workers=sum(map(len, active_gpu.values())),
                completed_history_peak_gpu_workers=peak, per_game=per_game,
                limitations=['Local receipt/rate audit; conservative resource amounts include reservation margin.',
                             'Unsettled operations remain held; actual provider invoice reconciliation is separate.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ledger', type=Path, default=DEFAULT)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    result = audit(CampaignBudget('generals', path=args.ledger))
    output = json.dumps(result, indent=2, allow_nan=False) + '\n'
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(output)
    print(output, end='')
    raise SystemExit(0 if result['audit_passed'] else 1)


if __name__ == '__main__':
    main()
