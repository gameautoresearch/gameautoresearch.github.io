#!/usr/bin/env python3
"""Observe Modal billing only for apps linked to this campaign's worker IDs.

Read-only provider request. Hourly, potentially lagged app totals are not final
invoices and never release holds in the authoritative allocation ledger.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import re

from tools.overnight_budget import CampaignBudget, DEFAULT


def owned_apps(root, snapshot):
    known = {o['provider_id']: o['game'] for o in snapshot['operations'] if o['provider_id']}
    apps, linked = defaultdict(list), set()
    names = ('app.json', 'sandbox.json', 'provider.json', 'allocation.json',
             'worker.json', 'lease.json', 'compute.json', 'session.json', 'launch.json')

    def direct_ids(node):
        if not isinstance(node, dict):
            return set()
        values = {v for v in node.values() if isinstance(v, str) and v in known}
        for key in ('provider', 'sandbox', 'worker'):
            sub = node.get(key)
            if isinstance(sub, dict):
                values.update(v for v in sub.values() if isinstance(v, str) and v in known)
        return values

    def walk(node, path, sibling_ids):
        if isinstance(node, dict):
            app = node.get('app_id')
            ids = direct_ids(node)
            explicit_worker = any(isinstance(v, str) and v.startswith('sb-') for v in node.values())
            for key in ('provider', 'sandbox', 'worker'):
                sub = node.get(key)
                if isinstance(sub, dict):
                    explicit_worker |= any(isinstance(v, str) and v.startswith('sb-') for v in sub.values())
            if not ids and not explicit_worker and len(sibling_ids) == 1:
                ids = sibling_ids
            if isinstance(app, str) and re.fullmatch(r'ap-[A-Za-z0-9]+', app) and ids:
                apps[app].append(dict(path=str(path.relative_to(root)),
                    provider_ids=sorted(ids), games=sorted({known[p] for p in ids})))
                linked.update(ids)
            for value in node.values():
                if isinstance(value, (dict, list)):
                    walk(value, path, set())
        elif isinstance(node, list):
            for value in node:
                walk(value, path, set())

    for game in snapshot['grants']:
        for directory, children, files in os.walk(root / game):
            children[:] = [n for n in children if n not in
                ('frozen', 'task', 'families', 'source-snapshots', 'host-execution-source',
                 'runtime', '__pycache__', '.git') and not n.startswith('release')]
            records = []
            for name in files:
                if not name.endswith(names):
                    continue
                path = Path(directory) / name
                try:
                    if path.stat().st_size <= 2_000_000:
                        records.append((path, json.loads(path.read_text())))
                except (OSError, ValueError):
                    pass
            sibling_ids = set().union(*(direct_ids(d) for _, d in records)) if records else set()
            for path, data in records:
                walk(data, path, sibling_ids)
    return dict(apps), sorted(set(known) - linked)


def merge_provider_ownership(local, snapshot, records):
    """Provider metadata overrides inferred local links, never links foreign IDs."""
    known = {o['provider_id']: o['game'] for o in snapshot['operations'] if o['provider_id']}
    found = {}
    for record in records:
        identity, app = record.get('provider_id'), record.get('app_id')
        if identity not in known or not isinstance(app, str) or not re.fullmatch(r'ap-[A-Za-z0-9]+', app):
            continue
        if identity in found and found[identity] != app:
            raise ValueError('Conflicting provider app identity for an owned worker')
        found[identity] = app
    result = defaultdict(list)
    for app, proofs in local.items():
        for proof in proofs:
            identities = sorted(set(proof['provider_ids']) - found.keys())
            if identities:
                result[app].append(dict(proof, provider_ids=identities,
                    games=sorted({known[p] for p in identities})))
    for identity, app in sorted(found.items()):
        result[app].append(dict(source='Modal SandboxList provider metadata',
            provider_ids=[identity], games=[known[identity]]))
    linked = {p for proofs in result.values() for proof in proofs for p in proof['provider_ids']}
    return dict(result), sorted(set(known) - linked)


def provider_inventory(snapshot, environment, max_pages=100):
    """Read only; retain metadata for recorded workers and discard unrelated IDs."""
    wanted = {o['provider_id'] for o in snapshot['operations'] if o['provider_id']}
    async def fetch():
        from modal.client import _Client
        from modal_proto import api_pb2
        client = await _Client.from_env()
        before, found, pages = None, {}, 0
        while wanted - found.keys() and pages < max_pages:
            response = await client.stub.SandboxList(api_pb2.SandboxListRequest(
                environment_name=environment, include_finished=True,
                **({'before_timestamp': before} if before is not None else {})),
                timeout=15, retry=None)
            pages += 1
            for item in response.sandboxes:
                if item.id in wanted:
                    found[item.id] = dict(provider_id=item.id, app_id=item.app_id,
                        started_at=item.task_info.started_at,
                        finished_at=item.task_info.finished_at,
                        gpu_type=item.task_info.gpu_type)
            if not response.sandboxes:
                break
            next_before = response.sandboxes[-1].created_at
            if before is not None and next_before >= before:
                raise ValueError('Provider inventory pagination made no progress')
            before = next_before
        return dict(records=list(found.values()), pages=pages,
            matched_workers=len(found), missing_workers=len(wanted - found.keys()),
            shared_app_usage_excluded=False)
    from modal._utils.async_utils import synchronizer
    return synchronizer.create_blocking(fetch)()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT.parent)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--environment', default='staging')
    parser.add_argument('--provider-inventory', action='store_true',
                        help='Also link recorded worker IDs through authoritative provider metadata')
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('Billing observations are immutable; choose a new output')
    budget = CampaignBudget('generals', path=args.root / 'campaign.sqlite3')
    snapshot = budget.snapshot()
    ownership, unlinked = owned_apps(args.root, snapshot)
    os.environ.setdefault('MODAL_DISABLE_API_PROXY', '1')
    inventory = None
    if args.provider_inventory:
        inventory = provider_inventory(snapshot, args.environment)
        ownership, unlinked = merge_provider_ownership(ownership, snapshot, inventory['records'])
    if not ownership:
        raise ValueError('No app is proven linked to an owned campaign worker')
    import modal
    now = datetime.now(timezone.utc)
    start = datetime.fromisoformat(budget.auth['started_at_utc'].replace('Z', '+00:00')) if 'started_at_utc' in budget.auth else datetime(2026, 11, 8, 15, 33, 49, tzinfo=timezone.utc)
    start = start.replace(minute=0, second=0, microsecond=0)
    end = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    report = modal.Environment.from_name(args.environment).billing.report(start=start, end=end, resolution='h')
    rows = [dict(row.items()) for row in report if row.object_id in ownership]
    result = dict(requested_at=now.isoformat(), source='Modal Environment.billing.report',
        environment=args.environment, owned_app_provenance=ownership,
        provider_inventory=inventory,
        unlinked_campaign_provider_ids=unlinked, rows=rows,
        observed_owned_app_billing_usd=sum(float(r['cost']) for r in rows),
        final_invoice=False, coverage_complete=False, budget_holds_released=False,
        limitations=['Metering can lag; the current hour is incomplete.',
                     'Hourly app rows can include other usage if an app was shared.',
                     'Unlinked workers and separately attributed builds/storage may be absent.'])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as file:
        file.write(json.dumps(result, default=str, indent=2) + '\n')
    print(json.dumps(dict(at=now.isoformat(), proven_owned_apps=len(ownership),
        unlinked_workers=len(unlinked), observed_owned_app_billing_usd=result['observed_owned_app_billing_usd'],
        final_invoice=False, coverage_complete=False)))


if __name__ == '__main__':
    main()
