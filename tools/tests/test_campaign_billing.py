import json
from pathlib import Path
import tempfile
import unittest

from tools.collect_overnight_billing import owned_apps, merge_provider_ownership


class BillingOwnershipTests(unittest.TestCase):
    def test_authoritative_inventory_replaces_inferred_link_and_ignores_foreign_worker(self):
        snapshot = {'operations': [{'provider_id': 'sb-owned', 'game': 'a'},
                                    {'provider_id': 'sb-missing', 'game': 'a'}]}
        local = {'ap-Wrong': [{'provider_ids': ['sb-owned'], 'games': ['a'], 'path': 'old.json'}]}
        records = [{'provider_id': 'sb-owned', 'app_id': 'ap-Right'},
                   {'provider_id': 'sb-foreign', 'app_id': 'ap-Foreign'}]
        apps, missing = merge_provider_ownership(local, snapshot, records)
        self.assertEqual(set(apps), {'ap-Right'})
        self.assertEqual(missing, ['sb-missing'])

    def test_partial_inventory_retains_only_unresolved_local_links(self):
        snapshot = {'operations': [{'provider_id': 'sb-first', 'game': 'a'},
                                    {'provider_id': 'sb-second', 'game': 'b'}]}
        local = {'ap-Local': [{'provider_ids': ['sb-first', 'sb-second'], 'games': ['a', 'b']}]}
        apps, missing = merge_provider_ownership(local, snapshot,
            [{'provider_id': 'sb-first', 'app_id': 'ap-Provider'}])
        self.assertEqual(apps['ap-Local'][0]['provider_ids'], ['sb-second'])
        self.assertEqual(apps['ap-Local'][0]['games'], ['b'])
        self.assertEqual(missing, [])

    def test_conflicting_authoritative_identity_is_rejected(self):
        snapshot = {'operations': [{'provider_id': 'sb-owned', 'game': 'a'}]}
        with self.assertRaisesRegex(ValueError, 'Conflicting'):
            merge_provider_ownership({}, snapshot,
                [{'provider_id': 'sb-owned', 'app_id': 'ap-First'},
                 {'provider_id': 'sb-owned', 'app_id': 'ap-Second'}])

    def test_foreign_worker_is_not_linked_via_a_neighboring_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / 'a' / 'job'
            folder.mkdir(parents=True)
            (folder / 'sandbox.json').write_text(json.dumps(
                {'provider_id': 'sb-owned', 'app_id': 'ap-Owned'}))
            (folder / 'old-sandbox.json').write_text(json.dumps(
                {'provider_id': 'sb-foreign', 'app_id': 'ap-Foreign'}))
            (folder / 'nested-sandbox.json').write_text(json.dumps(
                {'provider': {'provider_id': 'sb-foreign'}, 'app_id': 'ap-ForeignNested'}))
            frozen = folder / 'frozen'
            frozen.mkdir()
            (frozen / 'app.json').write_text(json.dumps(
                {'provider_id': 'sb-owned', 'app_id': 'ap-Historical'}))
            snapshot = {'operations': [{'provider_id': 'sb-owned', 'game': 'a'}], 'grants': {'a': {}}}
            apps, missing = owned_apps(root, snapshot)
            self.assertEqual(set(apps), {'ap-Owned'})
            self.assertEqual(missing, [])

    def test_app_only_metadata_needs_one_unambiguous_owned_worker(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / 'a' / 'job'
            folder.mkdir(parents=True)
            (folder / 'app.json').write_text(json.dumps({'app_id': 'ap-Owned'}))
            (folder / 'provider.json').write_text(json.dumps({'provider_id': 'sb-owned'}))
            snapshot = {'operations': [{'provider_id': 'sb-owned', 'game': 'a'}], 'grants': {'a': {}}}
            self.assertEqual(set(owned_apps(root, snapshot)[0]), {'ap-Owned'})
            (folder / 'worker.json').write_text(json.dumps({'provider_id': 'sb-other'}))
            snapshot['operations'].append({'provider_id': 'sb-other', 'game': 'a'})
            self.assertEqual(owned_apps(root, snapshot)[0], {})


if __name__ == '__main__':
    unittest.main()
