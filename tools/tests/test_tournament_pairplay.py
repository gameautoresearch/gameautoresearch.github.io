"""Offline tests for the Generals and Dou Dizhu tournaments (tools/tournament/pairplay.py and game modules).

The Modal runner is replaced by a stub that inspects the staged request and returns synthetic rows, so
these tests check entrant resolution against the real sealed bundles, schedules, staging, row validation,
scoring and result.json without creating any worker. Run with either game's research interpreter:

    PYTHONPATH=$PWD .venv-generals-ar/bin/python -m unittest tools.tests.test_tournament_pairplay
    PYTHONPATH=$PWD .venv-doudizhu-ar/bin/python -m unittest tools.tests.test_tournament_pairplay
"""
import sys

sys.dont_write_bytecode = True

from pathlib import Path
import json
import shutil
import tempfile
import unittest

from tools.tournament import pairplay, generals_worker, doudizhu_worker

REPO = Path(__file__).resolve().parents[2]
GENERALS_SESSION = REPO / 'evidence/six-game-5h-20261111/generals/deepseek-v41flash/smoke-1'
DDZ_SESSIONS = [REPO / 'evidence/ddz-3model-1h-20261111' / m / 'smoke-2' for m in ('opus55', 'deepseek-v4-pro', 'glm53')]


def fake_session(root, name, source, **changes):
    """A read-only-safe copy of a finished session: run record, final-train.json and the sealed bundle."""
    session = Path(root) / name / 'main'
    bundle = session.joinpath(*pairplay.FINAL_BUNDLE)
    bundle.mkdir(parents=True)
    for item in source.joinpath(*pairplay.FINAL_BUNDLE).iterdir():
        shutil.copyfile(item, bundle / item.name)
    shutil.copyfile(source / 'final-train.json', session / 'final-train.json')
    run = json.loads(Path(str(source) + '.harness-run.json').read_text())
    run['settings'].update(changes.pop('settings', {}))
    run.update(changes)
    Path(str(session) + '.harness-run.json').write_text(json.dumps(run))
    return session


class Stub:
    """Stands in for ModalPairRunner; `outcome(key, request)` gives A's points for one scheduled game."""

    def __init__(self, adapter, outcome, row):
        self.adapter, self.outcome, self.row, self.calls = adapter, outcome, row, []

    def __call__(self, pair, staging, folder):
        for local, _ in staging['files']:
            assert Path(local).is_file(), local
        self.calls.append((pair, staging['request']))
        rows = [dict(self.row(key), a_points=self.outcome(key), outcome='natural', fault_by=None)
                for key in sorted(self.adapter.expected_keys(self.adapter_schedule(staging['request'])))]
        pairplay.write_json(folder / 'sandbox.json', dict(worker_seconds=10.0))
        return dict(rows=rows, seconds=1.0)

    def adapter_schedule(self, request):
        if 'map_indices' in request:
            return dict(map_indices=request['map_indices'])
        return dict(deals=request['deals'])


class WorkerScheduleTests(unittest.TestCase):
    def test_generals_both_seats_and_one_seat_per_part(self):
        parts = generals_worker.schedule(range(70), batch_size=32)
        self.assertEqual([len(m) for _, m in parts], [32, 32, 6, 32, 32, 6])
        games = [(i, seat) for seat, maps in parts for i in maps]
        self.assertEqual(sorted(games), sorted((i, s) for i in range(70) for s in (0, 1)))
        for seat, maps in parts:
            self.assertEqual(len(set(maps)), len(maps))  # no map twice in one part, so no opposite-seat views

    def test_generals_points(self):
        self.assertEqual(generals_worker.points(1., False), 1.)
        self.assertEqual(generals_worker.points(-1., False), 0.)
        self.assertEqual(generals_worker.points(0., True), .5)
        self.assertEqual(generals_worker.points(0., False), .5)

    def test_doudizhu_team_orderings(self):
        self.assertEqual([doudizhu_worker.controller(0, r) for r in doudizhu_worker.ROLES], ['a', 'b', 'b'])
        self.assertEqual([doudizhu_worker.controller(1, r) for r in doudizhu_worker.ROLES], ['b', 'a', 'a'])
        self.assertTrue(doudizhu_worker.a_won(0, True))
        self.assertFalse(doudizhu_worker.a_won(1, True))
        self.assertTrue(doudizhu_worker.a_won(1, False))
        batches = doudizhu_worker.schedule(20, batch_size=32)
        self.assertEqual([len(b) for b in batches], [32, 8])
        for batch in batches:  # both orderings of every deal share a batch: same cards, same moment
            deals = [d for d, _ in batch]
            self.assertTrue(all(deals.count(d) == 2 for d in deals))


class ScoringTests(unittest.TestCase):
    def test_pair_summary_leaderboard_and_ci(self):
        rows = [dict(map_index=i, a_seat=s, a_points=1.0 if s == 0 else 0.5, outcome='natural', fault_by=None)
                for i in range(4) for s in (0, 1)]
        rows[-1].update(outcome='fault', fault_by='b', a_points=1.0)
        summary = pairplay.summarize_pair('x', 'y', rows, 'map_index', 'pairs/x-vs-y/')
        self.assertEqual(summary['games'], 8)
        self.assertAlmostEqual(summary['a_score_0_100'], 100 * 6.5 / 8)
        self.assertEqual(summary['faults'], 1)
        self.assertEqual(summary['faults_by'], {'b': 1})
        low, high = summary['ci95']
        self.assertLessEqual(low, summary['a_score_0_100'])
        self.assertGreaterEqual(high, summary['a_score_0_100'])
        board = pairplay.leaderboard(['x', 'y', 'z'], {('x', 'y'): rows})
        self.assertEqual([b['name'] for b in board], ['x', 'y', 'z'])
        self.assertAlmostEqual(board[1]['score_0_100'], 100 * 1.5 / 8)
        self.assertEqual(board[2]['games'], 0)

    def test_rows_must_match_schedule(self):
        key = lambda r: (r['map_index'], r['a_seat'])
        rows = [dict(map_index=0, a_seat=0, a_points=1.0, outcome='natural', fault_by=None)]
        with self.assertRaises(ValueError):
            pairplay.validate_rows(rows, {(0, 0), (0, 1)}, key)
        with self.assertRaises(ValueError):
            pairplay.validate_rows(rows * 2, {(0, 0)}, key)
        with self.assertRaises(ValueError):
            pairplay.validate_rows([dict(rows[0], a_points=0.3)], {(0, 0)}, key)
        with self.assertRaises(ValueError):
            pairplay.validate_rows([dict(rows[0], outcome='fault')], {(0, 0)}, key)
        pairplay.validate_rows(rows, {(0, 0)}, key)


class GameTestCase(unittest.TestCase):
    module = None
    probe = ()
    sources = ()

    def setUp(self):
        try:
            for name in self.probe:
                __import__(name)
            self.game = __import__('tools.tournament.' + self.module, fromlist=['Adapter'])
            self.adapter = self.game.Adapter()
        except ImportError as exc:
            self.skipTest('game modules unavailable in this interpreter: ' + str(exc))
        if not all(Path(s).is_dir() for s in self.sources):
            self.skipTest('test sessions not present')
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        if hasattr(self, 'tmp'):
            self.tmp.cleanup()


class GeneralsTests(GameTestCase):
    module = 'generals'
    probe = ('jax', 'integrations.generals.environment')
    sources = (GENERALS_SESSION,)

    def setUp(self):
        super().setUp()
        # The retained session's bundle binds the ruleset it was trained under. After a ruleset change it no
        # longer verifies (by design), so these tests wait for a retained session of the current ruleset.
        try:
            pairplay.resolve_entrant(*pairplay.parse_entrant(self.entrants()[0]), self.adapter.verify,
                                     self.adapter.release_engine_digest)
        except ValueError as exc:
            if 'mismatch' not in str(exc):
                raise
            self.skipTest('the retained test session was built under an earlier ruleset: ' + str(exc))

    def entrants(self, **changes):
        return ['one=' + str(fake_session(self.root, 'one', GENERALS_SESSION)),
                'two=' + str(fake_session(self.root, 'two', GENERALS_SESSION, **changes))]

    def test_smoke_tournament_writes_result(self):
        stub = Stub(self.adapter, lambda key: 1.0 if key[1] == 0 else 0.0,
                    lambda key: dict(map_index=key[0], a_seat=key[1], length=10))
        result = pairplay.run(self.adapter, self.entrants(), self.root / 'out', smoke=True, runner=stub)
        self.assertTrue(result['complete'])
        self.assertEqual(stub.calls[0][1]['map_indices'], [0, 1])
        self.assertEqual(stub.calls[0][1]['pool_path'], '/private/maps.npz')
        pair = result['pairs'][0]
        self.assertEqual((pair['a'], pair['b'], pair['games'], pair['a_score_0_100']), ('one', 'two', 4, 50.0))
        self.assertEqual(result['worker_seconds'], 10.0)
        self.assertEqual({b['score_0_100'] for b in result['leaderboard']}, {50.0})
        on_disk = json.loads((self.root / 'out/result.json').read_text())
        self.assertEqual(on_disk['entrants']['one']['bundle_sha256'], on_disk['entrants']['two']['bundle_sha256'])
        self.assertEqual(len((self.root / 'out/pairs/one-vs-two/games.jsonl').read_text().splitlines()), 4)
        with self.assertRaises(FileExistsError):
            pairplay.run(self.adapter, self.entrants(), self.root / 'out', smoke=True, runner=stub)

    def test_refusals(self):
        stub = Stub(self.adapter, lambda key: 1.0, lambda key: dict(map_index=key[0], a_seat=key[1]))
        specs = self.entrants(release_sha256='0' * 64)
        with self.assertRaises(ValueError):  # different release identity
            pairplay.run(self.adapter, specs, self.root / 'out1', smoke=True, runner=stub)
        session = Path(specs[0].split('=', 1)[1])
        fine = fake_session(self.root, 'five', GENERALS_SESSION)
        with self.assertRaises(ValueError):  # output inside an entrant's session
            pairplay.run(self.adapter, [specs[0], 'five=' + str(fine)], session / 'x', smoke=True, runner=stub)
        self.assertFalse((session / 'x').exists())
        tampered = fake_session(self.root, 'three', GENERALS_SESSION)
        policy = tampered.joinpath(*pairplay.FINAL_BUNDLE) / 'policy.py'
        policy.write_text(policy.read_text() + '\n# edited after sealing\n')
        with self.assertRaises(ValueError):
            pairplay.run(self.adapter, [specs[0], 'three=' + str(tampered)], self.root / 'out2', smoke=True, runner=stub)
        network = fake_session(self.root, 'four', GENERALS_SESSION, settings={'variant': 'network'})
        with self.assertRaises(ValueError):
            pairplay.run(self.adapter, [specs[0], 'four=' + str(network)], self.root / 'out3', smoke=True, runner=stub)
        self.assertFalse(any((self.root / n).exists() for n in ('out1', 'out2', 'out3')))

    def test_failed_pair_is_reported_not_scored(self):
        def broken(pair, staging, folder):
            raise RuntimeError('worker died')
        result = pairplay.run(self.adapter, self.entrants(), self.root / 'out', smoke=True, runner=broken)
        self.assertFalse(result['complete'])
        self.assertIn('worker died', result['pairs'][0]['error'])
        self.assertEqual(result['leaderboard'][0]['games'], 0)


class DoudizhuTests(GameTestCase):
    module = 'doudizhu'
    probe = ('simulators.doudizhu.autoresearch.bundle',)
    sources = tuple(DDZ_SESSIONS)

    def test_three_entrants_round_robin(self):
        specs = [n + '=' + str(fake_session(self.root, n, s)) for n, s in zip(('p', 'q', 'r'), DDZ_SESSIONS)]
        stub = Stub(self.adapter, lambda key: 1.0 if key[1] == 0 else 0.0,  # landlord side always wins
                    lambda key: dict(deal_index=key[0], ordering=key[1]))
        result = pairplay.run(self.adapter, specs, self.root / 'out', smoke=True, runner=stub)
        self.assertTrue(result['complete'])
        self.assertEqual([(p['a'], p['b']) for p in result['pairs']], [('p', 'q'), ('p', 'r'), ('q', 'r')])
        self.assertTrue(all(p['games'] == 8 and p['a_score_0_100'] == 50.0 for p in result['pairs']))
        self.assertEqual(result['worker_seconds'], 30.0)
        self.assertEqual({b['games'] for b in result['leaderboard']}, {16})
        request = stub.calls[0][1]
        self.assertEqual(request['deals'], 4)
        self.assertEqual((request['a'], request['b']), ('/private/a', '/private/b'))
        text = (self.root / 'out/result.json').read_text() + (self.root / 'out/plan.json').read_text()
        self.assertNotIn(str(request['deal_seed']), text)  # the private deal seed stays on the worker

    def test_mixed_releases_refused(self):
        other = REPO / 'evidence/ddz-3model-1h-20261111/release'
        specs = ['p=' + str(fake_session(self.root, 'p', DDZ_SESSIONS[0])),
                 'q=' + str(fake_session(self.root, 'q', DDZ_SESSIONS[1], settings={'release': str(other)}))]
        with self.assertRaises(ValueError):
            pairplay.run(self.adapter, specs, self.root / 'out', smoke=True, runner=lambda *a: None)
        self.assertFalse((self.root / 'out').exists())


if __name__ == '__main__':
    unittest.main()
