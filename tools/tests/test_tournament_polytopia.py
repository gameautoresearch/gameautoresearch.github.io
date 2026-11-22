"""Offline tests of the Polytopia tournament and its shared plumbing; the CPU grader is a stub.

    PYTHONPATH=$PWD .venv-polytopia-ar/bin/python -m unittest tools.tests.test_tournament_polytopia
"""
import sys

sys.dont_write_bytecode = True

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from simulators.polytopia.autoresearch.v1 import core
from tools.tournament import common, polytopia


def release(root, tag='r'):
    path = Path(root) / ('release-' + tag)
    path.mkdir()
    (path / 'benchmark.json').write_text(json.dumps(dict(tag=tag)))
    return path, core.seal(path)


def bundle(root, name, release_sha256, strength):
    path = Path(root) / ('bundle-' + name)
    path.mkdir()
    for f in ('final.npz', 'final.json', 'network.py', 'ar_arch.json'):
        (path / f).write_text(f + name)
    (path / 'strength.txt').write_text(str(strength))
    (path / 'binding.json').write_text(json.dumps(dict(benchmark_id=core.BENCHMARK, release_sha256=release_sha256)))
    core.seal(path)
    return path


class Grader:
    """Stands in for Provider.evaluate: the stronger network wins, equal strengths draw."""

    def __init__(self, fail_maps=()):
        self.calls = []
        self.fail_maps = set(fail_maps)

    def evaluate(self, bundle, seeds, opponent_seed, out, opponent_bundle=None):
        core.verify(bundle)
        self.calls.append((Path(bundle).name, Path(opponent_bundle).name, list(seeds)))
        common.write_json(Path(out).with_suffix('.worker.json'), dict(sandbox_id='sb-' + str(len(self.calls)), app_id='ap'))
        if self.fail_maps & set(seeds):
            raise RuntimeError('CPU evaluation failed')
        a = float((Path(bundle) / 'strength.txt').read_text())
        b = float((Path(opponent_bundle) / 'strength.txt').read_text())
        reward = (a > b) - (a < b)
        rows = [dict(seed=s, seat=seat, reward=float(reward), capped=seat == 2, turn=30, steps=300, scores=[1, 1])
                for s in seeds for seat in (1, 2)]
        common.write_json(out, dict(raw_score_0_100=0, details=rows))
        return {}

    def lookup(self, app_id, sandbox_id):
        return dict(provider_id=sandbox_id, started_at=100.0, finished_at=130.0)


class Plumbing(unittest.TestCase):
    def test_pairs_are_unordered_and_complete(self):
        self.assertEqual(common.pairs(['a', 'b', 'c']), [('a', 'b'), ('a', 'c'), ('b', 'c')])
        self.assertEqual(common.chunks(5, 2), [[0, 1], [2, 3], [4]])

    def test_leaderboard_gives_b_the_complement(self):
        pairs = [dict(a='x', b='y', games=4, a_points=3.0), dict(a='x', b='z', games=4, a_points=2.0),
                 dict(a='y', b='z', games=4, a_points=0.0)]
        board = {r['name']: r for r in common.leaderboard(['x', 'y', 'z'], pairs)}
        self.assertEqual(board['x']['score_0_100'], 62.5)
        self.assertEqual(board['y']['score_0_100'], 12.5)
        self.assertEqual(board['z']['score_0_100'], 75.0)
        self.assertEqual([r['name'] for r in common.leaderboard(['x', 'y', 'z'], pairs)], ['z', 'x', 'y'])

    def test_interval_resamples_units(self):
        rows = [dict(unit=u, seat=s, a_points=float(u % 2)) for u in range(10) for s in (1, 2)]
        low, high = common.bootstrap_ci(rows)
        self.assertLess(low, 50)
        self.assertGreater(high, 50)
        self.assertIsNone(common.bootstrap_ci([dict(unit=0, seat=0, a_points=None)]))


class Tournament(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()
        self.release, self.sha = release(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def args(self, *extra, out='out'):
        return ['--release', str(self.release), '--out', str(self.root / out), *extra]

    def test_every_pair_both_seats_on_one_private_schedule(self):
        strong, weak, mid = (bundle(self.root, n, self.sha, s) for n, s in (('s', 3), ('w', 1), ('m', 2)))
        grader = Grader()
        result = polytopia.main(self.args('--bundle', f'S={strong}', '--bundle', f'W={weak}', '--bundle', f'M={mid}',
                                          '--slice', '5', '--chunk', '2'), provider=grader)
        self.assertEqual([(p['a'], p['b']) for p in result['pairs']], [('S', 'W'), ('S', 'M'), ('W', 'M')])
        self.assertEqual(len(grader.calls), 3 * 3)
        schedule = json.loads((self.root / 'out/private/schedule.json').read_text())['seeds']
        for pair in ('bundle-s', 'bundle-w'), ('bundle-s', 'bundle-m'), ('bundle-w', 'bundle-m'):
            maps = [s for a, b, seeds in grader.calls if (a, b) == pair for s in seeds]
            self.assertEqual(sorted(maps), schedule)  # every pair plays every map once, A as candidate
        first = result['pairs'][0]
        self.assertEqual((first['games'], first['a_points'], first['a_score_0_100']), (10, 10.0, 100.0))
        self.assertEqual(first['by_a_seat'], {'1': dict(games=5, a_score_0_100=100.0), '2': dict(games=5, a_score_0_100=100.0)})
        self.assertEqual(first['truncations'], 5)
        self.assertTrue(first['complete'])
        self.assertEqual([r['name'] for r in result['leaderboard']], ['S', 'M', 'W'])
        self.assertEqual([r['score_0_100'] for r in result['leaderboard']], [100.0, 50.0, 0.0])
        self.assertEqual(result['worker_seconds'], 9 * 30.0)
        saved = json.loads((self.root / 'out/result.json').read_text())
        self.assertEqual(saved['release_sha256'], self.sha)
        self.assertEqual(saved['entrants']['S']['bundle_sha256'], core.verify(strong))
        rows = json.loads((self.root / 'out/pairs/S-vs-W/rows.json').read_text())
        self.assertEqual(len(rows), 10)
        self.assertNotIn('seed', rows[0])

    def test_a_failed_worker_is_recorded_not_scored(self):
        a, b = bundle(self.root, 'a', self.sha, 1), bundle(self.root, 'b', self.sha, 1)

        class Failing(Grader):
            def evaluate(self, bundle, seeds, opponent_seed, out, opponent_bundle=None):
                if len(self.calls) == 1:
                    self.calls.append(None)
                    raise RuntimeError('CPU evaluation failed')
                return super().evaluate(bundle, seeds, opponent_seed, out, opponent_bundle)

        result = polytopia.main(self.args('--bundle', f'A={a}', '--bundle', f'B={b}', '--slice', '4', '--chunk', '2',
                                          '--parallel', '1'), provider=Failing())
        pair = result['pairs'][0]
        self.assertEqual((pair['games'], pair['a_score_0_100'], pair['unplayed_games']), (4, 50.0, 4))
        self.assertFalse(pair['complete'])
        self.assertIn('CPU evaluation failed', pair['errors'][0]['error'])

    def test_session_entrants_use_their_fresh_final_bundle(self):
        a, b = bundle(self.root, 'a', self.sha, 2), bundle(self.root, 'b', self.sha, 1)
        sessions = []
        for name, path in (('x', a), ('y', b)):
            session = self.root / name / 'main'
            (session / 'final').mkdir(parents=True)
            session.with_name('main.harness-run.json').write_text(json.dumps(dict(settings=dict(
                variant='recipe', release=str(self.release)))))
            (session / 'final-train.json').write_text(json.dumps(dict(status='complete')))
            (session / 'final/state.json').write_text(json.dumps(dict(finalized=True, submission=dict(
                bundle=str(path), sha256=core.verify(path)))))
            sessions.append(session)
        result = polytopia.main(['--entrant', f'X={sessions[0]}', '--entrant', f'Y={sessions[1]}', '--out',
                                 str(self.root / 'out'), '--smoke'], provider=Grader())
        self.assertEqual(result['entrants']['X']['session'], str(sessions[0]))
        self.assertEqual(result['pairs'][0]['games'], 4)
        self.assertTrue(result['smoke'])
        (sessions[1] / 'final-train.json').write_text(json.dumps(dict(status='failed')))
        with self.assertRaises(ValueError):
            polytopia.main(['--entrant', f'X={sessions[0]}', '--entrant', f'Y={sessions[1]}', '--out',
                            str(self.root / 'out2'), '--smoke'], provider=Grader())

    def test_a_session_of_the_dollar_contract_is_read_from_its_ledger(self):
        from simulators.polytopia.autoresearch.contract import DEFAULT, TRAIN_SHAPE, rate
        from simulators.polytopia.autoresearch.ledger import Ledger
        path = bundle(self.root, 'a', self.sha, 2)
        final = self.root / 'x/main/final'
        ledger = Ledger(final, dict(DEFAULT))
        ledger.queue('final', 300., dict(kind='train', shape=dict(TRAIN_SHAPE), rate=rate(TRAIN_SHAPE)))
        ledger.settle('final', dict(provider_id='sb-1', started_at=1., finished_at=200.), 'complete')
        ledger.attach('final', path, core.verify(path))
        ledger.submit('final')
        with self.assertRaisesRegex(ValueError, 'no finalized submission'):
            polytopia.session_bundle(final.parent)
        ledger.finalize()
        sealed = final / 'submissions' / core.verify(path)  # where the backend keeps the read-only copy
        shutil.copytree(path, sealed)
        self.assertEqual(polytopia.session_bundle(final.parent), sealed)

    def test_refusals(self):
        other, other_sha = release(self.root, 'other')
        a, b = bundle(self.root, 'a', self.sha, 1), bundle(self.root, 'b', self.sha, 1)
        foreign = bundle(self.root, 'f', other_sha, 1)
        with self.assertRaises(ValueError):  # sealed against another release
            polytopia.main(self.args('--bundle', f'A={a}', '--bundle', f'F={foreign}'), provider=Grader())
        with self.assertRaises(ValueError):  # output inside an entrant
            polytopia.main(['--release', str(self.release), '--out', str(a / 'out'), '--bundle', f'A={a}',
                            '--bundle', f'B={b}'], provider=Grader())
        with self.assertRaises(ValueError):  # output inside the release
            polytopia.main(['--release', str(self.release), '--out', str(self.release / 'out'), '--bundle', f'A={a}',
                            '--bundle', f'B={b}'], provider=Grader())
        (self.root / 'taken').mkdir()
        with self.assertRaises(FileExistsError):
            polytopia.main(self.args('--bundle', f'A={a}', '--bundle', f'B={b}', out='taken'), provider=Grader())
        with self.assertRaises(ValueError):
            polytopia.main(self.args('--bundle', f'A={a}', '--bundle', f'A={b}'), provider=Grader())
        with self.assertRaises(ValueError):
            polytopia.main(['--out', str(self.root / 'o'), '--bundle', f'A={a}', '--bundle', f'B={b}'], provider=Grader())
        (a / 'final.json').write_text('changed')
        with self.assertRaises(ValueError):
            polytopia.main(self.args('--bundle', f'A={a}', '--bundle', f'B={b}'), provider=Grader())
        self.assertEqual(core.verify(self.release), self.sha)  # the release itself is never written

    def panel_release(self, tag, strengths, extra=None):
        import hashlib
        path = Path(self.root) / ('release-' + tag)
        path.mkdir()
        (path / 'benchmark.json').write_text(json.dumps(dict(tag='same', worker=extra or 'a')))
        (path / 'v1.py').write_text('engine')
        panel = []
        for name, strength in strengths.items():
            folder = path / 'host-private-panel' / name
            folder.mkdir(parents=True)
            for f in ('final.npz', 'final.json', 'network.py', 'ar_arch.json'):
                (folder / f).write_text(f + name if f == 'final.npz' else f)
            (folder / 'strength.txt').write_text(str(strength))
            panel.append(dict(name=name, directory='host-private-panel/' + name,
                              checkpoint_sha256=hashlib.sha256((folder / 'final.npz').read_bytes()).hexdigest(),
                              network_sha256=hashlib.sha256((folder / 'network.py').read_bytes()).hexdigest()))
        (path / 'host-private-panel' / 'inventory.json').write_text(json.dumps(dict(panel=panel)))
        return path, core.seal(path)

    def test_references_play_only_as_b_and_not_each_other(self):
        rel, sha = self.panel_release('p', dict(weak=0, mid=2))
        a, b = bundle(self.root, 'a', sha, 3), bundle(self.root, 'b', sha, 1)
        grader = Grader()
        out = self.root / 'ref'
        result = polytopia.main(['--release', str(rel), '--out', str(out), '--bundle', f'A={a}', '--bundle', f'B={b}',
                                 '--reference', 'weak', '--reference', 'mid', '--slice', '2'], provider=grader)
        self.assertEqual([(p['a'], p['b']) for p in result['pairs']],
                         [('A', 'B'), ('A', 'weak'), ('A', 'mid'), ('B', 'weak'), ('B', 'mid')])
        self.assertTrue(result['entrants']['mid']['reference'])
        board = {r['name']: r['score_0_100'] for r in result['leaderboard']}
        self.assertEqual(board, dict(A=100.0, B=100 / 3, mid=50.0, weak=0.0))
        with self.assertRaisesRegex(ValueError, 'No panel opponent'):
            polytopia.main(['--release', str(rel), '--out', str(self.root / 'ref3'), '--bundle', f'A={a}', '--bundle', f'B={b}',
                            '--reference', 'nope', '--slice', '2'], provider=Grader())
        (rel / 'host-private-panel' / 'mid' / 'final.npz').write_text('tampered')
        with self.assertRaisesRegex(ValueError, 'manifest mismatch|differs from the panel inventory'):
            polytopia.main(['--release', str(rel), '--out', str(self.root / 'ref2'), '--bundle', f'A={a}', '--bundle', f'B={b}',
                            '--reference', 'mid', '--slice', '2'], provider=Grader())

    def test_equivalent_release_admits_bundles_only_when_play_files_match(self):
        base, base_sha = self.panel_release('base', dict(weak=0))
        twin = self.root / 'release-twin'
        shutil.copytree(base, twin)
        (twin / 'manifest.json').unlink()
        (twin / 'benchmark.json').write_text(json.dumps(dict(tag='same', worker='b')))
        (twin / 'public').mkdir()
        (twin / 'public' / 'instructions.md').write_text('different text')
        twin_sha = core.seal(twin)
        self.assertNotEqual(twin_sha, base_sha)
        a, b = bundle(self.root, 'a', base_sha, 2), bundle(self.root, 'b', twin_sha, 1)
        args = ['--release', str(base), '--bundle', f'A={a}', '--bundle', f'B={b}', '--slice', '2']
        with self.assertRaisesRegex(ValueError, 'sealed against a different release'):
            polytopia.main(args + ['--out', str(self.root / 'e1')], provider=Grader())
        result = polytopia.main(args + ['--out', str(self.root / 'e2'), '--equivalent-release', str(twin)],
                                provider=Grader())
        self.assertEqual(result['entrants']['B']['release_sha256'], twin_sha)
        self.assertEqual(set(result['accepted_releases']), {base_sha, twin_sha})
        other = self.root / 'release-other'
        shutil.copytree(twin, other)
        (other / 'manifest.json').unlink()
        (other / 'v1.py').write_text('changed engine')
        core.seal(other)
        with self.assertRaisesRegex(ValueError, 'files that reach play'):
            polytopia.main(args + ['--out', str(self.root / 'e3'), '--equivalent-release', str(other)],
                           provider=Grader())


if __name__ == '__main__':
    unittest.main()
