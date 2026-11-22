"""Offline tests of the Pokémon tournament; the native worker is a stub.

Entrant verification uses the real native bundle verifier on a retained research-run bundle (read only);
the test is skipped where that evidence is absent.

    PYTHONPATH=$PWD .venv-pokemon/bin/python -m unittest tools.tests.test_tournament_pokemon
"""
import sys

sys.dont_write_bytecode = True

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from tools.tournament import common, pokemon

ROOT = Path(__file__).resolve().parents[2]
RETAINED = (ROOT / 'evidence/six-game-5h-20261111/pokemon/deepseek-v41flash/smoke-1/research/jobs/'
            'a7439f9227d64c5f9b78cfa4b8033399/output/stage/bundle')


class Worker:
    """Stands in for Provider.launch/collect: plays rows by a per-name strength table."""

    def __init__(self, root, strength, fault_at=None):
        self.root = Path(root)
        self.strength = strength
        self.fault_at = fault_at
        self.specs = {}
        self.stopped = []

    def launch(self, ledger, operation, kind, spec, files, seconds, stopped=None):
        assert kind == 'final' and spec['action'] == 'assess'
        self.specs[operation] = (spec, files)
        ledger.attach(operation, 'sb-' + operation[:6])

    def collect(self, ledger, operation):
        spec, _ = self.specs[operation]
        request = spec['assessment']
        a, b = request['candidate']['name'], request['panel'][0]['name']
        out = self.root / operation
        out.mkdir(parents=True)
        rows = []
        for index, _ in enumerate(request['seeds']):
            for seat in (0, 1):
                row = dict(opponent=b, seed_index=index, seat=seat, natural=True, truncated=False, fault=None,
                           fault_category=None, points=.5 + .5 * ((self.strength[a] > self.strength[b])
                                                                 - (self.strength[a] < self.strength[b])))
                if self.fault_at == (index, seat):
                    row.update(natural=False, points=None, fault='ValueError: illegal action',
                               fault_category='reference_illegal_action')
                rows.append(row)
                if row['fault']:
                    break
            if rows[-1]['fault']:
                break
        common.write_json(out / 'rows.json', rows)
        common.write_json(out / 'summary.json', dict(status='complete' if not self.fault_at else 'incomplete'))
        receipt = dict(provider_id='sb', started_at=10.0, finished_at=70.0, provider_verified_stopped=True)
        ledger.settle(operation, 60.0, receipt)
        return dict(status='settled', output=str(out), provider=receipt, successful=True)

    def verify_profile(self, operation, gpu):
        assert gpu is None

    def stop(self, operation):
        self.stopped.append(operation)


@unittest.skipUnless(RETAINED.is_dir(), 'retained Pokémon research bundle absent')
class Tournament(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()
        binding = json.loads((RETAINED / 'binding.json').read_text())
        self.release = self.root / 'release'
        assessment = self.release / 'private/assessment'
        assessment.mkdir(parents=True)
        (self.release / 'release.json').write_text(json.dumps(dict(
            release_sha256='r' * 64, protected_runtime_sha256=binding['runtime_sha256'],
            benchmark_id=binding['benchmark_id'])))
        (assessment / 'registry.json').write_text(json.dumps(dict(
            evaluator_request=dict(wheel_sha256={'w.whl': 'h'}, source_archive='x', public_sets='y'),
            panel=[dict(kind='power', name='power'),
                   dict(kind='sealed-original-bundle', name='p1', path='peer-1', bundle_sha256='z')])))

    def tearDown(self):
        self.tmp.cleanup()

    def run_main(self, entrants, worker, *extra):
        argv = ['--release', str(self.release), '--out', str(self.root / 'out'), *extra]
        for name, path in entrants:
            argv += ['--bundle', f'{name}={path}']
        return pokemon.main(argv, provider=worker, poll=0)

    def test_every_pair_is_a_one_peer_panel_with_both_seats(self):
        worker = Worker(self.root / 'w', dict(A=3, B=2, C=1))
        result = self.run_main([('A', RETAINED), ('B', RETAINED), ('C', RETAINED)], worker, '--slice', '5',
                               '--chunk', '2')
        self.assertEqual(len(worker.specs), 3 * 3)
        bank = json.loads((self.root / 'out/private/bank.json').read_text())['seeds']
        sha = json.loads((RETAINED / 'manifest.json').read_text())['bundle_sha256']
        seen = {}
        for spec, files in worker.specs.values():
            request = spec['assessment']
            self.assertEqual(request['panel'], [dict(kind='bundle-native', name=request['panel'][0]['name'],
                                                     path='/pilotjob/input/peer-0', bundle_sha256=sha)])
            self.assertEqual(request['candidate']['kind'], 'bundle-native')
            self.assertFalse(request['public_development'])
            names = {name for _, name in files}
            self.assertIn('peer-0/graph.json', names)
            self.assertIn('candidate/graph.json', names)
            self.assertIn('candidate/manifest.json', names)
            self.assertFalse(any(n.startswith('peer-1') for n in names))
            self.assertIn('wheels/w.whl', names)
            pair = (request['candidate']['name'], request['panel'][0]['name'])
            seen.setdefault(pair, []).extend(request['seeds'])
        self.assertEqual(set(seen), {('A', 'B'), ('A', 'C'), ('B', 'C')})
        for seeds in seen.values():
            self.assertEqual(sorted(seeds), sorted(bank))
        first = result['pairs'][0]
        self.assertEqual((first['a'], first['b'], first['games'], first['a_score_0_100']), ('A', 'B', 10, 100.0))
        self.assertEqual(set(first['by_a_seat']), {'0', '1'})
        self.assertEqual([(r['name'], r['score_0_100']) for r in result['leaderboard']],
                         [('A', 100.0), ('B', 50.0), ('C', 0.0)])
        self.assertEqual(result['worker_seconds'], 9 * 60.0)
        ledger = json.loads((self.root / 'out/ledger.json').read_text())['jobs']
        self.assertTrue(all(j['status'] == 'settled' and j['kind'] == 'final' for j in ledger))
        self.assertEqual(worker.stopped, [])

    def test_a_fault_stops_the_chunk_and_is_not_scored(self):
        worker = Worker(self.root / 'w', dict(A=1, B=1), fault_at=(1, 0))
        result = self.run_main([('A', RETAINED), ('B', RETAINED)], worker, '--smoke')
        pair = result['pairs'][0]
        self.assertEqual((pair['games'], pair['faults'], pair['unplayed_games'], pair['a_score_0_100']), (2, 1, 1, 50.0))
        self.assertEqual(pair['faults_by'], {'B': 1})
        self.assertFalse(pair['complete'])

    def test_launch_failure_stops_any_created_worker(self):
        class Broken(Worker):
            def launch(self, ledger, operation, kind, spec, files, seconds, stopped=None):
                ledger.attach(operation, 'sb')
                ledger.unknown(operation, dict(reason='create interrupted'))
                raise RuntimeError('create interrupted')

        worker = Broken(self.root / 'w', dict(A=1, B=1))
        result = self.run_main([('A', RETAINED), ('B', RETAINED)], worker, '--smoke')
        self.assertEqual(len(worker.stopped), 1)
        self.assertEqual(result['pairs'][0]['games'], 0)
        self.assertIn('create interrupted', result['pairs'][0]['errors'][0]['error'])

    def test_refusals(self):
        worker = Worker(self.root / 'w', dict(A=1, B=1))
        manifest = json.loads((self.release / 'release.json').read_text())
        (self.release / 'release.json').write_text(json.dumps(dict(manifest, protected_runtime_sha256='0' * 64)))
        with self.assertRaises(ValueError):  # bundle bound to another native runtime
            self.run_main([('A', RETAINED), ('B', RETAINED)], worker, '--smoke')
        (self.release / 'release.json').write_text(json.dumps(manifest))
        changed = self.root / 'changed'
        shutil.copytree(RETAINED, changed)
        (changed / 'config.json').chmod(0o644)
        (changed / 'config.json').write_text('{}')
        with self.assertRaises(ValueError):
            self.run_main([('A', RETAINED), ('B', changed)], worker, '--smoke')
        self.assertFalse((self.root / 'out').exists())

    def test_session_entrant_reads_the_locked_final_without_writing(self):
        import sqlite3
        session = self.root / 'seat' / 'main'
        (session / 'final').mkdir(parents=True)
        session.with_name('main.harness-run.json').write_text(json.dumps(dict(settings=dict(
            variant='recipe', release=str(self.release)))))
        (session / 'final-train.json').write_text(json.dumps(dict(status='complete')))
        sha = json.loads((RETAINED / 'manifest.json').read_text())['bundle_sha256']
        db = sqlite3.connect(session / 'final/ledger.sqlite3')
        db.execute('CREATE TABLE metadata(name TEXT PRIMARY KEY,value TEXT NOT NULL)')
        db.execute('INSERT INTO metadata VALUES(?,?)', ('final_locked', json.dumps(dict(
            bundle=str(RETAINED), bundle_sha256=sha, id=1, job='final'))))
        db.commit()
        db.close()
        before = sorted(p.name for p in (session / 'final').iterdir())
        result = pokemon.main(['--entrant', f'S={session}', '--bundle', f'R={RETAINED}', '--release',
                               str(self.release), '--out', str(self.root / 'out'), '--smoke'],
                              provider=Worker(self.root / 'w', dict(S=2, R=1)), poll=0)
        self.assertEqual(result['entrants']['S']['session'], str(session))
        self.assertEqual(result['entrants']['S']['bundle_sha256'], sha)
        self.assertEqual(sorted(p.name for p in (session / 'final').iterdir()), before)


if __name__ == '__main__':
    unittest.main()
