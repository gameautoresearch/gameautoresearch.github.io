"""Offline tests of the Clash tournament; the CPU grader (FastProvider.cpu_grade) and the release check are stubs.

    PYTHONPATH=$PWD .venv-clash-ar/bin/python -m unittest tools.tests.test_tournament_clash
"""
import sys

sys.dont_write_bytecode = True

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from simulators.clash.autoresearch import bundle as clash_bundle
from simulators.clash.autoresearch.common import HERE, digest, engine_binding
from simulators.clash.autoresearch.fast_entry import runtime_binding
from tools.tournament import clash, common


def make_bundle(root, name, strength, policy=None):
    """A real sealed bundle (starter source) whose single tensor encodes its test strength."""
    source = Path(root) / ('source-' + name)
    source.mkdir()
    for f in ('policy.py', 'learner.py', 'train.py'):
        shutil.copyfile(HERE / 'starter' / f, source / f)
    shutil.copyfile(HERE / 'recipes/starter.json', source / 'config.json')
    if policy is not None:
        (source / 'policy.py').write_text(policy)
    np.savez(source / 'weights.npz', strength=np.array([strength], np.float32))
    clash_bundle.seal(source, Path(root) / ('bundle-' + name), {'test': name})
    return Path(root) / ('bundle-' + name)


def strength(folder):
    with np.load(Path(folder) / 'weights.npz') as z:
        return float(z['strength'][0])


class Grader:
    """Stands in for FastProvider.cpu_grade: the stronger network wins, equal strengths tie."""

    def __init__(self, fail_first=False):
        self.calls = []
        self.fail_first = fail_first
        self.interface = None
        self.pair_workers = None

    def cpu_grade(self, bundle, schedule, opponents, out, seconds=600, split='public-development', stop=None):
        manifest = clash_bundle.verify(bundle)
        value = json.loads(Path(schedule).read_text())
        (name, folder), = opponents.items()
        assert [o['name'] for o in value['opponents']] == [name] and value['opponents'][0]['kind'] == 'checkpoint'
        clash_bundle.verify(folder, value['opponents'][0]['identity'])
        self.calls.append((Path(bundle).name, name, list(value['engine_seeds']), self.pair_workers, split))
        common.write_json(Path(out).with_suffix('.provider.json'), dict(cpu_only=True, provider=dict(
            provider_id='sb-%d' % len(self.calls), started_at=100.0, finished_at=160.0)))
        if self.fail_first and len(self.calls) == 1:
            raise RuntimeError('public practice failed: incomplete/invalid v3 grading')
        a, b = strength(bundle), strength(folder)
        rows = []
        for i, _ in enumerate(value['engine_seeds']):
            for seat in (0, 1):
                points = .5 if a == b else float(a > b)
                rows.append(dict(opponent=name, seed_index=i, seat=seat, winner=2 if a == b else seat if a > b else 1 - seat,
                                 points=points, decisions=400, natural=True, truncated=False, illegal=False, fault=False))
        for actor in value['opponents']:
            actor['bundle'] = '/private/' + actor['name']
        grade = dict(rows=rows, submission_identity=manifest['identity'], schedule_commitment=digest(value))
        common.write_json(out, grade)
        return grade


def fake_release(root):
    path = Path(root) / 'release'
    (path / 'public').mkdir(parents=True)
    (path / 'public/practice.json').write_text(json.dumps(dict(engine_seeds=[2100101], opponents=[])))
    (path / 'public/interface.json').write_text(json.dumps(dict(version='clash-evaluation-v3.1')))
    return path


class Tournament(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()
        self.release = fake_release(self.root)
        binding = dict(engine=engine_binding(), trusted_fast_runtime=runtime_binding())
        patcher = mock.patch.object(clash, 'verify_release', return_value=('releasesha', binding))
        patcher.start()
        self.addCleanup(patcher.stop)

    def tearDown(self):
        self.tmp.cleanup()

    def args(self, *extra, out='out'):
        return ['--release', str(self.release), '--out', str(self.root / out), *extra]

    def test_seeds_sets_the_schedule_size(self):
        a, b = (make_bundle(self.root, n, v) for n, v in (('a', 3), ('b', 1)))
        grader = Grader()
        clash.main(self.args('--bundle', f'A={a}', '--bundle', f'B={b}', '--seeds', '6', '--chunk', '3'),
                   provider=grader)
        schedule = json.loads((self.root / 'out/private/schedule.json').read_text())['engine_seeds']
        self.assertEqual(len(set(schedule)), 6)
        self.assertEqual(len(grader.calls), 2)

    def test_every_pair_both_seats_on_one_private_schedule(self):
        s, w, m = (make_bundle(self.root, n, v) for n, v in (('s', 3), ('w', 1), ('m', 2)))
        grader = Grader()
        result = clash.main(self.args('--bundle', f'S={s}', '--bundle', f'W={w}', '--bundle', f'M={m}',
                                      '--slice', '5', '--chunk', '2'), provider=grader)
        self.assertEqual([(p['a'], p['b']) for p in result['pairs']], [('S', 'W'), ('S', 'M'), ('W', 'M')])
        self.assertEqual(len(grader.calls), 3 * 3)
        self.assertEqual(grader.pair_workers, 1)
        self.assertEqual(grader.interface, dict(version='clash-evaluation-v3.1'))
        schedule = json.loads((self.root / 'out/private/schedule.json').read_text())['engine_seeds']
        self.assertEqual(len(set(schedule)), 5)
        self.assertNotIn(2100101, schedule)
        for a, b in (('bundle-s', 'W'), ('bundle-s', 'M'), ('bundle-w', 'M')):
            seeds = [x for c in grader.calls if c[:2] == (a, b) for x in c[2]]
            self.assertEqual(sorted(seeds), sorted(schedule))
        first = result['pairs'][0]
        self.assertEqual((first['games'], first['a_points'], first['a_score_0_100']), (10, 10.0, 100.0))
        self.assertEqual(set(first['by_a_seat']), {'0', '1'})
        self.assertTrue(first['complete'])
        self.assertEqual([r['name'] for r in result['leaderboard']], ['S', 'M', 'W'])
        self.assertEqual([r['score_0_100'] for r in result['leaderboard']], [100.0, 50.0, 0.0])
        self.assertEqual(result['worker_seconds'], 9 * 60.0)
        saved = json.loads((self.root / 'out/result.json').read_text())
        self.assertEqual(saved['entrants']['S']['bundle_sha256'], clash_bundle.verify(s)['identity'])
        self.assertEqual(saved['release_sha256'], 'releasesha')
        rows = json.loads((self.root / 'out/pairs/S-vs-W/rows.json').read_text())
        self.assertEqual(len(rows), 10)
        self.assertNotIn('seed', rows[0])

    def test_same_network_twice_scores_fifty(self):
        x = make_bundle(self.root, 'x', 1)
        result = clash.main(self.args('--bundle', f'A={x}', '--bundle', f'B={x}', '--smoke'), provider=Grader())
        self.assertEqual([r['score_0_100'] for r in result['leaderboard']], [50.0, 50.0])
        self.assertEqual(result['pairs'][0]['games'], 4)

    def test_failed_worker_is_recorded_not_scored(self):
        a, b = make_bundle(self.root, 'a', 2), make_bundle(self.root, 'b', 1)
        result = clash.main(self.args('--bundle', f'A={a}', '--bundle', f'B={b}', '--slice', '4', '--chunk', '2',
                                      '--parallel', '1'), provider=Grader(fail_first=True))
        pair = result['pairs'][0]
        self.assertEqual((pair['games'], pair['unplayed_games'], pair['a_score_0_100']), (4, 4, 100.0))
        self.assertFalse(pair['complete'])
        self.assertIn('incomplete/invalid', pair['errors'][0]['error'])
        self.assertEqual(result['worker_seconds'], 2 * 60.0)

    def test_tampered_worker_result_is_refused(self):
        a, b = make_bundle(self.root, 'a', 2), make_bundle(self.root, 'b', 1)

        class Lying(Grader):
            def cpu_grade(self, bundle, schedule, opponents, out, **kw):
                grade = super().cpu_grade(bundle, schedule, opponents, out, **kw)
                return dict(grade, rows=grade['rows'][:-1])

        result = clash.main(self.args('--bundle', f'A={a}', '--bundle', f'B={b}', '--smoke'), provider=Lying())
        self.assertEqual(result['pairs'][0]['games'], 0)
        self.assertIn('differ from the schedule', result['pairs'][0]['errors'][0]['error'])

    def test_session_entrants_use_their_fresh_final_bundle(self):
        sessions = []
        for name, value in (('x', 2), ('y', 1)):
            session = self.root / name / 'main'
            folder = session / 'final/jobs/final'
            folder.mkdir(parents=True)
            sealed = make_bundle(folder, 'n', value)
            sealed.rename(folder / 'bundle')
            session.with_name('main.harness-run.json').write_text(json.dumps(dict(settings=dict(
                variant='recipe', release=str(self.release)))))
            (session / 'final-train.json').write_text(json.dumps(dict(status='complete', submission=dict(
                identity=clash_bundle.verify(folder / 'bundle')['identity']))))
            sessions.append(session)
        result = clash.main(['--entrant', f'X={sessions[0]}', '--entrant', f'Y={sessions[1]}', '--out',
                             str(self.root / 'out'), '--smoke'], provider=Grader())
        self.assertEqual(result['entrants']['X']['session'], str(sessions[0]))
        self.assertEqual(result['entrants']['X']['bundle'], str(sessions[0] / 'final/jobs/final/bundle'))
        self.assertEqual(result['leaderboard'][0]['name'], 'X')
        record = json.loads((sessions[1] / 'final-train.json').read_text())
        record['submission']['identity'] = '0' * 64
        (sessions[1] / 'final-train.json').write_text(json.dumps(record))
        with self.assertRaises(ValueError):  # the bundle is not the one final training submitted
            clash.main(['--entrant', f'X={sessions[0]}', '--entrant', f'Y={sessions[1]}', '--out',
                        str(self.root / 'out2'), '--smoke'], provider=Grader())

    def test_refusals(self):
        a, b = make_bundle(self.root, 'a', 1), make_bundle(self.root, 'b', 1)
        with self.assertRaises(ValueError):  # the evaluator reserves this name
            clash.main(self.args('--bundle', f'candidate={a}', '--bundle', f'B={b}'), provider=Grader())
        bad = make_bundle(self.root, 'bad', 1, policy='import os\ndef act(model,observation,mask):\n    return model(observation,mask)[0].argmax(-1)\n')
        with self.assertRaises(ValueError):  # policy source refused by the evaluator's admission review
            clash.main(self.args('--bundle', f'A={a}', '--bundle', f'X={bad}'), provider=Grader())
        with self.assertRaises(ValueError):  # output inside an entrant bundle
            clash.main(self.args('--bundle', f'A={a}', '--bundle', f'B={b}', out='bundle-a/out'), provider=Grader())
        with self.assertRaises(ValueError):  # output inside the release
            clash.main(self.args('--bundle', f'A={a}', '--bundle', f'B={b}', out='release/out'), provider=Grader())
        (self.root / 'taken').mkdir()
        with self.assertRaises(FileExistsError):
            clash.main(self.args('--bundle', f'A={a}', '--bundle', f'B={b}', out='taken'), provider=Grader())
        weights = b / 'weights.npz'
        weights.chmod(0o644)
        np.savez(weights, strength=np.array([9], np.float32))
        with self.assertRaises(ValueError):  # bundle bytes changed after sealing
            clash.main(self.args('--bundle', f'A={a}', '--bundle', f'B={b}'), provider=Grader())


if __name__ == '__main__':
    unittest.main()
