"""Offline tests of the Tavern tournament; the CPU sandbox and its worker are stubs, the lobby plan is real.

    PYTHONPATH=$PWD .venv-tavern-ar/bin/python -m unittest tools.tests.test_tournament_tavern
"""
import sys

sys.dont_write_bytecode = True

import hashlib
import json
import shutil
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest import mock

import numpy as np

from simulators.tavern.autoresearch.contract import HERE as AUTO
from tools.tournament import common, tavern, tavern_worker

BINDING = dict(benchmark_id='tavern-four-small-v1', native_source_hash='test')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_bundle(root, name, strength, binding=BINDING):
    folder = Path(root) / ('bundle-' + name)
    folder.mkdir(parents=True)
    for f in ('policy.py', 'learner.py', 'train.py'):
        shutil.copyfile(AUTO / 'starter' / f, folder / f)
    config = json.loads((AUTO / 'recipes/starter.json').read_text())
    (folder / 'identity.json').write_text(json.dumps(dict(binding=binding, config=config, source={
        f: sha(folder / f) for f in ('policy.py', 'learner.py', 'train.py')})))
    np.savez(folder / 'weights.npz', strength=np.array([strength], np.float32))
    (folder / 'summary.json').write_text(json.dumps(dict(identity_sha256=sha(folder / 'identity.json'),
                                                         weights_sha256=sha(folder / 'weights.npz'))))
    (folder / 'resume.pt').write_text('never uploaded')
    return folder


def fake_release(root):
    path = Path(root) / 'release'
    (path / 'public').mkdir(parents=True)
    (path / 'private/registry').mkdir(parents=True)
    (path / 'binding.json').write_text(json.dumps(BINDING))
    rules = json.loads((AUTO / 'recipes/starter.json').read_text())['rules']
    opponents = [dict(policy_id=s, kind='script', style=s) for s in ('tempo', 'economy', 'synergy')]
    opponents.append(dict(policy_id='ref', kind='learned', path='learned/ref'))
    (path / 'private/registry/registry.json').write_text(json.dumps(dict(binding=dict(BINDING, rules=rules),
        opponents=opponents, profiles=dict(scripts=['tempo', 'economy', 'synergy'], mixed=['ref'] * 3))))
    (path / 'public/practice.json').write_text(json.dumps(dict(seeds=[50101, 50102, 50103, 50104])))
    return path


class Sandbox:
    count = 0

    def __init__(self):
        Sandbox.count += 1
        self.object_id = 'sb-%d' % Sandbox.count
        self.files = {}
        self.strength = {}


class Provider:
    """Stands in for harness.games.tavern.provider.ModalProvider and the uploaded worker.

    The stronger network places 1st, the weaker 2nd; equal networks are separated by physical seat
    (lower seat ahead), so a mirrored lobby reverses the order exactly as the real deterministic game would.
    """
    app_id = 'ap-test'

    def __init__(self, fault=None, fail_first=False):
        self.sandboxes = []
        self.terminated = []
        self.fault = fault
        self.fail_first = fail_first

    def create(self, gpu, timeout):
        assert gpu is False
        sb = Sandbox()
        self.sandboxes.append(sb)
        return sb

    def identity(self, sb):
        return sb.object_id

    def upload(self, sb, source, dest):
        names = sorted(p.name for p in Path(source).iterdir())
        assert names == sorted(tavern.FILES), names  # exactly the admitted files
        with np.load(Path(source) / 'weights.npz') as z:
            sb.strength[dest.rsplit('/', 1)[1]] = float(z['strength'][0])

    def copy(self, sb, local, remote):
        sb.files[remote] = Path(local).read_text()

    def command(self, sb, args, timeout=220):
        assert args == ['python', '/private/tournament_worker.py', '/private/tournament.json']
        assert sb.files['/private/tournament_worker.py'] == (Path(tavern.__file__).parent / 'tavern_worker.py').read_text()
        if self.fail_first and len(self.sandboxes) == 1:
            raise RuntimeError('Remote command failed: worker crashed')
        request = json.loads(sb.files['/private/tournament.json'])
        rows = []
        for lobby in request['lobbies']:
            seats = lobby['seats']
            a_seat, b_seat = seats.index('a'), seats.index('b')
            sa, sb_ = sb.strength['a'], sb.strength['b']
            a_first = sa > sb_ or (sa == sb_ and a_seat < b_seat)
            row = dict(unit=lobby['unit'], k=lobby['k'], swap=lobby['swap'], a_seat=a_seat, b_seat=b_seat, valid=True,
                       failure=None, fault_by=None, a_placement=1 if a_first else 2, b_placement=2 if a_first else 1,
                       process_seconds=3.0)
            if self.fault and (lobby['unit'], lobby['k'], lobby['swap']) == self.fault[0]:
                row.update(valid=False, failure=dict(kind=self.fault[1]), fault_by=self.fault[2], a_placement=None,
                           b_placement=None)
            rows.append(row)
        privacy = dict(private_paths_denied=3, credential_env_absent=True)
        sb.files['/private/tournament-result.json'] = json.dumps(dict(rows=rows, pending=[], privacy=dict(a=privacy, b=privacy)))
        return ''

    def read(self, sb, remote):
        return sb.files[remote].encode() if remote in sb.files else (_ for _ in ()).throw(FileNotFoundError(remote))

    def terminate(self, sb):
        self.terminated.append(sb.object_id)

    def allocation(self, app_id, sandbox_id):
        return dict(provider_id=sandbox_id, started_at=100.0, finished_at=150.0)


class Plan(unittest.TestCase):
    def test_every_lobby_has_a_mirror_and_each_entrant_every_seat(self):
        styles = ['tempo', 'economy', 'synergy', 'tempo', 'synergy', 'economy']
        plan = tavern_worker.lobbies([(0, 11), (1, 12)], styles)
        self.assertEqual(len(plan), 16)
        for lobby in plan:
            seats = lobby['seats']
            self.assertEqual((seats.count('a'), seats.count('b')), (1, 1))
            self.assertEqual(abs(seats.index('a') - seats.index('b')), 4)
            self.assertEqual(Counter(s[1] for s in seats if isinstance(s, list) or isinstance(s, tuple)),
                             Counter(styles))
            mirror = next(m for m in plan if (m['unit'], m['k'], m['swap']) == (lobby['unit'], lobby['k'], 1 - lobby['swap']))
            swapped = [{'a': 'b', 'b': 'a'}.get(s, s) if isinstance(s, str) else s for s in seats]
            self.assertEqual(mirror['seats'], swapped)
            self.assertEqual(mirror['seed'], lobby['seed'])
        for unit in (0, 1):
            group = [l for l in plan if l['unit'] == unit]
            self.assertEqual(sorted(l['seats'].index('a') for l in group), list(range(8)))
            self.assertEqual(sorted(l['seats'].index('b') for l in group), list(range(8)))

    def test_four_player_lobbies_seat_the_pair_opposite(self):
        styles = ['tempo', 'economy']
        plan = tavern_worker.lobbies([(0, 11), (1, 12)], styles)
        self.assertEqual(len(plan), 8)
        for lobby in plan:
            seats = lobby['seats']
            self.assertEqual(len(seats), 4)
            self.assertEqual(abs(seats.index('a') - seats.index('b')), 2)
            self.assertEqual([s[1] for s in seats if isinstance(s, tuple)], styles if lobby['k'] == 0 else styles[::-1])
        for unit in (0, 1):
            group = [l for l in plan if l['unit'] == unit]
            self.assertEqual(sorted(l['seats'].index('a') for l in group), list(range(4)))
        with self.assertRaises(ValueError):
            tavern_worker.lobbies([(0, 11)], ['tempo'])


class Tournament(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()
        self.release = fake_release(self.root)
        patcher = mock.patch.object(tavern, 'verify_release', return_value='releasesha')
        patcher.start()
        self.addCleanup(patcher.stop)

    def tearDown(self):
        self.tmp.cleanup()

    def args(self, *extra, out='out'):
        return ['--release', str(self.release), '--out', str(self.root / out), *extra]

    def test_every_pair_in_mirrored_lobbies(self):
        s, w, m = (make_bundle(self.root, n, v) for n, v in (('s', 3), ('w', 1), ('m', 2)))
        provider = Provider()
        result = tavern.main(self.args('--bundle', f'S={s}', '--bundle', f'W={w}', '--bundle', f'M={m}',
                                       '--slice', '3', '--chunk', '2'), provider=provider)
        self.assertEqual([(p['a'], p['b']) for p in result['pairs']], [('S', 'W'), ('S', 'M'), ('W', 'M')])
        self.assertEqual(len(provider.sandboxes), 3 * 2)
        self.assertEqual(sorted(provider.terminated), sorted(sb.object_id for sb in provider.sandboxes))
        first = result['pairs'][0]
        self.assertEqual((first['games'], first['a_points'], first['a_score_0_100']), (12, 12.0, 100.0))
        self.assertEqual(sorted(first['by_a_seat']), [str(s) for s in range(4)])
        self.assertTrue(first['complete'])
        self.assertAlmostEqual(first['a_placement_points_0_100'], 100.0)
        self.assertAlmostEqual(first['b_placement_points_0_100'], 200 / 3)
        self.assertEqual([r['name'] for r in result['leaderboard']], ['S', 'M', 'W'])
        self.assertEqual([r['score_0_100'] for r in result['leaderboard']], [100.0, 50.0, 0.0])
        self.assertEqual([r['name'] for r in result['placement_leaderboard']], ['S', 'M', 'W'])
        self.assertEqual(result['worker_seconds'], 6 * 50.0)
        self.assertEqual(result['script_seats']['styles'], ['tempo', 'economy'])
        self.assertEqual(result['players'], 4)
        seeds = json.loads((self.root / 'out/private/schedule.json').read_text())['seeds']
        self.assertEqual(len(set(seeds)), 3)
        self.assertTrue(all(0 <= s < 2 ** 64 for s in seeds))
        request = json.loads(next((self.root / 'out/private/requests').iterdir()).read_text())
        self.assertEqual(request['bundles'], dict(a='/private/entrants/a', b='/private/entrants/b'))
        self.assertTrue({l['seed'] for l in request['lobbies']} <= set(seeds))
        saved = json.loads((self.root / 'out/result.json').read_text())
        self.assertEqual(saved['entrants']['S']['files']['weights.npz'], sha(s / 'weights.npz'))
        rows = json.loads((self.root / 'out/pairs/S-vs-W/rows.json').read_text())
        self.assertEqual(len(rows), 12)
        self.assertFalse(any('seed' in r for r in rows))

    def test_same_network_twice_scores_fifty(self):
        x = make_bundle(self.root, 'x', 1)
        result = tavern.main(self.args('--bundle', f'A={x}', '--bundle', f'B={x}', '--smoke'), provider=Provider())
        self.assertEqual([r['score_0_100'] for r in result['leaderboard']], [50.0, 50.0])
        self.assertEqual(result['pairs'][0]['games'], 8)

    def test_faults_forfeit_only_when_an_entrant_caused_them(self):
        a, b = make_bundle(self.root, 'a', 1), make_bundle(self.root, 'b', 2)
        result = tavern.main(self.args('--bundle', f'A={a}', '--bundle', f'B={b}', '--slice', '1'),
                             provider=Provider(fault=((0, 0, 0), 'inference_deadline', 'b')))
        pair = result['pairs'][0]
        self.assertEqual((pair['games'], pair['a_points'], pair['faults'], pair['faults_by']), (4, 1.0, 1, {'B': 1}))
        result = tavern.main(self.args('--bundle', f'A={a}', '--bundle', f'B={b}', '--slice', '1', out='out2'),
                             provider=Provider(fault=((0, 1, 1), 'administrative_cap', None)))
        pair = result['pairs'][0]
        self.assertEqual((pair['games'], pair['a_points'], pair['truncations'], pair['complete']), (3, 0.0, 1, False))

    def test_failed_worker_is_recorded_and_terminated(self):
        a, b = make_bundle(self.root, 'a', 1), make_bundle(self.root, 'b', 2)
        provider = Provider(fail_first=True)
        result = tavern.main(self.args('--bundle', f'A={a}', '--bundle', f'B={b}', '--slice', '2', '--chunk', '1', '--parallel', '1'),
                             provider=provider)
        pair = result['pairs'][0]
        self.assertEqual((pair['games'], pair['unplayed_games']), (4, 4))
        self.assertIn('worker crashed', pair['errors'][0]['error'])
        self.assertEqual(len(provider.terminated), 2)
        self.assertEqual(result['worker_seconds'], 100.0)

    def test_session_entrants_use_their_final_submission(self):
        sessions = []
        for name, value in (('x', 2), ('y', 1)):
            session = self.root / name / 'main'
            folder = make_bundle(session / 'final/submissions', 'tmp', value)
            folder = folder.rename(folder.with_name('final'))
            (folder / 'resume.pt').unlink()
            files = {p.name: sha(p) for p in folder.iterdir()}
            (folder / 'submission.json').write_text(json.dumps(dict(name='final', files=files)))
            session.with_name('main.harness-run.json').write_text(json.dumps(dict(settings=dict(
                variant='recipe', release=str(self.release)))))
            (session / 'final-train.json').write_text(json.dumps(dict(status='complete', submission=dict(
                submission='final', files=files))))
            sessions.append(session)
        result = tavern.main(['--entrant', f'X={sessions[0]}', '--entrant', f'Y={sessions[1]}', '--out',
                              str(self.root / 'out'), '--smoke'], provider=Provider())
        self.assertEqual(result['entrants']['X']['bundle'], str(sessions[0] / 'final/submissions/final'))
        self.assertEqual(result['leaderboard'][0]['name'], 'X')
        weights = sessions[1] / 'final/submissions/final/weights.npz'
        np.savez(weights, strength=np.array([5], np.float32))
        with self.assertRaises(ValueError):
            tavern.main(['--entrant', f'X={sessions[0]}', '--entrant', f'Y={sessions[1]}', '--out',
                         str(self.root / 'out2'), '--smoke'], provider=Provider())

    def test_refusals(self):
        a = make_bundle(self.root, 'a', 1)
        foreign = make_bundle(self.root, 'f', 1, binding=dict(BINDING, native_source_hash='other'))
        with self.assertRaises(ValueError):  # bound to another task/native source
            tavern.main(self.args('--bundle', f'A={a}', '--bundle', f'F={foreign}'), provider=Provider())
        b = make_bundle(self.root, 'b', 1)
        (b / 'policy.py').write_text('# edited after training\n')
        with self.assertRaises(ValueError):
            tavern.main(self.args('--bundle', f'A={a}', '--bundle', f'B={b}'), provider=Provider())
        c = make_bundle(self.root, 'c', 1)
        with self.assertRaises(ValueError):
            tavern.main(self.args('--bundle', f'A={a}', '--bundle', f'C={c}', out='release/out'), provider=Provider())
        (self.root / 'taken').mkdir()
        with self.assertRaises(FileExistsError):
            tavern.main(self.args('--bundle', f'A={a}', '--bundle', f'C={c}', out='taken'), provider=Provider())


if __name__ == '__main__':
    unittest.main()
