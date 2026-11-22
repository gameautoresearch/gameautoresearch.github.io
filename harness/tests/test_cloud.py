"""The cloud coordinator's offline parts: staging, argv rewriting, record mirroring and pull filtering."""
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import time
import unittest

from harness import cloud, policy

ROOT = Path(__file__).resolve().parents[2]


def staged(game, directory):
    archive, digest = cloud.source_archive(game, directory)
    target = Path(directory) / 'repo'
    target.mkdir()
    with tarfile.open(archive) as tar:
        tar.extractall(target, filter='tar')
    return target, digest


class CloudTest(unittest.TestCase):
    def test_staged_source_has_the_same_identity_as_this_checkout(self):
        # Releases bind the source identity; a staged copy (no .git) must compute the same value.
        for game in ('clash', 'pokemon'):
            with tempfile.TemporaryDirectory() as temp:
                repo, _ = staged(game, temp)
                self.assertFalse((repo / '.git').exists())
                code = ('import json,sys;from harness import policy;'
                        'print(json.dumps([policy.source_identity(sys.argv[1]),'
                        'policy.source_identity(sys.argv[1],code_only=True)]))')
                out = subprocess.run([sys.executable, '-c', code, game], cwd=repo, capture_output=True, text=True,
                                     env=dict(os.environ, PYTHONPATH=str(repo)), check=True).stdout
                self.assertEqual(out.strip(), '["%s", "%s"]' % (policy.source_identity(game),
                                                                policy.source_identity(game, code_only=True)))

    def test_source_files_cover_the_game_and_skip_research_archives(self):
        names = set(cloud.source_files('polytopia'))
        self.assertIn('harness/cloud.py', names)
        self.assertIn('simulators/catalog.json', names)
        self.assertIn('simulators/polytopia/autoresearch/requirements-host.txt', names)
        self.assertFalse(any(name.startswith(cloud.SOURCE_EXCLUDE) for name in names))
        pokemon = set(cloud.source_files('pokemon'))
        self.assertTrue(any(name.startswith('vendor/engines/pokemon-showdown/') for name in pokemon))
        self.assertFalse(any(name.startswith('vendor/engines/crforge/') for name in pokemon))
        self.assertIn('integrations/pokemon', pokemon)  # compatibility links travel as links

    def test_archive_is_deterministic(self):
        with tempfile.TemporaryDirectory() as one, tempfile.TemporaryDirectory() as two:
            self.assertEqual(cloud.source_archive('hex', one)[1], cloud.source_archive('hex', two)[1])

    def test_requirements_are_flattened_with_includes(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = cloud.requirement_files('pokemon', temp)
            self.assertIn('-r requirements.txt', (folder / 'host.txt').read_text())
            self.assertIn('torch==', (folder / 'requirements.txt').read_text())
            self.assertEqual((folder / 'pins.txt').read_text(), '')
            hex_folder = cloud.requirement_files('hex', temp)
            self.assertIn('modal[api-proxy-support]==1.5.2', (hex_folder / 'pins.txt').read_text())

    def test_rewrite_stages_releases_and_scopes_sessions(self):
        argv = ['run', 'pokemon', 'recipe', '--release', 'evidence/r/pokemon-2', '--out', 'pokemon/opus/main',
                '--model', 'openrouter/x', '--execute']
        out, sessions, releases = cloud.rewrite('pokemon', 'camp', argv, stage=lambda game, path: (
            '/work/releases/pokemon/abc/' + Path(path).name, '/releases/pokemon/abc/pokemon-2.tar'))
        self.assertEqual(sessions, ['pokemon/opus/main'])
        self.assertEqual(releases, ['/work/releases/pokemon/abc/pokemon-2'])
        self.assertEqual(out[out.index('--release') + 1], '/work/releases/pokemon/abc/pokemon-2')
        self.assertEqual(cloud.release_tar(releases[0]), '/releases/pokemon/abc/pokemon-2.tar')
        self.assertEqual(out[out.index('--out') + 1], '/work/camp/pokemon/opus/main')
        refuse = [argv + ['--key-file', 'k'], argv + ['--api-base', 'http://127.0.0.1:18081/v1'],
                  ['run', 'pokemon', 'recipe', '--out', '/abs'], ['run', 'pokemon', 'recipe', '--out', '../up'],
                  ['run', 'pokemon', 'recipe'], ['prepare', 'pokemon', '--out', 'x']]
        for bad in refuse:
            with self.assertRaises(SystemExit):
                cloud.rewrite('pokemon', 'camp', bad, stage=lambda game, path: (path, None))
        public = argv + ['--api-base', 'https://openrouter.ai/api/v1']
        self.assertIn('https://openrouter.ai/api/v1', cloud.rewrite('pokemon', 'camp', public, stage=lambda g, p: (p, None))[0])

    def test_a_continuation_seed_travels_as_a_campaign_input(self):
        with tempfile.TemporaryDirectory() as temp:
            seed = Path(temp) / 'astra6-a1'
            (seed / 'workspace/training').mkdir(parents=True)
            (seed / 'seed.json').write_text('{}')
            (seed / 'workspace/training/train.py').write_text('x')
            argv = ['run', 'polytopia', 'recipe', '--out', 'polytopia/astra6/c1', '--continue-from', str(seed),
                    '--model', 'openrouter/x', '--execute']
            sent = []
            staged, inputs = cloud.stage_inputs('camp', argv, lambda folder, remote, files: sent.append(
                (folder, remote, sorted(p.name for p in files))))
            self.assertEqual(len(inputs), 1)
            self.assertTrue(inputs[0].startswith('inputs/astra6-a1-'))
            self.assertEqual(sent, [(seed, 'camp/' + inputs[0], ['seed.json', 'train.py'])])
            self.assertEqual(cloud.stage_inputs('camp', argv)[1], inputs)  # named by content: stable
            out, sessions, _ = cloud.rewrite('polytopia', 'camp', staged, stage=lambda g, p: (p, None))
            self.assertEqual(sessions, ['polytopia/astra6/c1'])
            self.assertEqual(out[out.index('--continue-from') + 1], '/work/camp/' + inputs[0])
            (seed / 'workspace/training/train.py').write_text('y')
            self.assertNotEqual(cloud.stage_inputs('camp', argv)[1], inputs)
            with self.assertRaises(SystemExit):
                cloud.stage_inputs('camp', ['run', '--continue-from', str(seed / 'missing')])

    def test_app_names_are_valid_and_distinct(self):
        names = {cloud.app_name('two-game-3h-20261114', f'polytopia/{model}/main')
                 for model in ('opus-5.5', 'deepseek-v4-pro', 'kimi-k3.1-pro', 'glm-5.3')}
        self.assertEqual(len(names), 4)
        for name in names:
            self.assertLessEqual(len(name), 56)
            self.assertRegex(name, r'^[a-z0-9-]+$')

    def test_mirror_copies_new_and_changed_files_only(self):
        with tempfile.TemporaryDirectory() as source, tempfile.TemporaryDirectory() as target:
            (Path(source) / 'a/b').mkdir(parents=True)
            (Path(source) / 'a/b/x.json').write_text('1')
            (Path(source) / 'top.log').write_text('log')
            self.assertEqual(cloud.mirror(source, target), 2)
            self.assertEqual(cloud.mirror(source, target), 0)
            time.sleep(1.1)
            (Path(source) / 'a/b/x.json').write_text('22')
            self.assertEqual(cloud.mirror(source, target), 1)
            self.assertEqual((Path(target) / 'a/b/x.json').read_text(), '22')

    def test_release_travels_as_one_tar_and_verifies_after_unpacking(self):
        from harness.tests.support import release_fixture
        with tempfile.TemporaryDirectory() as temp:
            release, _ = release_fixture('hex', Path(temp) / 'here')
            archive = cloud.release_archive(release, Path(temp) / 'release.tar')
            again = cloud.release_archive(release, Path(temp) / 'again.tar')
            self.assertEqual(archive.read_bytes(), again.read_bytes())
            there = Path(temp) / 'there'
            there.mkdir()
            with tarfile.open(archive) as tar:
                tar.extractall(there, filter='tar')
            self.assertEqual(policy.verify_release('hex', there / 'hex')['sha256'],
                             policy.verify_release('hex', release)['sha256'])

    def test_restore_brings_a_recorded_session_back(self):
        with tempfile.TemporaryDirectory() as store, tempfile.TemporaryDirectory() as work:
            session = Path(store) / 'camp/pokemon/opus/main'
            (session / 'final').mkdir(parents=True)
            (session / 'session.json').write_text('{}')
            session.with_name('main.harness-run.json').write_text('{"settings": {"release": "/work/releases/p/k/r"}}')
            self.assertEqual(cloud.restore('camp', 'pokemon/opus/main', store, work), '/work/releases/p/k/r')
            self.assertTrue((Path(work) / 'camp/pokemon/opus/main/session.json').is_file())
            self.assertTrue((Path(work) / 'camp/pokemon/opus/main.harness-run.json').is_file())
            with self.assertRaises(FileNotFoundError):
                cloud.restore('camp', 'pokemon/kimi/main', store, work)

    def test_pull_keeps_one_sessions_records(self):
        out = 'pokemon/opus/main'
        self.assertTrue(cloud.wanted('pokemon/opus/main/session.json', out))
        self.assertTrue(cloud.wanted('pokemon/opus/main.harness-run.json', out))
        self.assertTrue(cloud.wanted('pokemon__opus__main.coordinator.log', out))
        self.assertFalse(cloud.wanted('pokemon/opus/main-r2/session.json', out))
        self.assertFalse(cloud.wanted('pokemon/kimi/main/session.json', out))
        self.assertTrue(cloud.wanted('anything', None))



class TerminateBoxesTest(unittest.TestCase):
    """A coordinator that ended without closing its session must not leave the researcher box running."""

    class Box:
        def __init__(self, name):
            self.object_id, self.ended = name, False

        def terminate(self):
            self.ended = True

    def test_terminates_every_box_of_the_session(self):
        boxes = [self.Box('sb-1'), self.Box('sb-2')]
        ended = cloud.terminate_boxes(lambda: dict(box_tag='a1-abc'), 'clash/m/a1', sandboxes=boxes)
        self.assertEqual(ended, ['sb-1', 'sb-2'])
        self.assertTrue(all(box.ended for box in boxes))

    def test_a_session_without_a_plan_or_tag_touches_nothing(self):
        box = self.Box('sb-1')
        self.assertEqual(cloud.terminate_boxes(lambda: None, 's', sandboxes=[box]), [])
        self.assertEqual(cloud.terminate_boxes(lambda: dict(), 's', sandboxes=[box]), [])
        self.assertFalse(box.ended)

if __name__ == '__main__':
    unittest.main()


class SqliteSidecars(unittest.TestCase):
    def test_mirror_removes_a_wal_gone_from_the_source_and_restore_drops_a_stale_one(self):
        import os, sqlite3
        with tempfile.TemporaryDirectory() as tmp:
            source, target = Path(tmp) / 'work', Path(tmp) / 'records'
            source.mkdir()
            db = source / 'ledger.sqlite3'
            connection = sqlite3.connect(db)
            connection.execute('PRAGMA journal_mode=WAL')
            connection.execute('CREATE TABLE t(x)')
            connection.execute("INSERT INTO t VALUES ('open')")
            connection.commit()
            cloud.mirror(source, target)
            self.assertTrue((target / 'ledger.sqlite3-wal').is_file())
            connection.execute("UPDATE t SET x='final'")
            connection.commit()
            connection.close()          # checkpoints into the database and removes the WAL
            self.assertFalse((source / 'ledger.sqlite3-wal').exists())
            cloud.mirror(source, target)
            self.assertFalse((target / 'ledger.sqlite3-wal').exists())
            self.assertEqual(sqlite3.connect(target / 'ledger.sqlite3').execute('SELECT x FROM t').fetchone()[0], 'final')
            # A stale sidecar left by an older mirror is older than its database: found and dropped.
            stale = target / 'ledger.sqlite3-wal'
            stale.write_bytes(b'old')
            os.utime(stale, (1, 1))
            self.assertEqual(cloud.stale_sidecars(target), [stale])
