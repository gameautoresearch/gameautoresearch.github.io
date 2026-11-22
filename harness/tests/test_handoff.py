"""The shared worker-to-host hand-off: the in-worker publish script and the host's verified collection."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from harness import handoff


class FolderVolume:
    """A local directory standing in for a Modal Volume (read_file yields chunks; missing files raise)."""

    def __init__(self, root):
        self.root = Path(root)

    def read_file(self, path):
        target = self.root / path
        if not target.is_file():
            raise FileNotFoundError(path)
        data = target.read_bytes()
        for start in range(0, len(data), 7):
            yield data[start:start + 7]


class FakeModal:
    def __init__(self, root):
        self.root = root
        self.created = []
        outer = self

        class Volume:
            @staticmethod
            def from_name(name, environment_name=None, **options):
                outer.created.append((name, environment_name, options))
                return FolderVolume(outer.root)

        self.Volume = Volume


def publish(mount, job, files):
    """Run the real in-worker script against a local folder (no `sync` on this host)."""
    code = handoff.PUBLISH.replace(repr(handoff.MOUNT), repr(str(mount)), 1)
    code = code.replace("subprocess.run(['sync',mount],check=True,timeout=120)", 'pass')
    return subprocess.run([sys.executable, '-c', code, job, *(f'{k}={v}' for k, v in files.items())],
                          capture_output=True, text=True)


class HandoffTest(unittest.TestCase):
    def setUp(self):
        self.temp = Path(tempfile.mkdtemp())
        self.mount = self.temp / 'mount'
        (self.temp / 'bundle.tar').write_bytes(b'network' * 1000)
        (self.temp / 'log.txt').write_text('trained\n')

    def session(self):
        return handoff.SessionVolume('polytopia', self.temp / 'session/research', modal_module=FakeModal(self.mount))

    def test_publish_then_collect_round_trip(self):
        done = publish(self.mount, 'run1', {'bundle.tar': self.temp / 'bundle.tar', 'log.txt': self.temp / 'log.txt'})
        self.assertEqual(done.returncode, 0, done.stderr)
        manifest = json.loads(done.stdout.strip().splitlines()[-1])
        self.assertEqual(set(manifest), {'bundle.tar', 'log.txt'})
        self.assertEqual(oct((self.mount / 'run1').stat().st_mode & 0o777), oct(0o700))
        got = self.session().collect('run1', self.temp / 'out', wait=0)
        self.assertEqual((self.temp / 'out/bundle.tar').read_bytes(), b'network' * 1000)
        self.assertEqual(got['log.txt']['sha256'], handoff.sha256(self.temp / 'log.txt'))

    def test_unpublished_job_is_retried_then_reported(self):
        with self.assertRaises(handoff.NotPublished):
            self.session().collect('missing', self.temp / 'out', attempts=2, wait=0)

    def test_corrupted_file_is_refused(self):
        publish(self.mount, 'run2', {'bundle.tar': self.temp / 'bundle.tar'})
        (self.mount / 'run2/files/bundle.tar').write_bytes(b'tampered')
        with self.assertRaises(ValueError):
            self.session().collect('run2', self.temp / 'out', wait=0)
        self.assertFalse((self.temp / 'out/bundle.tar').exists())

    def test_names_and_binding(self):
        with self.assertRaises(ValueError):
            handoff.publish_command('../x', {'a': '/b'})
        with self.assertRaises(ValueError):
            handoff.publish_command('job', {'MANIFEST.json': '/b'})
        refused = publish(self.mount, 'run3', {'../escape': self.temp / 'log.txt'})
        self.assertNotEqual(refused.returncode, 0)
        first = self.session()
        second = self.session()
        self.assertEqual(first.name, second.name)
        self.assertTrue(first.name.startswith('polytopia-handoff-session-research-'))
        other = handoff.SessionVolume('polytopia', self.temp / 'session/final', modal_module=FakeModal(self.mount))
        self.assertNotEqual(first.name, other.name)
        # The same path re-created from scratch gets a new Volume.
        (self.temp / 'session/research' / handoff.BINDING).unlink()
        again = self.session()
        self.assertNotEqual(first.name, again.name)
        self.assertLessEqual(len(again.name), 64)
        with self.assertRaises(ValueError):
            handoff.SessionVolume('hex', self.temp / 'session/research')

    def test_mounts_create_a_v2_volume_in_the_project_environment(self):
        fake = FakeModal(self.mount)
        session = handoff.SessionVolume('hex', self.temp / 's/research', modal_module=fake)
        self.assertEqual(list(session.mounts()), [handoff.MOUNT])
        name, environment, options = fake.created[-1]
        self.assertEqual((environment, options), ('game-autoresearch-project', dict(version=2, create_if_missing=True)))


if __name__ == '__main__':
    unittest.main()
