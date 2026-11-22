"""Pause and resume: a paused session keeps everything and continues from that point.

The operator's `pause` stops the researcher and its workers without finalizing; `resume` reopens the same
ledger, budget, submissions and workspace in a new box, with the wall clock that was left."""
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest

from harness import policy, registry, session, task, workspace
from harness.tests.support import FakeBackend, Process, Researcher, release_fixture
from harness.transport import LocalBox

EDIT = '# an edit made before the pause\n'


class PauseAndResume(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.release, self.meta = release_fixture('doudizhu', self.root / 'release')
        self._backend, self._modal = registry.backend, policy.require_modal
        registry.backend = lambda game: FakeBackend
        policy.require_modal = lambda: None

    def tearDown(self):
        registry.backend, policy.require_modal = self._backend, self._modal
        self.temp.cleanup()

    def start(self, out, box, script, **kwargs):
        """Run a session on a thread; `script(researcher)` plays the researcher. Returns (thread, result holder)."""
        holder = {}

        def launcher(folder):
            return Process(lambda: script(Researcher(folder / 'task', box)))

        def work():
            try:
                holder['result'] = session.run('doudizhu', 'network', self.release, out, model='test/model',
                                               smoke=True, launcher=launcher,
                                               find_box=lambda: LocalBox(box / 'ar') if (box / 'ar').is_dir() else None,
                                               **kwargs)
            except BaseException as exc:
                holder['error'] = exc
        thread = threading.Thread(target=work, daemon=True)
        thread.start()
        return thread, holder

    def wait_for(self, check, what, seconds=60):
        deadline = time.time() + seconds
        while not check():
            self.assertLess(time.time(), deadline, 'timed out waiting for ' + what)
            time.sleep(0.05)

    def test_a_paused_session_keeps_its_state_and_resumes_from_that_point(self):
        out, box, release_hold = self.root / 'sessions' / 'one', self.root / 'box1', threading.Event()
        seen = {}

        def before(ar):
            seen['run'] = ar('run', 'first', '--seconds', '120')
            ar('wait', 'first')
            seen['submit'] = ar('submit', 'first')
            name = next(item for item in self.meta['writable'] if item.endswith('.py'))
            ar.edit(name, (ar.workspace / name).read_text() + EDIT)
            ar.edit(workspace.NOTES, 'first run submitted; next: a longer one\n')
            seen['ready'] = True
            release_hold.wait(60)  # still "thinking" when the operator pauses

        thread, holder = self.start(out, box, before, wall_seconds=3000)
        self.wait_for(lambda: seen.get('ready'), 'the researcher')
        with self.assertRaisesRegex(ValueError, 'not paused'):
            session.resume(out, launcher=lambda *a: None)
        asked = session.request_pause(out, note='look at the checkpoints')
        thread.join(timeout=60)
        release_hold.set()
        self.assertNotIn('error', holder, holder.get('error'))
        result = holder['result']
        # Paused: recorded, nothing finalized, nothing reported as a finished session.
        self.assertEqual(result['status'], 'paused')
        state = json.loads((out / 'session.json').read_text())
        self.assertFalse(state['finalized'])
        self.assertEqual((state['paused'], len(state['submissions'])), (1, 1))
        self.assertFalse((out / 'session-result.json').exists() or (out / 'completion.json').exists())
        record = json.loads((out / session.PAUSE).read_text())
        self.assertEqual((record['state'], record['index'], record['request']['id']),
                         ('paused', 1, asked['requested']['id']))
        self.assertEqual(record['request']['note'], 'look at the checkpoints')
        self.assertTrue(2000 < record['wall_remaining_seconds'] <= 3000)
        self.assertEqual(record['budget']['used_seconds'], 120)
        # The workspace as it was in the box, edit and notes included.
        name = next(item for item in self.meta['writable'] if item.endswith('.py'))
        seed = out / record['seed']
        self.assertTrue((seed / task.SEED_FILES / name).read_text().endswith(EDIT))
        self.assertEqual((seed / task.SEED_FILES / workspace.NOTES).read_text(), 'first run submitted; next: a longer one\n')
        self.assertEqual(record['workspace_sources'][name], 'box')
        with self.assertRaisesRegex(ValueError, 'already paused'):
            session.request_pause(out)

        # Resume: a new box holding that workspace, the same ledger and submissions, the clock that was left.
        box2, after = self.root / 'box2', {}

        def later(ar):
            after['instructions'] = (ar.workspace / 'instructions.md').read_text()
            after['file'] = (ar.workspace / name).read_text()
            after['notes'] = (ar.workspace / workspace.NOTES).read_text()
            after['status'] = ar('status')
            after['budget'] = ar('budget')
            ar('run', 'second', '--seconds', '120')
            ar('wait', 'second')
            after['submit'] = ar('submit', 'second')
            after['finalize'] = ar('finalize')

        resumed = session.resume(out, launcher=lambda folder, task_dir: Process(lambda: later(Researcher(task_dir, box2))),
                                 find_box=lambda: LocalBox(box2 / 'ar') if (box2 / 'ar').is_dir() else None)
        self.assertTrue(after['instructions'].startswith('This research session was paused by its operator'))
        self.assertTrue(after['file'].endswith(EDIT))
        self.assertEqual(after['notes'], 'first run submitted; next: a longer one\n')
        self.assertEqual([job['name'] for job in after['status']['jobs']], ['first'])
        self.assertEqual(after['budget']['used_seconds'], 120)
        self.assertNotIn('error', after['finalize'])
        self.assertEqual(resumed['status'], 'researched')
        self.assertEqual((resumed['pauses'], resumed['budget']['used_seconds']), (1, 240))
        state = json.loads((out / 'session.json').read_text())
        self.assertTrue(state['finalized'])
        self.assertEqual([item['name'] for item in state['submissions']], ['first', 'second'])
        self.assertIsNone(state['paused'])
        self.assertTrue(time.time() + 1900 < state['deadline'] < time.time() + 3000)
        self.assertEqual(json.loads((out / session.PAUSE).read_text())['state'], 'resumed')
        self.assertTrue((out / 'session-result.json').is_file() and (out / 'task-resume-001/task.toml').is_file())
        with self.assertRaisesRegex(ValueError, 'not paused'):
            session.resume(out, launcher=lambda *a: None)
        with self.assertRaisesRegex(ValueError, 'unfinalized'):
            session.request_pause(out)

    def test_a_session_can_be_paused_again_and_a_stale_request_does_not_pause_the_resumed_session(self):
        out, box, hold = self.root / 'sessions' / 'two', self.root / 'boxa', threading.Event()
        ready = {}

        def idle(ar):
            ready['one'] = True
            hold.wait(60)
        thread, holder = self.start(out, box, idle, wall_seconds=3000)
        self.wait_for(lambda: ready.get('one'), 'the researcher')
        session.request_pause(out)
        thread.join(timeout=60)
        hold.set()
        self.assertEqual(holder['result']['status'], 'paused')
        # No box was read here before the researcher sent a file: the host's copies are the starter's.
        first = json.loads((out / session.PAUSE).read_text())
        # A request written while the session is paused (as the cloud path can leave one) asks for nothing new.
        session.write_json(out / session.PAUSE_REQUEST, dict(id='stale', requested_at=time.time(), note=None))
        box2, hold2, ready2, holder2 = self.root / 'boxb', threading.Event(), {}, {}

        def idle2(ar):
            ready2['one'] = True
            hold2.wait(60)

        def again():
            holder2['result'] = session.resume(
                out, launcher=lambda folder, task_dir: Process(lambda: idle2(Researcher(task_dir, box2))),
                find_box=lambda: LocalBox(box2 / 'ar') if (box2 / 'ar').is_dir() else None)
        second = threading.Thread(target=again, daemon=True)
        second.start()
        self.wait_for(lambda: ready2.get('one'), 'the resumed researcher')
        time.sleep(2.5)  # the stale request was not acted on
        self.assertTrue(second.is_alive())
        self.assertEqual(json.loads((out / session.PAUSE).read_text())['state'], 'resumed')
        session.request_pause(out, note='again')
        second.join(timeout=60)
        hold2.set()
        self.assertEqual(holder2['result']['status'], 'paused')
        record = json.loads((out / session.PAUSE).read_text())
        self.assertEqual((record['index'], record['state'], record['request']['note']), (2, 'paused', 'again'))
        self.assertLess(record['wall_remaining_seconds'], first['wall_remaining_seconds'])
        self.assertEqual(sorted(p.name for p in (out / 'pauses').iterdir()), ['001', '002'])

    def test_the_cancellation_of_a_job_stopped_at_a_pause_is_not_a_fault_of_the_resumed_session(self):
        """Live, 2026-11-19: Harbor records a CancelledError for the job the host stopped at the pause; the session
        that then finished normally was reported as failed because of it."""
        out = self.root / 'sessions' / 'three'

        def job(name, exceptions, tokens, result=True):
            folder = out / 'harbor' / name
            (folder / 'trial' / 'agent').mkdir(parents=True)
            if result:
                (folder / 'result.json').write_text(json.dumps({'stats': dict(
                    n_input_tokens=tokens, n_cache_tokens=0, n_output_tokens=1, cost_usd=0.5,
                    evals={'e': {'exception_stats': {kind: ['trial'] for kind in exceptions}}})}))
            else:  # stopped before Harbor wrote its result: the usage is read from the trajectory it kept
                (folder / 'trial' / 'agent' / 'trajectory.json').write_text(json.dumps({'final_metrics': dict(
                    total_prompt_tokens=tokens, total_cached_tokens=0, total_completion_tokens=1, total_cost_usd=0.5)}))
            time.sleep(0.02)
        job('three', ['CancelledError'], 100)
        job('three-resume001', [], 10, result=False)
        job('three-resume002', ['AgentTimeoutError'], 1000)
        totals = session.harbor_totals(out, current='three-resume002')
        self.assertEqual(totals['exceptions'], {'AgentTimeoutError': 1})
        self.assertEqual(totals['paused_exceptions'], {'CancelledError': 1})
        self.assertEqual((totals['usage']['n_input_tokens'], totals['usage']['cost_usd']), (1110, 1.5))
        self.assertEqual(session.harbor_totals(out)['exceptions'], {'CancelledError': 1, 'AgentTimeoutError': 1})


if __name__ == '__main__':
    unittest.main()
