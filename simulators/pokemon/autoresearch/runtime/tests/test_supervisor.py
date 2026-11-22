"""Real subprocess fault fixtures for the trusted trainer supervisor; no games, cloud or GPU.

The last class runs the real starter under the real supervisor and sealer on CPU,
against the dry run's stand-in for the environment service.
"""
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import shutil
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from simulators.pokemon.autoresearch.runtime import sdk as old_sdk, supervisor as SUP


CONFIG=dict(seed=1,training_seed_start=1,environments=1,rollout_steps=1,minibatch=1,ppo_epochs=1)
SCRIPT='''import argparse,json,os,signal,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--out');p.add_argument('--seconds',type=float)
a,_=p.parse_known_args();out=Path(a.out);out.mkdir()
(out/'checkpoint.pt').write_bytes(b'complete-checkpoint')
(out/'checkpoint.tmp').write_bytes(b'incomplete-write')
(out/'progress.jsonl').write_text('{"samples":1}\\n')
(out/'observed.json').write_text(json.dumps(dict(seconds=a.seconds,
    deadline=float(os.environ['POKEMON_TRAINING_DEADLINE_MONOTONIC']),
    hard=float(os.environ['POKEMON_TRAINING_HARD_DEADLINE_MONOTONIC']))))
MODE
'''

@contextmanager
def service(*args):yield 'fixture-socket'

class TrainingRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.source=self.root/'candidate';self.source.mkdir()
        for name in old_sdk.EDITABLE:(self.source/name).write_text('# fixture\n')
        (self.source/'config.json').write_text(json.dumps(CONFIG))
        self.out=self.root/'stage'

    def run_trainer(self,body,seconds=.4,**options):
        (self.source/'train.py').write_text(SCRIPT.replace('MODE',body))
        return SUP.train(self.source,CONFIG,self.out,seconds,device='cpu',development=True,**options)

    def receipt(self):
        return json.loads((self.out/'supervisor.json').read_text())

    def test_timeout_preserves_checkpoint_before_temp_cleanup_and_never_seals(self):
        with patch.object(SUP,'native_service',service),patch.object(SUP,'seal')as seal:
            with self.assertRaises(TimeoutError):self.run_trainer('time.sleep(30)')
        seal.assert_not_called()
        receipt=json.loads((self.out/'supervisor.json').read_text())
        self.assertEqual(receipt['termination_reason'],'hard_deadline')
        self.assertEqual(receipt['returncode'],-signal.SIGKILL)
        self.assertFalse(receipt['successful'])
        # No result.json or graph.json: nothing to package, so the stage fails as it always did.
        self.assertEqual((receipt['phase'],receipt['error_type'],receipt['early_packaging_error_type']),('running','HardDeadline','ValueError'))
        self.assertNotIn('cut_short',receipt)
        recovery=json.loads((self.out/'recovery/manifest.json').read_text())
        self.assertFalse(recovery['eligible_for_submission'])
        self.assertTrue(recovery['checkpoint_retained'])
        self.assertEqual((self.out/'recovery/checkpoint.pt').read_bytes(),b'complete-checkpoint')
        self.assertEqual(recovery['files']['checkpoint.pt']['sha256'],hashlib.sha256(b'complete-checkpoint').hexdigest())
        self.assertNotIn('checkpoint.tmp',recovery['files'])
        self.assertFalse((self.out/'bundle').exists())
        self.assertFalse((self.out/'completed.json').exists())
        self.assertEqual((self.out/'recovery/source/train.py').read_bytes(),(self.source/'train.py').read_bytes())

    COMPLETE="(out/'result.json').write_text('{}');(out/'graph.json').write_text('{}')\n"

    def test_trainer_killed_at_its_hard_deadline_with_a_complete_set_is_sealed_as_cut_short(self):
        with patch.object(SUP,'native_service',service),patch.object(SUP,'seal',return_value={'bundle_sha256':'fixture'})as seal:
            returned=self.run_trainer(self.COMPLETE+'time.sleep(30)')
        receipt=self.receipt()
        self.assertEqual((receipt['successful'],receipt['cut_short'],receipt['termination_reason'],receipt['phase'],receipt['returncode']),
            (True,True,'hard_deadline','complete',-signal.SIGKILL))
        self.assertEqual((returned['cut_short'],receipt['bundle_sha256']),(True,'fixture'))
        self.assertNotIn('stopped_early',receipt);self.assertNotIn('error_type',receipt)
        # Sealed like a finished run: from the trusted copies, with the flag already in the binding it is given.
        (output,_,_,checkpoint,binding),_=seal.call_args
        self.assertEqual((output,checkpoint,binding['cut_short']),(self.out/'bundle',self.out/'checkpoint.pt',True))
        self.assertEqual((self.out/'checkpoint.pt').read_bytes(),b'complete-checkpoint')
        self.assertTrue(all((self.out/name).is_file()for name in SUP.STAGE_FILES))
        self.assertFalse((self.out/'recovery').exists())
        self.assertFalse((self.out/'checkpoint.tmp').exists())

    def test_cut_off_trainer_whose_set_does_not_seal_fails_as_before(self):
        def seal(output,*args):
            output.mkdir();(output/'manifest.json').write_text('incomplete')
            raise ValueError('graph does not match the checkpoint')
        with patch.object(SUP,'native_service',service),patch.object(SUP,'seal',side_effect=seal):
            with self.assertRaises(SUP.HardDeadline):self.run_trainer(self.COMPLETE+'time.sleep(30)')
        receipt=self.receipt()
        self.assertEqual((receipt['successful'],receipt['termination_reason'],receipt['phase'],receipt['early_packaging_error_type']),
            (False,'hard_deadline','running','ValueError'))
        self.assertNotIn('cut_short',receipt)
        self.assertFalse((self.out/'bundle').exists())
        self.assertFalse(any((self.out/name).exists()for name in SUP.STAGE_FILES))
        recovery=json.loads((self.out/'recovery/manifest.json').read_text())
        self.assertEqual((recovery['eligible_for_submission'],recovery['checkpoint_retained']),(False,True))

    def stop_after(self,delay):
        request=self.root/'stop-request'
        timer=threading.Timer(delay,request.write_text,args=('stop\n',));timer.start();self.addCleanup(timer.cancel)
        return request

    def test_stop_request_ends_the_trainer_and_seals_what_it_wrote_as_stopped_early(self):
        started=time.monotonic()
        with patch.object(SUP,'native_service',service),patch.object(SUP,'seal',return_value={'bundle_sha256':'fixture'})as seal:
            self.run_trainer(self.COMPLETE+'time.sleep(60)',seconds=60,stop_request=self.stop_after(.3))
        self.assertLess(time.monotonic()-started,20)  # ended on the request, not at the 60-second deadline
        receipt=self.receipt()
        self.assertEqual((receipt['successful'],receipt['stopped_early'],receipt['termination_reason'],receipt['phase'],receipt['returncode']),
            (True,True,'stop_requested','complete',-signal.SIGKILL))
        self.assertNotIn('cut_short',receipt)
        seal.assert_called_once()
        self.assertFalse((self.out/'recovery').exists())

    def test_stop_request_before_a_complete_set_fails_and_never_seals(self):
        with patch.object(SUP,'native_service',service),patch.object(SUP,'seal')as seal:
            with self.assertRaises(SUP.StopRequested):self.run_trainer('time.sleep(60)',seconds=60,stop_request=self.stop_after(.3))
        seal.assert_not_called()
        receipt=self.receipt()
        self.assertEqual((receipt['successful'],receipt['termination_reason'],receipt['error_type']),(False,'stop_requested','StopRequested'))
        self.assertNotIn('stopped_early',receipt)
        self.assertTrue(receipt['recovery']['checkpoint_retained']);self.assertFalse(receipt['recovery']['eligible_for_submission'])
        self.assertFalse((self.out/'bundle').exists())

    def test_no_stop_request_leaves_a_finished_run_unflagged(self):
        with patch.object(SUP,'native_service',service),patch.object(SUP,'seal',return_value={'bundle_sha256':'fixture'}):
            self.run_trainer(self.COMPLETE,stop_request=self.root/'never-created')
        receipt=self.receipt()
        self.assertEqual((receipt['successful'],receipt['termination_reason']),(True,'completed'))
        self.assertFalse({'cut_short','stopped_early'}&set(receipt))

    def test_trainer_that_fails_by_itself_with_a_complete_set_is_still_a_failure(self):
        with patch.object(SUP,'native_service',service),patch.object(SUP,'seal')as seal:
            with self.assertRaises(RuntimeError):self.run_trainer(self.COMPLETE+'raise SystemExit(7)',stop_request=self.root/'never-created')
        seal.assert_not_called()
        receipt=self.receipt()
        self.assertEqual((receipt['successful'],receipt['termination_reason'],receipt['returncode']),(False,'trainer_error',7))
        self.assertFalse({'cut_short','stopped_early'}&set(receipt))

    def test_external_kill_is_distinct_from_supervisor_timeout(self):
        with patch.object(SUP,'native_service',service):
            with self.assertRaises(RuntimeError):self.run_trainer('os.kill(os.getpid(),signal.SIGKILL)')
        self.assertEqual(json.loads((self.out/'supervisor.json').read_text())['termination_reason'],'signal_9')
        self.assertTrue((self.out/'recovery/checkpoint.pt').exists())

    def test_error_exit_keeps_failure_and_diagnostics(self):
        with patch.object(SUP,'native_service',service),patch.object(SUP,'seal')as seal:
            with self.assertRaises(RuntimeError):self.run_trainer('raise SystemExit(7)')
        seal.assert_not_called()
        receipt=json.loads((self.out/'supervisor.json').read_text())
        self.assertEqual(receipt['returncode'],7)
        self.assertEqual(receipt['termination_reason'],'trainer_error')
        self.assertEqual(receipt['recovery']['source_sha256'],old_sdk.source_signature(self.source))

    def test_initialization_and_export_fit_reserved_budget(self):
        body="""time.sleep(.03)
# Deliberately use the old relative-clock convention after initialization.
time.sleep(a.seconds)
time.sleep(.025)
(out/'result.json').write_text('{}');(out/'graph.json').write_text('{}')
# Record only the public timing diagnostics needed by the assertion.
(out/'result.json').write_text((out/'observed.json').read_text())
"""
        with patch.object(SUP,'native_service',service),patch.object(SUP,'seal',return_value={'bundle_sha256':'fixture'})as seal:
            self.run_trainer(body,seconds=.8)
        seal.assert_called_once()
        receipt=json.loads((self.out/'supervisor.json').read_text())
        observed=json.loads((self.out/'result.json').read_text())
        self.assertTrue(receipt['successful'])
        self.assertLess(observed['seconds'],.61)
        self.assertAlmostEqual(observed['hard']-observed['deadline'],.2)
        self.assertLess(receipt['process_wall_seconds'],.8)
        self.assertFalse((self.out/'recovery').exists())

    def test_monotonic_budget_subtracts_setup_and_does_not_extend_deadline(self):
        train,hard=SUP.child_budget(1000,180,now=100)
        self.assertEqual((train,hard),(720,900))
        with self.assertRaises(TimeoutError):SUP.child_budget(1000,180,now=821)

    def test_setup_exhaustion_records_failure_without_starting_child(self):
        with patch.object(SUP,'native_service',service),patch.object(SUP,'child_budget',side_effect=TimeoutError('setup')),\
             patch.object(SUP.subprocess,'Popen')as popen:
            with self.assertRaises(TimeoutError):self.run_trainer('pass')
        popen.assert_not_called()
        receipt=json.loads((self.out/'supervisor.json').read_text())
        self.assertFalse(receipt['successful']);self.assertIsNone(receipt['returncode'])

    @unittest.skipUnless(hasattr(signal,'setitimer'),'POSIX worker signal')
    def test_absolute_worker_alarm_still_retains_partial_outputs(self):
        def alarm(*_):raise TimeoutError('outer worker alarm fixture')
        old=signal.signal(signal.SIGALRM,alarm)
        try:
            signal.setitimer(signal.ITIMER_REAL,.2)
            with patch.object(SUP,'native_service',service):
                with self.assertRaises(TimeoutError):self.run_trainer('time.sleep(30)',seconds=5)
        finally:
            signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old)
        receipt=json.loads((self.out/'supervisor.json').read_text())
        self.assertEqual(receipt['termination_reason'],'supervisor_interrupted_TimeoutError')
        self.assertTrue(receipt['recovery']['checkpoint_retained'])

    def retain_fixture(self,make):
        output=self.root/'output';output.mkdir();train=output/'train';train.mkdir()
        make(train)
        self.out.mkdir()
        return SUP.retain_failed_outputs(output,self.source,self.out,CONFIG)

    def test_recovery_rejects_symlink_artifacts(self):
        secret=self.root/'private';secret.write_text('must not be copied')
        record=self.retain_fixture(lambda train:(train/'checkpoint.pt').symlink_to(secret))
        self.assertFalse(record['checkpoint_retained']);self.assertIn('checkpoint.pt',record['errors'])

    def test_recovery_rejects_linked_parent_directory(self):
        output=self.root/'output';output.mkdir();(output/'train').symlink_to(self.source,target_is_directory=True);self.out.mkdir()
        record=SUP.retain_failed_outputs(output,self.source,self.out,CONFIG)
        self.assertFalse(record['checkpoint_retained']);self.assertIn('train_directory',record['errors'])

    def test_recovery_rejects_hardlinks(self):
        other=self.root/'other';other.write_bytes(b'not owned artifact')
        record=self.retain_fixture(lambda train:os.link(other,train/'checkpoint.pt'))
        self.assertFalse(record['checkpoint_retained'])

    def test_recovery_bounds_size_and_ignores_partial_temp_checkpoint(self):
        def create(train):
            (train/'checkpoint.pt').write_bytes(b'123456789')
            (train/'checkpoint.tmp').write_bytes(b'partial')
        with patch.dict(SUP.RECOVERY_LIMITS,{'checkpoint.pt':8}):record=self.retain_fixture(create)
        self.assertFalse(record['checkpoint_retained']);self.assertNotIn('checkpoint.tmp',record['files'])

    @unittest.skipUnless(hasattr(os,'mkfifo'),'POSIX worker FIFO')
    def test_recovery_does_not_block_on_fifo(self):
        record=self.retain_fixture(lambda train:os.mkfifo(train/'checkpoint.pt'))
        self.assertFalse(record['checkpoint_retained'])

    def test_failed_sealing_removes_partial_bundle_and_keeps_checkpoint(self):
        def seal(output,*args):
            output.mkdir();(output/'manifest.json').write_text('incomplete')
            raise ValueError('invalid graph')
        body="(out/'result.json').write_text('{}');(out/'graph.json').write_text('{}')"
        with patch.object(SUP,'native_service',service),patch.object(SUP,'seal',side_effect=seal):
            with self.assertRaises(ValueError):self.run_trainer(body)
        self.assertFalse((self.out/'bundle').exists());self.assertTrue((self.out/'recovery/checkpoint.pt').exists())
        self.assertFalse((self.out/'checkpoint.pt').exists())
        self.assertEqual(json.loads((self.out/'supervisor.json').read_text())['phase'],'sealing')

    def test_invalid_time_is_rejected_before_output_creation(self):
        for seconds in [float('nan'),float('inf'),True,0,-1,3601]:
            with self.assertRaises(ValueError):SUP.train(self.source,CONFIG,self.out,seconds,development=True)
        self.assertFalse(self.out.exists())


ROOT=Path(__file__).resolve().parents[5]
ENDLESS='time.monotonic()<stop and '


class RealStarter(unittest.TestCase):
    """The starter itself, ended by the supervisor, is sealed by the real sealer from what it had written."""

    def setUp(self):
        import torch  # noqa: F401  (the sealer and the trainer need it)
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.source=self.root/'candidate';self.source.mkdir();self.out=self.root/'stage'
        spec=importlib.util.spec_from_file_location('pokemon_dry_run',ROOT/'simulators/pokemon/autoresearch/task/dryrun.py')
        dry=importlib.util.module_from_spec(spec);spec.loader.exec_module(dry)
        for name in('policy.py','learner.py'):shutil.copyfile(old_sdk.STARTER/name,self.source/name)
        starter=(old_sdk.STARTER/'train.py').read_text()
        self.assertEqual(starter.count(ENDLESS),1)
        # A researcher's trainer that never stops by itself, on random contract-shaped observations instead of the engine.
        name='integrations.pokemon.autoresearch.runtime.env_client'
        (self.source/'train.py').write_text('import sys,types\n_stand_in=types.ModuleType(%r)\nexec(compile(%r,%r,"exec"),_stand_in.__dict__)\n'
            'sys.modules[%r]=_stand_in\n'%(name,dry.ENV_CLIENT,name,name)+starter.replace(ENDLESS,''))
        self.config=dict(json.loads((old_sdk.HERE.parent/'recipes/smoke.json').read_bytes()),environments=8,rollout_steps=8)

    def rows(self):
        log=self.out/'trainer.log'
        return [line for line in log.read_text().splitlines()if line.startswith('{')]if log.is_file()else[]

    def sealed(self,receipt,flag):
        import torch
        from simulators.pokemon.autoresearch.runtime import bundle,graph_export
        saved=json.loads((self.out/'supervisor.json').read_text())
        self.assertEqual((saved['successful'],saved[flag],saved['phase'],saved['returncode']),(True,True,'complete',-signal.SIGKILL),saved)
        self.assertEqual(receipt['bundle_sha256'],saved['bundle_sha256'])
        manifest=bundle.verify(self.out/'bundle',saved['bundle_sha256'])
        self.assertEqual(set(manifest['files']),set(bundle.FILES))
        binding=json.loads((self.out/'bundle/binding.json').read_bytes())
        checkpoint=torch.load(self.out/'checkpoint.pt',weights_only=True)
        self.assertEqual((binding[flag],binding['iteration'],binding['source_sha256']),(True,checkpoint['iteration'],old_sdk.source_signature(self.source)))
        self.assertGreaterEqual(checkpoint['iteration'],1)
        # The sealed weights are the last checkpoint's, and the graph written before training runs on them.
        weights=torch.load(self.out/'bundle/weights.pt',weights_only=True)
        self.assertTrue(all(torch.equal(value,checkpoint['model'][name])for name,value in weights.items()))
        logits,_=graph_export.run(graph_export.load_graph(self.out/'bundle/graph.json'),weights,graph_export.example_observations())
        self.assertEqual(tuple(logits.shape),(1,14))
        result=json.loads((self.out/'result.json').read_bytes())
        self.assertIn(checkpoint['iteration']-result['iterations'],(0,1))  # result.json may trail by the interrupted update
        self.assertGreaterEqual(len((self.out/'progress.jsonl').read_text().splitlines()),checkpoint['iteration'])
        self.assertFalse((self.out/'recovery').exists())
        return saved

    def test_starter_cut_off_at_its_hard_deadline_is_sealed_from_its_last_checkpoint(self):
        with patch.object(SUP,'native_service',service):
            receipt=SUP.train(self.source,self.config,self.out,20,device='cpu',development=True)
        saved=self.sealed(receipt,'cut_short')
        self.assertEqual(saved['termination_reason'],'hard_deadline')
        self.assertGreaterEqual(saved['process_wall_seconds'],20)
        self.assertLess(saved['packaging_seconds'],30)  # the worker leaves 30 s between the stage deadline and its alarm

    def test_starter_stopped_on_request_is_sealed_from_its_last_checkpoint(self):
        request=self.root/'stop-request';done=threading.Event()
        def ask():
            while not done.is_set()and len(self.rows())<3:time.sleep(.2)  # two updates are certainly complete
            request.write_text('stop\n')
        watcher=threading.Thread(target=ask,daemon=True);watcher.start()
        try:
            with patch.object(SUP,'native_service',service):
                receipt=SUP.train(self.source,self.config,self.out,600,device='cpu',development=True,stop_request=request)
        finally:done.set();watcher.join(timeout=10)
        saved=self.sealed(receipt,'stopped_early')
        self.assertEqual(saved['termination_reason'],'stop_requested')
        self.assertLess(saved['process_wall_seconds'],300)

    def test_starter_stopped_before_its_first_update_fails(self):
        request=self.root/'stop-request';request.write_text('stop\n')
        with patch.object(SUP,'native_service',service):
            with self.assertRaises(SUP.StopRequested):
                SUP.train(self.source,self.config,self.out,600,device='cpu',development=True,stop_request=request)
        saved=json.loads((self.out/'supervisor.json').read_text())
        self.assertEqual((saved['successful'],saved['termination_reason']),(False,'stop_requested'))
        self.assertFalse((self.out/'bundle').exists())


if __name__=='__main__':unittest.main()
