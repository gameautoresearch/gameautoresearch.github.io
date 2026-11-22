import io,json
from types import SimpleNamespace
from harness.games.generals.provider import capture_training_diagnostics


def process(rc=0):
    return SimpleNamespace(poll=lambda:rc,stdout=io.StringIO('trainer output'),stderr=io.StringIO('trainer error'))


def test_supervisor_failure_is_salvage_even_when_parent_exec_returned_zero(tmp_path):
    sandbox=SimpleNamespace(filesystem=SimpleNamespace(read_text=lambda p:json.dumps(dict(status='completed',returncode=1))))
    value=capture_training_diagnostics(sandbox,process(),tmp_path,checkpoint_valid=True)
    assert value['lifecycle']=='salvaged_checkpoint_after_process_failure'
    assert value['failure_scope']=='child'
    assert value['training_completion_verified'] is False
    assert value['parent_exec_returncode']==0 and value['child_returncode']==1
    assert (tmp_path/'trainer-stdout.log').read_text()=='trainer output'
    assert (tmp_path/'trainer-stderr.log').read_text()=='trainer error'
    assert json.loads((tmp_path/'training-lifecycle.json').read_text())==value


def test_clean_completion_requires_child_and_parent_success(tmp_path):
    sandbox=SimpleNamespace(filesystem=SimpleNamespace(read_text=lambda p:json.dumps(dict(status='completed',returncode=0))))
    value=capture_training_diagnostics(sandbox,process(),tmp_path,checkpoint_valid=True)
    assert value['lifecycle']=='clean_trainer_completion'
    assert value['training_completion_verified'] is True


def test_missing_supervisor_does_not_invent_failure_or_clean_exit(tmp_path):
    def unavailable(path):raise FileNotFoundError(path)
    sandbox=SimpleNamespace(filesystem=SimpleNamespace(read_text=unavailable))
    value=capture_training_diagnostics(sandbox,process(),tmp_path,checkpoint_valid=True)
    assert value['lifecycle']=='valid_checkpoint_training_exit_unverified'
    assert value['child_returncode'] is None and value['training_completion_verified'] is False
    assert (tmp_path/'trainer-stderr.log').exists()


def test_failed_export_never_becomes_valid_from_clean_supervisor(tmp_path):
    sandbox=SimpleNamespace(filesystem=SimpleNamespace(read_text=lambda p:json.dumps(dict(status='completed',returncode=0))))
    value=capture_training_diagnostics(sandbox,process(),tmp_path,checkpoint_valid=False)
    assert value['lifecycle']=='no_valid_exported_checkpoint'
    assert value['checkpoint_valid'] is False


def test_parent_failure_is_distinguished_from_successful_child(tmp_path):
    sandbox=SimpleNamespace(filesystem=SimpleNamespace(read_text=lambda p:json.dumps(dict(status='completed',returncode=0))))
    value=capture_training_diagnostics(sandbox,process(1),tmp_path,checkpoint_valid=True)
    assert value['lifecycle']=='salvaged_checkpoint_after_process_failure'
    assert value['failure_scope']=='parent_exec'
    assert value['child_returncode']==0 and value['training_completion_verified'] is False
