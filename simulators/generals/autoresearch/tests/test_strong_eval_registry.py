from pathlib import Path
import json
import pytest
from simulators.generals.autoresearch.strong_eval import registry as r
from simulators.generals.autoresearch.contract import sha


def fixture(tmp_path,monkeypatch):
 monkeypatch.setattr(r,'ROOT',tmp_path)
 sources={}
 for field,folder in [('averagejoe_files','vendor/engines/AverageJoe'),('engine_files','vendor/engines/generals-bots'),('framework_files','simulators/generals/averagejoe')]:
  p=tmp_path/folder/'model.py';p.parent.mkdir(parents=True,exist_ok=True);p.write_text('unchangedmodel\n');sources[field]={'model.py':sha(p)}
 archive=tmp_path/'archived';ext=archive/'simulators/generals/averagejoe/strong_campaign/runner.py';ext.parent.mkdir(parents=True);ext.write_text('exacttrainingextension\n');sources['strong_campaign_files']={'runner.py':sha(ext)}
 account=archive/'tools/open_credit_campaign.py';account.parent.mkdir(parents=True);account.write_text('timeboundedaccounting\n');sources['accounting_files']={'tools/open_credit_campaign.py':sha(account)}
 p=tmp_path/'source.json';p.write_text(json.dumps(dict(sources=sources,sha256=r.digest_legacy(sources))))
 return p,archive,ext


def test_training_extensions_require_and_verify_archived_bytes(tmp_path,monkeypatch):
 source,archive,ext=fixture(tmp_path,monkeypatch)
 with pytest.raises(ValueError,match='Archived training extensions required'):r.verify_source_manifest(source)
 assert r.verify_source_manifest(source,archive)['sources']['strong_campaign_files']
 ext.write_text('changedtrainingextension\n')
 with pytest.raises(ValueError,match='Archived training extension changed'):r.verify_source_manifest(source,archive)


def test_model_engine_and_framework_still_need_exact_compatibility(tmp_path,monkeypatch):
 source,archive,ext=fixture(tmp_path,monkeypatch)
 (tmp_path/'vendor/engines/AverageJoe/model.py').write_text('different architecture\n')
 with pytest.raises(ValueError,match='Checkpoint source incompatible'):r.verify_source_manifest(source,archive)


def test_unknown_training_source_category_cannot_be_ignored(tmp_path,monkeypatch):
 source,archive,ext=fixture(tmp_path,monkeypatch);d=json.loads(source.read_text());d['sources']['unreviewed_files']={'hidden.py':'anything'};d['sha256']=r.digest_legacy(d['sources']);source.write_text(json.dumps(d))
 with pytest.raises(ValueError,match='Unknown training source'):r.verify_source_manifest(source,archive)


def test_manifest_path_cannot_escape_selected_source():
 for name in ('/absolute.py','../outside.py','folder/../../outside.py'):
  with pytest.raises(ValueError,match='Unsafe manifest path'):r.relative_name(name)
 assert r.relative_name('folder/policy.py')==Path('folder/policy.py')
