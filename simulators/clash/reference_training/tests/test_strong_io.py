import concurrent.futures,importlib,threading
import pytest

@pytest.mark.parametrize('module_name',['provider','dev_remote'])
def test_concurrent_collection_never_double_settles_or_downloads(tmp_path,monkeypatch,module_name):
 module=importlib.import_module('simulators.clash.reference_training.strong.'+module_name)
 monkeypatch.setattr(module,'OWN',tmp_path);folder=tmp_path/('cpu-development' if module_name=='dev_remote' else '')/'attempt';folder.mkdir(parents=True)
 entered=threading.Event();release=threading.Event();calls=[]
 def real_collect(name):
  calls.append(name);entered.set();assert release.wait(3);return {'done':True}
 monkeypatch.setattr(module,'_collect',real_collect)
 with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
  first=pool.submit(module.collect,'attempt')
  try:
   assert entered.wait(3)
   assert module.collect('attempt')=={'collection_in_progress':True,'name':'attempt'}
  finally:release.set()
  assert first.result(timeout=3)=={'done':True}
 assert calls==['attempt']
