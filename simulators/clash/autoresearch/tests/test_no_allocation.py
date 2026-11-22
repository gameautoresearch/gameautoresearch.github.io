import pytest
from simulators.clash.autoresearch.ledger import Ledger
from simulators.clash.autoresearch.common import dump

def test_precreate_inventory_releases_only_unattempted_reservation(tmp_path):
 ledger=Ledger(tmp_path/'ledger.json',600);ledger.reserve('failed',150);ledger.record('failed',strong_job='stage-1');path=tmp_path/'proof.json';dump(path,{'stage':'stage-1','provider_inventory_checked':True,'worker_count':0,'worker_create_never_attempted':True})
 ledger.settle_no_allocation('failed',path);state=ledger.read();assert ledger.used()==0 and state['allocations'][0]['reserved_seconds']==150 and state['allocations'][0]['status']=='failed'
 with pytest.raises(ValueError,match='known, settled'):ledger.settle_no_allocation('failed',path)
 with pytest.raises(ValueError,match='without submission'):ledger.finalize()

def test_unknown_or_known_worker_cannot_be_zeroed(tmp_path):
 ledger=Ledger(tmp_path/'ledger.json',600);ledger.reserve('known',150);ledger.record('known',strong_job='stage-1',sandbox_id='real-worker');path=tmp_path/'proof.json';dump(path,{'stage':'stage-1','provider_inventory_checked':True,'worker_count':0,'worker_create_never_attempted':True})
 with pytest.raises(ValueError,match='known, settled'):ledger.settle_no_allocation('known',path)
 assert ledger.used()==150
 dump(path,{'stage':'stage-1','provider_inventory_checked':True,'worker_count':0,'worker_create_never_attempted':False})
 with pytest.raises(ValueError,match='pre-create'):ledger.settle_no_allocation('known',path)
