import hashlib
from pathlib import Path
import pytest
from simulators.generals.averagejoe.strong_campaign.resume_safe import retain_file, refresh_fits


def test_ancestor_pool_is_retained_and_portable(tmp_path):
    ancestor = tmp_path / 'ancestor' / 'pool.npz'
    ancestor.parent.mkdir()
    ancestor.write_bytes(b'full verified active pool')
    sha = hashlib.sha256(ancestor.read_bytes()).hexdigest()
    current = tmp_path / 'current'
    target = current / 'output' / 'pools' / 'retained-active.npz'
    assert retain_file(ancestor, sha, current, target).read_bytes() == ancestor.read_bytes()
    ancestor.unlink()
    portable = tmp_path / 'portable'
    (portable / 'output/pools').mkdir(parents=True)
    (portable / 'output/pools/retained-active.npz').write_bytes(target.read_bytes())
    recorded = '/retained/runs/old-run/output/pools/retained-active.npz'
    copied = retain_file(recorded, sha, portable, portable / 'checked.npz')
    assert copied.read_bytes() == b'full verified active pool'


def test_existing_ancestor_with_wrong_bytes_is_rejected(tmp_path):
    p = tmp_path / 'pool.npz'
    p.write_bytes(b'changed')
    with pytest.raises(ValueError, match='Referenced input changed'):
        retain_file(p, hashlib.sha256(b'original').hexdigest(), tmp_path, tmp_path / 'out.npz')


def test_late_cold_refresh_is_deferred_but_warm_refresh_can_finish():
    assert not refresh_fits(381, 452, warmed=False, iteration_cost=12)
    assert refresh_fits(381, 452, warmed=True, iteration_cost=12)
    assert refresh_fits(100, 3400, warmed=False, iteration_cost=12)
    assert not refresh_fits(430, 452, warmed=True, iteration_cost=12)
