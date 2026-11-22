"""The editable-simulator starter (`prepare(starter='slowsim')`): the plain simulator plays the engine's games, the
box shows it as editable and nothing of the engine, and an unprivileged process cannot import the engine."""
import json
import re
from pathlib import Path

import pytest

from simulators.generals.autoresearch import prepare as prep
from simulators.generals.autoresearch.slow_sim import parity

HERE = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def release(tmp_path_factory):
    out = tmp_path_factory.mktemp('slowsim') / 'generals'
    prep.prepare(out, include_existing=False, private_maps=2, starter='slowsim')
    return out


def test_plain_simulator_plays_the_engine_games(release):
    stats = parity.run(release / 'public/train-maps.npz', games=6, ticks=120, seed=3)
    assert stats['ticks'] == 720 and stats['illegal_ids'] > 0 and stats['max_encode_diff'] <= 1e-6
    trades = parity.trade_cases(release / 'public/train-maps.npz', seed=5, cases=96)
    assert trades['cases'] == 96 and trades['trades'] > 0


def test_box_holds_the_simulator_as_editable_and_no_engine(release):
    meta = json.loads((release / 'public/workspace_manifest.json').read_text())
    assert meta['editable_simulator'] is True
    sim = {'simulator/generals_sim/' + n for n in prep.SLOW_SIM}
    assert sim <= set(meta['writable'])
    shown = {n for n in meta['files'] if n.startswith('simulator/')}
    assert shown == sim | {'simulator/README.md'}
    box = release / 'public/agent_workspace'
    text = '\n'.join(p.read_text() for p in box.rglob('*.md'))
    for word in ('jax', 'vmap', 'jit', 'packed', 'generals-bots', 'integrations.generals.environment'):
        assert word not in text.lower(), word
    # Nothing tells the researcher what to do with the simulator or how quick it is, only what it is. (The brief's
    # "the optimizer" of a learner and "spend it proportionally faster" of the budget are about other things.)
    hint = re.compile(r'\b(slow\w*|fast|speed\w*|vectori[sz]\w*|throughput|optimi[sz]e[sd]?|efficien\w*|profil\w*)\b', re.I)
    for name in ('public/brief.md', 'public/agent_workspace/simulator/README.md'):
        assert not hint.search((release / name).read_text()), (name, hint.search((release / name).read_text()))
    for path in (release / 'public/agent_workspace/simulator').rglob('*.py'):
        assert not hint.search(path.read_text()), path


def test_unprivileged_tree_has_no_engine(release):
    root = release / 'frozen/unprivileged'
    files = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}
    assert not any(f.startswith('generals/') for f in files)
    assert (root / 'integrations/generals/environment.py').read_text() == prep.SLOW_ENVIRONMENT
    assert not re.search(r'^\s*(import|from)\s', prep.SLOW_ENVIRONMENT, re.M)
    prep.audit_release(release)
