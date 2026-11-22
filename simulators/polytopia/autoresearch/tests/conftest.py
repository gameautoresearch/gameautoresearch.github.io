"""Shared setup: the repository root on sys.path and one staged copy of the worker's Python runtime."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(scope='session')
def runtime(tmp_path_factory):
    """The pruned package with the starter overlaid, exactly as a release's runtime/ holds it, first on sys.path."""
    from simulators.polytopia.autoresearch import build_context
    repo = tmp_path_factory.mktemp('runtime') / 'repo'
    modules = build_context.stage_source(repo)
    for name in [name for name in sys.modules if name == 'polytopia_bot' or name.startswith('polytopia_bot.')]:
        del sys.modules[name]
    sys.path.insert(0, str(repo / 'src'))
    yield repo, modules
    sys.path.remove(str(repo / 'src'))
