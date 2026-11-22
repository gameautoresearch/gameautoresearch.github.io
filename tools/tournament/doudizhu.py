"""Dou Dizhu head-to-head tournament on the release's private grading deals.

Team scheme (see `doudizhu_worker.py`): every unordered pair (A, B) plays each scheduled private deal twice
on identical cards, once with A as landlord against B as both peasants and once with B as landlord against
A as both peasants. A side's points are 1 for a win and 0 for a loss, so each deal is zero-sum and both
entrants hold every seat once. Each entrant runs as three untrusted `PolicyProcess` children (one per role,
uid 65534, pipe protocol, the submitted `policy.py`), never as an in-process panel bot. The worker is
uploaded into a sandbox built from the release's grading image, CPU-only like the release grader (4 cores,
16 GiB, no network, no secrets), in Modal environment `game-autoresearch-project`. One sandbox per pair.

    PYTHONPATH=$PWD MODAL_PROFILE=duper MODAL_DISABLE_API_PROXY=1 .venv-doudizhu-ar/bin/python \
        -m tools.tournament.doudizhu --entrant A=SESSION --entrant B=SESSION --out NEW_DIR [--smoke] [--deals N]

Default schedule: every deal of the release's `private/grade-seeds.json` (256); `--smoke` plays deals 0-3.
"""
import sys

sys.dont_write_bytecode = True

from contextlib import contextmanager
from pathlib import Path
import json
import tempfile

from tools.tournament import pairplay

HERE = Path(__file__).resolve().parent
SMOKE_DEALS = 4


def _torch_available():
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


class Adapter:
    game = 'doudizhu'
    unit = 'deal_index'
    unit_flag = 'deals'
    gpu = None
    worker_source = HERE / 'doudizhu_worker.py'
    worker_env = {'CUDA_VISIBLE_DEVICES': ''}
    scoring = ('Per game from A\'s side: 1 if A\'s side (landlord, or the two peasants) wins the deal, else 0. '
               'Each deal is played twice on identical cards: A landlord vs B both peasants, and B landlord vs A '
               'both peasants. Each entrant runs one isolated policy process per role. A policy fault forfeits '
               'the batch\'s unfinished games to the opponent (outcome "fault", counted in faults). ci95: '
               'bootstrap over whole deals (both orderings together). Leaderboard: mean points over all of an '
               'entrant\'s games.')

    def __init__(self):
        from harness.policy import ENVIRONMENT
        if ENVIRONMENT != pairplay.ENVIRONMENT:
            raise RuntimeError('Unexpected Modal environment ' + ENVIRONMENT)
        # The host venv may lack torch; tensors are then re-verified by PolicyProcess inside the worker.
        self.check_tensors = _torch_available()

    def gpu_matches(self, reported):
        return True

    @property
    def python(self):
        from harness.games.doudizhu.provider import PYTHON
        return PYTHON

    def verify(self, bundle, expected):
        from simulators.doudizhu.autoresearch.bundle import verify
        return verify(bundle, expected, check_tensors=self.check_tensors)

    def release_engine_digest(self, release):
        from simulators.doudizhu.autoresearch.contract import digest
        return digest(json.loads((Path(release) / 'release.json').read_text())['engine'])

    def schedule(self, release, smoke, units):
        seeds = json.loads((Path(release) / 'private/grade-seeds.json').read_text())
        available = int(seeds['deals'])
        count = units or (SMOKE_DEALS if smoke else available)
        if not 1 <= count <= available:
            raise ValueError('The release has ' + str(available) + ' private grading deals')
        return dict(deal_seed=seeds['deal_seed'], split=seeds['split'], deals=count, batch_size=32)

    def public_schedule(self, schedule):
        # The private deal seed stays out of every host output.
        return dict(deals=schedule['deals'], deal_indices='0..' + str(schedule['deals'] - 1),
                    orderings={'0': 'A landlord, B both peasants', '1': 'B landlord, A both peasants'},
                    source='release private/grade-seeds.json (' + schedule['split'] + ')',
                    games_per_pair=2 * schedule['deals'], batch_size=schedule['batch_size'])

    def expected_keys(self, schedule):
        return {(d, o) for d in range(schedule['deals']) for o in (0, 1)}

    @staticmethod
    def row_key(row):
        return (row['deal_index'], row['ordering'])

    @contextmanager
    def staging(self, a, b, release, schedule):
        from simulators.doudizhu.autoresearch.bundle import archive_bundle
        with tempfile.TemporaryDirectory(prefix='dd-tournament-') as temporary:
            files = []
            for label, entrant in (('a', a), ('b', b)):
                archive = Path(temporary) / (label + '.tar')
                archive_bundle(entrant['bundle'], archive)
                files.append((archive, '/private/' + label + '.tar'))
            yield dict(directories=[], files=files,
                       request=dict(a='/private/a', b='/private/b', deal_seed=schedule['deal_seed'],
                                    split=schedule['split'], deals=schedule['deals'],
                                    batch_size=schedule['batch_size']))

    def allocation_info(self, app_id, sandbox_id):
        from harness.games.doudizhu.provider import allocation_info
        return allocation_info(app_id, sandbox_id)

    def modal_runner(self, release, timeout):
        import modal
        from harness.games.doudizhu.provider import ModalProvider
        app = modal.App('game-autoresearch-tournament-doudizhu')
        context = app.run(environment_name=pairplay.ENVIRONMENT)
        context.__enter__()
        try:
            provider = ModalProvider(release, app)  # audits the release; same grading image
        except BaseException:
            context.__exit__(None, None, None)
            raise
        return context, pairplay.ModalPairRunner(self, app, provider.image, timeout)


if __name__ == '__main__':
    sys.exit(pairplay.main(Adapter()))
