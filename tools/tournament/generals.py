"""Generals head-to-head tournament on the release's private grading maps.

Every unordered pair (A, B) plays every scheduled private map twice: A in seat 0 against B in seat 1, then
the seats swapped. Both networks run as untrusted `PolicyProcess` children (uid 65534, pipe protocol, the
submitted `policy.py`), never as in-process panel bots; see `generals_worker.py`, which is uploaded into a
sandbox built from the release's grading image (one L40S for inference, 4 CPU cores, 16 GiB, no network,
no secrets) in Modal environment `game-autoresearch-project`. One sandbox per pair.

    PYTHONPATH=$PWD MODAL_PROFILE=duper MODAL_DISABLE_API_PROXY=1 .venv-generals-ar/bin/python \
        -m tools.tournament.generals --entrant A=SESSION --entrant B=SESSION --out NEW_DIR [--smoke] [--maps N]

Default schedule: every map in the release's `private/grade-maps.npz` (128); `--smoke` plays maps 0-1.
"""
import sys

sys.dont_write_bytecode = True

from contextlib import contextmanager
from pathlib import Path
import json

from tools.tournament import pairplay

HERE = Path(__file__).resolve().parent
SMOKE_MAPS = 2


class Adapter:
    game = 'generals'
    unit = 'map_index'
    unit_flag = 'maps'
    python = 'python'
    worker_source = HERE / 'generals_worker.py'
    worker_env = {'JAX_PLATFORMS': 'cpu', 'GENERALS_POLICY_PLATFORM': 'cuda', 'GENERALS_CPU_INFERENCE_PROFILE': 'default'}
    scoring = ('Per game from A\'s side: capture win 1, capture loss 0, the turn cap (or a simultaneous '
               'outcome) 0.5, as in the release grader. Each map is played with A in seat 0 and in seat 1. '
               'A policy fault (protocol, deadline, illegal action) forfeits that part\'s unfinished games to '
               'the opponent (outcome "fault", counted in faults, never as natural results). ci95: bootstrap '
               'over whole maps (both seats together). Leaderboard: mean points over all of an entrant\'s games.')

    def __init__(self):
        from harness import hardware
        from harness.policy import ENVIRONMENT
        if ENVIRONMENT != pairplay.ENVIRONMENT:
            raise RuntimeError('Unexpected Modal environment ' + ENVIRONMENT)
        self.gpu = hardware.GPU
        self.gpu_matches = hardware.matches

    def verify(self, bundle, expected):
        from simulators.generals.autoresearch.bundle import verify
        return verify(bundle, expected, check_tensors=True)

    def release_engine_digest(self, release):
        from simulators.generals.autoresearch.contract import digest
        return digest(json.loads((Path(release) / 'release.json').read_text())['engine'])

    def schedule(self, release, smoke, units):
        available = len(json.loads((Path(release) / 'private/grade-seeds.json').read_text())['seeds'])
        count = units or (SMOKE_MAPS if smoke else available)
        if not 1 <= count <= available:
            raise ValueError('The release has ' + str(available) + ' private grading maps')
        return dict(pool_path=str(Path(release) / 'private/grade-maps.npz'), map_indices=list(range(count)),
                    split='heldout', batch_size=32)

    def public_schedule(self, schedule):
        return dict(maps=len(schedule['map_indices']), map_indices=schedule['map_indices'], seats=[0, 1],
                    source='release private/grade-maps.npz (heldout)', games_per_pair=2 * len(schedule['map_indices']),
                    batch_size=schedule['batch_size'])

    def expected_keys(self, schedule):
        return {(i, seat) for i in schedule['map_indices'] for seat in (0, 1)}

    @staticmethod
    def row_key(row):
        return (row['map_index'], row['a_seat'])

    @contextmanager
    def staging(self, a, b, release, schedule):
        from simulators.generals.autoresearch.bundle import FILES
        files = [(schedule['pool_path'], '/private/maps.npz')]
        for label, entrant in (('a', a), ('b', b)):
            files += [(Path(entrant['bundle']) / name, '/private/' + label + '/' + name) for name in (*FILES, 'manifest.json')]
        yield dict(directories=['/private/a', '/private/b'], files=files,
                   request=dict(a='/private/a', b='/private/b', pool_path='/private/maps.npz',
                                map_indices=schedule['map_indices'], batch_size=schedule['batch_size']))

    def allocation_info(self, app_id, sandbox_id):
        from simulators.generals.autoresearch.provider import allocation_info
        return allocation_info(app_id, sandbox_id)

    def modal_runner(self, release, timeout):
        import modal
        from harness.games.generals.provider import ModalProvider
        from simulators.generals.autoresearch.provider import read_rpc
        app = modal.App('game-autoresearch-tournament-generals')
        context = app.run(environment_name=pairplay.ENVIRONMENT)
        context.__enter__()
        try:
            provider = ModalProvider(release, app)  # audits the release and builds its grading image
        except BaseException:
            context.__exit__(None, None, None)
            raise
        return context, pairplay.ModalPairRunner(self, app, provider.image, timeout, rpc=read_rpc)


if __name__ == '__main__':
    sys.exit(pairplay.main(Adapter()))
