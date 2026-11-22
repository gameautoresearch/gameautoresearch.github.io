"""Hex research substrate: wire, openings, scoring, bundles, the arena and the release. CPU only.

MoHex games run only when a MoHex binary is available (HEX_MOHEX, /opt/mohex/mohex or PATH).
"""
import importlib.util
import json
import math
from pathlib import Path
import shutil

import numpy as np
import pytest
import torch

from simulators.hex.autoresearch import arena, bundle, contract, grading, wire
from simulators.hex.autoresearch.contract import ANCHOR, HERE, PANEL

STARTER = HERE / 'starter'


def has_mohex():
    try:
        from simulators.hex.mohex import locate
        locate()
        return True
    except FileNotFoundError:
        return False


def package(folder, policy_source=None, seed=0):
    """A sealed package of a randomly initialized starter network (or a replacement policy.py)."""
    source = folder / 'source'
    source.mkdir(parents=True)
    for name in contract.EDITABLE:
        shutil.copyfile(STARTER / name, source / name)
    if policy_source is not None:
        (source / 'policy.py').write_text(policy_source)
    spec = importlib.util.spec_from_file_location('starter_policy', STARTER / 'policy.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch.manual_seed(seed)
    config = {'hidden': 32, 'depth': 1}
    (source / 'model.json').write_text(json.dumps(config))
    torch.save(module.create(config).state_dict(), source / 'weights.pt')
    bundle.seal(source, folder / 'bundle', {'kind': 'test'})
    return folder / 'bundle'


def rows_for(opponents, openings, won):
    return [{'opponent': o, 'opening_index': i, 'seat': s, 'won': won(o, i, s), 'natural': True,
             'truncated': False, 'plies': 30} for o in opponents for i in range(openings) for s in ('black', 'white')]


# ----- wire
def test_wire_round_trip_and_rejections():
    board = np.zeros((3, wire.CELLS), np.int8)
    board[0, 40] = 1
    to_play = np.array([2, 1, 1], np.int8)
    legal = board == 0
    raw = wire.encode(board, to_play, legal)
    b, t, l = wire.decode(raw[4:])
    assert np.array_equal(b, board) and np.array_equal(t, to_play) and np.array_equal(l, legal)
    with pytest.raises(ValueError):
        wire.decode(wire.encode(board[:, :-1], to_play, legal[:, :-1])[4:])
    with pytest.raises(ValueError):
        wire.decode(wire.encode(board, np.array([0, 1, 1], np.int8), legal)[4:])
    with pytest.raises(ValueError):
        wire.decode(wire.encode(board, to_play, np.zeros_like(legal))[4:])


# ----- openings
def test_opening_bank_is_seeded_distinct_and_never_a_repeat_cell():
    a, b = arena.opening_bank(7, 64), arena.opening_bank(7, 64)
    assert a == b and len(set(a)) == 64 and all(x != y for x, y in a)
    assert arena.opening_bank(8, 64) != a
    assert len(set(arena.opening_bank(1, 169 * 168))) == 169 * 168
    with pytest.raises(ValueError):
        arena.opening_bank(1, 169 * 168 + 1)


# ----- scoring
def test_summary_weights_opponents_equally_and_needs_the_full_schedule():
    rows = rows_for(['random', 'mohex-100'], 4, lambda o, i, s: o == 'random')
    result = grading.summary(rows, replicates=200)
    assert result['raw_score'] == 50 and result['per_opponent'] == {'mohex-100': 0.0, 'random': 1.0}
    assert result['raw_score_ci95'] == [50.0, 50.0] and 'anchor' not in result
    with pytest.raises(ValueError):
        grading.summary(rows[:-1])
    with pytest.raises(ValueError):
        grading.summary(rows + rows[:1])


def test_perfect_player_against_perfect_anchor_is_zero_elo():
    """Each opening is won from exactly one seat by the theoretically winning side."""
    rows = rows_for([p['name'] for p in PANEL], 8,
                    lambda o, i, s: True if o != ANCHOR else (s == 'black') == (i % 2 == 0))
    result = grading.summary(rows, replicates=200)
    assert math.isclose(result['raw_score'], 100 * (8 + 0.5) / 9)
    anchor = result['anchor']
    assert anchor['win_rate'] == 0.5 and anchor['elo_vs_anchor'] == 0
    assert anchor['openings_won_one_seat'] == 8 and anchor['openings_won_both_seats'] == 0
    assert math.isclose(grading.elo(0, 16), 400 * math.log10(1 / 31)) and math.isclose(grading.elo(0.75, 100), 400 * math.log10(3))


def test_grade_is_the_raw_score():
    rows = rows_for(['random'], 2, lambda o, i, s: s == 'black')
    graded = grading.grade(rows)
    assert graded['status'] == 'complete' and graded['raw_score'] == 50 and graded['normalized_quality'] is None
    assert grading.grade(rows, development=True)['status'] == 'development_only'


# ----- bundles
def test_bundle_seal_verify_archive_and_drift(tmp_path):
    sealed = package(tmp_path)
    manifest = bundle.verify(sealed)
    assert manifest['contract'] == contract.DEFAULT['version']
    bundle.archive_bundle(sealed, tmp_path / 'b.tar')
    bundle.unpack_bundle(tmp_path / 'b.tar', tmp_path / 'copy')
    assert bundle.verify(tmp_path / 'copy')['bundle_sha256'] == manifest['bundle_sha256']
    with pytest.raises(ValueError):
        bundle.verify(sealed, expected='0' * 64)
    (sealed / 'policy.py').write_text('# changed\n')
    with pytest.raises(ValueError):
        bundle.verify(sealed)


def test_bundle_rejects_non_tensor_weights(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    for name in contract.EDITABLE:
        shutil.copyfile(STARTER / name, source / name)
    (source / 'model.json').write_text('{}')
    torch.save({'w': torch.ones(2), 'bad': torch.tensor(float('nan'))}, source / 'weights.pt')
    with pytest.raises(ValueError):
        bundle.seal(source, tmp_path / 'bundle')


# ----- arena
def test_arena_against_random_completes_with_natural_results(tmp_path):
    sealed = package(tmp_path)
    result = arena.evaluate(sealed, [{'name': 'random', 'kind': 'random'}], opening_seed=3, policy_seed=5,
                            openings=6, split='practice', development=True, workers=1)
    rows = result['rows']
    assert len(rows) == 12 and all(r['natural'] and not r['truncated'] for r in rows)
    assert {(r['opening_index'], r['seat']) for r in rows} == {(i, s) for i in range(6) for s in ('black', 'white')}
    for r in rows:
        assert r['moves'][:2] == r['opening'] and len(set(r['moves'])) == r['plies']
        from simulators.hex import reference
        _, winner, length = reference.play(r['moves'])
        assert length == r['plies'] and (winner == (1 if r['seat'] == 'black' else 2)) == r['won']
    again = arena.evaluate(sealed, [{'name': 'random', 'kind': 'random'}], opening_seed=3, policy_seed=5,
                           openings=6, split='practice', development=True, workers=1)
    assert [r['moves'] for r in again['rows']] == [r['moves'] for r in rows]


@pytest.mark.parametrize('body, message', [
    ('def act(policy, board, to_play, legal):\n    return [int(np.argmin(l)) if not l.all() else 0 for l in legal]\n',
     'Illegal'),
    ('def act(policy, board, to_play, legal):\n    return [0]\n', 'cells for'),
])
def test_arena_invalidates_bad_policies(tmp_path, monkeypatch, body, message):
    source = (STARTER / 'policy.py').read_text().split('\ndef act(')[0] + '\nimport numpy as np\n' + body
    sealed = package(tmp_path, source)
    with pytest.raises(arena.PolicyFailure, match=message):
        arena.evaluate(sealed, [{'name': 'random', 'kind': 'random'}], opening_seed=3, policy_seed=5,
                       openings=4, split='practice', development=True, workers=1)


def starter_with(tmp_path, create_prefix='', act_prefix=''):
    """The starter policy with code run first in create() or act()."""
    source = (STARTER / 'policy.py').read_text()
    source = source.replace('\ndef create(config):\n', '\ndef create(config):\n' + create_prefix, 1)
    source = source.replace('\ndef act(policy, board, to_play, legal):\n',
                            '\ndef act(policy, board, to_play, legal):\n' + act_prefix, 1)
    return package(tmp_path, source)


def test_a_late_move_is_replaced_by_a_random_move_and_play_continues(tmp_path, monkeypatch):
    # The third call overruns the move clock; the call count lives in the policy's scratch HOME,
    # which survives the restart of the stalled process, so only that one call is late.
    slow_third = ('    import os, time\n'
                  '    path = os.path.join(os.environ["HOME"], "calls")\n'
                  '    calls = (int(open(path).read()) if os.path.exists(path) else 0) + 1\n'
                  '    open(path, "w").write(str(calls))\n'
                  '    if calls == 3:\n'
                  '        time.sleep(30)\n')
    sealed = starter_with(tmp_path, act_prefix=slow_third)
    monkeypatch.setitem(contract.DEFAULT, 'call_seconds', 1)
    started = __import__('time').monotonic()
    result = arena.evaluate(sealed, [{'name': 'random', 'kind': 'random'}], opening_seed=3, policy_seed=5,
                            openings=2, split='practice', development=True, workers=1, batch_size=1)
    assert __import__('time').monotonic() - started < 25  # the sleeping call was abandoned, not awaited
    assert (result['call_seconds'], result['move_timeouts'], result['random_moves']) == (1, 1, 1)
    assert sum(r['random_moves'] for r in result['rows']) == 1
    assert all(r['natural'] for r in result['rows']) and result['candidate_decisions'] >= 3
    from simulators.hex import reference
    for r in result['rows']:  # every game, the substituted move included, is a legal finished game
        _, winner, length = reference.play(r['moves'])
        assert length == r['plies'] and (winner == (1 if r['seat'] == 'black' else 2)) == r['won']


def test_the_move_clock_grows_with_the_positions_in_a_call(tmp_path):
    """Inference is capped per move: a fixed allowance per position plus one constant per call."""
    assert arena.move_clock(1) == pytest.approx(0.11) and arena.move_clock(64) == pytest.approx(0.74)
    assert arena.move_clock(4, position_seconds=0.1, call_seconds=1) == pytest.approx(1.4)
    # A policy that spends 50 ms on every position: on time under a 500 ms allowance, late under a 10 ms one.
    sealed = starter_with(tmp_path, act_prefix='    import time; time.sleep(0.05 * len(to_play))\n')
    board = np.zeros((4, wire.CELLS), np.int8)
    to_play, legal = np.ones(4, np.int8), board == 0
    with arena.PolicyProcess(sealed, development=True, position_seconds=0.5, call_seconds=1) as generous:
        assert len(generous.choose(board, to_play, legal)) == 4 and generous.timeouts == 0
    with arena.PolicyProcess(sealed, development=True, position_seconds=0.01, call_seconds=0.05) as tight:
        with pytest.raises(arena.PolicyTimeout):
            tight.choose(board, to_play, legal)
        assert tight.timeouts == 1
        assert tight.process.poll() is None  # the stalled process was replaced by a fresh one


def test_loading_the_model_is_not_on_the_move_clock(tmp_path, monkeypatch):
    sealed = starter_with(tmp_path, create_prefix='    import time; time.sleep(3)\n')
    monkeypatch.setitem(contract.DEFAULT, 'call_seconds', 1)
    result = arena.evaluate(sealed, [{'name': 'random', 'kind': 'random'}], opening_seed=3, policy_seed=5,
                            openings=1, split='practice', development=True, workers=1)
    assert (result['move_timeouts'], result['random_moves']) == (0, 0)


def test_a_policy_that_does_not_load_in_time_is_invalid(tmp_path, monkeypatch):
    sealed = starter_with(tmp_path, create_prefix='    import time; time.sleep(30)\n')
    monkeypatch.setitem(contract.DEFAULT, 'policy_startup_seconds', 2)
    with pytest.raises(arena.PolicyFailure, match='did not load within 2 s'):
        arena.evaluate(sealed, [{'name': 'random', 'kind': 'random'}], opening_seed=3, policy_seed=5,
                       openings=1, split='practice', development=True, workers=1)


def test_official_grading_refuses_an_unisolated_host(tmp_path):
    import os
    import sys
    if sys.platform == 'linux' and os.geteuid() == 0:
        pytest.skip('this host can isolate the candidate')
    with pytest.raises(RuntimeError, match='root-owned Linux'):
        arena.evaluate(package(tmp_path), [{'name': 'random', 'kind': 'random'}], opening_seed=1, policy_seed=1,
                       openings=1)


def test_panel_names_are_checked(tmp_path):
    with pytest.raises(ValueError):
        arena.evaluate(package(tmp_path), [{'name': 'mohex-7', 'kind': 'mohex'}], opening_seed=1, policy_seed=1,
                       openings=1, development=True)


@pytest.mark.skipif(not has_mohex(), reason='MoHex binary not built')
def test_arena_against_mohex_levels(tmp_path):
    sealed = package(tmp_path)
    panel = [{'name': 'mohex-100', 'kind': 'mohex'}, {'name': 'mohex-2k', 'kind': 'mohex'}]
    result = arena.evaluate(sealed, panel, opening_seed=11, policy_seed=2, openings=2, split='practice',
                            development=True, workers=2)
    assert len(result['rows']) == 8
    # An untrained network does not beat MoHex.
    assert sum(r['won'] for r in result['rows']) <= 1
    summary = grading.summary(result['rows'], replicates=100)
    assert summary['games'] == 8


@pytest.mark.skipif(not has_mohex(), reason='MoHex binary not built')
def test_mohex_conventions_match_the_engine():
    from simulators.hex.engine import parse_cell
    from simulators.hex.mohex import MoHex
    with MoHex(contract.MOHEX_STARTUP) as m:
        for line in contract.MOHEX_LEVELS['mohex-100']:
            m.query(line)
        for row in range(1, 13):
            m.play(1, parse_cell(f'e{row}'))
            m.play(2, parse_cell(f'a{row}'))
        m.play(1, parse_cell('e13'))
        assert m.query('final_score') == 'B+'  # black joins top (row 1) to bottom (row 13)
        m.new_game()
        for col in 'abcdefghijkl':
            m.play(1, parse_cell(f'{col}1'))
            m.play(2, parse_cell(f'{col}5'))
        m.play(1, parse_cell('a13'))
        m.play(2, parse_cell('m5'))
        assert m.query('final_score') == 'W+'  # white joins column a to column m
        assert m.query('param_game').splitlines()[0].endswith('allow_swap 0')


# ----- release and dry run
@pytest.fixture(scope='module')
def release(tmp_path_factory):
    from simulators.hex.autoresearch.prepare import prepare
    out = tmp_path_factory.mktemp('hex-release') / 'release'
    prepare(out)
    return out


def test_release_layout_and_secrecy(release):
    from simulators.hex.autoresearch.prepare import audit_release
    assert audit_release(release)['official_grading_ready']
    public = release / 'public'
    names = {p.relative_to(public).as_posix() for p in public.rglob('*') if p.is_file()}
    assert not any('grade-seeds' in n or n.endswith('.pt') for n in names)
    box = public / 'agent_workspace'
    assert (box / 'simulator/simulators/hex/engine.py').is_file()
    # MoHex is held out: the researcher gets neither its source nor its client, the workers still build it.
    assert not (box / 'simulator/vendor').exists() and not (box / 'simulator/simulators/hex/mohex.py').exists()
    assert (release / 'frozen/vendor/engines/benzene/src/mohex/MoHexPlayer.cpp').is_file()
    assert (box / 'training/train.py').read_text() == (STARTER / 'train.py').read_text()
    assert json.loads((release / 'private/grade-seeds.json').read_text())['openings'] == 64
    frozen = release / 'frozen'
    assert (frozen / 'simulators/hex/autoresearch/arena.py').is_file()
    assert (frozen / 'vendor/engines/benzene/src/util/Misc.cpp').read_text().count('lexically_normal') == 3
    assert not (frozen / 'simulators/hex/autoresearch/starter').exists()


def test_release_detects_drift(release, tmp_path):
    from simulators.hex.autoresearch.prepare import audit_release
    copy = tmp_path / 'copy'
    shutil.copytree(release, copy)
    target = copy / 'frozen/simulators/hex/engine.py'
    target.chmod(0o644)
    target.write_text(target.read_text() + '\n')
    with pytest.raises(ValueError, match='drift'):
        audit_release(copy)


def test_dry_run_trains_and_answers_in_the_box_layout(release, tmp_path, monkeypatch):
    box = tmp_path / 'workspace'
    shutil.copytree(release / 'public/agent_workspace', box)
    for path in box.rglob('*'):
        path.chmod(0o755 if path.is_dir() else 0o644)
    shutil.copyfile(HERE / 'recipes/smoke.json', box / 'configs/smoke.json')
    spec = importlib.util.spec_from_file_location('hex_dryrun', HERE / 'task/dryrun.py')
    dryrun = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dryrun)
    monkeypatch.setattr(dryrun, 'TRAINER_SECONDS', 6)
    monkeypatch.setattr('sys.argv', ['dryrun.py', '--workspace', str(box), '--config', 'configs/smoke.json'])
    result = dryrun.main()
    assert result['passed'], result
    assert result['progress_rows'] >= 1 and result['positions_answered'] > 64
    # A policy that answers an occupied cell fails the dry run.
    policy = (box / 'training/policy.py').read_text()
    (box / 'training/policy.py').write_text(policy + '\n\ndef act(policy, board, to_play, legal):\n'
                                            '    return [int((~legal[i]).nonzero()[0][0]) if (~legal[i]).any() '
                                            'else 0 for i in range(len(legal))]\n')
    result = dryrun.main()
    assert not result['passed'] and 'illegal' in result['error']


# ----- the calibrated panel
def test_panel_is_spaced_and_rated():
    elos = [p['elo'] for p in PANEL]
    assert elos == sorted(elos) and PANEL[-1]['name'] == ANCHOR and PANEL[-1]['elo'] == 0
    gaps = np.diff(elos)
    assert gaps.min() >= 100 and gaps.max() <= 450
    arena.check_panel(list(PANEL))
    with pytest.raises(ValueError):
        arena.check_panel([{'name': 'x', 'kind': 'heuristic', 'epsilon': 1.0}])


def test_panel_elo_recovers_a_rating_and_reports_bounds():
    rng = np.random.default_rng(4)
    true = -700.0

    def won(o, i, s):
        rating = next(p['elo'] for p in PANEL if p['name'] == o)
        return bool(rng.random() < 1 / (1 + 10 ** ((rating - true) / 400)))
    rows = rows_for([p['name'] for p in PANEL], 400, won)
    rated = grading.panel_elo(rows, replicates=200)
    assert abs(rated['elo'] - true) < 3 * rated['se'] and rated['ci95'][0] < true < rated['ci95'][1]
    assert not rated['at_bound']
    everything = grading.panel_elo(rows_for([p['name'] for p in PANEL], 4, lambda o, i, s: True), replicates=50)
    nothing = grading.panel_elo(rows_for([p['name'] for p in PANEL], 4, lambda o, i, s: False), replicates=50)
    assert everything['at_bound'] and everything['elo'] == grading.HIGH
    assert nothing['at_bound'] and nothing['elo'] == grading.LOW
    assert grading.panel_elo(rows_for(['someone-else'], 2, lambda o, i, s: True)) is None


def test_epsilon_opponents_are_reproducible(tmp_path):
    sealed = package(tmp_path)
    panel = [{'name': 'heuristic-e50', 'kind': 'heuristic', 'epsilon': 0.5}, {'name': 'heuristic', 'kind': 'heuristic'}]
    first = arena.evaluate(sealed, panel, opening_seed=3, policy_seed=5, openings=4, development=True, workers=1)
    again = arena.evaluate(sealed, panel, opening_seed=3, policy_seed=5, openings=4, development=True, workers=1)
    assert [r['moves'] for r in first['rows']] == [r['moves'] for r in again['rows']]


def test_ladder_orders_cheap_players():
    from simulators.hex.autoresearch import ladder
    players = [{'name': 'random', 'kind': 'random'}, {'name': 'heuristic-e70', 'kind': 'heuristic', 'epsilon': 0.7},
               {'name': 'heuristic', 'kind': 'heuristic'}]
    rows = ladder.play([(a, b) for i, a in enumerate(players) for b in players[i + 1:]], openings=8, workers=1)
    assert len(rows) == 3 * 16 and {r['a_seat'] for r in rows} == {'black', 'white'}
    ratings = ladder.fit(rows, [p['name'] for p in players])
    assert ratings['random']['elo'] < ratings['heuristic-e70']['elo'] < ratings['heuristic']['elo']
    table = {(r['a'], r['b']): r['a_win_rate'] for r in ladder.pairwise(rows)}
    assert table[('random', 'heuristic')] < 0.2


def test_mohex_pool_replaces_a_crashed_process_and_counts_it(monkeypatch):
    """A dead MoHex is replaced and asked again; a live one that refuses, or repeated crashes, is a fault."""
    from simulators.hex import mohex

    class Fake:
        crashes = 0

        def __init__(self, params, seed=1, binary=None):
            self.process = type('P', (), {'returncode': None, 'poll': lambda s: s.returncode})()

        def query(self, line):
            if line.startswith('reg_genmove'):
                if Fake.crashes:
                    Fake.crashes -= 1
                    self.process.returncode = -11
                    raise mohex.MoHexFault('MoHex exited (code -11)')
                return 'g7'
            return ''

        def new_game(self):
            pass

        def play(self, colour, cell):
            pass

        def close(self):
            pass
    monkeypatch.setattr(mohex, 'MoHex', Fake)
    with arena.MoHexPool(2, seed=1) as pool:
        Fake.crashes = 2
        assert pool.genmoves([('mohex-100', [0, 1], 1)]) == [84]
        assert len(pool.restarts) == 2 and pool.restarts[0]['level'] == 'mohex-100'
        Fake.crashes = 3
        with pytest.raises(mohex.MoHexFault):
            pool.genmoves([('mohex-2k', [0, 1], 1)])


def test_a_running_job_holds_its_whole_grant_so_parallel_grants_fit_the_budget(tmp_path):
    import time
    from simulators.hex.autoresearch.contract import DEFAULT, TRAIN_SHAPE, rate
    from simulators.hex.autoresearch.ledger import Ledger
    r = rate(TRAIN_SHAPE)
    ledger = Ledger(tmp_path, dict(DEFAULT, budget_usd=600 * r, min_run_seconds=1))
    ledger.queue('a', 400, {})
    ledger.start('a', 'sb-a', time.time())  # just started: almost no time spent yet
    assert ledger.used() == pytest.approx(400 * r)
    assert ledger.public()['remaining_usd'] == pytest.approx(200 * r)
    with pytest.raises(ValueError, match='exceeds remaining'):
        ledger.queue('b', 400, {})
    ledger.queue('b', 200, {})
    # Milestones keep their own measure: a running worker's time so far.
    assert ledger.progress() == pytest.approx(200 * r, abs=5 * r)
    ledger.finish('a', ledger.jobs()[0]['started'] + 100)
    assert ledger.used() == pytest.approx(300 * r)  # a's 100 seconds and b's hold


def test_a_seconds_mode_ledger_runs_one_job_at_a_time_in_training_worker_seconds(tmp_path):
    import time
    from simulators.hex.autoresearch.contract import (DEFAULT, PRACTICE_SHAPE, TRAIN_SHAPE, rate, seconds_weight,
                                                      validate_session)
    from simulators.hex.autoresearch.ledger import Ledger
    config = dict(DEFAULT, budget_mode='seconds', budget_seconds=600, min_run_seconds=1, max_concurrent_jobs=1,
                  max_concurrent_gpus=1, max_concurrent_cpu_cores=None)
    del config['budget_usd']
    ledger = Ledger(tmp_path, config)
    assert (ledger.mode, ledger.unit, ledger.budget()) == ('seconds', 'seconds', 600)
    assert 'budget_usd' not in ledger.config
    assert seconds_weight(TRAIN_SHAPE) == 1.0
    assert seconds_weight(PRACTICE_SHAPE) == pytest.approx(rate(PRACTICE_SHAPE) / rate(TRAIN_SHAPE))
    ledger.queue('a', 400, {})  # a payload without a recorded rate is the training worker at weight 1
    with pytest.raises(ValueError, match='One job at a time'):
        ledger.queue('b', 10, {'shape': dict(PRACTICE_SHAPE), 'rate': seconds_weight(PRACTICE_SHAPE)})
    ledger.start('a', 'sb-a', time.time())
    ledger.finish('a', ledger.jobs()[0]['started'] + 100)
    assert ledger.public()['charged_or_reserved_seconds'] == pytest.approx(100)
    assert ledger.public()['remaining_seconds'] == pytest.approx(500)
    weight = seconds_weight(PRACTICE_SHAPE)
    with pytest.raises(ValueError, match='exceeds remaining'):
        ledger.queue('b', 501 / weight, {'shape': dict(PRACTICE_SHAPE), 'rate': weight})
    ledger.queue('b', 400, {'shape': dict(PRACTICE_SHAPE), 'rate': weight})
    assert ledger.used() == pytest.approx(100 + 400 * weight)
    with pytest.raises(ValueError, match='not dollars'):
        validate_session(dict(config, budget_usd=1.0))
    with pytest.raises(ValueError, match='opened in seconds budget mode'):
        Ledger(tmp_path, dict(DEFAULT, budget_usd=1.0))
