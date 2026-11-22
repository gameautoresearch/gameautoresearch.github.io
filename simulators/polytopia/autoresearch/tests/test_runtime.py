"""The staged worker runtime and the starter it carries (what a release's runtime/ holds)."""
import json
import subprocess
import sys
import time
import types
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
POLICY_CAP, CRITIC_CAP = 500_000, 500_000


def counts(arch):
    """The networks the trainer builds for an architecture: the provider's parameter check, in process."""
    import jax
    from polytopia_bot.rl.gpu_resident import LearnerConfig, critic_template, network_spec_of, policy_template
    spec = network_spec_of(LearnerConfig(), arch)
    size = lambda tree: int(sum(leaf.size for leaf in jax.tree.leaves(tree)))  # noqa: E731
    return dict(policy=size(policy_template(spec)), critic=size(critic_template(spec)))


def test_runtime_is_the_minimal_package_with_the_starter_overlaid(runtime):
    repo, modules = runtime
    import polytopia_bot
    assert Path(polytopia_bot.__file__).is_relative_to(repo)
    assert len(modules) == 37
    package = repo / 'src' / 'polytopia_bot'
    names = {p.stem for p in package.rglob('*.py')} | {p.name for p in package.rglob('*') if p.is_dir()}
    for research in ('distributed', 'vector_actor', 'inference_server', 'search', 'belief', 'learning', 'full_rl',
                     'cpu_arena'):
        assert research not in names, research
    for name in ('src/polytopia_bot/rl/network.py', 'src/polytopia_bot/rl/ppo.py', 'src/polytopia_bot/rl/gpu_resident.py',
                 'src/polytopia_bot/rl/infra.py', 'src/polytopia_bot/rl/data.py', 'src/polytopia_bot/rl/mode.py',
                 'experiments/gpu_resident_train.py'):
        assert (repo / name).read_bytes() == (HERE / 'starter' / name).read_bytes(), name
    assert not [p for p in repo.rglob('*') if 'hidden' in p.parts or p.name.startswith('reference')]


@pytest.mark.parametrize('recipe', ['starter.json', 'smoke.json'])
def test_public_recipes_are_valid_learner_configs(runtime, recipe):
    from dataclasses import fields
    from polytopia_bot.rl.gpu_resident import LearnerConfig
    from simulators.polytopia.autoresearch.v1.core import validate_config
    cfg = validate_config(json.loads((HERE / 'recipes' / recipe).read_text()))
    cfg.pop('_doc')
    assert set(cfg.pop('trainer')) == {'games', 'sets', 'ring'}
    assert set(cfg) <= {f.name for f in fields(LearnerConfig)}
    LearnerConfig(**cfg).validate()


def test_the_trainer_and_the_host_read_one_game_mode(runtime):
    """The staged package's mode module is the file the host loads by path; the trainer's defaults come from it."""
    from polytopia_bot.rl import mode
    from polytopia_bot.rl.gpu_resident import LearnerConfig
    from simulators.polytopia.autoresearch.v1.core import MODE
    names = ('SIZE', 'PLAYERS', 'MAPS', 'ROUND_LIMIT', 'MAP_SEED_LOW', 'MAP_SEED_HIGH', 'MAP_TABLE', 'FROZEN')
    assert {name: getattr(mode, name) for name in names} == {name: getattr(MODE, name) for name in names}
    default = LearnerConfig()
    assert {key: getattr(default, key) for key in mode.FROZEN} == mode.FROZEN
    assert (default.size, default.maps, default.max_turns) == (11, 'drylands', mode.ROUND_LIMIT)
    for bad in (None, 0, 200.0, True):
        with pytest.raises(ValueError, match='max_turns'):
            LearnerConfig(max_turns=bad).validate()


def test_param_counts_match_known_sizes(runtime):
    # On the 11x11 board, observation version 3 (what the trainer builds): 2x32 = 191,587. The policy's parameter
    # embedding is (17, max(size * size, 128), width): 128 targets here, where the 14x14 board had 196 (228,579).
    arch = json.loads((HERE / 'recipes' / 'starter_arch.json').read_text())
    arch.pop('_doc')
    c = counts(arch)
    assert c['policy'] == 191_587 and c['critic'] == c['policy']     # the starter's critic: the same network
    assert counts(dict(width=32, blocks=2, obs_version=1))['policy'] == 179_651
    assert counts(dict(width=64, blocks=2))['policy'] == 469_155
    c = counts(dict(width=64, blocks=2))
    assert c['policy'] < POLICY_CAP and c['critic'] < CRITIC_CAP
    assert counts(dict(width=96, blocks=3))['policy'] > POLICY_CAP


def test_starter_network_rejects_removed_options(runtime):
    for removed in (dict(block_pool=2), dict(attention=1), dict(global_pool=True, extra_blocks=1)):
        with pytest.raises(Exception):
            counts(dict(width=32, blocks=2, **removed))


# ---- a stopped or cut-off run keeps its last network: the trainer's side --------------------------------------------

def loop_trainer(tmp_path, updates, **config):
    """The starter trainer's real loop, checkpoints and exports on CPU: only the CUDA rollout (`_step`) and the PPO
    update are stood in for. Each update adds 1 to every policy weight; `updates(trainer)` runs after each one."""
    import jax
    from polytopia_bot.rl import gpu_resident as G, ppo
    from polytopia_bot.rl.runner import save_checkpoint
    t = object.__new__(G.GpuResidentTrainer)
    t.jax = jax
    t.torch = types.SimpleNamespace(cuda=types.SimpleNamespace(get_device_name=lambda index: 'cpu stand-in',
                                                                max_memory_allocated=lambda: 0))
    t.config = cfg = G.LearnerConfig(**dict(dict(batch=64, minibatch=64, checkpoint_every=1000, snapshot_every=1000,
                                                 training_seconds=120), **config))
    cfg.validate()
    t.root, t.log, t.commit = tmp_path / 'output' / 'run', (lambda line: None), (lambda: None)
    (t.root / 'snapshots').mkdir(parents=True)
    t.G, t.sets, t.ring, t.L, t.device_index, t.ring_bytes, t.round_limit = 8, 1, 4, 8, 0, 0, cfg.max_turns
    t.version, t.gsets, t.clock, t.calls = 0, [], {}, {}
    t.stop_request = str(tmp_path / 'stop-request')
    t._build_network(dict(width=8, blocks=1))
    t.opt = ppo.adam_init((t.params, t.critic))
    t.n_params = t.n_critic = int(sum(x.size for x in jax.tree.leaves(t.params)))
    t.seglog = types.SimpleNamespace(closed_rows=cfg.batch, recorded=0, forced_rows=0)
    t.stats = dict(steps=0, finished=0, terminated=0, truncated=0, drawn=0, nonok=0, maps={})
    t._step = lambda: None

    def update():
        t.params = jax.tree.map(lambda x: x + 1, t.params)
        t.version += 1
        updates(t)
        return dict(policy_loss=.5, value_loss=.25)
    t._update = update
    save_checkpoint(t.root / 'initial.npz', t.params, t.run_config, stage='initial')   # as v1/train.py does
    return t


def collect(run, mode, part='network'):
    """The host's own collection (provider.EXPORT_SCRIPT and provider.unpack) of a run folder, as in the worker."""
    from harness.games.polytopia.provider import EXPORT_SCRIPT, unpack
    logs = [str(run.parent / 'trainer.stdout'), str(run.parent / 'trainer.stderr')]
    done = subprocess.run([sys.executable, '-c', EXPORT_SCRIPT, str(run), mode, part, *logs, '4096', '4096'],
                          capture_output=True)
    return None if done.returncode else unpack(done.stdout)


def policy_shift(files, tmp_path, trainer):
    """How many updates the collected final policy holds (every update added 1 to each weight)."""
    import jax
    import numpy as np
    from polytopia_bot.rl.runner import load_checkpoint
    folder = tmp_path / 'collected'
    folder.mkdir(exist_ok=True)
    for name in ('final.npz', 'final.json', 'initial.npz', 'initial.json'):
        (folder / name).write_bytes(files[name])
    template = jax.tree.map(lambda x: x * 0, trainer.params)
    final, initial = (load_checkpoint(folder / (name + '.npz'), template) for name in ('final', 'initial'))
    shifts = {round(float(np.asarray(a - b).mean()), 3) for a, b in zip(jax.tree.leaves(final), jax.tree.leaves(initial))}
    assert len(shifts) == 1
    return shifts.pop()


def test_trainer_exports_a_whole_policy_early_and_keeps_it_current(runtime, tmp_path):
    seen = []
    t = loop_trainer(tmp_path, lambda t: seen.append(sorted(p.name for p in (t.root / 'exports').glob('*'))
                                                     if (t.root / 'exports').exists() else []),
                     training_updates=5, export_seconds=0.)
    t.run()
    exports = t.root / 'exports'
    assert sorted(p.name for p in exports.iterdir()) == ['update-000000004', 'update-000000005']   # the newest two
    assert all(name.startswith('update-') or name.startswith('writing-') for names in seen for name in names)
    for folder in exports.iterdir():
        assert sorted(p.name for p in folder.iterdir()) == sorted(['final.npz', 'final.json', 'final-critic.npz',
                                                                    'final-critic.json', 'export.json'])
    record = json.loads((exports / 'update-000000005/export.json').read_text())
    assert (record['update'], record['samples']) == (5, 5 * 64)
    done = json.loads((t.root / 'training-complete.json').read_text())
    assert (done['updates'], done['stopped']) == (5, False)
    files = collect(t.root, 'final')
    assert 'training-complete.json' in files and 'export.json' not in files and policy_shift(files, tmp_path, t) == 5.0
    assert len(files['training.jsonl'].splitlines()) == 5


def test_a_trainer_that_ends_at_its_time_limit_leaves_a_run_the_host_collects(runtime, tmp_path):
    """The ending every failed final of 2026-11-13 took (the loop stops because its time is up, not by an update
    count or a stop request), with periodic exports, both clock snapshots and a checkpoint on the way: the trainer's
    own files are whole and the host's collection of them succeeds. The failure was not in what the trainer leaves."""
    def updates(t):
        time.sleep(0.02)
        for minutes in (5, 10):          # v1/train.py's clock snapshots, written from the loop thread
            if t.version == minutes and not (t.root / f'clock-{minutes}m.npz').exists():
                t._write(f'clock-{minutes}m', t._host_state())
                (t.root / f'clock-{minutes}m-receipt.json').write_text(json.dumps(dict(requested_minutes=minutes)))
    t = loop_trainer(tmp_path, updates, export_seconds=0.05, checkpoint_every=7, snapshot_every=11,
                     training_seconds=1)       # what `--seconds` sets
    t.run()
    done = json.loads((t.root / 'training-complete.json').read_text())
    assert done['stopped'] is False and done['updates'] == t.version > 10 and done['seconds'] >= 1
    files = collect(t.root, 'final')
    assert files is not None and 'training-complete.json' in files and 'export.json' not in files
    assert policy_shift(files, tmp_path, t) == float(t.version)         # the final network, not an older export
    assert len(files['training.jsonl'].splitlines()) == t.version
    assert json.loads(files['extras.json']).keys() == {p.name for p in t.root.glob('clock-*')} and len(json.loads(files['extras.json'])) == 10
    extras = collect(t.root, 'final', 'extras')
    assert set(extras) == set(json.loads(files['extras.json'])) | {'export-checksums.json'}
    # the same folder read as a cut-off run would be: its own end still wins over the periodic exports
    assert 'training-complete.json' in collect(t.root, 'early')


def test_trainer_exports_after_its_first_update_and_then_every_export_seconds(runtime, tmp_path):
    t = loop_trainer(tmp_path, lambda t: None, training_updates=4, export_seconds=3600.)
    t.run()
    assert [p.name for p in (t.root / 'exports').iterdir()] == ['update-000000001']   # a usable policy early


def test_a_stop_request_ends_the_loop_at_its_next_step_with_a_final_checkpoint(runtime, tmp_path):
    def updates(t):
        if t.version == 3:
            Path(t.stop_request).write_text('stop\n')   # what the host writes into the worker on `ar stop`
    t = loop_trainer(tmp_path, updates, export_seconds=3600.)
    t.run()
    done = json.loads((t.root / 'training-complete.json').read_text())
    assert (t.version, done['updates'], done['stopped'], t.result['stopped']) == (3, 3, True, True)
    files = collect(t.root, 'final')
    assert policy_shift(files, tmp_path, t) == 3.0 and json.loads(files['training-complete.json'])['stopped'] is True


def test_a_trainer_ended_mid_run_is_collected_from_its_newest_whole_export(runtime, tmp_path):
    """Cut off at its time limit: the process dies between updates, with no final checkpoint."""
    def updates(t):
        if t.version == 4:
            t.ckpt.wait()               # update 3's export has landed
            raise SystemExit('killed')  # the provider ends the command
    t = loop_trainer(tmp_path, updates, export_seconds=0.)
    with pytest.raises(SystemExit):
        t.run()
    assert not (t.root / 'training-complete.json').exists() and not (t.root / 'final.npz').exists()
    assert collect(t.root, 'final') is None          # not a finished run
    files = collect(t.root, 'early')
    assert json.loads(files['export.json'])['update'] == 3 and policy_shift(files, tmp_path, t) == 3.0
    assert 'training-complete.json' not in files
    assert [json.loads(line)['update'] for line in files['training.jsonl'].splitlines()] == [1, 2, 3]
    sums = json.loads(files['export-checksums.json'])
    import hashlib
    assert all(hashlib.sha256(files[name]).hexdigest() == value for name, value in sums.items())


def test_an_export_is_never_visible_half_written(runtime, tmp_path, monkeypatch):
    from polytopia_bot.rl import runner
    t = loop_trainer(tmp_path, lambda t: None)
    state = t._host_state()
    t._export(state, dict(update=1, samples=64, elapsed=1.))
    real = runner.save_checkpoint

    def dies_on_the_critic(path, *args, **kwargs):
        if 'critic' in Path(path).name:
            raise OSError('the process ended here')
        return real(path, *args, **kwargs)
    monkeypatch.setattr(runner, 'save_checkpoint', dies_on_the_critic)
    with pytest.raises(OSError):
        t._export(state, dict(update=2, samples=128, elapsed=2.))
    assert sorted(p.name for p in (t.root / 'exports').iterdir()) == ['update-000000001', 'writing-2']
    (t.root / 'protocol.json').write_text('{}')
    (t.root / 'training.jsonl').write_text(json.dumps(dict(update=1, policy_loss=.5, value_loss=.25)) + '\n')
    assert json.loads(collect(t.root, 'early')['export.json'])['update'] == 1   # the half-written one is not taken


def test_a_run_with_no_whole_export_has_nothing_to_collect(runtime, tmp_path):
    t = loop_trainer(tmp_path, lambda t: None)
    (t.root / 'protocol.json').write_text('{}')
    (t.root / 'training.jsonl').write_text('')
    assert collect(t.root, 'early') is None


# ---- training and evaluation use one outcome rule -------------------------------------------------------------------

def test_game_outcomes_is_the_evaluators_rule(runtime):
    """Elimination keeps the engine's reward; a capped game goes to the higher in-game score; equal scores draw."""
    import numpy as np
    from polytopia_bot.rl.infra import game_outcomes
    terminated = np.array([0, 0, 0, 1, 1], bool)
    truncated = np.array([1, 1, 1, 0, 0], bool)
    engine = np.array([[0, 0], [0, 0], [0, 0], [1, -1], [-1, 1]], np.float32)      # the engine's terminal reward
    scores = np.array([[3000, 2500], [2500, 3000], [2750, 2750], [100, 9000], [9000, 100]], np.int32)
    out = game_outcomes(terminated, truncated, engine, scores)
    assert out.dtype == np.float32 and out.tolist() == [[1, -1], [-1, 1], [0, 0], [1, -1], [-1, 1]]
    # The grader's rule as it stood, written out: for every seat of every case the function gives the same number.
    rng = np.random.default_rng(0)
    for _ in range(500):
        score = rng.integers(0, 4, 2) * 100
        capped = bool(rng.integers(2))
        reward = np.array([1., -1.] if rng.integers(2) else [-1., 1.], np.float32)
        for seat in (1, 2):
            expected = float(reward[seat - 1])
            if capped:
                expected = float(np.sign(score[seat - 1] - score[2 - seat]))
            assert float(game_outcomes([not capped], [capped], reward, score)[0, seat - 1]) == expected
    # One definition, called by both; neither keeps a rule of its own.
    grade = (HERE / 'v1' / 'grade.py').read_text()
    trainer = (HERE / 'starter/src/polytopia_bot/rl/gpu_resident.py').read_text()
    assert 'game_outcomes(' in grade and 'np.sign' not in grade
    assert "game_outcomes(term, trunc, host['rewards'][d], host['scores'][d])" in trainer
    for gone in ('np.sign', 'boot_store', 'BOOT_STORE', '_store_final_views', 'bootstraps from the critic'):
        assert gone not in trainer, gone


def test_trainer_scores_a_finished_game_as_the_evaluator_does(runtime):
    """The trainer's own outcome path (`_outcomes` -> SegmentLog -> GAE) on one engine tick's staged results: capped
    games with unequal scores give +1 / -1 to the right seats, equal scores 0, an elimination the engine's reward."""
    import numpy as np
    from polytopia_bot.rl import gpu_resident as G
    games = 6
    t = object.__new__(G.GpuResidentTrainer)
    t.Gs, t.map_lo, t.map_types = games, 200000, ['drylands']
    t.map_kind, t.playing_kind = np.zeros(1, np.int8), np.zeros(games, np.int8)
    t.stats = dict(steps=0, finished=0, terminated=0, truncated=0, drawn=0, nonok=0, maps={})
    t.seglog = log = G.SegmentLog(games, 2, 4, 100)
    every = np.arange(games)
    for step, seat in enumerate((0, 1, 0)):         # seat 1 decides, seat 2 decides, seat 1 decides: 3 rows a game
        log.record(step, every, np.full(games, seat, np.int32))
    #            capped,  capped,  capped, eliminated, eliminated, still playing
    done = np.array([1, 1, 1, 1, 1, 0], np.uint8)
    terminated = np.array([0, 0, 0, 1, 1, 0], np.uint8)
    truncated = np.array([1, 1, 1, 0, 0, 0], np.uint8)
    scores = np.array([[3000, 2500], [2500, 3000], [2750, 2750], [100, 9000], [9000, 100], [1, 2]], np.int32)
    rewards = np.array([[0, 0], [0, 0], [0, 0], [1, -1], [-1, 1], [0, 0]], np.float32)   # the engine's: eliminations only
    gs = types.SimpleNamespace(stage_event=types.SimpleNamespace(synchronize=lambda: None), games=every,
                               seeds_host=np.full(games, 200000, np.int32),
                               stage_out=dict(scores=scores, rewards=rewards, status=np.zeros(games, np.int32),
                                              flags=np.stack([done, terminated, truncated])))
    t._outcomes(gs)
    assert (t.stats['finished'], t.stats['terminated'], t.stats['truncated'], t.stats['drawn']) == (5, 2, 3, 1)
    closed = {key: np.concatenate([event[key] for event in log.events]) for key in log.KEYS}
    assert closed['terminal'].all()                  # every finished game is an outcome: nothing bootstraps
    outcome = {(int(g), int(s)): float(r) for g, s, r in zip(closed['game'], closed['seat'], closed['reward'])}
    assert outcome == {(0, 0): 1, (0, 1): -1,       # capped, seat 1 has the higher score
                       (1, 0): -1, (1, 1): 1,       # capped, seat 2 has the higher score
                       (2, 0): 0, (2, 1): 0,        # capped, equal scores: a draw
                       (3, 0): 1, (3, 1): -1,       # eliminated: the engine's reward, whatever the scores say
                       (4, 0): -1, (4, 1): 1}
    assert log.count[5].tolist() == [2, 1] and log.count[:5].sum() == 0      # the unfinished game stays open
    # Through the batch assembly's bookkeeping: with a zero critic, each row's return is its seat's game result.
    take = log.take(15, 2, 100)
    stale, ring, call = G.value_sources(take, np.zeros(100, np.int64), 0)
    assert not ring.any() and not call.any()        # no boundary is valued for a finished game
    segments = take['segments']
    _, returns = G.segment_gae(np.zeros(take['n'], np.float32), take['seg'], take['pos'], segments['length'],
                               segments['terminal'], segments['reward'], np.zeros(len(segments['length']), np.float32),
                               1., 1.)
    for row in range(take['n']):
        seg = take['seg'][row]
        assert returns[row] == outcome[(int(segments['game'][seg]), int(segments['seat'][seg]))]
    assert take['n'] == 15 and sorted(set(returns.tolist())) == [-1.0, 0.0, 1.0]
