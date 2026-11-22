"""Native fixed8 checkpoint evaluation and exact public-action replay.

No JVM is started or Java transition implementation called. The optional pure
Python opponent is candidates/crforge/python/crforge_env/scripted.py, with facts
read by the same formulas as vector.py:_SpellReader/unit_facts, DirectHit.java,
and combat/LevelScaling.java at f0659dafabfb9d2a4a8d3b22a9d14de306f5e4de
(Apache-2.0). All schema/table inputs come from validated frozen native exports.

Evaluation is public practice, never a private/held-out panel. A source-bound
checkpoint is transferred as weights, not resumed: its original build identity
is verified, while the selected evaluation backend gets its own build receipt.
Each episode owns independent policy RNGs, so changing episode order does not
change its actions. CUDA evaluation may copy public outputs to CPU for bots and
evidence; simulator transitions still execute solely in the native CUDA engine.
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import argparse
import hashlib
import importlib
import json
import sys
import time

HERE = Path(__file__).resolve().parent
INTEGRATION = HERE.parent
ROOT = INTEGRATION.parent.parent
DATA = HERE / 'generated-v2/task-data.json'
PRACTICE = INTEGRATION / 'configs/eval-practice.json'
PIN = 'f0659dafabfb9d2a4a8d3b22a9d14de306f5e4de'
FORMAT = 'clash-native-evaluation-v1'
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(INTEGRATION))


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest_json(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def no_jvm():
    import jpype
    if jpype.isJVMStarted():
        raise RuntimeError('native evaluation requires a process with no started JVM')


class StaticSchema(SimpleNamespace):
    """Frozen public layout; ScriptedBot never receives a world or table reader."""

    def __init__(self, data):
        s = data['metadata']['schema']
        super().__init__(**(s | {k: data['schema'][k] for k in ('header', 'entity', 'info', 'flags', 'kinds')}))
        self.width_units = s['scales']['width_units']
        self.height_units = s['scales']['height_units']
        # env/src/main/java/org/crforge/env/Schema.java at PIN.
        self.unknown, self.wait = -1, 0
        self.winner_none, self.winner_draw = -1, 2

    def field(self, observation, name):
        import numpy as np
        indices = self.header[name]
        if len(indices) == 1:
            value = observation[..., indices[0]]
            return int(value) if np.ndim(value) == 0 else value
        return observation[..., list(indices)]

    def entities(self, observation):
        count = int(self.field(observation, 'entity_count'))
        if not 0 <= count <= self.max_entities:
            raise ValueError('public observation entity count exceeds frozen schema')
        return observation[self.header_ints:].reshape(self.max_entities, self.entity_ints)[:count]

    def play(self, slot, col, row):
        if not (0 <= slot < self.hand_slots and 0 <= col < self.cols and 0 <= row < self.rows):
            raise ValueError('play outside public action grid')
        return 1 + slot * self.cols * self.rows + row * self.cols + col


def pure_opponent_modules():
    """Verify the exact pinned files before importing any pure opponent code."""
    no_jvm()
    vendored = ROOT / 'candidates/crforge'
    lock = json.loads((INTEGRATION / 'rules.lock.json').read_text())
    if lock['engine_commit'] != PIN:
        raise ValueError('scripted policy upstream pin differs')
    hashes = {}
    # __init__ imports the other public wrappers; none starts Java at import.
    for path in sorted((vendored / 'python/crforge_env').glob('*.py')):
        relative = path.relative_to(vendored).as_posix()
        actual = sha(path)
        if lock['engine_source_sha256'].get(relative) != actual:
            raise ValueError('pinned pure Python policy source differs: ' + relative)
        hashes[relative] = actual
    sys.path.insert(0, str(vendored / 'python'))
    scripted = importlib.import_module('crforge_env.scripted')
    vector = importlib.import_module('crforge_env.vector')
    for module, name in ((scripted, 'scripted'), (vector, 'vector')):
        if Path(module.__file__).resolve() != (vendored / f'python/crforge_env/{name}.py').resolve():
            raise ValueError('unexpected installed pure Python opponent module')
    no_jvm()
    return scripted.ScriptedBot, vector.SpellFacts, vector.UnitFacts, hashes


def frozen_facts(data):
    """Read only supported fixed8 rows, fail closed on newly reachable mechanics."""
    from data_codegen import scaling
    _, SpellFacts, UnitFacts, hashes = pure_opponent_modules()
    if data['rules']['level'] != 11 or data['rules']['tower_level'] != 11:
        raise ValueError('frozen level-11 facts required')
    vocab = {name: i for i, name in enumerate(data['vocabulary'])}
    types, spells, units = {}, {}, {}
    modes = {'NONE': 0, 'CARD_DAMAGE': 1, 'KING_DAMAGE': 3, 'TOWER_DAMAGE': 5}

    def crown(percent, damage):
        return ((max(percent, -100) + 100) * damage + 99) // 100

    def projectile(name):
        row = data['projectiles'][name]
        if row['spawnCharacter'] or row['spawnProjectile']:
            raise ValueError('unsupported frozen spell spawn/roll')
        damage = scaling(row['damage'], row, modes[row['damageMode']])
        return row, damage, crown(row['crownTowerDamagePercent'], damage)

    for name in data['rules']['deck']:
        card = data['cards'][name]
        unit = card['unit']
        if unit is not None:
            types[vocab[name]] = 'building' if unit['building'] else 'troop'
            units[vocab[name]] = UnitFacts(vocab[unit['name']], unit['range'],
                                          unit['targetOnlyBuildings'], unit['building'],
                                          scaling(unit['hitpoints'], unit, 2))
        else:
            types[vocab[name]] = 'spell'
            if card['projectile']:
                row, damage, tower = projectile(card['projectile'])
                waves = max(card['projectileWaves'], 1)
                fact = SpellFacts(card['radius'] or row['radius'], damage * waves,
                                  tower * waves, row['aoeToAir'], row['aoeToGround'])
            elif card['areaEffect']:
                row = data['areas'][card['areaEffect']]
                hits = max(row['lifeDurationMs'] // row['hitSpeedMs'] if row['hitSpeedMs'] > 0 else 1, 1)
                maximum = 0
                if row['projectile']:
                    _, damage, tower = projectile(row['projectile'])
                    maximum = max(row['maximumTargets'], 0) * hits
                elif row['typedDamage']:
                    damage = scaling(row['typedDamage']['baseDamage'], row, 1) * hits
                    tower = scaling(row['typedDamage']['towerDamage'], row, 1) * hits
                else:
                    damage = scaling(row['damage'], row, 1) * hits
                    tower = crown(row['crownTowerDamagePercent'], damage)
                fact = SpellFacts(card['radius'] or row['radius'], damage, tower, max_targets=maximum)
            else:
                raise ValueError('unsupported frozen spell without effect')
            if fact.radius > 0:
                spells[vocab[name]] = fact
    receipt = {'upstream_commit': PIN, 'source_sha256': hashes,
               'card_types': types, 'spell_facts': {k: asdict(v) for k, v in spells.items()},
               'unit_facts': {k: asdict(v) for k, v in units.items()}}
    return types, spells, units, receipt


def scripted_bot(data, facts=None):
    ScriptedBot, _, _, _ = pure_opponent_modules()
    types, spells, units, _ = frozen_facts(data) if facts is None else facts
    adapter = SimpleNamespace(schema=StaticSchema(data), vocabulary_size=len(data['vocabulary']))
    return ScriptedBot(types, spells, units).bind(adapter)


def _validate_original_build(saved, verified):
    """Validate a weight-transfer receipt without requiring its original hardware."""
    from build_receipt import identity_sha256
    build = saved.get('backend_build', {})
    identity = build.get('identity', {})
    cuda = saved['config']['backend'] == 'native-cuda'
    expected_backend = 'native-cuda' if cuda else 'native-cpu'
    if identity.get('backend') != expected_backend or identity.get('data_binding') != verified:
        raise ValueError('checkpoint original backend/data build differs')
    if build.get('identity_sha256') != identity_sha256(identity):
        raise ValueError('checkpoint original build identity checksum differs')
    if not isinstance(build.get('library_sha256'), str) or len(build['library_sha256']) != 64:
        raise ValueError('checkpoint original binary identity missing')
    paths = sorted(p for p in HERE.rglob('*') if p.is_file() and '.build' not in p.parts and
                   ((p.suffix in ('.hpp', '.cuh') or p.name == 'cuda_engine.cu') if cuda else p.suffix in ('.hpp', '.cpp')))
    sources = {p.relative_to(HERE).as_posix(): sha(p) for p in paths}
    if identity.get('source_sha256') != sources:
        raise ValueError('checkpoint original build sources differ')
    runtime = saved.get('execution_runtime', {})
    if not all(key in runtime for key in ('torch', 'numpy', 'python', 'cuda', 'gpu')):
        raise ValueError('checkpoint original execution runtime is missing')
    # A different hardware/runtime is allowed only because this is evaluation,
    # never an optimizer/world/RNG resume. Keep the original receipt verbatim.
    return build


def load_checkpoint(path, data_path=DATA, device='cpu'):
    import torch
    from data_binding import validate_data
    from model import Policy
    from train import binding
    no_jvm()
    verified = validate_data()
    if sha(data_path) != verified['task_data_sha256']:
        raise ValueError('evaluation data differs from compiled data')
    data = json.loads(Path(data_path).read_text())
    saved = torch.load(path, map_location='cpu', weights_only=False)
    if saved.get('format') != 'clash-native-ppo-v2' or saved.get('binding') != binding(data_path):
        raise ValueError('native checkpoint source/data/rules binding differs')
    c = saved['config']
    if c.get('backend') not in ('native-cpu', 'native-cpu-cuda', 'native-cuda'):
        raise ValueError('checkpoint native backend is invalid')
    import match_forms
    form_rules = dict(data['rules'], decision_ticks=match_forms.variant(c.get('match_form', 'full'))['decision_ticks'])
    if c.get('task', data['task']) != data['task'] or c.get('rules', form_rules) != form_rules or saved.get('metadata') != data['metadata']:
        raise ValueError('checkpoint task/rules/schema metadata differs')
    for key in ('width', 'embedding'):
        if type(c.get(key)) is not int or c[key] < 1:
            raise ValueError('checkpoint model configuration differs: ' + key)
    _validate_original_build(saved, verified)
    # Keep model construction from perturbing the caller's random stream.
    with torch.random.fork_rng(devices=[]):
        policy = Policy(StaticSchema(data), len(data['vocabulary']), len(data['buff_names']), c['width'], c['embedding'])
    expected = policy.state_dict()
    weights = saved['model']
    if expected.keys() != weights.keys():
        raise ValueError('checkpoint model state keys differ')
    for key, value in expected.items():
        actual = weights[key]
        if not isinstance(actual, torch.Tensor) or actual.shape != value.shape or actual.dtype != value.dtype:
            raise ValueError('checkpoint model shape/dtype differs: ' + key)
        if not torch.isfinite(actual).all():
            raise ValueError('checkpoint nonfinite model state: ' + key)
        if key in ('global_scale', 'entity_scale', 'flag_bits') and not torch.equal(actual, value):
            raise ValueError('checkpoint frozen model schema buffer differs: ' + key)
    policy.load_state_dict(weights, strict=True)
    policy.to(device).eval()
    policy.device_indices = True
    model_digest = hashlib.sha256()
    for key, value in sorted(weights.items()):
        model_digest.update(key.encode()); model_digest.update(str(value.dtype).encode())
        model_digest.update(canonical(list(value.shape)).encode())
        model_digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    receipt = {'path': str(Path(path).resolve()), 'sha256': sha(path), 'model_sha256': model_digest.hexdigest(),
               'config': c, 'update': saved['update'], 'samples': saved['samples'],
               'original_backend_build': saved['backend_build'], 'original_execution_runtime': saved['execution_runtime'],
               'original_auxiliary_builds': saved.get('auxiliary_builds', {}),
               'transfer': 'weights only; no optimizer, simulator or training RNG restoration'}
    return policy, data, receipt


def cpu(value):
    import numpy as np
    import torch
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.array(value, copy=True)


def public_outputs(env):
    return {key: cpu(getattr(env, key)) for key in ('observations', 'masks', 'infos', 'rewards')}


def public_digest(outputs):
    """Hash exact canonical bytes, dtype and shape of all public engine outputs."""
    h = hashlib.sha256()
    for key in ('observations', 'masks', 'infos', 'rewards'):
        value = outputs[key]
        h.update(canonical([key, value.dtype.str, list(value.shape)]).encode())
        h.update(value.astype(value.dtype.newbyteorder('<'), copy=False).tobytes(order='C'))
    return h.hexdigest()


def policy_seed(base, engine_seed, side, role, opponent):
    key = [int(base), int(engine_seed), int(side), role, opponent]
    return int.from_bytes(hashlib.sha256(canonical(key).encode()).digest()[:8], 'little') & ((1 << 63) - 1)


class PublicActor:
    """Only a single seat observation and legal mask cross the policy boundary."""

    def __init__(self, kind, seed, *, model=None, data=None, greedy=False):
        import numpy as np
        import torch
        self.kind, self.seed = kind, int(seed)
        self.rng = np.random.default_rng(seed)
        self.model, self.greedy = model, greedy
        self.bot = scripted_bot(data) if kind == 'scripted' else None
        self.generator = None
        if kind == 'checkpoint':
            if model is None:
                raise ValueError('checkpoint actor needs a loaded policy')
            self.device = next(model.parameters()).device
            self.generator = torch.Generator(device=self.device).manual_seed(seed)
        elif kind not in ('wait', 'random', 'scripted'):
            raise ValueError('unsupported public opponent')

    def __call__(self, observation, mask):
        import numpy as np
        import torch
        if self.kind == 'wait':
            return 0
        if self.kind == 'random':
            return int(self.rng.choice(np.flatnonzero(mask)))
        if self.kind == 'scripted':
            return self.bot(observation, mask, self.rng)
        with torch.inference_mode():
            # An owned public snapshot prevents a subsequent engine read from
            # changing the actor's input. Full capacity preserves every row.
            batch = self.model.pack(observation[None], mask[None], device=self.device, trim=False)
            logits, _ = self.model(batch)
            if not torch.isfinite(logits).all():
                raise ValueError('nonfinite evaluation policy logits')
            if self.greedy:
                return int(logits.argmax(-1).cpu()[0])
            probabilities = torch.softmax(logits, -1)
            return int(torch.multinomial(probabilities, 1, generator=self.generator).cpu()[0, 0])

    def state(self):
        return {'kind': self.kind, 'seed': self.seed, 'numpy': self.rng.bit_generator.state,
                'torch_state_sha256': None if self.generator is None else hashlib.sha256(self.generator.get_state().cpu().numpy().tobytes()).hexdigest()}


def classify(info, schema, *, decision_limit=False, exception=None, illegal_attempts=0, overflow=False):
    """Natural results, truncation and invalid transitions remain separate."""
    fields = {name: int(info[index]) for name, index in schema.info.items()}
    winner = fields['winner']
    fault = fields['fault'] != 0 or exception is not None
    illegal = fields['illegal_0'] + fields['illegal_1']
    truncated = bool(fields['truncated'] or decision_limit)
    natural = winner in (0, 1, 2) and not fault and not truncated
    valid = natural and not (illegal or illegal_attempts or fields['not_modelled'] or overflow)
    return {'info': fields, 'winner': winner, 'natural': natural, 'draw': natural and winner == 2,
            'truncated': truncated, 'engine_truncated': bool(fields['truncated']),
            'decision_limit': bool(decision_limit), 'illegal': illegal, 'illegal_attempts': illegal_attempts,
            'not_modelled': fields['not_modelled'], 'entity_overflow': bool(overflow),
            'simulator_fault': fields['fault'], 'exception': exception, 'valid': valid}


def run_episode(env, actors, schema, *, max_decisions=0, max_ticks=0):
    """Run one already-reset native lane. No policy receives infos or rewards."""
    output = public_outputs(env)
    trace = {'initial_public_digest': public_digest(output), 'initial_rng': [actor.state() for actor in actors], 'steps': []}
    illegal_attempts = 0
    overflow = False
    exception = None
    while True:
        info = output['infos'][0]
        if info[schema.info['winner']] >= 0 or info[schema.info['truncated']] or info[schema.info['fault']]:
            break
        if max_decisions and len(trace['steps']) >= max_decisions:
            break
        actions = []
        try:
            for side, actor in enumerate(actors):
                observation, mask = output['observations'][0, side], output['masks'][0, side]
                action = actor(observation.copy(), mask.copy())
                if not isinstance(action, int):
                    raise ValueError('policy action must be an integer')
                if not -(1 << 31) <= action < (1 << 31):
                    raise ValueError('policy action exceeds native int32 ABI')
                illegal_attempts += int(not (0 <= action < schema.actions and mask[action] != 0))
                actions.append(action)
        except Exception as exc:
            exception = {'phase': 'policy', 'type': type(exc).__name__, 'message': str(exc)}
            break
        error = None
        try:
            env.step([actions], max_ticks)
        except Exception as exc:
            error = {'phase': 'engine', 'type': type(exc).__name__, 'message': str(exc)}
        output = public_outputs(env)
        overflow |= bool((output['observations'][..., schema.header['entity_overflow'][0]] != 0).any())
        trace['steps'].append({'actions': actions, 'public_digest': public_digest(output), 'exception': error})
        if error is not None:
            exception = error
            break
    limit = bool(max_decisions and len(trace['steps']) >= max_decisions and
                 output['infos'][0, schema.info['winner']] < 0 and not output['infos'][0, schema.info['truncated']] and exception is None)
    trace['final_rng'] = [actor.state() for actor in actors]
    trace['final_public_digest'] = public_digest(output)
    trace['outcome'] = classify(output['infos'][0], schema, decision_limit=limit, exception=exception,
                                illegal_attempts=illegal_attempts, overflow=overflow)
    return trace


def read_schedule(path):
    schedule = json.loads(Path(path).read_text())
    frozen = json.loads(PRACTICE.read_text())
    # Runtime accepts subsets of the frozen PUBLIC schedule, never arbitrary
    # held-out/evaluation seeds. Config keys are explicit and closed.
    if set(schedule) != set(frozen) or schedule['visibility'] != 'public practice' or schedule['policy_seed'] != frozen['policy_seed']:
        raise ValueError('only the frozen public practice schedule is supported')
    for key in ('engine_seeds', 'opponents'):
        if not schedule[key] or len(set(schedule[key])) != len(schedule[key]) or any(value not in frozen[key] for value in schedule[key]):
            raise ValueError('schedule is not a public practice subset: ' + key)
    return schedule


def summarize(games):
    outcomes = [game['outcome'] for game in games]
    wins = sum(v['natural'] and v['winner'] == game['learner_side'] for game, v in zip(games, outcomes))
    draws = sum(v['draw'] for v in outcomes)
    losses = sum(v['natural'] and not v['draw'] and v['winner'] != game['learner_side'] for game, v in zip(games, outcomes))
    valid = bool(games) and all(v['valid'] for v in outcomes)
    return {'games': len(games), 'natural': sum(v['natural'] for v in outcomes), 'wins': wins, 'losses': losses,
            'draws': draws, 'truncations': sum(v['truncated'] for v in outcomes),
            'illegal_actions': sum(v['illegal'] for v in outcomes), 'illegal_attempts': sum(v['illegal_attempts'] for v in outcomes),
            'not_modelled': sum(v['not_modelled'] for v in outcomes),
            'simulator_faults': sum(bool(v['simulator_fault']) for v in outcomes),
            'exceptions': sum(v['exception'] is not None for v in outcomes), 'entity_overflows': sum(v['entity_overflow'] for v in outcomes),
            'valid': valid, 'official_score': (wins + .5 * draws) / len(games) if valid else None,
            'score_scope': 'public practice only'}


def evaluate_checkpoints(checkpoints, out, *, backend='native-cpu', schedule=PRACTICE,
                         data_path=DATA, max_decisions=0, max_ticks=0, greedy=False,
                         cuda_launch_threads=None, cuda_render_mode=None):
    import numpy as np
    import torch
    from build_receipt import backend_build_binding
    from data_binding import validate_data
    from train import archive_owned_source, binding
    no_jvm()
    if backend not in ('native-cpu', 'native-cuda') or max_decisions < 0 or max_ticks < 0:
        raise ValueError('invalid native evaluation backend/limits')
    if not checkpoints:
        raise ValueError('at least one source-bound checkpoint is required')
    if backend != 'native-cuda' and (cuda_launch_threads is not None or cuda_render_mode is not None):
        raise ValueError('CUDA execution settings require the native-cuda evaluation backend')
    practice = read_schedule(schedule)
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    bound = binding(data_path)
    device = 'cuda' if backend == 'native-cuda' else 'cpu'
    if device == 'cuda' and torch.cuda.device_count() != 1:
        raise ValueError('native CUDA evaluation requires exactly one visible GPU')
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    if device == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = False
    manifest = {'format': FORMAT, 'binding': bound, 'data_binding': validate_data(),
                'config': {'backend': backend, 'schedule': practice, 'max_decisions': max_decisions,
                           'max_ticks': max_ticks, 'greedy': bool(greedy)},
                'schedule_source': {'path': str(Path(schedule).resolve()), 'sha256': sha(schedule)},
                'runtime': {'python': list(sys.version_info[:3]), 'torch': torch.__version__, 'numpy': np.__version__,
                            'cuda': torch.version.cuda if device == 'cuda' else None,
                            'gpu': torch.cuda.get_device_name() if device == 'cuda' else None,
                            'neural_precision': 'fp32', 'cuda_matmul_allow_tf32': False if device == 'cuda' else None},
                'checkpoints': [], 'games': []}
    for checkpoint_index, path in enumerate(checkpoints):
        model, data, receipt = load_checkpoint(path, data_path, device)
        # Every checkpoint of one evaluation plays the match form it was trained on.
        form = torch.load(path, map_location='cpu', weights_only=False)['config'].get('match_form', 'full')
        if manifest['config'].setdefault('match_form', form) != form:
            raise ValueError('checkpoints of one evaluation were trained on different match forms')
        facts = frozen_facts(data)
        manifest['pure_opponent'] = facts[3]
        manifest['checkpoints'].append(receipt)
        schema = StaticSchema(data)
        if backend == 'native-cuda':
            from runtime import NativeCUDA, build_cuda
            library = build_cuda(form)
            def factory(seeds):
                return NativeCUDA(seeds, launch_threads=cuda_launch_threads, render_mode=cuda_render_mode, form=form)
        else:
            from runtime import NativeCPU
            from build import build
            library = build(form)
            def factory(seeds):
                return NativeCPU(seeds, form=form)
        manifest['evaluation_backend_build'] = backend_build_binding(library)
        for opponent in practice['opponents']:
            for seed in practice['engine_seeds']:
                for side in (0, 1):
                    actors = [None, None]
                    actors[side] = PublicActor('checkpoint', policy_seed(practice['policy_seed'], seed, side, 'learner', opponent), model=model, greedy=greedy)
                    actors[1 - side] = PublicActor(opponent, policy_seed(practice['policy_seed'], seed, 1 - side, 'opponent', opponent), data=data)
                    env = factory([seed])
                    try:
                        if backend == 'native-cuda':
                            engine_policy = {'cuda_launch_threads': env.launch_threads, 'cuda_render_mode': env.render_mode}
                            if manifest['runtime'].get('engine_policy', engine_policy) != engine_policy:
                                raise ValueError('native CUDA evaluation engine policy changed between episodes')
                            manifest['runtime']['engine_policy'] = engine_policy
                            manifest['config'].update(engine_policy)
                        tape = run_episode(env, actors, schema, max_decisions=max_decisions, max_ticks=max_ticks)
                    finally:
                        env.close()
                    game = {'checkpoint_index': checkpoint_index, 'opponent': opponent, 'engine_seed': seed,
                            'learner_side': side, 'outcome': tape['outcome'], 'decisions': len(tape['steps']),
                            'final_public_digest': tape['final_public_digest']}
                    filename = f'game-{len(manifest["games"]):05d}.json'
                    tape.update({key: game[key] for key in ('checkpoint_index', 'opponent', 'engine_seed', 'learner_side')})
                    (out / filename).write_text(json.dumps(tape, indent=2, sort_keys=True) + '\n')
                    manifest['games'].append(game | {'tape': filename, 'tape_sha256': sha(out / filename)})
        no_jvm()
    if binding(data_path) != bound:
        raise ValueError('native source changed during evaluation')
    manifest['summary'] = summarize(manifest['games'])
    manifest['by_checkpoint'] = [summarize([g for g in manifest['games'] if g['checkpoint_index'] == i]) for i in range(len(checkpoints))]
    manifest['wall_seconds'] = time.monotonic() - start
    manifest['jvm_started'] = False
    manifest.update(archive_owned_source(out, bound['source']))
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    return manifest


def replay_evaluation(manifest_path, out, *, data_path=DATA):
    """Replay saved actions in NativeCPU, checking EVERY public transition."""
    from data_binding import validate_data
    from train import binding
    from runtime import NativeCPU
    from build import build
    from build_receipt import backend_build_binding
    no_jvm()
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('format') != FORMAT or manifest.get('binding') != binding(data_path) or manifest.get('data_binding') != validate_data():
        raise ValueError('native evaluation replay source/data/rules binding differs')
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    count = 0
    for game in manifest['games']:
        relative = Path(game['tape'])
        if relative.is_absolute() or len(relative.parts) != 1:
            raise ValueError('unsafe action tape path')
        tape_path = manifest_path.parent / relative
        if sha(tape_path) != game['tape_sha256']:
            raise ValueError('action tape checksum differs')
        tape = json.loads(tape_path.read_text())
        for key in ('checkpoint_index', 'opponent', 'engine_seed', 'learner_side', 'outcome', 'final_public_digest'):
            if tape[key] != game[key]:
                raise ValueError('action tape/manifest metadata differs: ' + key)
        env = NativeCPU([game['engine_seed']], form=manifest['config'].get('match_form', 'full'))
        try:
            if public_digest(public_outputs(env)) != tape['initial_public_digest']:
                raise ValueError('native CPU replay initial public digest differs')
            for index, frame in enumerate(tape['steps']):
                error = None
                try:
                    env.step([frame['actions']], manifest['config']['max_ticks'])
                except Exception as exc:
                    error = {'phase': 'engine', 'type': type(exc).__name__, 'message': str(exc)}
                if error != frame['exception'] or public_digest(public_outputs(env)) != frame['public_digest']:
                    raise ValueError(f'native CPU replay differs game {game["tape"]} step {index}')
                count += 1
            if public_digest(public_outputs(env)) != tape['final_public_digest']:
                raise ValueError('native CPU replay final public digest differs')
        finally:
            env.close()
    if manifest['summary'] != summarize(manifest['games']):
        raise ValueError('evaluation summary differs from episode outcomes')
    result = {'status': 'passed', 'manifest_sha256': sha(manifest_path), 'games': len(manifest['games']),
              'exact_public_transitions': count, 'backend': 'native-cpu', 'jvm_started': False,
              'backend_build': backend_build_binding(build(manifest['config'].get('match_form', 'full'))), 'binding': binding(data_path)}
    no_jvm()
    (out / 'replay.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    evaluate = sub.add_parser('evaluate')
    evaluate.add_argument('--checkpoint', nargs='+', type=Path, required=True)
    evaluate.add_argument('--backend', choices=('native-cpu', 'native-cuda'), default='native-cpu')
    evaluate.add_argument('--schedule', type=Path, default=PRACTICE)
    evaluate.add_argument('--max-decisions', type=int, default=0)
    evaluate.add_argument('--max-ticks', type=int, default=0)
    evaluate.add_argument('--greedy', action='store_true')
    evaluate.add_argument('--cuda-launch-threads', type=int)
    evaluate.add_argument('--cuda-render-mode', type=int)
    evaluate.add_argument('--out', type=Path, required=True)
    replay = sub.add_parser('replay')
    replay.add_argument('--manifest', type=Path, required=True)
    replay.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'evaluate':
        result = evaluate_checkpoints(args.checkpoint, args.out, backend=args.backend, schedule=args.schedule,
                                      max_decisions=args.max_decisions, max_ticks=args.max_ticks, greedy=args.greedy,
                                      cuda_launch_threads=args.cuda_launch_threads, cuda_render_mode=args.cuda_render_mode)
        print(json.dumps(result['summary'], indent=2))
    else:
        result = replay_evaluation(args.manifest, args.out)
        print(json.dumps({k: result[k] for k in ('status', 'games', 'exact_public_transitions')}, indent=2))


if __name__ == '__main__':
    main()
