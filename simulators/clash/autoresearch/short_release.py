"""Short-v1 release constructor: the shortened Clash benchmark on its scripted evaluation panel.

`prepare(out)` writes a NEW standalone release for `clash-autoresearch-short-v1`: 60 s regulation + 30 s
overtime at 2 decisions per simulated second (common.MATCH_FORM, native/match_forms.py), the same fixed
8-card level-11 mirror, full arena, 7,024-int observations and 2,307 masked actions. No base release is
read: the evaluation interface, opponent registry, public practice and protected schedule are new.

Panel: the four trusted scripted opponents (standard, pressure, intermediate, teacher) of the v3/v4 panel,
which read only the public observation and need no retraining. The two learned panel opponents of fast-v4
(retained-deepseek, incumbent-v4) were trained on the pinned 180+120 s, 4-per-second form and are dropped.
Public practice plays new public seeds; the protected schedule plays 32 fresh secret seeds that are never
shown to the researcher. Neither schedule reuses an earlier seed bank.
"""
import json
import secrets
import shutil
from pathlib import Path

from .common import EDITABLE, HERE, MATCH_FORM, DATA, digest, dump, engine_binding, setup_native, sha
from .fast_entry import runtime_binding
from .bundle import verify as verify_bundle
from .release_v3 import INTERFACE_V31, PUBLIC_CAPACITY_SEEDS, PUBLIC_DEMO_SEEDS, PUBLIC_SEEDS, VALIDATION_SEEDS, \
    fixed_panel, resolve_schedule

BENCHMARK_ID = 'clash-autoresearch-short-v1'
FORMAT = 5
# New public practice seeds for the short form; never used by an earlier Clash schedule.
PRACTICE_SEEDS = list(range(2111301, 2111309))
PRACTICE_POLICY_SEED = 20261113
PROTECTED_SEEDS = 32
DROPPED_OPPONENTS = {
    'retained-deepseek': 'learned on the pinned 180+120 s, 4-decisions-per-second form; not retrained',
    'incumbent-v4': 'learned on the pinned 180+120 s, 4-decisions-per-second form; not retrained'}
PUBLIC_DOCS = {'benchmark.json': 'benchmark-short-v1.json', 'short-contract.md': 'CONTRACT-short-v1.md'}


def interface():
    """The short evaluation interface: v3.1 grading (three CPU pair workers) on the short match form."""
    setup_native()
    import match_forms
    form = match_forms.describe(MATCH_FORM)
    return {**INTERFACE_V31, 'version': 'clash-evaluation-short-v1', 'match_form': form,
            'decisions_per_second': form['decisions_per_second'], 'decision_ticks': form['decision_ticks'],
            'regulation_seconds': form['regulation_seconds'], 'overtime_seconds': form['overtime_seconds'],
            'legacy_engine_binding_unchanged': False}


def observation_schema():
    data = json.loads(DATA.read_text())
    return {'actions': 2307, 'entity_rows': 200, 'observation_ints': 7024, 'schema_version': 4,
            'last_request_refusal_index': 62, 'last_run_refusal_index': 63, 'public_enemy_play_count_index': 114,
            'observer_whitelist': 'supervisor reads only other-seat public placed count/history; never forwards '
                                  'other-seat arrays',
            'source_schema': data['schema']}


def registry():
    return {'fixed_panel': fixed_panel(), 'learned': [], 'dropped': DROPPED_OPPONENTS,
            'strong_panel_qualified': False, 'teacher_sha256': sha(HERE / 'visible_teacher.py'),
            'match_form': MATCH_FORM}


def used_seeds():
    return set(PUBLIC_SEEDS + VALIDATION_SEEDS + PUBLIC_DEMO_SEEDS + PUBLIC_CAPACITY_SEEDS + PRACTICE_SEEDS)


def protected_schedule(count=PROTECTED_SEEDS):
    seeds = []
    excluded = used_seeds()
    while len(seeds) < count:
        seed = secrets.randbelow(2 ** 31 - 1) + 1
        if seed not in excluded and seed not in seeds:
            seeds.append(seed)
    return {'engine_seeds': seeds, 'policy_seed': secrets.randbelow(2 ** 31 - 1) + 1, 'opponents': fixed_panel(),
            'match_form': MATCH_FORM, 'schedule_kind': 'protected-development-short-v1'}


def runtime_source():
    """Every top-level module here: worker trainer and evaluator, ledger, bundle code and this constructor."""
    return {str(p.relative_to(HERE)): sha(p) for p in sorted(HERE.glob('*.py'))}


def check_panel(root, release):
    """The protected schedule and its opponents still match the release's commitments; both schedules
    play this runtime's match form."""
    root = Path(root)
    schedule = json.loads((root / 'private/grade.json').read_text())
    if digest(schedule) != release['private_schedule_commitment']:
        raise ValueError('frozen schedule drift')
    practice = json.loads((root / 'public/practice.json').read_text())
    for value in (schedule, practice):
        if value.get('match_form') != MATCH_FORM:
            raise ValueError('schedule match form differs from the research runtime')
        for opponent in value['opponents']:
            if opponent['kind'] == 'checkpoint':
                identity = release['opponent_identities'][opponent['name']]
                if opponent['identity'] != identity or (value is schedule and verify_bundle(opponent['bundle'], identity)['identity'] != identity):
                    raise ValueError('opponent identity differs')
    resolve_schedule(root, practice)
    return schedule


def prepare(out):
    from harness import workspace
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    public = out / 'public'
    public.mkdir()
    (out / 'private').mkdir(mode=0o700)
    for n in EDITABLE:
        shutil.copyfile(HERE / 'starter' / n, public / n)
    shutil.copyfile(HERE / 'recipes/starter.json', public / 'config.json')
    shutil.copyfile(HERE / 'recipes/smoke.json', public / 'smoke-config.json')
    for name, source in PUBLIC_DOCS.items():
        shutil.copyfile(HERE / source, public / name)
    value = interface()
    dump(public / 'interface.json', value)
    dump(public / 'observation-schema.json', observation_schema())
    dump(public / 'opponent-registry.json', registry())
    practice = {'engine_seeds': PRACTICE_SEEDS, 'policy_seed': PRACTICE_POLICY_SEED, 'opponents': fixed_panel(),
                'match_form': MATCH_FORM, 'schedule_kind': 'public-practice-short-v1'}
    dump(public / 'practice.json', practice)
    schedule = protected_schedule()
    dump(out / 'private/grade.json', schedule)
    for n in ('config.json', 'smoke-config.json'):
        if json.loads((public / n).read_text()).get('match_form') != MATCH_FORM:
            raise ValueError(f'{n} must name match_form {MATCH_FORM!r}')
    # The researcher sees only the canonical workspace built from this public tree.
    meta = workspace.stage('clash', public)
    release = {'format': FORMAT, 'benchmark_id': BENCHMARK_ID, 'mode': 'provisional-development', 'grading': 'parallel3',
               'match_form': value['match_form'], 'engine': engine_binding(), 'interface_sha256': digest(value),
               'teacher_sha256': sha(HERE / 'visible_teacher.py'), 'opponent_identities': {},
               'dropped_opponents': DROPPED_OPPONENTS, 'registry_commitment': digest(registry()),
               'private_schedule_commitment': digest(schedule),
               'protected_schedule': f'{PROTECTED_SEEDS} fresh secret engine seeds on the scripted panel '
                                     '(development only; never shown to the researcher)',
               'private_final_feedback_allowed_for_training': False, 'normalized_reference_score': None,
               'trusted_fast_runtime': runtime_binding(), 'runtime_source_sha256': runtime_source(),
               'public_files': {str(p.relative_to(public)): sha(p) for p in sorted(public.iterdir()) if p.is_file()},
               'workspace_sha256': meta['sha256'],
               'training_resources': {'gpu': 'NVIDIA L40S', 'cpu_cores': 4, 'memory_gib': 16}}
    dump(out / 'release.json', release)
    verify(out)
    return {'release': str(out), 'commitment': digest(release), 'benchmark_id': BENCHMARK_ID,
            'match_form': MATCH_FORM, 'protected_seeds': PROTECTED_SEEDS}


def verify(release):
    from harness import workspace
    p = Path(release)
    r = json.loads((p / 'release.json').read_text())
    if r.get('format') != FORMAT or r.get('benchmark_id') != BENCHMARK_ID or r['engine'] != engine_binding():
        raise ValueError('short release engine/version binding differs; prepare an explicit new release')
    if r['trusted_fast_runtime'] != runtime_binding():
        raise ValueError('fast runtime source differs; prepare explicit new release')
    if r['runtime_source_sha256'] != runtime_source():
        raise ValueError('research runtime source differs; prepare explicit new release')
    if any(sha(p / 'public' / n) != h for n, h in r['public_files'].items()):
        raise ValueError('short public release drift')
    value = json.loads((p / 'public/interface.json').read_text())
    if value != interface() or digest(value) != r['interface_sha256']:
        raise ValueError('evaluation interface differs from the short contract')
    if r.get('workspace_sha256') != workspace.verify(p)['sha256']:
        raise ValueError('researcher workspace differs from the release')
    check_panel(p, r)
    return r
