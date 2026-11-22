"""Build the Harbor task that puts Terminus-2 in the uniform workspace.

One builder for every game and both variants. The only inputs that differ are
the release's workspace, the game's brief and the session's budgets.
"""
import json
from pathlib import Path, PurePosixPath
import shutil

from harness import agents, budget, registry, workspace
from harness.broker import VARIANTS

HERE = Path(__file__).resolve().parent
# CPU torch in the researcher's box matches the game's training workers where that version is declared.
BOX_TORCH = '2.15.0'
TORCH_ARG = 'ARG TORCH=' + BOX_TORCH
# Where an installed researcher agent's build-time install layer goes.
AGENT_INSTALL = '# AGENT_INSTALL'
ARTIFACTS = ['/workspace/training', '/workspace/configs', '/workspace/NOTES.md']
VERIFIER = '''#!/bin/bash
# Records whether the researcher submitted. Private grading is host-owned and happens elsewhere.
mkdir -p /logs/verifier
ar _verify > /logs/verifier/session.json 2>&1 || true
if grep -q '"submitted": true' /logs/verifier/session.json; then echo 1 > /logs/verifier/reward.txt; else echo 0 > /logs/verifier/reward.txt; fi
'''
# With restarts, each step's verifier also reports whether an unfinished session still has usable allocation;
# Harbor starts the next step (a new researcher conversation in the same box) only while it does.
STEP_VERIFIER = '''#!/bin/bash
# Records whether the researcher submitted, and whether the session should be resumed by another researcher step.
mkdir -p /logs/verifier
ar _verify > /logs/verifier/session.json 2>&1 || true
submitted=0; resume=0
grep -q '"submitted": true' /logs/verifier/session.json && submitted=1
grep -q '"resume": true' /logs/verifier/session.json && resume=1
printf '{"reward": %s, "resume": %s}\\n' "$submitted" "$resume" > /logs/verifier/reward.json
'''
STEP = '''
[[steps]]
name = "{name}"
{gate}
[steps.agent]
timeout_sec = {timeout:.1f}
user = "researcher"

[steps.verifier]
timeout_sec = 600.0
'''
CONTINUE = '''You are resuming an unfinished research session in this workspace. An earlier researcher
conversation stopped while usable allocation remained. Nothing was reset: `NOTES.md` holds its notes,
`training/` and `configs/` hold its latest edits, and `ar status` lists every run so far.

Read `NOTES.md`, `ar status` and `ar budget` first, then continue the research from where the notes leave
off: run further experiments, submit whenever a result is better, and keep `NOTES.md` current. `ar finalize`
is refused until the allocation is used. The complete task instructions follow.

---

'''
CONTINUED = '''This session continues an earlier research session of yours on this same task, on the same game
and starter. `training/`, `configs/` and `NOTES.md` are exactly as that session left them: read `NOTES.md` first.
{scratch}This session has its own ledger and its own new allocation (below), separate from what the earlier
session spent: `ar status` lists only this session's runs, and nothing the earlier session submitted carries over,
so submit again here whenever a result is better.

---

'''
RESUMED = '''This research session was paused by its operator and is now resumed, in a new box. Nothing was reset:
`training/`, `configs/` and `NOTES.md` are as they were at the pause, `ar status` lists every run so far, `ar budget`
shows what is left of the same allocation, your submissions stand, and any shared worker storage (`/scratch`, where
this game has it) is unchanged. Whatever was running at the pause was stopped: a worker and its commands are gone, and
a training run keeps only what it had already saved. The wall-clock limit below is what was left when the session was
paused.

Read `NOTES.md`, `ar status` and `ar budget` first, then continue from where the notes leave off.

---

'''
SCRATCH_COPIED = ('The files the earlier session kept in its `/scratch` storage (its run folders and checkpoints) '
                  'are in this session\'s `/scratch`, as a copy.\n')
# A continuation seed: the editable files and notes an earlier session ended with, and where they came from.
SEED_RECORD = 'seed.json'
SEED_FILES = 'workspace'
# A sandbox lives at most 24 hours; all steps together stay inside that.
MAX_TASK_SECONDS = 82800.0
TOML = '''schema_version = "1.4"
# Harbor saves these from the box when the session ends.
artifacts = {artifacts}
{top}
[metadata]
name = "{game}-autoresearch-{variant}"
description = "Improve the {game} starter agent under an allocated-worker budget ({variant} variant)."

[agent]
# Wall-clock stop for the researcher. The research budget itself is allocated worker time.
timeout_sec = {wall:.1f}
user = "researcher"
{agent_network}

[verifier]
timeout_sec = 600.0

[environment]
build_timeout_sec = 1800.0
# The researcher's box is CPU-only and offline. Training runs on workers started by the host broker.
cpus = 4
memory_mb = 16384
network_mode = "no-network"
'''


def duration(seconds):
    seconds = float(seconds)
    for size, unit in ((3600, 'hour'), (60, 'minute')):
        if seconds >= size and seconds % size == 0:
            count = int(seconds // size)
            return f'{count} {unit}' + ('' if count == 1 else 's')
    if seconds >= 3600:
        return f'{seconds / 3600:.2f} hours'
    return f'{seconds / 60:g} minutes' if seconds >= 120 else f'{seconds:g} seconds'


def brief(game, release=None):
    """The game's brief: the copy a release sealed (`public/brief.md`, where the release chose a starter), else
    the game's `task/brief.md`."""
    sealed = workspace.public(release) / 'brief.md' if release is not None else None
    source = sealed if sealed is not None and sealed.is_file() else workspace.task_dir(game) / 'brief.md'
    text = source.read_text().strip()
    lines = text.splitlines()
    if lines and lines[0].startswith('# '):
        lines = lines[1:]
    return '\n'.join(lines).strip()


def instruction(game, variant, *, research_seconds, final_train_seconds, wall_seconds, hardware, smoke=False,
                use_budget=False, full_session=False, research_usd=None, budget_mode=None, release=None):
    """The full prompt: common instructions, this game's brief, what is graded, this session's numbers."""
    if variant not in VARIANTS:
        raise ValueError('variant must be recipe or network')
    if research_usd is not None:
        research = f'${research_usd:,.2f} of worker time'
        allocation = (f'- Research allocation: ${research_usd:,.2f} of allocated worker time in total, each worker '
                      'second charged at the price of its shape.')
    else:
        research = duration(research_seconds)
        allocation = f'- Research allocation: {research} of allocated worker time in total.'
    graded = (workspace.TEMPLATES / f'variant-{variant}.md').read_text().format(
        research=research, final_train=duration(final_train_seconds))
    session = ['## This session', '', allocation]
    if variant == 'recipe':
        session.append(f'- Fresh final training of your recipe by the host: {duration(final_train_seconds)}, '
                       'outside your research allocation.')
    spec = registry.spec(game)
    if spec.get('move_flops_cap'):
        size = (f'- Model size: no parameter cap. Compute per move is capped: one forward pass for one board may '
                f'take at most {spec["move_flops_cap"]:,} FLOPs, counted from the traced computation (the brief says how it is '
                f'measured). Weights are stored up to {spec["parameter_cap"]:,} parameters. A policy over either '
                'is rejected.')
    else:
        size = (f'- Model size: at most {spec["parameter_cap"]:,} parameters and buffers in the policy that is '
                'graded. A larger one is rejected.')
    session += [size,
                f'- Workers: {hardware}.',
                budget.describe(budget_mode or (budget.USD if research_usd is not None else budget.SECONDS)),
                f'- Wall-clock limit for this terminal session: {duration(wall_seconds)}.',
                '- `ar budget` is authoritative if it disagrees with these numbers.']
    if use_budget:
        session.append('- Use the whole allocation. `ar finalize` is refused while a useful share of it is unspent: '
                       'it is accepted once less than 10% of the allocation (or less than one minimum run) is '
                       'left, or within 10 minutes of the wall-clock limit. Until then keep experimenting and '
                       'submit again whenever a result is better. If you stop before that, another researcher '
                       'conversation continues this session from your `NOTES.md` and files.')
    if full_session:
        session += ['', 'This is a full session: use your whole allocation and session time to make your agent as '
                        'strong as you can. Keep a good network submitted and keep improving it. `ar finalize` is '
                        'accepted only once at most 10% of the allocation or 20 minutes of session time remain; '
                        'if you stop earlier, the latest submission is graded.']
    if smoke:
        session += ['', 'This is a short end-to-end smoke session, not a research campaign: make one small '
                        'legitimate change, run one short training job with `--config configs/smoke.json`, '
                        'practice it once, submit and finalize.']
    editable_simulator = bool(release and workspace.enabled(release)
                              and workspace.manifest(release).get('editable_simulator'))
    common = workspace.instructions(bool(registry.spec(game).get('leases')), editable_simulator).strip()
    return '\n\n'.join([common, '## This game\n\n' + brief(game, release), graded.strip(),
                        '\n'.join(session)]) + '\n'


def make_seed(game, variant, files, notes, out, prior):
    """Write a continuation seed: exactly the release's editable files and the notes, with `prior` (where they
    came from: session, model, what it spent) recorded beside their digests. `out` must not exist."""
    out = Path(out)
    out.mkdir(parents=True)
    names = dict(files)
    names[workspace.NOTES] = notes
    for name, text in names.items():
        target = workspace.safe(out / SEED_FILES, name)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
    record = dict(schema=1, game=game, variant=variant, prior=prior,
                  files={name: workspace.sha(workspace.safe(out / SEED_FILES, name)) for name in sorted(names)})
    record['sha256'] = workspace.digest({k: v for k, v in record.items()})
    (out / SEED_RECORD).write_text(json.dumps(record, indent=2, sort_keys=True) + '\n')
    return record


def read_seed(seed, game, variant, meta):
    """A continuation seed checked against this release: the same game and variant, exactly the declared editable
    files plus the notes, bytes matching their recorded digests. Returns (record, {name: text})."""
    seed = Path(seed)
    record = json.loads((seed / SEED_RECORD).read_text())
    body = {k: v for k, v in record.items() if k != 'sha256'}
    if record.get('schema') != 1 or workspace.digest(body) != record.get('sha256'):
        raise ValueError('Continuation seed record is damaged')
    if (record['game'], record['variant']) != (game, variant):
        raise ValueError(f"Continuation seed is for {record['game']} {record['variant']}, not {game} {variant}")
    expected = set(meta['writable']) | {workspace.NOTES}
    if set(record['files']) != expected:
        raise ValueError('Continuation seed must hold exactly the editable files and the notes: '
                         + ', '.join(sorted(expected)))
    files = {}
    for name, digest in record['files'].items():
        path = workspace.safe(seed / SEED_FILES, name)
        if not path.is_file() or path.is_symlink() or workspace.sha(path) != digest:
            raise ValueError(f'Continuation seed file {name} is missing or changed')
        files[name] = path.read_text()
    return record, files


def seed_summary(seed, game, variant, release):
    """What a session binding records about the seed it continues."""
    record, _ = read_seed(seed, game, variant, workspace.verify(release))
    return dict(path=str(Path(seed).resolve()), sha256=record['sha256'], prior=record['prior'])


def build(game, variant, release, out, *, research_seconds, final_train_seconds, wall_seconds, hardware,
          smoke=False, agent=None, full_session=False, restarts=0, use_budget=False, research_usd=None,
          budget_mode=None, continue_from=None):
    """Write a Harbor task directory for one session. `out` must not exist.

    `restarts` > 0 makes it a multi-step task: when the researcher stops while the broker still reports usable
    allocation, Harbor starts another researcher conversation in the same box, up to that many times.

    `continue_from` (a seed from `make_seed`) starts the workspace from an earlier session's editable files and
    notes instead of the release's starter, and says so at the top of the instructions.
    """
    meta = workspace.verify(release)
    if meta['game'] != game:
        raise ValueError('Release belongs to a different game')
    seed = read_seed(continue_from, game, variant, meta) if continue_from is not None else None
    spec = registry.spec(game)
    adapter = agents.resolve(spec, agent)
    out = Path(out)
    out.mkdir(parents=True)
    environment = out / 'environment'
    box = environment / 'workspace'
    shutil.copytree(workspace.public(release) / workspace.WORKSPACE, box)
    for path in box.rglob('*'):
        path.chmod(0o755 if path.is_dir() else 0o644)
    box.chmod(0o755)
    if {p.name for p in box.iterdir()} != set(workspace.TOP):
        raise ValueError('Release workspace is not the canonical layout')
    text = instruction(game, variant, research_seconds=research_seconds, final_train_seconds=final_train_seconds,
                       wall_seconds=wall_seconds, hardware=hardware, smoke=smoke, use_budget=use_budget,
                       full_session=full_session, research_usd=research_usd, budget_mode=budget_mode,
                       release=release)
    if seed is not None:
        record, files = seed
        for name, content in files.items():
            target = workspace.safe(box, name)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content)
            target.chmod(0o644)
        scratch = SCRATCH_COPIED if record['prior'].get('scratch_copy') else ''
        # A paused session resumed (session.resume) is the same session; a continued one is a new session.
        text = (RESUMED if record['prior'].get('paused') else CONTINUED.format(scratch=scratch)) + text
    if adapter['instruction']:
        text = text.rstrip() + '\n\n' + adapter['instruction'].strip() + '\n'
    (out / 'instruction.md').write_text(text)
    (box / 'instructions.md').write_text(text)
    if spec.get('leases'):
        # The researcher's own launcher: it only calls `ar`, and is not one of the files that are trained.
        (box / 'training/launch.py').write_text(workspace.launcher(game))
    (environment / 'ar').mkdir()
    shutil.copyfile(HERE / 'box/ar.py', environment / 'ar/ar.py')
    dry_run = workspace.task_dir(game) / 'dryrun.py'
    if dry_run.is_file():
        # The game's own CPU dry run of the editable files; `ar check` runs it inside the box.
        shutil.copyfile(dry_run, environment / 'ar/dryrun.py')
    # `ar` offers only the options this game's backend acts on.
    (environment / 'ar/box.json').write_text(json.dumps(dict(
        game=game, variant=variant, writable=meta['writable'], run_seed=bool(spec.get('run_seed')),
        eval_seconds=bool(spec.get('eval_seconds')), leases=bool(spec.get('leases')),
        dry_run=dry_run.is_file()), indent=2) + '\n')
    dockerfile = (HERE / 'box/Dockerfile').read_text()
    if dockerfile.count(TORCH_ARG) != 1:
        raise ValueError('Box Dockerfile lost its torch version argument')
    dockerfile = dockerfile.replace(TORCH_ARG, 'ARG TORCH=' + spec.get('box_torch', BOX_TORCH))
    if dockerfile.count(AGENT_INSTALL) != 1:
        raise ValueError('Box Dockerfile lost its agent install marker')
    dockerfile = dockerfile.replace(AGENT_INSTALL, agents.box_install(adapter))
    # Editable files outside training/ and configs/ (an editable simulator): the researcher owns them and the
    # folders that hold them, as it owns training/.
    extra = sorted(name for name in meta['writable'] if not name.startswith(('training/', 'configs/')))
    if extra:
        folders = sorted({str(PurePosixPath(name).parent) for name in extra})
        dockerfile += ('RUN cd /workspace && chown researcher:researcher ' + ' '.join(folders + extra)
                       + ' && chmod 755 ' + ' '.join(folders) + ' && chmod 644 ' + ' '.join(extra) + '\n')
    (environment / 'Dockerfile').write_text(dockerfile)
    (out / 'tests').mkdir()
    (out / 'tests/test.sh').write_text(VERIFIER)
    (out / 'tests/test.sh').chmod(0o755)
    # An installed agent needs egress during its own phase; the environment baseline stays offline.
    network = f'network_mode = "{adapter["agent_network"]}"' if adapter['agent_network'] else ''
    restarts = int(restarts)
    # The trial's own reward stays "the researcher submitted", read from the last step that ran.
    toml = TOML.format(artifacts=json.dumps(ARTIFACTS), game=game, variant=variant, wall=float(wall_seconds),
                       top='multi_step_reward_strategy = "final"\n' if restarts > 0 else '',
                       agent_network=network)
    if restarts > 0:
        later = min(float(wall_seconds), max(600.0, (MAX_TASK_SECONDS - float(wall_seconds)) / restarts))
        names = ['research'] + [f'continue{index}' for index in range(1, restarts + 1)]
        for index, name in enumerate(names):
            step = out / 'steps' / name
            (step / 'tests').mkdir(parents=True)
            (step / 'instruction.md').write_text(text if index == 0 else CONTINUE + text)
            (step / 'tests/test.sh').write_text(STEP_VERIFIER)
            (step / 'tests/test.sh').chmod(0o755)
            toml += STEP.format(name=name, timeout=float(wall_seconds) if index == 0 else later,
                                gate='min_reward = { resume = 1.0 }' if index < restarts else '')
    (out / 'task.toml').write_text(toml)
    return dict(task=str(out), game=game, variant=variant, agent=adapter['name'], workspace_sha256=meta['sha256'],
                instruction_sha256=workspace.sha(out / 'instruction.md'),
                files=sum(1 for p in box.rglob('*') if p.is_file()))
