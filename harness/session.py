"""Run one research session: Terminus-2 in the box, the broker and the game backend on the host.

    harbor run -a terminus-2 -e modal   researcher (CPU box, offline, no credentials)
            |  ar requests / replies
    Broker  ->  Backend                 host: validates files, enforces the variant, dispatches workers

Model calls are made by Harbor on the host. No key enters the box, the task
directory or a command line.
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time
import uuid

from harness import agents, budget, policy, registry, task, workspace
from harness.transport import Bridge, serve
from harness.broker import Broker, write_json

ROOT = Path(__file__).resolve().parents[1]
APP = 'game-autoresearch-researchers'
# How long final training keeps re-reading a just-finished job, or retrying a finalize refused while the backend
# settles it, before accepting the outcome.
SETTLE_GRACE_SECONDS = 180
# How many times a researcher that stops with usable allocation left is started again in the same box.
RESTARTS = 3
MAX_WALL = 82800.0  # a Modal sandbox lives at most 24 hours
# Pause and resume: the operator's request, the session's pause state, and the launch options a resume reuses.
PAUSE_REQUEST = 'pause.request'
PAUSE = 'pause.json'
LAUNCH = 'launch.json'


def harbor_binary(explicit=None):
    path = Path(explicit or os.environ.get('HARNESS_HARBOR') or ROOT / '.venv-harness/bin/harbor')
    if not path.is_file():
        raise FileNotFoundError(
            f'Harbor is not installed at {path}. Create it once with:\n'
            f'  uv venv .venv-harness --python 3.12 && uv pip install --python .venv-harness "{policy.HARBOR}" '
            '"modal[api-proxy-support]" socksio\n'
            'or point HARNESS_HARBOR at a harbor executable of that version.')
    return str(path)


def harbor_command(binary, task_dir, out, session, model, *, agent=None, api_base=None, reasoning_effort=None,
                   model_info=None, agent_kwargs=(), box_tag=None):
    adapter = agents.get(agent)
    command = [binary, 'run', '-p', str(task_dir), '-a', adapter['harbor'], '-m', model, '-e', 'modal',
               '-o', str(out), '--job-name', session, '-n', '1', '-y',
               '--ek', 'app_name=' + APP, '--ek', 'labels=' + json.dumps({'ar_session': box_tag or session}),
               # The box keeps the task's cpus/memory as a hard cap but reserves only Modal's minimum: Modal bills
               # max(reservation, use), and a researcher's box mostly waits on the model (2026-11-17: 4 CPU / 16 GiB
               # reserved for the whole session was ~30 % of a Pokemon campaign's Modal bill).
               '--cpus', 'limit', '--memory', 'limit']
    for item in agents.command_kwargs(adapter, api_base=api_base, reasoning_effort=reasoning_effort,
                                      model_info=model_info, extra=agent_kwargs):
        command += ['--ak', item]
    return command


def harbor_environment(key_file=None, key_env=None):
    """The model key reaches Harbor through its environment only."""
    env = dict(os.environ, MODAL_ENVIRONMENT=policy.ENVIRONMENT)
    policy.direct_modal(env)
    if not policy.in_modal():  # a cloud coordinator authenticates as its own container
        env['MODAL_PROFILE'] = policy.MODAL_WORKSPACE
    if key_file:
        path = Path(key_file)
        if path.stat().st_mode & 0o077:
            raise ValueError('Model key file must have owner-only permissions')
        env[key_env or agents.get()['key_env']] = path.read_text().strip()
    return env


def launch_harbor(command, env, log_path):
    log = open(log_path, 'ab')
    return subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=ROOT)


def default_wall(research_seconds):
    return min(MAX_WALL, 2 * research_seconds + 3600)


def wall_limit(game, smoke=False, research_seconds=None, wall_seconds=None, budget_mode=None):
    """The researcher's wall-clock limit: explicit; in dollar mode the game's declared one; in seconds mode
    derived from the allocation."""
    if wall_seconds:
        return float(wall_seconds)
    if research_seconds is None or budget_mode == budget.USD:
        return float(registry.spec(game)['smoke_wall_seconds' if smoke else 'wall_seconds'])
    return float(default_wall(research_seconds))


def plan(game, variant, release, out, *, model, agent=None, smoke=False, research_seconds=None,
         final_train_seconds=None, wall_seconds=None, research_usd=None, budget_mode=None, use_budget=None,
         restarts=None, continue_from=None, **extra):
    """Everything a session would do, without creating anything.

    `use_budget` (default: every session but a smoke) refuses the researcher's finalize while a useful share of
    the allocation is unspent; `restarts` (default RESTARTS with it, else 0) is how often a researcher that
    stops anyway is started again in the same box.
    """
    chosen = registry.research(game, smoke, budget_mode, research_seconds, research_usd)
    research, usd = chosen['seconds'], chosen['usd']
    _, final_train = registry.budgets(game, smoke, final_train_seconds=final_train_seconds, budget_mode=chosen['mode'])
    spec = registry.spec(game)
    adapter = agents.resolve(spec, agent)
    use_budget = (not smoke) if use_budget is None else bool(use_budget)
    restarts = (RESTARTS if use_budget else 0) if restarts is None else int(restarts)
    settings = dict(game=game, variant=variant, release=str(Path(release).resolve()),
                    session=str(Path(out).resolve()),
                    agent=adapter['name'], agent_summary=adapter['summary'], harbor=policy.HARBOR, model=model,
                    smoke=bool(smoke), use_budget=use_budget, restarts=restarts, budget_mode=chosen['mode'],
                    budget_limits=budget.limits(chosen['mode']),
                    research_seconds=research, final_train_seconds=final_train if variant == 'recipe' else None,
                    wall_seconds=wall_limit(game, smoke, research, wall_seconds, chosen['mode']),
                    hardware=spec['hardware'],
                    environment=policy.ENVIRONMENT, modal_workspace=policy.MODAL_WORKSPACE, **extra)
    if usd is not None:  # a dollar-mode session has no research_seconds
        settings['research_usd'] = usd
    if continue_from is not None:  # recorded only when used, so earlier bindings read the same
        settings['continue_from'] = task.seed_summary(continue_from, game, variant, release)
    return settings


def harbor_outcome(job_dir):
    """What Harbor recorded for its one trial. Harbor exits 0 even when the trial errored."""
    path = Path(job_dir) / 'result.json'
    # One folder per researcher step that ran (1 + restarts), inside the trial's folder.
    steps = len({item.name for item in Path(job_dir).glob('*/steps/*') if item.is_dir()}) or None
    if not path.is_file():
        return dict(exceptions={}, usage=None, steps=steps)
    stats = json.loads(path.read_text()).get('stats', {})
    exceptions = {}
    for evaluation in stats.get('evals', {}).values():
        for kind, trials in evaluation.get('exception_stats', {}).items():
            exceptions[kind] = exceptions.get(kind, 0) + len(trials)
    usage = {key: stats.get(key) for key in ('n_input_tokens', 'n_cache_tokens', 'n_output_tokens', 'cost_usd')}
    return dict(exceptions=exceptions, usage=usage, steps=steps)


def researcher_records(out, job_dir):
    """Where the researcher's final notes and whole trajectory are, relative to the session folder.

    Harbor saves both per researcher step, under `steps/STEP/`. Two other copies look like them and are not:
    the trial's own `agent/trajectory.json` is a snapshot taken before the last turns (it lacked the final submit
    and finalize in all seven Polytopia sessions of 2026-11-14), and `research/agent_workspace/NOTES.md` is the
    host's copy of the editable files, which notes never enter. The newest step's notes are also copied to the
    session folder as `NOTES.md`."""
    out, steps = Path(out), []
    for step in list(Path(job_dir).glob('*/steps/*')) + list(Path(job_dir).glob('*/*/steps/*')):
        trajectory, notes = step / 'agent' / 'trajectory.json', step / 'artifacts' / 'workspace' / workspace.NOTES
        if trajectory.is_file() or notes.is_file():
            steps.append(dict(step=step.name, at=max(p.stat().st_mtime for p in (trajectory, notes) if p.is_file()),
                              trajectory=str(trajectory.relative_to(out)) if trajectory.is_file() else None,
                              notes=str(notes.relative_to(out)) if notes.is_file() else None))
    steps.sort(key=lambda item: item['at'])
    final_notes = next((item['notes'] for item in reversed(steps) if item['notes']), None)
    if final_notes:
        shutil.copyfile(out / final_notes, out / workspace.NOTES)
    return dict(steps=[{k: v for k, v in item.items() if k != 'at'} for item in steps],
                notes=workspace.NOTES if final_notes else None,
                trajectory=steps[-1]['trajectory'] if steps else None)


# Reaching the wall-clock limit is how a session normally ends when the researcher keeps working.
EXPECTED_EXCEPTIONS = ('AgentTimeoutError',)


def serve_until(stop, find_box, broker, backend, errors):
    box = None
    while not stop.is_set():
        try:
            if box is None or not box.alive():
                box = find_box()
            if box is not None:
                serve(box, broker)
            backend.tick()
        except Exception as exc:
            box = None
            with open(errors, 'a') as log:
                log.write(json.dumps(dict(time=time.time(), type=type(exc).__name__, message=str(exc)[:500])) + '\n')
        stop.wait(1.0)


def run(game, variant, release, out, *, model, agent=None, smoke=False, research_seconds=None,
        final_train_seconds=None, wall_seconds=None, research_usd=None, budget_mode=None, api_base=None,
        key_file=None, key_env=None, reasoning_effort=None, model_info=None, agent_kwargs=(), harbor=None,
        backend_options=None, final=True, launcher=None, find_box=None, full_session=False, use_budget=None,
        restarts=None, continue_from=None):
    """One full session. `launcher` and `find_box` exist so tests can stand in for Harbor and Modal."""
    out = Path(out).resolve()
    release = Path(release).resolve()
    recorded = {key: value for key, value in (backend_options or {}).items() if not key.startswith('_')}
    settings = plan(game, variant, release, out, model=model, agent=agent, smoke=smoke,
                    research_seconds=research_seconds, final_train_seconds=final_train_seconds,
                    wall_seconds=wall_seconds, research_usd=research_usd, budget_mode=budget_mode, api_base=api_base,
                    reasoning_effort=reasoning_effort, backend_options=recorded, full_session=bool(full_session),
                    use_budget=use_budget, restarts=restarts, continue_from=continue_from)
    adapter = agents.get(settings['agent'])
    settings = json.loads(json.dumps(settings, default=str))
    research, final_train, wall = settings['research_seconds'], settings['final_train_seconds'], settings['wall_seconds']
    usd = settings.get('research_usd')
    live = launcher is None
    if live:
        policy.require_modal()
        binary = harbor_binary(harbor)
    # The box's Modal tag must be unique among all concurrent sessions: brokers find their box by it, and
    # sessions in different folders often share a folder name (e.g. three models' `smoke-1`).
    settings['box_tag'] = out.name + '-' + uuid.uuid4().hex[:12]
    policy.bind_run(game, release, out, settings)
    out.mkdir(parents=True)
    session = out.name
    box_tag = settings['box_tag']
    write_json(out / 'plan.json', settings)
    task.build(game, variant, release, out / 'task', research_seconds=research, research_usd=usd,
               budget_mode=settings['budget_mode'],
               final_train_seconds=final_train or 0, wall_seconds=wall, hardware=settings['hardware'], smoke=smoke,
               agent=settings['agent'], full_session=settings['full_session'],
               restarts=settings['restarts'], use_budget=settings['use_budget'], continue_from=continue_from)
    # What a resume needs to start the researcher again and no binding records (never a key, only where it is read).
    write_json(out / LAUNCH, dict(api_base=api_base, key_file=str(key_file) if key_file else None, key_env=key_env,
                                  reasoning_effort=reasoning_effort, model_info=model_info,
                                  agent_kwargs=list(agent_kwargs), harbor=str(harbor) if harbor else None))
    backend = registry.backend(game)(release, out / 'research', smoke=smoke, budget_seconds=research,
                                     budget_usd=usd, options=backend_options)
    return _conduct(out, game, settings, backend, task_dir=out / 'task', job=session, box_tag=box_tag,
                    deadline=time.time() + wall, binary=binary if live else None,
                    start=None if live else (lambda: launcher(out)), find_box=find_box, api_base=api_base,
                    key_file=key_file, key_env=key_env, reasoning_effort=reasoning_effort, model_info=model_info,
                    agent_kwargs=agent_kwargs, backend_options=backend_options, final=final)


def pending_pause(out):
    """The operator's pause request this session has not acted on yet, else None.

    `harness pause` writes `pause.request`; a request counts once, by its id (records mirrored to the cloud are
    never deleted, so a handled request stays on disk)."""
    out = Path(out)
    path = out / PAUSE_REQUEST
    if not path.is_file():
        return None
    try:
        request = json.loads(path.read_text())
        if not isinstance(request, dict) or not request.get('id'):
            raise ValueError
    except ValueError:
        return None  # half-written: read again on the next pass
    state = json.loads((out / PAUSE).read_text()) if (out / PAUSE).is_file() else {}
    return None if request['id'] in state.get('handled', []) else request


def request_pause(out, note=None):
    """Ask the session's coordinator to pause it. Returns at once; the coordinator acts within seconds."""
    out = Path(out).resolve()
    policy.run_binding(out)
    state = json.loads((out / 'session.json').read_text()) if (out / 'session.json').is_file() else None
    if state is None or state['finalized']:
        raise ValueError('Only a running, unfinalized session can be paused')
    paused = json.loads((out / PAUSE).read_text()) if (out / PAUSE).is_file() else {}
    if paused.get('state') == 'paused':
        raise ValueError('This session is already paused; resume it with `python -m harness resume`')
    request = dict(id=uuid.uuid4().hex, requested_at=time.time(), note=note)
    write_json(out / PAUSE_REQUEST, request)
    return dict(requested=request, session=str(out),
                note='The coordinator snapshots the workspace, stops the researcher and its workers and writes '
                     f'{PAUSE} (state "paused"). Nothing is finalized; resume with `python -m harness resume`.')


def wait_or_pause(process, out):
    """(Harbor's return code, None), or (None, request) when a pause request arrives while it runs."""
    done = {}

    def waiter():
        try:
            done['code'] = process.wait()
        except BaseException as exc:
            done['error'] = exc
    thread = threading.Thread(target=waiter, daemon=True)
    thread.start()
    while thread.is_alive():
        request = pending_pause(out)
        if request is not None:
            return None, request
        thread.join(1.0)
    if 'error' in done:
        raise done['error']
    return done.get('code'), None


def snapshot_workspace(find_box, backend, out, meta):
    """The researcher's editable files and notes as they are now: from the box, else the newest copies the host
    has (the files sent with the last `ar` command, the notes Harbor last saved). Never raises."""
    names, got, error = list(meta['writable']) + [workspace.NOTES], {}, None
    try:
        box = find_box() if find_box is not None else None
        got = box.snapshot(names) if box is not None else {}
    except Exception as exc:
        error = type(exc).__name__ + ': ' + str(exc)[:300]
    files, sources = {}, {}
    try:
        host = workspace.editable(backend.release, backend.workspace)
    except Exception:
        host = {}
    starter = workspace.public(backend.release) / workspace.WORKSPACE
    for name in meta['writable']:
        if isinstance(got.get(name), str):
            files[name], sources[name] = got[name], 'box'
        elif name in host:
            files[name], sources[name] = host[name], 'host copy of the last ar command'
        else:
            files[name], sources[name] = (starter / name).read_text(), 'release starter'
    try:  # the box's own files must still be a valid editable set; otherwise keep the host's last validated one
        workspace.validate(meta, files)
    except Exception as exc:
        error = (error + '; ' if error else '') + 'box files rejected: ' + str(exc)[:200]
        files = {name: host.get(name, (starter / name).read_text()) for name in meta['writable']}
        sources = {name: 'host copy (box files rejected)' for name in files}
    notes = got.get(workspace.NOTES)
    if not isinstance(notes, str):
        saved = sorted(Path(out, 'harbor').glob('*/*/steps/*/artifacts/workspace/' + workspace.NOTES),
                       key=lambda item: item.stat().st_mtime)
        earlier = sorted(Path(out, 'pauses').glob('*/seed/' + task.SEED_FILES + '/' + workspace.NOTES))
        notes = (saved[-1].read_text() if saved else earlier[-1].read_text() if earlier
                 else (starter / workspace.NOTES).read_text())
        sources[workspace.NOTES] = 'host copy'
    else:
        sources[workspace.NOTES] = 'box'
    return dict(files=files, notes=notes, sources=sources, error=error)


def record_pause(out, game, settings, broker, backend, request, snapshot, job, remaining):
    """Write the pause: a seed of the workspace, what was left of the wall clock and budget, and the session state."""
    out = Path(out)
    before = json.loads((out / PAUSE).read_text()) if (out / PAUSE).is_file() else {}
    index = int(before.get('index', 0)) + 1
    folder = out / 'pauses' / f'{index:03d}'
    task.make_seed(game, settings['variant'], snapshot['files'], snapshot['notes'], folder / 'seed',
                   dict(paused=True, index=index, session=str(out), model=settings.get('model')))
    record = dict(schema=1, state='paused', index=index, request=request, paused_at=time.time(),
                  wall_remaining_seconds=remaining, harbor_job=job, seed=str((folder / 'seed').relative_to(out)),
                  workspace_sources=snapshot['sources'], snapshot_error=snapshot['error'], budget=backend.budget(),
                  handled=list(before.get('handled', [])) + [request['id']])
    write_json(folder / 'pause.json', record)
    write_json(out / PAUSE, record)
    broker.state['paused'] = index
    broker.save()
    return record


def trajectory_usage(job_dir):
    """Token usage of a Harbor job that never wrote its result (a paused one), from the trajectory it kept."""
    total = {}
    for path in Path(job_dir).glob('*/agent/trajectory.json'):
        try:
            metrics = json.loads(path.read_text()).get('final_metrics') or {}
        except ValueError:
            continue
        for ours, theirs in (('n_input_tokens', 'total_prompt_tokens'), ('n_cache_tokens', 'total_cached_tokens'),
                             ('n_output_tokens', 'total_completion_tokens'), ('cost_usd', 'total_cost_usd')):
            if metrics.get(theirs) is not None:
                total[ours] = total.get(ours, 0) + metrics[theirs]
    return total or None


def harbor_jobs(out):
    """Every Harbor job folder of a session, oldest first: the first run and one per resume."""
    root = Path(out) / 'harbor'
    return sorted((p for p in root.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime) if root.is_dir() else []


def harbor_totals(out, current=None):
    """`harbor_outcome` over all of a session's Harbor jobs: usage summed, steps counted.

    `exceptions` are those of the job named `current` (all jobs when None): a job the host stopped at a pause
    records its own cancellation, which is the pause and not a fault; those are reported as `paused_exceptions`."""
    exceptions, earlier, usage, steps = {}, {}, {}, 0
    for job in harbor_jobs(out):
        one = harbor_outcome(job)
        into = exceptions if current is None or job.name == current else earlier
        for kind, count in one['exceptions'].items():
            into[kind] = into.get(kind, 0) + count
        spent = one['usage'] if (job / 'result.json').is_file() else trajectory_usage(job)
        for key, value in (spent or {}).items():
            if value is not None:
                usage[key] = usage.get(key, 0) + value
        steps += one['steps'] or 0
    return dict(exceptions=exceptions, paused_exceptions=earlier, usage=usage or None, steps=steps or None)


def _conduct(out, game, settings, backend, *, task_dir, job, box_tag, deadline, binary, start, find_box, api_base,
             key_file, key_env, reasoning_effort, model_info, agent_kwargs, backend_options, final):
    """The researcher's part of a session, from the first run or after a resume: Harbor in its box, the broker
    serving it, then either the normal close (settle, finalize, record) or a pause (settle, record, nothing closed)."""
    variant = settings['variant']
    research, final_train = settings['research_seconds'], settings['final_train_seconds']
    usd = settings.get('research_usd')
    adapter = agents.get(settings['agent'])
    live = start is None
    result = dict(settings, status='failed', error=None, harbor_returncode=None)
    stop = threading.Event()
    bridge = None
    paused = None
    with backend:
        broker = Broker(backend, out, variant, research_seconds=research, research_usd=usd,
                        final_train_seconds=final_train, deadline=deadline,
                        use_budget=settings['use_budget'], full_session=settings['full_session'])
        if broker.state.get('deadline') != deadline or broker.state.get('paused'):
            # A resumed session: the wall clock runs again from where the pause stopped it.
            broker.state.update(deadline=deadline, paused=None)
            broker.save()
        if live:
            command = harbor_command(binary, task_dir, out / 'harbor', job, settings['model'], agent=settings['agent'],
                                     api_base=api_base, reasoning_effort=reasoning_effort, model_info=model_info,
                                     agent_kwargs=agent_kwargs, box_tag=box_tag)
            write_json(out / ('harbor-command.json' if job == out.name else f'harbor-command-{job}.json'), command)
            environment = harbor_environment(key_file, key_env or adapter['key_env'])
            process = launch_harbor(command, environment, out / 'harbor.log')
            # The box is reached with Harbor's own Modal client; see harness/transport.py.
            bridge = Bridge(Path(binary).parent / 'python', ['--app', APP, '--session', box_tag,
                                                              '--environment', policy.ENVIRONMENT],
                            env={k: v for k, v in environment.items() if not k.endswith('_API_KEY')}, cwd=ROOT)
            find_box = bridge.find
        else:
            process = start()
        server = threading.Thread(target=serve_until, args=(stop, find_box, broker, backend, out / 'broker-errors.jsonl'),
                                  daemon=True)
        server.start()
        try:
            result['harbor_returncode'], request = wait_or_pause(process, out)
            if request is not None:
                # Pause: no more requests are answered, the workspace is read while the box still exists, and
                # only then are the researcher and (below) its box and workers stopped.
                remaining = max(0., deadline - time.time())
                stop.set()
                server.join(timeout=120)
                snapshot = snapshot_workspace(find_box, backend, out, broker.meta)
                paused = dict(request=request, snapshot=snapshot, remaining=remaining)
                for end in ('terminate', 'kill'):
                    if hasattr(process, end):
                        getattr(process, end)()
                        try:
                            process.wait(timeout=20)
                            break
                        except Exception:
                            continue
        except BaseException as exc:  # operator interrupt: stop the researcher, still settle workers below
            result['error'] = type(exc).__name__ + ': ' + str(exc)
            process.kill() if hasattr(process, 'kill') else None
        finally:
            stop.set()
            server.join(timeout=120)
            if bridge is not None:
                try:
                    result['box_terminated_by_host'] = bridge.terminate()
                except Exception as exc:
                    result['box_cleanup_error'] = type(exc).__name__ + ': ' + str(exc)[:300]
                bridge.close()
        if paused is not None and result['error'] is None:
            # Paused: every worker is stopped and settled, and nothing is finalized. The session stays open in
            # its ledger and session.json; `resume` continues it with what is left of its budget and wall clock.
            try:
                backend.settle()
            except Exception as exc:
                result['error'] = 'pause settle: ' + type(exc).__name__ + ': ' + str(exc)
            record = record_pause(out, game, settings, broker, backend, paused['request'], paused['snapshot'], job,
                                  paused['remaining'])
            result.update(status='paused', pause=record, budget=record['budget'],
                          model_usage=harbor_totals(out)['usage'] if live else None)
            write_json(out / 'pauses' / f"{record['index']:03d}" / 'result.json', result)
            return result
        try:
            # The researcher is gone. Nothing of theirs keeps running. The latest submission is the answer; with
            # none, the broker registers the newest complete run of the graded configuration.
            backend.settle()
            if not broker.state['finalized']:
                broker.finalize('host')
        except Exception as exc:
            result['error'] = (result['error'] or '') + ' close: ' + type(exc).__name__ + ': ' + str(exc)
        budget = backend.budget()
        unit = 'usd' if 'budget_usd' in budget else 'seconds'
        total = float(budget.get('budget_' + unit) or (usd if unit == 'usd' else research) or 0)
        result.update(finalized=broker.state['finalized'], closed_by=broker.state['closed_by'],
                      submission=broker.submission, budget=budget,
                      allocation_used_fraction=round(float(budget.get('used_' + unit) or 0) / total, 4) if total else None,
                      automatic_answer=bool(broker.submission and broker.submission.get('auto')),
                      automatic_answer_error=broker.state.get('automatic_answer_error'))
    pauses = len(list((out / 'pauses').glob('*/pause.json'))) if (out / 'pauses').is_dir() else 0
    if pauses:
        result['pauses'] = pauses
    if live:
        outcome = harbor_totals(out, current=job)
        if outcome['paused_exceptions']:
            result['harbor_exceptions_of_paused_jobs'] = outcome['paused_exceptions']
        result.update(harbor_exceptions=outcome['exceptions'], model_usage=outcome['usage'],
                      researcher_steps=outcome['steps'],
                      researcher_records=researcher_records(out, out / 'harbor'))
        unexpected = sorted(set(outcome['exceptions']) - set(EXPECTED_EXCEPTIONS))
        if unexpected and result['error'] is None:
            result['error'] = 'Harbor trial failed: ' + ', '.join(unexpected) + ' (see harbor.log)'
    if result['error'] is None and result['harbor_returncode'] == 0:
        result['status'] = 'researched' if broker.submission else 'no_submission'
    write_json(out / 'session-result.json', result)
    if variant == 'recipe' and final and broker.submission and result['error'] is None:
        result['final_training'] = final_training(out, backend_options=backend_options)
        write_json(out / 'session-result.json', result)
    write_json(out / 'completion.json', policy.completion(out))
    return result


def resume(out, *, api_base=None, key_file=None, key_env=None, model_info=None, agent_kwargs=None, harbor=None,
           backend_options=None, final=True, wall_seconds=None, launcher=None, find_box=None):
    """Continue a paused session where it stopped: the same ledger, budget, submissions and worker storage, the
    workspace as it was, a new box, and the wall clock from what was left. Options not given are the first run's."""
    out = Path(out).resolve()
    binding = policy.run_binding(out)
    settings, game = binding['settings'], binding['game']
    state = json.loads((out / PAUSE).read_text()) if (out / PAUSE).is_file() else {}
    if state.get('state') != 'paused':
        raise ValueError('This session is not paused')
    if json.loads((out / 'session.json').read_text())['finalized']:
        raise ValueError('This session is finalized')
    policy.verify_release(game, settings['release'], code_only=True)
    live = launcher is None
    if live:
        policy.require_modal()
    launch = json.loads((out / LAUNCH).read_text()) if (out / LAUNCH).is_file() else {}
    api_base = api_base or launch.get('api_base')
    key_file = key_file or launch.get('key_file')
    key_env = key_env or launch.get('key_env')
    model_info = model_info if model_info is not None else launch.get('model_info')
    agent_kwargs = agent_kwargs if agent_kwargs is not None else launch.get('agent_kwargs') or ()
    binary = harbor_binary(harbor or launch.get('harbor')) if live else None
    index = state['index']
    remaining = float(wall_seconds or state['wall_remaining_seconds'])
    if remaining < 60:
        raise ValueError('Less than a minute of the wall-clock limit was left at the pause; give --wall-seconds')
    task_dir = out / f'task-resume-{index:03d}'
    job = f'{out.name}-resume{index:03d}'
    box_tag = out.name + '-' + uuid.uuid4().hex[:12]
    task.build(game, settings['variant'], settings['release'], task_dir,
               research_seconds=settings['research_seconds'], research_usd=settings.get('research_usd'),
               budget_mode=settings['budget_mode'], final_train_seconds=settings['final_train_seconds'] or 0,
               wall_seconds=remaining, hardware=settings['hardware'], smoke=settings['smoke'],
               agent=settings['agent'], full_session=settings['full_session'], restarts=settings['restarts'],
               use_budget=settings['use_budget'], continue_from=out / state['seed'])
    options = dict(settings.get('backend_options') or {}, **(backend_options or {}))
    backend = registry.backend(game)(settings['release'], out / 'research', smoke=settings['smoke'], resume=True,
                                     options=options)
    stale = pending_pause(out)  # a request that arrived while the session was already paused asks for nothing new
    state.update(state='resumed', resumed_at=time.time(), resumed_box_tag=box_tag, resumed_job=job,
                 resumed_wall_seconds=remaining,
                 handled=list(state.get('handled', [])) + ([stale['id']] if stale else []))
    write_json(out / 'pauses' / f'{index:03d}' / 'pause.json', state)
    write_json(out / PAUSE, state)
    return _conduct(out, game, settings, backend, task_dir=task_dir, job=job, box_tag=box_tag,
                    deadline=time.time() + remaining, binary=binary,
                    start=None if live else (lambda: launcher(out, task_dir)), find_box=find_box, api_base=api_base,
                    key_file=key_file, key_env=key_env, reasoning_effort=settings.get('reasoning_effort'),
                    model_info=model_info, agent_kwargs=agent_kwargs, backend_options=options, final=final)


def final_training(out, backend_options=None, poll=2.0):
    """Recipe variant: train the final recipe from scratch on a fresh worker, then register that network.

    This runs in its own backend session with its own allowance, so nothing
    from the research session (checkpoints, caches, ledger) can reach it.
    """
    out = Path(out).resolve()
    binding = policy.run_binding(out)
    settings, game = binding['settings'], binding['game']
    session = json.loads((out / 'session.json').read_text())
    if settings['variant'] != 'recipe' or not session['finalized'] or not session['submissions']:
        raise ValueError('Final training needs a finalized recipe session with a submission')
    if (out / 'final').exists():
        raise FileExistsError('Final training was already started for this session; it is never repeated')
    policy.verify_release(game, settings['release'], code_only=True)
    recipe = session['submissions'][-1]
    folder = out / recipe['path'] / 'files'
    files = {name: (folder / name).read_text() for name in recipe['files']}
    allowance = float(settings['final_train_seconds'])
    config = workspace.CONFIGS[1 if settings['smoke'] else 0]
    # A replicate retrain (`harness retrain`) carries an explicit seed; a session's own final does not.
    seed = settings.get('seed')
    mode = registry.spec(game)['retrain_seed']
    expected = dict(recipe['files'])
    if seed is not None and mode == 'config':
        files[config] = seeded_config(files[config], seed)
        expected[config] = workspace.validate({'writable': [config]}, {config: files[config]})[config]
    run_seed = int(seed) if seed is not None and mode == 'run' else 1
    backend = registry.backend(game)(settings['release'], out / 'final', smoke=settings['smoke'],
                                     budget_seconds=allowance, options=backend_options)
    backend.final = True
    record = dict(recipe=recipe['index'], recipe_sha256=recipe['sha256'], allowance_seconds=allowance, config=config,
                  status='failed', error=None)
    if seed is not None:
        record['seed'] = dict(value=int(seed), mode=mode, note=SEED_NOTES[mode])
    with backend:
        try:
            if workspace.apply(backend.release, backend.workspace, files) != expected:
                raise RuntimeError('Recipe bytes differ from the submission')
            # One run takes the whole allowance, less whatever the game's ledger holds back for teardown.
            seconds = min(allowance, float(backend.budget().get('allocatable_seconds', allowance)))
            record['seconds'] = seconds
            record['run'] = backend.run('final', seconds, config, run_seed)
            deadline = time.time() + 2 * seconds + 3600
            while True:
                backend.tick()
                status = backend.status()
                if 'final' not in status.get('active', []):
                    break
                if time.time() > deadline:
                    raise TimeoutError('Final training did not settle')
                time.sleep(poll)
            job = next(j for j in status['jobs'] if j['name'] == 'final')
            # A backend can report the job inactive a moment before its own outcome record is final (Pokémon
            # read "failed" while the worker record was still gaining successful=true). Re-read for a while.
            settle_deadline = time.time() + SETTLE_GRACE_SECONDS
            while job['status'] != 'complete' and time.time() < settle_deadline:
                time.sleep(poll)
                backend.tick()
                status = backend.status()
                job = next(j for j in status['jobs'] if j['name'] == 'final')
            record['job'] = job
            if job['status'] != 'complete':
                raise RuntimeError('Final training ended as ' + str(job['status']))
            record['submission'] = backend.submit('final')
            # The controller may still be settling the finished job (Tavern refused finalize with
            # "Resolve active work"); retry that refusal only, for a bounded time.
            settle_deadline = time.time() + SETTLE_GRACE_SECONDS
            while True:
                try:
                    record['native'] = backend.finalize()
                    break
                except ValueError as exc:
                    if 'active work' not in str(exc).lower() or time.time() > settle_deadline:
                        raise
                    time.sleep(poll)
                    backend.tick()
            record['status'] = 'complete'
        except Exception as exc:
            record['error'] = type(exc).__name__ + ': ' + str(exc)
            backend.settle()
        record['budget'] = backend.budget()
    write_json(out / 'final-train.json', record)
    return record


SEED_NOTES = {'run': 'passed to the trainer as its run seed',
              'config': "written into the config's `seed` field; the recipe's other bytes are unchanged",
              'host': "a label only: the game's fresh final draws its own secret training seeds every time"}


def seeded_config(text, seed):
    """The config with its `seed` field replaced; a config without one cannot be seeded by the host."""
    value = json.loads(text)
    if not isinstance(value, dict) or 'seed' not in value:
        raise ValueError('This recipe\'s config has no `seed` field to set')
    value['seed'] = int(seed)
    return json.dumps(value, indent=2) + '\n'


def retrain(game, release, out, *, seed, session=None, starter=False, smoke=False, final_train_seconds=None,
            label=None, backend_options=None, train=True):
    """Host-only replicate: train one recipe from scratch with an explicit seed, as its own recipe "session".

    The recipe is a finished session's last submission, or the release's own unmodified starter. `out` gets
    the same layout as a recipe session that went straight to its fresh final (binding, session.json with
    one submission, recipes/001, final/, final-train.json), so tournaments take it as an ordinary entrant.
    Nothing is written into the source session, and an existing `out` is refused.
    """
    if bool(session) == bool(starter):
        raise ValueError('Retrain exactly one of: --session SESSION or --starter')
    release = Path(release).resolve()
    out = Path(out).resolve()
    meta = workspace.manifest(release)
    if starter:
        base = workspace.public(release) / workspace.WORKSPACE
        files = {name: (base / name).read_text() for name in meta['writable']}
        source = dict(kind='starter', workspace_sha256=meta['sha256'])
    else:
        source_dir = Path(session).resolve()
        source_settings = policy.run_binding(source_dir)['settings']
        state = json.loads((source_dir / 'session.json').read_text())
        if source_settings['variant'] != 'recipe' or not state['submissions']:
            raise ValueError('The source must be a recipe session with a submission')
        recipe = state['submissions'][-1]
        folder = source_dir / recipe['path'] / 'files'
        files = {name: (folder / name).read_text() for name in recipe['files']}
        if workspace.validate({'writable': list(recipe['files'])}, files) != recipe['files']:
            raise ValueError('The source recipe no longer matches its recorded hashes')
        source = dict(kind='session', session=str(source_dir), recipe=recipe['index'], recipe_sha256=recipe['sha256'],
                      model=source_settings.get('model'), release=source_settings.get('release'))
    # The recipe must fit this release's editable set exactly, before anything is bound or spent.
    hashes = workspace.validate(meta, files)
    if registry.spec(game)['retrain_seed'] == 'config':
        for name in workspace.CONFIGS:
            seeded_config(files[name], seed)
    _, final_train = registry.budgets(game, smoke, None, final_train_seconds)
    settings = json.loads(json.dumps(dict(
        game=game, variant='recipe', kind='retrain', release=str(release), session=str(out), smoke=bool(smoke),
        research_seconds=0.0, final_train_seconds=final_train, seed=int(seed), source=source,
        model=label or ('starter' if starter else source.get('model')), hardware=registry.spec(game)['hardware'],
        environment=policy.ENVIRONMENT, modal_workspace=policy.MODAL_WORKSPACE,
        backend_options={k: v for k, v in (backend_options or {}).items() if not k.startswith('_')}), default=str))
    policy.bind_run(game, release, out, settings)
    out.mkdir(parents=True)
    folder = out / 'recipes' / '001'
    for name, text in files.items():
        target = folder / 'files' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
    entry = dict(index=1, kind='recipe', files=hashes, sha256=workspace.digest(hashes), path='recipes/001',
                 note='host retrain of ' + source['kind'], time=time.time())
    write_json(folder / 'recipe.json', entry)
    write_json(out / 'session.json', dict(schema=1, game=game, variant='recipe', finalized=True, closed_by='host-retrain',
                                          research_seconds=0.0, final_train_seconds=final_train, deadline=None,
                                          submissions=[entry]))
    write_json(out / 'plan.json', settings)
    if not train:
        return dict(settings, trained=False)
    return final_training(out, backend_options=backend_options)


def official(game, settings):
    """A session counts as official only at the game's own declared budgets, in the game's default mode."""
    research, final_train = registry.budgets(game)
    return (not settings['smoke'] and settings.get('budget_mode', registry.spec(game)['budget_mode'])
            == registry.spec(game)['budget_mode'] and settings['research_seconds'] == research
            and settings.get('research_usd') == registry.research_usd(game)
            and (settings['variant'] == 'network' or settings['final_train_seconds'] == final_train))


def grade(out, backend_options=None):
    """Host-private grading of the session's answer: the submitted network, or the recipe's freshly trained one."""
    out = Path(out).resolve()
    binding = policy.run_binding(out)
    settings, game = binding['settings'], binding['game']
    if (out / 'grade.json').exists():
        raise FileExistsError('This session is already graded')
    policy.require_modal()
    policy.verify_release(game, settings['release'], code_only=True)
    session = json.loads((out / 'session.json').read_text())
    if not session['finalized'] or not session['submissions']:
        raise ValueError('Nothing to grade: the session has no finalized submission')
    root = out / 'research'
    if settings['variant'] == 'recipe':
        trained = out / 'final-train.json'
        if not trained.is_file() or json.loads(trained.read_text())['status'] != 'complete':
            raise ValueError('The recipe has no completed fresh final training to grade')
        root = out / 'final'
    backend = registry.backend(game)(settings['release'], root, smoke=settings['smoke'], resume=True,
                                     options=backend_options)
    backend.final = root.name == 'final'
    backend.official = official(game, settings)
    with backend:
        result = backend.grade()
    write_json(out / 'grade.json', dict(variant=settings['variant'], graded_root=root.name,
                                        official=backend.official, result=result))
    write_json(out / 'completion.json', policy.completion(out))
    return result
