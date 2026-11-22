#!/usr/bin/env python3
"""Normalize the six-game 5-hour recipe campaign into the trajectory-visualization input contract.

Reads the retained records under ``evidence/six-game-5h-20261111/`` read-only (SQLite opened with
``mode=ro``) and writes ``reports/six-game-5h-2026-11-12/normalized-runs.json``. Standard library only.

    python3 tools/viz/six_game_5h_timelines.py [--output PATH] [--reconcile]

Render with:

    python3 .agents/skills/trajectory-visualization/scripts/build.py \
        reports/six-game-5h-2026-11-12/normalized-runs.json \
        --output reports/six-game-5h-2026-11-12/trajectories.html
"""
import argparse
import hashlib
import json
import re
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN = ROOT / 'evidence/six-game-5h-20261111'
DEFAULT_OUTPUT = ROOT / 'reports/six-game-5h-2026-11-12/normalized-runs.json'

GAMES = [  # display order: by game, DeepSeek seat first
    ('generals', 'Generals', 'glm53-prime'),
    ('doudizhu', 'Dou Dizhu', 'glm53-prime'),
    ('pokemon', 'Pokémon', 'opus55'),
    ('polytopia', 'Polytopia', 'opus55'),
    ('tavern', 'Tavern', 'glm53'),
    ('clash', 'Clash', 'glm53'),
]
SEAT_NAMES = {'deepseek-v41flash': 'DeepSeek V4.1 Flash', 'opus55': 'Opus 5.5',
              'glm53-prime': 'GLM-5.3 Prime', 'glm53': 'GLM-5.3'}
GPU_PRACTICE_GAMES = {'generals', 'doudizhu', 'pokemon'}
RESEARCH_SECONDS = 18000.0
WALL_MINUTES = 330

BUCKETS = [
    {'id': 'train-new', 'label': 'Train: new recipe version', 'color': '#edb447',
     'description': 'A training run whose editable source/config differs from every earlier run of the session '
                    '(source snapshot hashes; see timing note).'},
    {'id': 'train-repeat', 'label': 'Train: repeat of an earlier version', 'color': '#5b9dc9',
     'description': 'A training run whose editable source/config matches an earlier run of the session '
                    '(often a longer or reseeded rerun).'},
    {'id': 'practice', 'label': 'Public practice', 'color': '#a37cc6',
     'description': 'Public practice evaluation. On an L40S worker in Generals, Dou Dizhu and Pokémon; on CPU '
                    'workers outside the GPU allocation in Clash, Polytopia and Tavern.'},
    {'id': 'gpu-wait', 'label': 'Never got a GPU', 'color': '#ed896d',
     'description': 'Worker created at the provider but never started (L40S capacity); settled by the operator '
                    'reaper, 0 s charged. Interval: sandbox creation to settlement.'},
    {'id': 'final', 'label': 'Fresh final training (host)', 'color': '#43a98e',
     'description': 'The host retraining the last submitted recipe from scratch (up to 3,600 s). One event per '
                    'attempt.'},
]

# Operator recovery facts from the owning readout (reports/six-game-5h-2026-11-12.md), keyed by agent ID.
OPERATOR_NOTES = {
    'generals-deepseek-v41flash': 'Agent finalize was refused while a never-started practice job stayed unresolved; '
                                  'after the reaper settled it, the operator re-ran the standard host close and the '
                                  'host trained the fresh final.',
    'doudizhu-deepseek-v41flash': 'Fresh final retried: attempt 1 cancelled by the operator\'s duplicate finals '
                                  'manager (operator bug), attempt 2 never got a GPU, attempt 3 trained.',
    'polytopia-deepseek-v41flash': 'Fresh final retried: attempt 1 never got a GPU; attempt 2 trained.',
    'pokemon-deepseek-v41flash': 'Final worker succeeded but final_training read it as failed (backend race); the '
                                 'operator verified completion and ran the remaining submit and finalize steps.',
    'pokemon-opus55': 'Final worker succeeded but final_training read it as failed (backend race); the operator '
                      'verified completion and ran the remaining submit and finalize steps.',
    'tavern-deepseek-v41flash': 'Final trained and was submitted, then finalize raced the controller settling the '
                                'job; the operator re-ran finalize once no work was active.',
}


# ----------------------------------------------------------------------------- helpers

def load(path):
    return json.loads(Path(path).read_text())


def maybe(path):
    path = Path(path)
    return load(path) if path.is_file() else None


def ro_query(db, sql, args=()):
    con = sqlite3.connect(f'file:{db}?mode=ro', uri=True)
    con.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in con.execute(sql, args)]
    finally:
        con.close()


def clean(text, limit=320):
    """One-line, path-sanitized error summary (last meaningful line of a traceback)."""
    if not text:
        return ''
    text = str(text).replace(str(ROOT) + '/', '').replace(str(ROOT), '.')
    text = re.sub(r'/Users/[^\s\'"]+', '<local path>', text)
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    if not lines:
        return ''
    line = lines[-1] if len(lines) > 1 else lines[0]
    line = line.strip('; ').strip()
    if line.endswith('; termination:'):
        line = line[:-len('; termination:')].strip()
    if line in ('termination:', ''):
        return ''
    timeout = re.search(r'TimeoutExpired: .* timed out after ([0-9.]+) seconds', line)
    if timeout:
        line = f'subprocess.TimeoutExpired: trainer command timed out after {float(timeout.group(1)):.0f} s (worker deadline)'
    if line[-1] not in '.…':
        line += '.'
    return line if len(line) <= limit else line[:limit - 1] + '…'


def norm_status(raw):
    raw = (raw or '').lower()
    if raw in ('complete', 'completed', 'succeeded', 'success'):
        return 'succeeded'
    if raw in ('stopped', 'cancelled', 'canceled'):
        return 'stopped'
    if raw in ('failed', 'error'):
        return 'failed'
    return raw or 'unknown'


def receipt_times(prov):
    """Provider start/finish from a receipt; None when the provider never started the task."""
    if not prov:
        return None, None
    s, f = prov.get('started_at') or None, prov.get('finished_at') or None
    return s, f


def job(name, kind, start, end, status, error='', charged=None, resource='L40S', provider_id=None,
        created=None, timing='', extra=''):
    return dict(name=name, kind=kind, start=start, end=end, status=norm_status(status), error=clean(error),
                charged=charged, resource=resource, provider_id=provider_id, created=created, timing=timing,
                extra=extra)


def mins(seconds):
    return f'{seconds / 60:.0f} min'


# ----------------------------------------------------------------------------- per-game adapters
# Each returns a list of job dicts for one ledger directory (research or a final attempt).

def sqlite_ledger_jobs(ledger_dir, final=False):
    """Generals / Dou Dizhu: ledger.sqlite3 jobs + jobs/<name>/allocation.json provider receipts."""
    ledger_dir = Path(ledger_dir)
    rows = ro_query(ledger_dir / 'ledger.sqlite3',
                    'select name,status,grant,started,ended,charged,payload,provider_id,error from jobs')
    events = ro_query(ledger_dir / 'ledger.sqlite3', 'select at,kind,data from events order by id')
    out = []
    for r in rows:
        kind = 'final' if final else (json.loads(r['payload'] or '{}').get('kind') or 'train')
        alloc = maybe(ledger_dir / 'jobs' / r['name'] / 'allocation.json') or {}
        prov = alloc.get('provider') or {}
        s, f = receipt_times(prov)
        timing = 'provider receipt (jobs/<name>/allocation.json)'
        if s is None and r['started']:
            s, f, timing = r['started'], r['ended'], 'ledger started/ended'
        if s is None:  # never started: bound by this job's ledger events
            ats = [e['at'] for e in events if json.loads(e['data'] or '{}').get('name') == r['name']]
            if ats:
                s, f, timing = min(ats), max(ats), 'ledger events (queued to last event); no provider start recorded'
        out.append(job(r['name'], 'practice' if kind == 'practice' else kind, s, f, r['status'], r['error'],
                       r['charged'], provider_id=r['provider_id'] or prov.get('provider_id'),
                       created=prov.get('created_at'), timing=timing, extra=f'grant {r["grant"]:.0f} s'))
    return out


def pokemon_jobs(ledger_dir, final=False):
    """Pokémon: ledger.sqlite3 jobs with JSON receipt; success from jobs/<id>/output/completed.json."""
    ledger_dir = Path(ledger_dir)
    rows = ro_query(ledger_dir / 'ledger.sqlite3',
                    'select id,name,kind,seconds,status,actual_seconds,receipt,created from jobs')
    out = []
    for r in rows:
        rec = json.loads(r['receipt'] or '{}')
        jdir = ledger_dir / 'jobs' / r['id']
        req = maybe(jdir / 'request.json') or {}
        kind = 'final' if final else ('practice' if req.get('action') == 'assess' or r['name'].startswith('eval-')
                                      else 'train')
        done = maybe(jdir / 'output/completed.json')
        err_log = jdir / 'output/worker-error.log'
        if done and done.get('status') == 'complete':
            status, error = 'complete', ''
        elif err_log.is_file():
            status, error = 'failed', err_log.read_text(errors='replace')
        else:
            status, error = r['status'], 'no completion record retained'
        s, f = receipt_times(rec)
        out.append(job(r['name'], kind, s, f, status, error, r['actual_seconds'], provider_id=rec.get('provider_id'),
                       created=rec.get('created_at'), timing='provider receipt (ledger receipt JSON)',
                       extra=f'grant {r["seconds"]:.0f} s'))
    return out


def polytopia_jobs(ledger_dir, final=False):
    """Polytopia: gpu-ledger.json (L40S training) + state.json runs/evaluations (CPU practice)."""
    ledger_dir = Path(ledger_dir)
    gpu = load(ledger_dir / 'gpu-ledger.json')
    state = maybe(ledger_dir / 'state.json') or {}
    runs = state.get('runs') or {}
    out = []
    for j in gpu['jobs']:
        prov = j.get('provider') or {}
        s, f = receipt_times(prov)
        st = runs.get(j['name'], {})
        error = j.get('failure') or st.get('error') or ''
        if s is None and st.get('started_at'):
            s, f = st.get('started_at'), st.get('finished_at')
        out.append(job(j['name'], 'final' if final else 'train', s, f, j['status'], error, j.get('seconds'),
                       provider_id=prov.get('provider_id'), created=prov.get('created_at'),
                       timing='provider receipt (gpu-ledger.json)', extra=f'grant {j["grant"]:.0f} s'))
    if not final:
        for name, ev in (state.get('evaluations') or {}).items():
            res = ev.get('result') or {}
            extra = f'practice of {ev.get("run")}'
            if res.get('raw_score_0_100') is not None:
                extra += f'; public practice score {res["raw_score_0_100"]:.1f} over {res.get("games")} games'
            out.append(job(name, 'practice', ev.get('started_at'), ev.get('finished_at'), ev.get('status'),
                           ev.get('error'), None, resource='CPU',
                           timing='state.json evaluation started/finished (CPU workers; no per-worker receipt kept)',
                           extra=extra))
    return out


def clash_jobs(ledger_dir, final=False):
    """Clash: allocation-ledger.json (L40S) + session-jobs.json practice + jobs/<run>/practice.provider.json (CPU)."""
    ledger_dir = Path(ledger_dir)
    ledger = load(ledger_dir / 'allocation-ledger.json')
    out = []
    for a in ledger['allocations']:
        rec = a.get('receipt') or {}
        s, f = receipt_times(rec)
        out.append(job(a['name'], 'final' if final else 'train', s, f, a['status'], a.get('error'),
                       a.get('charged_seconds'), provider_id=rec.get('provider_id'), created=rec.get('created_at'),
                       timing='provider receipt (allocation-ledger.json)',
                       extra=f'reserved {a.get("reserved_seconds", 0):.0f} s'))
    if not final:
        for j in maybe(ledger_dir / 'session-jobs.json') or []:
            if j.get('kind') != 'practice':
                continue
            pp = maybe(ledger_dir / 'jobs' / j.get('run', j['name'][5:]) / 'practice.provider.json') or {}
            prov = pp.get('provider') or {}
            s, f = receipt_times(prov)
            timing = 'CPU provider receipt (jobs/<run>/practice.provider.json)'
            if s is None:
                s, f, timing = j.get('started_at'), j.get('finished_at'), 'session-jobs.json started/finished'
            res = j.get('result') or {}
            extra = f'practice of {j.get("run")}'
            if res.get('raw_score_0_100') is not None:
                extra += f'; public practice score {res["raw_score_0_100"]:.1f} over {res.get("games")} games'
            out.append(job(j['name'], 'practice', s, f, j.get('status'), j.get('error'), None, resource='CPU',
                           provider_id=prov.get('provider_id'), timing=timing, extra=extra))
    return out


def tavern_jobs(ledger_dir, final=False):
    """Tavern: worker-ledger.sqlite allocations (L40S) + state.json + <name>-provider.json CPU practice receipts."""
    ledger_dir = Path(ledger_dir)
    rows = ro_query(ledger_dir / 'worker-ledger.sqlite', 'select name,reserved,actual,receipt,created from allocations')
    state = maybe(ledger_dir / 'state.json') or {}
    runs = state.get('runs') or {}
    out = []
    for r in rows:
        rec = json.loads(r['receipt'] or '{}')
        s, f = receipt_times(rec)
        st = runs.get(r['name'], {})
        out.append(job(r['name'], 'final' if final else 'train', s, f, st.get('status') or 'unknown',
                       st.get('error'), r['actual'], provider_id=rec.get('provider_id'), created=rec.get('created_at'),
                       timing='provider receipt (worker-ledger.sqlite)', extra=f'reserved {r["reserved"]:.0f} s'))
    if not final:
        for name, p in (state.get('practice') or {}).items():
            prov = maybe(ledger_dir / f'practice{name}-provider.json') or {}
            s, f = receipt_times(prov)
            res = maybe(ledger_dir / f'practice-{name}/result.json') or {}
            extra = f'practice of {name}'
            if res.get('raw_score') is not None:
                extra += f'; public practice score {res["raw_score"]:.1f} over {res.get("observed_cases")} lobbies'
            out.append(job(f'practice-{name}', 'practice', s, f, p.get('status'), p.get('error'), None,
                           resource='CPU', provider_id=prov.get('provider_id'),
                           timing='CPU provider receipt (practice<name>-provider.json)', extra=extra))
    return out


ADAPTERS = {
    'generals': (sqlite_ledger_jobs, 'research'),
    'doudizhu': (sqlite_ledger_jobs, 'research'),
    'pokemon': (pokemon_jobs, 'research'),
    'polytopia': (polytopia_jobs, 'research'),
    'tavern': (tavern_jobs, 'research'),
    'clash': (clash_jobs, 'research'),
}


# ----------------------------------------------------------------------------- recipe identity

def source_identity(session, run):
    """Hash of the run's editable source snapshot: every file in sources/<run>/run.json except the
    configs/{smoke,train}.json that this run did not select."""
    meta = maybe(session / 'sources' / run / 'run.json')
    if not meta:
        return None, None
    files = dict(meta['files'])
    selected = meta.get('config')
    for cfg in ('configs/smoke.json', 'configs/train.json'):
        if cfg != selected:
            files.pop(cfg, None)
    blob = json.dumps(sorted(files.items())).encode()
    return hashlib.sha256(blob).hexdigest(), selected


# ----------------------------------------------------------------------------- session build

def requests(session):
    return [json.loads(l) for l in (session / 'requests.jsonl').open()]


def final_attempts(game, session):
    fn, _ = ADAPTERS[game]
    dirs = sorted(session.glob('final-attempt-*'), key=lambda p: int(p.name.rsplit('-', 1)[1]))
    dirs.append(session / 'final')
    out = []
    for i, d in enumerate(dirs, 1):
        if not d.is_dir():
            continue
        for j in fn(d, final=True):
            j['attempt'] = i
            j['attempt_dir'] = d.name
            out.append(j)
    return out


def build_agent(game, game_label, seat, reaper, stats, finals_log):
    session = CAMPAIGN / game / seat / 'main'
    rel_session = f'{game}/{seat}/main'
    agent_id = f'{game}-{seat}'
    plan = load(session / 'plan.json')
    sess = load(session / 'session.json')
    reqs = requests(session)
    fn, sub = ADAPTERS[game]
    research = fn(session / sub)
    finals = final_attempts(game, session)
    st = stats[(game, seat)]

    reaped = {r['receipt']['provider_id']: r for r in reaper if r['session'] == rel_session}
    reaped_names = {r['job']: r for r in reaper if r['session'] == rel_session}

    def reaper_entry(j, is_final):
        r = reaped.get(j.get('provider_id'))
        if r is None and not is_final:
            r = reaped_names.get(j['name'])
        return r

    for j in research + finals:
        r = reaper_entry(j, j['kind'] == 'final')
        if r:
            j['reaper'] = r
            j['start'] = r['receipt']['created_at']
            j['end'] = r['at']
            j['timing'] = 'sandbox created_at (provider receipt) to reaper settlement (reaper-log.jsonl)'

    missing = [f'{agent_id}/{j["name"]}' for j in research + finals if j['start'] is None or j['end'] is None]

    first_req = min(r['time'] for r in reqs)
    first_job = min((j['start'] for j in research if j['start'] is not None), default=first_req)
    origin = min(first_req, first_job)

    # recipe identity, in start order
    events, seen = [], {}
    research.sort(key=lambda j: (j['start'] if j['start'] is not None else float('inf')))
    counts = {'train': 0, 'train_new': 0, 'train_repeat': 0, 'practice': 0, 'gpu_wait': 0,
              'train_gpu_seconds': 0.0, 'practice_gpu_seconds': 0.0, 'gpu_wait_train': 0, 'gpu_wait_practice': 0}
    for j in research:
        if j['start'] is None:
            continue
        s, e = j['start'] - origin, j['end'] - origin
        detail_bits = []
        if j.get('reaper'):
            bucket = 'gpu-wait'
            status = 'no GPU (0 s charged)'
            title = f'{"Practice" if j["kind"] == "practice" else "Training run"} {j["name"]}: never got a GPU'
            detail_bits.append('Provider created the sandbox but never started a task (no task id); the backend '
                               'marked it uncertain and the operator reaper terminated it and settled the job '
                               f'failed, charged 0. Waited {mins(e - s)}.')
            if j['error']:
                detail_bits.append(f'Ledger error: {j["error"]}')
            counts['gpu_wait'] += 1
            counts['gpu_wait_practice' if j['kind'] == 'practice' else 'gpu_wait_train'] += 1
            include = False
            if j['kind'] != 'practice':
                ident, cfg = source_identity(session, j['name'])
                if ident:
                    detail_bits.append(f'Source {ident[:10]} ({cfg}); '
                                       + (f'same source as {seen[ident]}.' if ident in seen else 'not run before.'))
        elif j['kind'] == 'practice':
            bucket = 'practice'
            status = j['status']
            gpu = game in GPU_PRACTICE_GAMES
            include = gpu
            title = f'Practice {j["name"]}'
            detail_bits.append(f'Public practice on {"an L40S worker (charged to the research allocation)" if gpu else "a CPU worker outside the GPU allocation"}. '
                               f'{j["extra"]}.')
            if j['charged'] is not None:
                detail_bits.append(f'Charged {j["charged"]:.0f} s.')
                if gpu:
                    counts['practice_gpu_seconds'] += j['charged'] or 0
            counts['practice'] += 1
        else:
            ident, cfg = source_identity(session, j['name'])
            counts['train'] += 1
            if ident is None:
                bucket = 'train-new'
                detail_bits.append('No source snapshot retained; classified as new.')
            elif ident in seen:
                bucket = 'train-repeat'
                detail_bits.append(f'Source {ident[:10]} ({cfg}) repeats {seen[ident]}.')
            else:
                bucket = 'train-new'
                detail_bits.append(f'Source {ident[:10]} ({cfg}): first run of this version.')
            counts['train_new' if bucket == 'train-new' else 'train_repeat'] += 1
            if ident and ident not in seen:
                seen[ident] = j['name']
            status = j['status']
            include = True
            title = f'Train {j["name"]}'
            detail_bits.append(f'{j["extra"]}; charged {j["charged"] or 0:.0f} s on L40S.')
            counts['train_gpu_seconds'] += j['charged'] or 0
        if j['error'] and not j.get('reaper'):
            detail_bits.append(f'Error: {j["error"]}')
        detail_bits.append(f'Timing: {j["timing"]}.')
        events.append(dict(id=f'r-{j["name"]}', start=round(s, 3), end=round(e, 3), bucket=bucket, title=title,
                           short_label=j['name'], detail=' '.join(detail_bits), status=status,
                           include_in_aggregate=include))

    final_end = None
    for j in finals:
        if j['start'] is None:
            continue
        s, e = j['start'] - origin, j['end'] - origin
        label = f'Fresh final attempt {j["attempt"]}' if len([f for f in finals]) > 1 else 'Fresh final'
        bits = [f'Host retrain of the last submitted recipe ({j["attempt_dir"]}/).']
        if j.get('reaper'):
            bucket, status, include = 'gpu-wait', 'no GPU (0 s charged)', False
            bits.append('Provider created the sandbox but never started a task; the operator reaper terminated it '
                        f'and settled the attempt failed, charged 0. Waited {mins(e - s)}.')
            title = f'{label}: never got a GPU'
        else:
            bucket, status = 'final', j['status']
            ran = bool(j['charged'])
            include = ran
            title = label
            bits.append(f'{j["extra"]}; charged {j["charged"] or 0:.0f} s on L40S.' if ran else
                        'No GPU time charged; no provider start/finish recorded (interval = ledger events).')
            if j['status'] == 'succeeded':
                final_end = max(final_end or 0, e)
        if j['error'] and not j.get('reaper'):
            bits.append(f'Error: {j["error"]}')
        for fl in finals_log:
            if fl.get('session') == rel_session and fl.get('action') == 'set-aside' and fl.get('attempt') == j['attempt']:
                bits.append('Finals log: ' + clean(fl.get('reason'), 400))
        if j is finals[-1] and agent_id in OPERATOR_NOTES:
            bits.append('Operator recovery: ' + OPERATOR_NOTES[agent_id])
        bits.append(f'Timing: {j["timing"]}.')
        events.append(dict(id=f'f-{j["attempt"]}-{j["name"]}', start=round(s, 3), end=round(e, 3), bucket=bucket,
                           title=title, short_label=f'final {j["attempt"]}' if len(finals) > 1 else 'final',
                           detail=' '.join(bits), status=status, include_in_aggregate=include))

    # markers
    markers = []
    train_events = [ev for ev in events if ev['bucket'] in ('train-new', 'train-repeat')]

    def latest_train(at):
        before = [ev for ev in train_events if ev['start'] <= at]
        return before[-1]['id'] if before else None

    n = 0
    for r in reqs:
        if r['request'].get('op') == 'submit' and not r['error']:
            n += 1
            at = r['time'] - origin
            m = dict(at=round(at, 3), label=f'S{n}', title=f'Submit {n} (ar submit) at {mins(at)}')
            eid = latest_train(at)
            if eid:
                m['event_id'] = eid
                m['title'] += f'; latest training run started before it: {eid[2:]}'
            markers.append(m)
    fin = [r for r in reqs if r['request'].get('op') == 'finalize' and not r['error']]
    refused = [r for r in reqs if r['request'].get('op') == 'finalize' and r['error']]
    if fin:
        finalize_at = fin[0]['time'] - origin
        m = dict(at=round(finalize_at, 3), label='F', title=f'Researcher finalize at {mins(finalize_at)}')
    else:
        finalize_at = sess['finalized_at'] - origin
        m = dict(at=round(finalize_at, 3), label='F',
                 title=f'Host close at {mins(finalize_at)} (session.json finalized_at; no researcher finalize '
                       f'succeeded, {len(refused)} refused)')
    eid = latest_train(finalize_at)
    if eid:
        m['event_id'] = eid
    markers.append(m)

    # end of timeline
    recovery_ats = []
    ft = maybe(session / 'final-train.json') or {}
    if isinstance(ft.get('operator_recovery'), dict) and ft['operator_recovery'].get('at'):
        recovery_ats.append(ft['operator_recovery']['at'])
    orc = maybe(session / 'operator-recovery.json')
    if orc and orc.get('at'):
        recovery_ats.append(orc['at'])
    session_end = max([r['time'] for r in reqs] + [sess.get('finalized_at') or 0] + recovery_ats) - origin
    end = max([session_end, final_end or 0] + [ev['end'] for ev in events] + [mk['at'] for mk in markers])

    used = st['research_used_seconds']
    cost = st['model_cost_usd']
    notional = plan['model'].startswith('anthropic/')
    facts = [
        {'label': 'research wall', 'value': f'{st["research_wall_minutes"]:.0f} of {WALL_MINUTES} min'},
        {'label': 'research GPU used', 'value': f'{used:,.0f} of {RESEARCH_SECONDS:,.0f} s ({100 * used / RESEARCH_SECONDS:.0f}%)'},
        {'label': 'runs requested', 'value': st['runs']},
        {'label': 'practice requested', 'value': st['practice']},
        {'label': 'submits', 'value': st['submits']},
        {'label': 'model cost', 'value': f'${cost:.2f}' + (' (notional; subscription)' if notional else '')},
    ]
    summary = (f'{counts["train"]} training runs on a GPU ({counts["train_new"]} new versions, '
               f'{counts["train_repeat"]} repeats), {counts["practice"]} practice jobs, '
               f'{counts["gpu_wait"]} jobs that never got a GPU; '
               f'{"finalized" if fin else "host-closed"} at {mins(finalize_at)} of {WALL_MINUTES}, '
               f'using {100 * used / RESEARCH_SECONDS:.0f}% of the research GPU allowance.')

    agent = dict(id=agent_id, name=f'{game_label} · {SEAT_NAMES[seat]}', subtitle=plan['model'], summary=summary,
                 end_seconds=round(end, 3), end_label='Fresh final complete', facts=facts, events=events,
                 markers=markers)
    finals_ok = [ev for ev in events if ev['bucket'] == 'final' and ev['status'] == 'succeeded']
    info = dict(counts=counts, used=used, stats=st, finalize_at=finalize_at, by_researcher=bool(fin),
                refused=len(refused), missing=missing, origin=origin, final_attempts=len(finals),
                final_seconds=[f['charged'] for f in finals if f['status'] == 'succeeded'],
                final_trained=sum(1 for _ in finals_ok))
    return agent, info


def end_cell(agent_id, info):
    pct = 100 * info['used'] / RESEARCH_SECONDS
    at = info['finalize_at'] / 60
    if info['by_researcher']:
        title = f'Finalized at {at:.0f} of {WALL_MINUTES} min; used {pct:.0f}% of GPU allowance'
    else:
        title = (f'Finalize refused {info["refused"]}×; host close at {at:.0f} min; used {pct:.0f}% of GPU allowance')
    detail = OPERATOR_NOTES.get(agent_id, 'No operator recovery; the session closed and its final trained normally.')
    if info['final_seconds']:
        detail += f' Fresh final charged {info["final_seconds"][-1]:,.0f} s of 3,600.'
    return {'title': title, 'detail': detail}


def outcome(game, seat, other_label):
    res = load(CAMPAIGN / game / 'tournament/result.json')
    p = res['pairs'][0]
    a_score = p['a_score_0_100']
    score = a_score if p['a'] == seat else 100 - a_score
    lo, hi = p['ci95']
    if p['a'] != seat:
        lo, hi = 100 - hi, 100 - lo
    won = score > 50
    detail = (f'Head-to-head score 0–100 for this seat: {score:.1f}; 95% CI {lo:.1f}–{hi:.1f} over {p["games"]} '
              f'games, both seats/sides, private maps/deals/seeds; {p.get("faults", 0)} faults, '
              f'{p.get("truncations", 0)} truncations.')
    if game == 'generals':
        detail += f' {p["truncations"]}/{p["games"]} games hit the turn cap (draw = 0.5); draw-dominated.'
    if game == 'clash':
        detail += ' The CI includes 50: a lean, not a clear win.'
    if game == 'tavern':
        detail += ' A game is a shared 8-player lobby; a point = placed above the other entrant.'
    detail += ' Compares the two fresh-final networks of this game only.'
    return {'title': f'Tournament vs {other_label}', 'value': f'{score:.1f} ({"won" if won else "lost"})',
            'detail': detail}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument('--reconcile', action='store_true', help='print the reconciliation table to stdout')
    args = ap.parse_args()

    reaper = [json.loads(l) for l in (CAMPAIGN / 'reaper-log.jsonl').open() if l.strip()]
    stats = {(r['game'], r['seat']): r for r in load(CAMPAIGN / 'stats.json')}
    finals_log = [json.loads(l) for l in (CAMPAIGN / 'finals-log.jsonl').open() if l.strip()]

    agents, infos, outcomes, cells = [], {}, [], {}
    for game, label, challenger in GAMES:
        seats = ['deepseek-v41flash', challenger]
        for seat in seats:
            agent, info = build_agent(game, label, seat, reaper, stats, finals_log)
            agents.append(agent)
            infos[agent['id']] = info
            other = seats[1] if seat == seats[0] else seats[0]
            outcomes.append(dict(agent_id=agent['id'], **outcome(game, seat, SEAT_NAMES[other])))
            cells[agent['id']] = end_cell(agent['id'], info)

    data = {
        'schema_version': 1,
        'title': 'Six games, two models, five hours.',
        'subtitle': ('Twelve Terminus-2 recipe sessions: DeepSeek V4.1 Flash against one challenger per game, each with '
                     '18,000 s of L40S research allocation in a 330-minute window, then a host retrain of the last '
                     'submitted recipe for up to 3,600 s. The final verdict is a head-to-head tournament per game.'),
        'kicker': 'Six-game recipe campaign / 2026-11-11–12',
        'time_origin': ('Elapsed wall time from each session\'s own start: the earlier of its first researcher request '
                        '(requests.jsonl) and its first research job start. Sessions were launched at different '
                        'times (Polytopia ~14 min, Tavern ~70 min and Clash ~1.7 h after Generals/Dou Dizhu/Pokémon), '
                        'so equal positions on different rows are not simultaneous.'),
        'timing_note': (
            'Job intervals are provider start→finish receipts where retained: Generals/Dou Dizhu '
            'jobs/<name>/allocation.json, Pokémon ledger receipt JSON, Polytopia gpu-ledger.json, Clash '
            'allocation-ledger.json and CPU practice.provider.json, Tavern worker-ledger.sqlite and CPU '
            'practice<name>-provider.json. Polytopia CPU practice uses state.json evaluation start/finish (no '
            'per-worker receipt). Never-started jobs run from sandbox created_at to the reaper settlement time. '
            'One Dou Dizhu final attempt has no provider start and spans its ledger queued→last event. '
            'New vs repeat: each run\'s source snapshot (sources/<run>/run.json file SHA-256s, excluding the '
            'unselected one of configs/smoke.json and configs/train.json) is hashed and compared with every earlier '
            'run of the same session in start order; run length and seed are not part of the identity. '
            'S markers are accepted ar submit requests; F is the successful researcher finalize request (Generals '
            'DeepSeek: host close time from session.json). Each marker links the latest training run started before '
            'it. Authoring time between runs was not measured.'),
        'buckets': BUCKETS,
        'aggregate': {
            'label': 'L40S worker time by purpose',
            'note': ('Cumulative allocated L40S worker time (provider start→finish) across the 12 sessions: research '
                     'training, GPU practice in Generals/Dou Dizhu/Pokémon, and fresh finals. Excludes CPU practice '
                     '(Clash/Polytopia/Tavern) and never-started workers (0 s charged). Polytopia training workers '
                     'carry 16 CPU cores and 64 GiB; the others 4 cores and 16 GiB.'),
        },
        'agents': agents,
        'comparisons': [{'label': 'How the session ended', 'cells': cells}],
        'outcomes': outcomes,
        'notes': [
            'One attempt per cell; no seed repeats. Pokémon, Generals, Clash and Tavern differences rest on 64–288 games.',
            'Each tournament score compares the two fresh-final networks of one game only; there is no cross-game '
            'ranking or reference normalization.',
            'L40S capacity starvation: 9 jobs (Generals ×4, Dou Dizhu ×3, Polytopia ×2, two of them fresh finals) '
            'were created but never started; their "Never got a GPU" intervals are waiting time, charged 0 s, and '
            'blocked their sessions while unresolved.',
            'Fresh finals shorter than 3,600 s are the recipe\'s own stopping point (e.g. Generals GLM-5.3 Prime, '
            'Tavern DeepSeek), not truncation by the host.',
            'Gaps between jobs are not measured thinking or coding time; model request latency is not shown.',
        ],
        'sources': [
            {'label': 'Campaign readout', 'href': '../six-game-5h-2026-11-12.md'},
            {'label': 'Tournament tool', 'href': '../../tools/tournament/README.md'},
        ],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=1, ensure_ascii=False) + '\n')
    print(f'wrote {args.output.relative_to(ROOT) if args.output.is_relative_to(ROOT) else args.output}')

    if args.reconcile:
        hdr = ('agent', 'train ev', 'runs(stats)', 'gpu-wait tr', 'practice ev', 'gpu-wait pr', 'practice(stats)',
               'train GPU s', 'GPU practice s', 'used(stats)', 'diff', 'finals', 'missing')
        print(' | '.join(hdr))
        for aid, i in infos.items():
            c, st = i['counts'], i['stats']
            tot = c['train_gpu_seconds'] + c['practice_gpu_seconds']
            print(' | '.join(str(x) for x in (
                aid, c['train'], st['runs'], c['gpu_wait_train'], c['practice'], c['gpu_wait_practice'],
                st['practice'], f'{c["train_gpu_seconds"]:.0f}', f'{c["practice_gpu_seconds"]:.0f}',
                f'{i["used"]:.0f}', f'{i["used"] - tot:.1f}', i['final_attempts'], ','.join(i['missing']) or '-')))


if __name__ == '__main__':
    main()
