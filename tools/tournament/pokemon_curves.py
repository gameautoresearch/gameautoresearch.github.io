"""Research curves for the Pokémon 3v3 campaigns: every intermediate research network of each counted attempt.

    python -m tools.tournament.pokemon_curves enumerate --out DIR/manifest.json
    python -m tools.tournament.pokemon_curves analyze --manifest DIR/manifest.json --result DIR/parts... --out DIR

`enumerate` lists, for the 21 counted attempts of poke3v3-gpt6-20261115 (Sol, Astra, Luna) and
poke3v3-batch-20261114 (DeepSeek, Kimi, GLM, Inkling), every research training job from the session's own stage
records (`research/stages/<run>.json`), with its provider start/finish times (hours since the session started:
`session.json` deadline - `session-result.json` wall_seconds), the bundle it sealed, public-practice scores of that
run (`research/practice/*.json`), and whether the session's final recipe has exactly that run's files
(`final.json` submission -> `recipes/NNN/recipe.json` files == `sources/<run>/run.json` files). Each bundle is checked
with the release's native verifier and binding; unsuccessful or non-verifying jobs are recorded as skipped.
Also lists each attempt's fresh-final bundle and the three starter anchors. Reads only; writes only --out.

`analyze` turns the anchor games (tools.tournament.pokemon_checkpoints results) into per-bundle lift, per-attempt
summaries, the final-bundle sanity check against analysis-combined.json, and the figures.
"""
import sys

sys.dont_write_bytecode = True

import argparse
import json
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GPT6 = ROOT / 'runs/local/poke3v3-gpt6-20261115'
BATCH = ROOT / 'runs/cloud/poke3v3-batch-20261114'
RELEASE = ROOT / 'evidence/releases-20261114-main3/pokemon'
COUNTED = {
    'sol61': (GPT6, ('a1', 'a2', 'a3')), 'astra6': (GPT6, ('a1', 'a2', 'a3')), 'luna6': (GPT6, ('a1', 'a2', 'a3')),
    'deepseek-v4pro': (BATCH, ('a2', 'b1', 'b2')), 'kimi-k3': (BATCH, ('a1', 'a2', 'a3')),
    'glm53': (BATCH, ('a1', 'a2', 'a3')), 'inkling2s': (BATCH, ('a1', 'b2', 'b3')),
}
LABELS = {'sol61': 'GPT-6.1 Sol', 'astra6': 'GPT-6 Astra', 'luna6': 'GPT-6 Luna', 'deepseek-v4pro': 'DeepSeek V4 Pro',
          'kimi-k3': 'Kimi K3', 'glm53': 'GLM-5.3', 'inkling2s': 'Inkling 2 Small'}
SHORT = {'sol61': 'sol', 'astra6': 'astra', 'luna6': 'luna', 'deepseek-v4pro': 'ds', 'kimi-k3': 'kimi',
         'glm53': 'glm', 'inkling2s': 'ink'}
ANCHORS = ('s1r', 's2', 's3r')


def read(path):
    return json.loads(Path(path).read_text())


def check(bundle, manifest, expected=None):
    """(sha, None) when the bundle verifies against the release; (None, reason) otherwise."""
    from tools.tournament import pokemon
    try:
        sealed = pokemon.native().verify(bundle, expected)
        binding = read(Path(bundle) / 'binding.json')
        if binding.get('runtime_sha256') != manifest['protected_runtime_sha256']:
            return None, 'bound to a different native runtime than the release'
        if binding.get('benchmark_id') != manifest['benchmark_id']:
            return None, 'different benchmark'
        return sealed['bundle_sha256'], None
    except Exception as exc:
        return None, f'{type(exc).__name__}: {str(exc)[:300]}'


def session_clock(session):
    state, result = read(session / 'session.json'), read(session / 'session-result.json')
    return state['deadline'] - result['wall_seconds'], result


def final_files(session):
    path = read(session / 'final.json')['submission']['path']
    recipe = read(session / path / 'recipe.json')
    return path, recipe['files']


def practices(session):
    """{run name: [(time finished, score or None, eval name)]} from the session's practice records."""
    out = {}
    for path in sorted((session / 'research/practice').glob('*.json')):
        record = read(path)
        if record.get('kind') != 'practice':
            continue
        finished = (record.get('provider') or {}).get('finished_at')
        score = record.get('score_0_100') if record.get('score_valid') else None
        out.setdefault(record.get('run'), []).append(dict(eval=record.get('name'), finished_at=finished, score=score,
                                                          bundle_sha256=record.get('bundle_sha256')))
    return out


def allocation_before(session):
    """{operation: allocated seconds settled before that job was created} from the research ledger (read-only)."""
    db = session / 'research/ledger.sqlite3'
    connection = sqlite3.connect(f'file:{db}?mode=ro', uri=True)
    try:
        rows = connection.execute('SELECT id, created, actual_seconds FROM jobs ORDER BY created').fetchall()
    finally:
        connection.close()
    out, used = {}, 0.0
    for ident, created, seconds in rows:
        out[ident] = used
        used += seconds or 0.0
    return out


def enumerate_attempt(model, base, attempt, manifest):
    session = base / 'pokemon' / model / attempt
    start, result = session_clock(session)
    recipe_path, recipe = final_files(session)
    practice = practices(session)
    before = allocation_before(session)
    jobs, skipped = [], []
    for stage_path in sorted((session / 'research/stages').glob('*.json')):
        stage = read(stage_path)
        if stage.get('kind') != 'train':
            continue
        name, operation = stage['name'], stage['operation']
        provider = stage.get('provider') or {}
        run = session / 'sources' / name / 'run.json'
        files = read(run)['files'] if run.is_file() else None
        row = dict(model=model, attempt=attempt, run=name, operation=operation,
                   job_dir=str(session / 'research/jobs' / operation),
                   requested_seconds=stage.get('seconds'),
                   start_h=(provider['started_at'] - start) / 3600 if provider.get('started_at') else None,
                   end_h=(provider['finished_at'] - start) / 3600 if provider.get('finished_at') else None,
                   allocation_before_s=before.get(operation),
                   is_final_recipe=files is not None and files == recipe,
                   practice=practice.get(name, []))
        bundle = session / 'research/jobs' / operation / 'output/stage/bundle'
        reason = None
        if not stage.get('successful'):
            reason = 'host marked the run unsuccessful: ' + str(stage.get('error'))[:300]
        elif not bundle.is_dir():
            reason = 'no sealed bundle in the job output'
        else:
            sha, reason = check(bundle, manifest, stage.get('bundle_sha256'))
        if reason:
            skipped.append(dict(row, reason=reason))
            continue
        jobs.append(dict(row, bundle=str(bundle), bundle_sha256=sha))
    jobs.sort(key=lambda r: r['end_h'])
    final_bundle = session / 'final/submissions/final'
    sha, reason = check(final_bundle, manifest)
    final = dict(bundle=str(final_bundle), bundle_sha256=sha, error=reason, recipe=recipe_path,
                 matching_runs=[j['run'] for j in jobs + skipped if j['is_final_recipe']])
    return dict(model=model, attempt=attempt, session=str(session), session_start=start,
                research_wall_h=(result['wall_seconds']) / 3600, jobs=jobs, skipped=skipped, final=final)


def cmd_enumerate(args):
    from tools.tournament import pokemon
    manifest = pokemon.release_manifest(RELEASE)
    attempts = [enumerate_attempt(model, base, attempt, manifest)
                for model, (base, names) in COUNTED.items() for attempt in names]
    anchors = {}
    for name in ANCHORS:
        bundle = BATCH / 'pokemon/starter' / name / 'final/submissions/final'
        sha, reason = check(bundle, manifest)
        if reason:
            raise SystemExit(f'anchor {name} does not verify: {reason}')
        anchors['starter-' + name] = dict(bundle=str(bundle), bundle_sha256=sha)
    entrants = {}
    for a in attempts:
        prefix = f"{SHORT[a['model']]}-{a['attempt']}"
        for i, job in enumerate(a['jobs'], 1):
            job['entrant'] = f'{prefix}-r{i:02d}'
            entrants[job['entrant']] = dict(bundle=job['bundle'], bundle_sha256=job['bundle_sha256'])
        if a['final']['bundle_sha256']:
            a['final']['entrant'] = f'{prefix}-final'
            entrants[a['final']['entrant']] = dict(bundle=a['final']['bundle'],
                                                   bundle_sha256=a['final']['bundle_sha256'])
    value = dict(release=str(RELEASE), release_sha256=manifest['release_sha256'], anchors=anchors,
                 entrants=entrants, attempts=attempts)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise SystemExit(f'{out} exists')
    out.write_text(json.dumps(value, indent=2) + '\n')
    print(json.dumps(dict(attempts=len(attempts), research_bundles=sum(len(a['jobs']) for a in attempts),
                          skipped=[(a['model'], a['attempt'], s['run'], s['reason'][:80]) for a in attempts
                                   for s in a['skipped']],
                          finals=sum(bool(a['final']['bundle_sha256']) for a in attempts),
                          per_attempt={f"{a['model']}/{a['attempt']}": len(a['jobs']) for a in attempts}), indent=1))


def main(argv=None):
    p = argparse.ArgumentParser(prog='python -m tools.tournament.pokemon_curves')
    sub = p.add_subparsers(dest='command', required=True)
    e = sub.add_parser('enumerate')
    e.add_argument('--out', required=True)
    a = sub.add_parser('analyze')
    a.add_argument('--manifest', required=True)
    a.add_argument('--result', action='append', required=True, help='a pokemon_checkpoints output folder')
    a.add_argument('--combined', default=str(GPT6 / 'analysis-combined.json'))
    a.add_argument('--out', required=True)
    args = p.parse_args(argv)
    if args.command == 'enumerate':
        cmd_enumerate(args)
    else:
        from tools.tournament import pokemon_curves_report
        pokemon_curves_report.run(args)


if __name__ == '__main__':
    main()
