"""Entrants of a Polytopia checkpoint-Elo tournament (read-only on every session), from the tournament's config.

Every research network each session sealed (a `package` job of its research ledger: training/launch.py or a manual
`ar package`; its bundle verified against the ledger's artifact digest), each session's fresh final (the recipe
retrained by the host), the anchors (Elo 0), and references from the release's host-private panel (checksums checked
against its inventory). For each research network: the run, the lease it was packaged from and that lease's grant,
its practice score(s), which formal submissions match it exactly (trainer sources, architecture and training config),
and where it sits in the session: research allocation charged (training-worker hours; leases charge seconds x their
shape's rate) and hours since the session's first `ar` request, at its lease's start and when it was sealed.

    PYTHONPATH=$PWD .venv-polytopia-ar/bin/python -m tools.tournament.polytopia_elo.entrants --config CONFIG.json

Writes entrants.json and entrants-meta.json beside the config. Pull cloud sessions first (a bundle missing locally is
skipped and listed in the meta, never entered unverified).
"""
import argparse, hashlib, json, sqlite3, sys
from pathlib import Path

from simulators.polytopia.autoresearch.v1 import core
from tools.tournament.polytopia import equivalent_releases
from tools.tournament.polytopia_elo import config as configs


def session_bundle(session):
    """A recipe session's (or host retrain's) fresh final network, from the host's own `final-train.json`: complete,
    finalized, and the sealed bundle `final/submissions/<sha>` verifying to that sha. Ledgers are not opened: a pulled
    copy can pair a final database with a stale WAL the coordinator mirror never deletes (and opening one read-write
    changes the record)."""
    record = Path(session) / 'final-train.json'
    if not record.is_file():
        raise ValueError(f'{session} has no final-train.json')
    final = json.loads(record.read_text())
    native = final.get('native') or {}
    if final.get('status') != 'complete' or not native.get('finalized') or not native.get('submission'):
        raise ValueError(f"{session} final is {final.get('status')}: {final.get('error')}")
    bundle = Path(session) / 'final/submissions' / native['submission']
    if core.verify(bundle) != native['submission']:
        raise ValueError(f'Final bundle of {session} differs from its record')
    return bundle

# Workspace file -> native path inside a run's sealed source (harness/workspace.py TRAINING['polytopia']).
NATIVE = {'training/policy.py': 'src/polytopia_bot/rl/network.py', 'training/learner.py': 'src/polytopia_bot/rl/ppo.py',
          'training/train.py': 'src/polytopia_bot/rl/gpu_resident.py',
          'training/collector.py': 'experiments/gpu_resident_train.py', 'configs/architecture.json': 'arch.json'}
PLAY_FILES = ('final.npz', 'final.json', 'network.py', 'ar_arch.json')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def first_request(session):
    times = []
    for line in (session / 'requests.jsonl').read_text().splitlines():
        try:
            times.append(json.loads(line)['time'])
        except (ValueError, KeyError):
            pass
    return min(times)


def snapshot(db_path):
    """A private copy of a (possibly live, WAL-mode) SQLite database with its -wal/-shm, so nothing of the session's
    own files is opened for writing."""
    import shutil, tempfile
    tmp = Path(tempfile.mkdtemp(prefix='elo-ledger-'))
    main = Path(db_path)
    wal = Path(str(db_path) + '-wal')
    # A pulled cloud copy keeps a WAL the coordinator mirror never deletes: one older than the database is stale (its
    # frames predate the final checkpoint) and would roll the database back, so it is left out.
    fresh_wal = wal.is_file() and wal.stat().st_mtime >= main.stat().st_mtime
    for suffix in ('', '-wal', '-shm') if fresh_wal else ('',):
        src = Path(str(db_path) + suffix)
        if src.is_file():
            shutil.copyfile(src, tmp / (Path(db_path).name + suffix))
    return sqlite3.connect(str(tmp / Path(db_path).name))


def ledger_rows(session):
    db = snapshot(Path(session) / 'research/ledger.sqlite3')
    jobs = {}
    for name, status, grant, started, ended, charged, payload, artifact, artifact_sha in db.execute(
            'select name,status,grant,started,ended,charged,payload,artifact,artifact_sha from jobs'):
        try:
            payload = json.loads(payload or '{}')
        except ValueError:
            payload = {}
        jobs[name] = dict(name=name, status=status, grant=grant, started=started, ended=ended, charged=charged or 0.0,
                          payload=payload, artifact=artifact, artifact_sha=artifact_sha)
    events = [dict(at=at, kind=kind, data=json.loads(data or '{}'))
              for at, kind, data in db.execute('select at,kind,data from events order by id')]
    db.close()
    return jobs, events


def charged_at(jobs, rates, t):
    """Research allocation (training-worker seconds) charged by moment t: settled jobs in full, running ones pro rata."""
    total = 0.0
    for j in jobs.values():
        rate = rates.get(j['name'], 0.0)
        if not rate or not j['started'] or j['started'] > t:
            continue
        end = j['ended'] if j['ended'] else t
        if j['ended'] and j['ended'] <= t:
            total += j['charged']
        else:
            total += (min(end, t) - j['started']) * rate
    return total


def submissions_of(session):
    state = json.loads((session / 'session.json').read_text())
    return [(s['index'], s['files']) for s in state.get('submissions', [])]


def matches(bundle, config_name, files):
    """True when the sealed sources equal a submission's files (training sources, architecture, train config)."""
    source = bundle / 'source'
    for workspace_name, native in NATIVE.items():
        path = source / native
        if workspace_name not in files:
            continue
        if not path.is_file() or sha(path) != files[workspace_name]:
            return False
    recipe = source / 'recipe.json'
    return recipe.is_file() and sha(recipe) == files.get('configs/train.json')


SKIPPED = []


def research_entrants(key, label, session):
    session = Path(session)
    if not (session / 'research/ledger.sqlite3').is_file():
        print(f'skip {key}: no research ledger at {session}', file=sys.stderr)
        return [], None
    t0 = first_request(session)
    jobs, events = ledger_rows(session)
    rates = {}
    for e in events:
        if e['kind'] == 'queued' and 'rate' in e['data']:
            rates[e['data']['name']] = float(e['data']['rate'])
    packaged_at = {e['data']['name']: e['at'] for e in events if e['kind'] == 'packaged'}
    practice = {}
    for j in jobs.values():
        if j['payload'].get('kind') == 'practice' and j['status'] == 'complete':
            report = session / 'research/jobs' / j['name'] / 'practice.json'
            if report.is_file():
                practice.setdefault(j['payload'].get('run'), []).append(
                    round(json.loads(report.read_text())['raw_score_0_100'], 2))
    subs = submissions_of(session)
    out = []
    for j in sorted(jobs.values(), key=lambda x: packaged_at.get(x['name'], x['started'] or 0)):
        if j['payload'].get('kind') != 'package' or j['status'] != 'complete' or not j['artifact']:
            continue
        # The copy beside the session being read comes first: the ledger records the path where the job ran (a cloud
        # coordinator's, or this machine's original folder, which may since have been cleaned and restored elsewhere).
        bundle = session / 'research/jobs' / j['name'] / 'bundle'
        if not bundle.is_dir():
            bundle = Path(j['artifact'])
        # A bundle the last pull caught mid-write (or never got) is skipped and recorded, never entered unverified.
        try:
            whole = core.verify(bundle) == j['artifact_sha']
        except (OSError, ValueError, KeyError) as exc:
            whole = False
        if not whole:
            SKIPPED.append(dict(entrant=f"{key}:{j['name']}", bundle=str(bundle), reason='missing or does not match its ledger digest'))
            print('SKIP', key, j['name'], 'bundle missing or not matching its ledger digest', flush=True)
            continue
        lease = jobs.get(j['payload'].get('lease'), {})
        sealed = packaged_at.get(j['name'], j['started'])
        out.append(dict(
            name=f"{key}:{j['name']}", kind='research', model=key, label=label, run=j['name'], bundle=str(bundle),
            bundle_sha256=j['artifact_sha'], release_sha256=json.loads((bundle / 'binding.json').read_text())['release_sha256'],
            lease=lease.get('name'), grant_seconds=lease.get('grant'), lease_shape=(lease.get('payload') or {}).get('shape'),
            config=j['payload'].get('config'), practice=practice.get(j['name'], []),
            submissions=[i for i, files in subs if matches(bundle, j['payload'].get('config'), files)],
            gpu_h_start=round(charged_at(jobs, rates, lease.get('started') or sealed) / 3600, 4),
            gpu_h_end=round(charged_at(jobs, rates, sealed) / 3600, 4),
            wall_h_start=round(((lease.get('started') or sealed) - t0) / 3600, 4),
            wall_h_end=round((sealed - t0) / 3600, 4)))
    total = charged_at(jobs, rates, max([j['ended'] or 0 for j in jobs.values()] + [0])) / 3600
    end_wall = (max([j['ended'] or 0 for j in jobs.values()] + [t0]) - t0) / 3600
    return out, dict(total_gpu_h=round(total, 4), end_wall_h=round(end_wall, 4))


def final_entrant(key, label, session, totals):
    try:
        bundle = session_bundle(session)
    except Exception as exc:  # no complete final: reported, not entered
        print(f'skip {key} final: {exc}', file=sys.stderr)
        return None
    return dict(name=f'{key}:final', kind='final', model=key, label=label, run='final', bundle=str(bundle),
                bundle_sha256=core.verify(bundle),
                release_sha256=json.loads((Path(bundle) / 'binding.json').read_text())['release_sha256'],
                gpu_h_end=(totals or {}).get('total_gpu_h'), wall_h_end=(totals or {}).get('end_wall_h'),
                submissions=[], practice=[], grant_seconds=None)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    a = p.parse_args()
    c = configs.load(a.config)
    sessions, anchors, refs = c['sessions'], c['anchors'], c['references']
    out = Path(c['dir']) / 'entrants.json'
    release = Path(c['release']).resolve()
    accepted = equivalent_releases(release, c['equivalent_releases'])
    entrants, totals = [], {}
    continues = c.get('continues', {})
    for key, (label, session) in sessions.items():
        research, tot = research_entrants(key, label, Path(session))
        # A session whose own fresh final failed on a host fault may name a host retrain of its last submission (same
        # recipe and seed, `harness retrain`) to stand in for it: "final_sessions": {key: retrain dir}.
        replacement = c.get('final_sessions', {}).get(key)
        final = final_entrant(key, label, Path(replacement) if replacement else Path(session), tot) if tot is not None else None
        if final and replacement:
            final['final_source'] = str(replacement)
        if final:
            research.append(final)
        totals[key] = tot
        if key in continues:
            # A session that continues an earlier one (harness --continue-from): its networks are placed after the
            # earlier session's whole spend and session time, on the same model's line.
            parent = continues[key]
            if parent not in totals or totals[parent] is None:
                raise ValueError(f'{key} continues {parent}, which must be listed (and read) before it')
            shift = dict(gpu=totals[parent]['total_gpu_h'], wall=totals[parent]['end_wall_h'])
            for e in research:
                e.update(continues=parent, gpu_h_offset=shift['gpu'], wall_h_offset=shift['wall'])
                for field, base in (('gpu_h_start', 'gpu'), ('gpu_h_end', 'gpu'), ('wall_h_start', 'wall'),
                                    ('wall_h_end', 'wall')):
                    if e.get(field) is not None:
                        e[field] = round(e[field] + shift[base], 4)
        entrants += research
    for name, session in anchors.items():
        bundle = session_bundle(session)
        entrants.append(dict(name=name, kind='anchor', model='anchor', label='anchor',
                             run=name, bundle=str(bundle), bundle_sha256=core.verify(bundle),
                             release_sha256=json.loads((Path(bundle) / 'binding.json').read_text())['release_sha256'],
                             submissions=[], practice=[], grant_seconds=None, gpu_h_end=None, wall_h_end=None))
    inventory = {e['name']: e for e in json.loads((release / 'host-private-panel/inventory.json').read_text())['panel']}
    for name in refs:
        folder = release / 'host-private-panel' / name
        if sha(folder / 'final.npz') != inventory[name]['checkpoint_sha256'] or \
                sha(folder / 'network.py') != inventory[name]['network_sha256']:
            raise ValueError(f'reference {name} differs from the panel inventory')
        entrants.append(dict(name=name, kind='reference', model='reference', label='ladder reference', run=name,
                             bundle=str(folder), bundle_sha256=inventory[name]['checkpoint_sha256'],
                             release_sha256=core.verify(release), submissions=[], practice=[], grant_seconds=None,
                             gpu_h_end=None, wall_h_end=None))
    for e in entrants:
        if e['release_sha256'] not in accepted:
            raise ValueError(f"{e['name']} is sealed against a release outside {sorted(accepted)}")
        missing = [f for f in PLAY_FILES if not (Path(e['bundle']) / f).is_file()]
        if missing:
            raise ValueError(f"{e['name']} lacks {missing}")
    names = [e['name'] for e in entrants]
    if len(set(names)) != len(names):
        raise ValueError('duplicate entrant names')
    out.write_text(json.dumps(entrants, indent=1) + '\n')
    meta = dict(release=str(release), release_sha256=core.verify(release), accepted=accepted, totals=totals, skipped=SKIPPED,
                counts={k: sum(e['kind'] == k for e in entrants) for k in ('research', 'final', 'anchor', 'reference')})
    out.with_name('entrants-meta.json').write_text(json.dumps(meta, indent=1) + '\n')
    print(json.dumps(meta['counts']), len(entrants), 'entrants ->', out)


if __name__ == '__main__':
    main()
