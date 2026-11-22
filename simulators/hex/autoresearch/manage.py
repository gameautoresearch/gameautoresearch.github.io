"""Host-private grading of a finalized session.

Called by the host only, with a provider (harness.games.hex.provider.ModalProvider);
nothing here is reachable from a researcher box. The grade is the raw panel score
anchored on MoHex: there is no starter/reference normalization to calibrate first.
"""
from pathlib import Path
import json
import time

from .bundle import verify
from .contract import digest, write_json
from .ledger import Ledger


def panel(release, which):
    return json.loads((Path(release) / 'private/panel.json').read_text())[which]


def grade_session(release, session, provider, development=False, openings=None):
    """Grade the submission held at every milestone point. `openings` (development only) grades a prefix."""
    root = Path(release)
    ledger = Ledger(session)
    state = ledger.finalize()
    seeds = json.loads((root / 'private/grade-seeds.json').read_text())
    full_commitment = digest(seeds)
    if openings is not None:
        if not development or not 0 < openings <= seeds['openings']:
            raise ValueError('A smaller private schedule is development-only')
        seeds = dict(seeds, openings=openings)
    opponents = panel(root, 'official')
    attempt = Path(session) / 'grading-private' / str(time.time_ns())
    attempt.mkdir(parents=True, mode=0o700, exist_ok=False)
    attempt.parent.chmod(0o700)
    write_json(attempt / 'schedule-commitment.json', {
        'full_seed_commitment': full_commitment, 'executed_schedule_commitment': digest(seeds),
        'openings': seeds['openings'], 'panel': opponents, 'gpu_count': 0, 'release': str(root),
        'development': development}, exclusive=True)
    results, cache = {}, {}
    for point in state['points']:
        sub = next((s for s in state['submissions'] if s['id'] == point['submission']), None)
        if sub is None:
            results[point['fraction']] = {'status': 'no_submission', 'raw_score': 0}
            continue
        job = next(j for j in ledger.jobs() if j['name'] == sub['job'])
        verify(job['artifact'], sub['artifact_sha'], check_tensors=False)
        if sub['artifact_sha'] not in cache:
            cache[sub['artifact_sha']] = provider.evaluate(job['artifact'], opponents, seeds, development)
        result = cache[sub['artifact_sha']]
        if result.get('allocation'):
            write_json(attempt / (point['fraction'] + '-allocation.json'), result['allocation'], exclusive=True)
        write_json(attempt / (point['fraction'] + '.json'), result['private'], exclusive=True)
        results[point['fraction']] = dict(result['public'], submission=sub['id'], artifact_sha256=sub['artifact_sha'])
    write_json(attempt / 'grades.json', results, exclusive=True)
    write_json(Path(session) / 'latest-grade.json', {'attempt': str(attempt), 'results': results})
    return results
