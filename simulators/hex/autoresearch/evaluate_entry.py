"""Practice or private grading inside an evaluation worker. Reads a root-only request, writes public/private results."""
from pathlib import Path
import argparse
import json

from .arena import evaluate, PolicyFailure
from .bundle import unpack_bundle
from .contract import write_json
from .grading import grade


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--request', default='/private/evaluation.json')
    p.add_argument('--public', default='/job/evaluation-public.json')
    p.add_argument('--private', default='/private/evaluation-private.json')
    a = p.parse_args()
    c = json.loads(Path(a.request).read_text())
    archive = Path(c['candidate'] + '.tar')
    if archive.is_file():
        unpack_bundle(archive, c['candidate'])
    development = c.get('development', False)
    try:
        result = evaluate(c['candidate'], c['panel'], opening_seed=c['opening_seed'], policy_seed=c['policy_seed'],
                          openings=c['openings'], split=c['split'], development=development, workers=c.get('workers'))
    except PolicyFailure as exc:
        result = {'status': 'invalid_policy', 'raw_score': 0, 'error_type': type(exc).__name__, 'error': str(exc),
                  'natural_losses_awarded': 0}
        write_json(a.private, result)
        write_json(a.public, result)
        return
    write_json(a.private, result)
    public = grade(result['rows'], development=development)
    public.update(seconds=result['seconds'], candidate_decisions=result['candidate_decisions'], split=result['split'],
                  mohex_restarts=len(result['mohex_restarts']), position_seconds=result['position_seconds'],
                  call_seconds=result['call_seconds'],
                  move_timeouts=result['move_timeouts'], random_moves=result['random_moves'])
    write_json(a.public, public)


if __name__ == '__main__':
    main()
