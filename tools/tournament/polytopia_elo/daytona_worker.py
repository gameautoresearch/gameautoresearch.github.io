"""Inside a Daytona worker sandbox: play one call (a job as JSON in argv[1]) and print its result as one JSON line.

The sandbox holds the release's runtime, grader and maps (the snapshot) and the entrants' play files on the
tournament volume at /bundles; job.py plays exactly as in the Modal backend.
"""
import json
import sys

sys.path.insert(0, '/opt/elo')
import job as runner  # noqa: E402

spec = json.loads(sys.argv[1])
res = runner.play('/bundles/' + spec['a'].replace(':', '_'), '/bundles/' + spec['b'].replace(':', '_'),
                  spec['seeds'], spec['b'])
print('RESULT ' + json.dumps(res), flush=True)
