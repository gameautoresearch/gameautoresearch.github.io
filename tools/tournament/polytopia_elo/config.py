"""A tournament's config: one JSON file per campaign, its directory is the tournament's output folder.

    {
      "name": "polytopia-empty15-20261118",          # labels the Modal app, Daytona resources and the report
      "root": "/abs/path/of/the/checkout",            # optional: base of relative paths (default: this repository)
      "title": "polytopia-empty15",                   # figure / report title (default: name)
      "release": "path/to/release",                   # the tournament release (relative paths: repository root)
      "equivalent_releases": ["path/to/other"],       # play-equivalent releases (tools/tournament/polytopia.py)
      "sessions": {"astra": ["GPT-6 Astra", "runs/local/.../astra6/a1"], ...},
      "anchors": {"ppo15": "runs/local/.../anchors/ppo15-s1"},     # Elo 0 (the mean of the anchors)
      "references": ["random", "starter-20m", "starter-65m", "starter-200m"],  # from the release's host-private panel
      "final_sessions": {"astra": "runs/.../retrain"}, # optional: a host retrain standing in for a failed final
      "continues": {"astra+": "astra"}                # optional: a session continuing an earlier one (listed before
    }                                                 # it); its networks sit after that session's GPU-hours and time
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def load(path):
    path = Path(path).resolve()
    c = json.loads(path.read_text())
    root = Path(c.get('root', REPO))
    resolve = lambda p: str((root / p).resolve()) if not Path(p).is_absolute() else p
    c['release'] = resolve(c['release'])
    c['equivalent_releases'] = [resolve(p) for p in c.get('equivalent_releases', [])]
    c['sessions'] = {k: [label, resolve(p)] for k, (label, p) in c.get('sessions', {}).items()}
    c['anchors'] = {k: resolve(p) for k, p in c.get('anchors', {}).items()}
    c['final_sessions'] = {k: resolve(p) for k, p in c.get('final_sessions', {}).items()}
    c.setdefault('references', [])
    c.setdefault('title', c['name'])
    c['dir'] = str(path.parent)
    return c
