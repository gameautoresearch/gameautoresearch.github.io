"""Admit pinned upstream code without changing its model or learning equations."""
from pathlib import Path
import hashlib
import json
import sys
ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
UPSTREAM = ROOT / 'vendor/engines/AverageJoe'
ENGINE = ROOT / 'vendor/engines/generals-bots'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def activate():
    pin=json.loads((HERE/'upstream.json').read_text())
    for name,expected in pin['files'].items():
        if sha(UPSTREAM/name)!=expected:raise ValueError('AverageJoe source drift: '+name)
    sys.path[:0]=[str(UPSTREAM),str(ENGINE)]
    return pin

def binding():
    pin=activate()
    engine={str(p.relative_to(ENGINE)):sha(p) for p in (ENGINE/'generals').rglob('*.py')}
    own={str(p.relative_to(HERE)):sha(p) for p in HERE.glob('*.py')}
    payload={'averagejoe_commit':pin['commit'],'averagejoe_files':pin['files'],'engine_files':engine,'framework_files':own,'recipe_sha256':sha(UPSTREAM/'configs/custom/L_7d_gae90.yaml')}
    return {'sha256':hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest(),'sources':payload}
