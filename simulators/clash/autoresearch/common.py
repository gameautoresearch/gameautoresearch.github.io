from pathlib import Path
import hashlib,json,sys,importlib.util,os,tempfile
HERE=Path(__file__).resolve().parent
GAME=HERE.parent
ROOT=GAME.parent.parent
NATIVE=GAME/'native'
DATA=NATIVE/'generated-v2/task-data.json'
EDITABLE=('policy.py','learner.py','train.py')
# The match form this research runtime trains, practises and grades (native/match_forms.py): 60 s + 30 s
# overtime at 2 decisions per simulated second. The pinned 180+120 s, 4-per-second form is at git tag
# full-size-20261113. Trusted: the editable config must name the same form, and bundles bind it.
MATCH_FORM='short-v1'
# The host writes this file inside a training worker to ask the trusted loop to end at its next update; the
# run then checkpoints, exports and is collected exactly like one that reached its time limit.
STOP_REQUEST='/tmp/clash-ar-stop-requested'
def dump(path,value):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
 payload=json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n'
 fd,name=tempfile.mkstemp(prefix='.'+p.name+'-',dir=p.parent)
 try:
  with os.fdopen(fd,'w') as stream:stream.write(payload);stream.flush();os.fsync(stream.fileno())
  os.replace(name,p)
 finally:
  if os.path.exists(name):os.unlink(name)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def setup_native():
 sys.path.insert(0,str(NATIVE));sys.path.insert(0,str(GAME));sys.path.insert(0,str(ROOT/'vendor/engines/crforge/python'))
def engine_binding():
 setup_native()
 from data_binding import validate_data
 import match_forms
 return {'match_form':match_forms.describe(MATCH_FORM),'data':validate_data(),'source':{str(p.relative_to(GAME)):sha(p) for p in sorted(NATIVE.rglob('*')) if p.is_file() and p.suffix in ('.py','.cpp','.cu','.hpp','.cuh') and not any(v in ('.build','__pycache__') or v.startswith('generated') for v in p.parts)},'rules':sha(GAME/'rules.lock.json')}
def load_module(path,name):
 spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod
