from pathlib import Path
import hashlib,json,sys
HERE=Path(__file__).resolve().parent
GAME=HERE.parent
ROOT=GAME.parents[1]
sys.path.insert(0,str(GAME))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');t.replace(p)
def binding():
 import taverngym as tg
 ENGINE=ROOT/"vendor/engines/AutoTavern"
 if not tg.build_is_current(ENGINE) or tg.build_info()['source_hash']!='217577b83fc4525e':raise ValueError('Only qualified admission-v2 native runtime accepted')
 files=[*sorted((ENGINE/'crates').rglob('*.rs')),ENGINE/'crates/taverngym/data/cards.json',ENGINE/'Cargo.lock',*sorted(GAME.glob('*.py')),*sorted(HERE.glob('*.py'))]
 return dict(schema='tavern-reference-population-v1',native_source_hash=tg.build_info()['source_hash'],files={str(p.relative_to(ROOT)):sha(p) for p in files},observations=[tg.OBS_FLOATS,tg.OBS_CARDS,tg.N_ACTIONS])
