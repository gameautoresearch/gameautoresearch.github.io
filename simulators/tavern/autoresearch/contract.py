"""Frozen task and content-addressed source boundary."""
from pathlib import Path
import hashlib,json,os
HERE=Path(__file__).resolve().parent
GAME=HERE.parent
ROOT=GAME.parents[1]
ENGINE=ROOT/'vendor/engines/AutoTavern'
EDITABLE=('policy.py','learner.py','train.py')
RULES=json.loads((HERE/'recipes/starter.json').read_text())['rules']
# The small form: four players, tavern tiers 1-4, three minion types (the eight-player form is at tag full-size-20261113).
BENCHMARK_ID='tavern-four-small-v1'
PLAYERS=RULES['num_players']
def points(placement,players=PLAYERS):return 100*(players-placement)/(players-1)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def binding():
 import taverngym as tg
 if not tg.build_is_current(ENGINE):raise ValueError('Native build stale; rebuild current CPU simulator')
 files=[*sorted((ENGINE/'crates').rglob('*.rs')),ENGINE/'crates/taverngym/data/cards.json',ENGINE/'Cargo.lock',GAME/'rollout.py',GAME/'adapter.py']
 return {'benchmark_id':BENCHMARK_ID,'files':{str(p.relative_to(ROOT)):sha(p) for p in files},'rules':RULES,'native_source_hash':tg.build_info()['source_hash'],'observations':[tg.OBS_FLOATS,tg.OBS_CARDS,tg.N_ACTIONS]}
def validate(c):
 if c['rules']!=RULES:raise ValueError('Frozen game rules required')
 if not 0<=c['gamma']<=1 or c['opponent'] not in ('scripted','selfplay') or not c['learning_seats'] or len(set(c['learning_seats']))!=len(c['learning_seats']) or any(s not in range(PLAYERS) for s in c['learning_seats']):raise ValueError('Invalid learning configuration')
 if c['native_threads']>4 or c.get('torch_threads',1)>4 or c['n_games'] not in range(1,257):raise ValueError('Resource/cohort bounds')
 if c['network']['hidden'] not in range(8,257) or c['network']['embedding'] not in range(1,33):raise ValueError('Architecture bounds')
 if c.get('start_turn',1)!=1:raise ValueError('Full-from-start task only')

def verify_private_release(release,full=False):
 release=Path(release);manifest=json.loads((release/'release.json').read_text());private=release/'private'
 inventory=private/'opponents.json';panel=private/('panel-full.json' if full else 'panel.json')
 if not inventory.exists() or sha(inventory)!=manifest.get('opponent_inventory_sha256'):raise ValueError('Missing/mutated committed opponent inventory')
 if not panel.exists() or sha(panel)!=manifest.get('full_private_panel_sha256' if full else 'private_panel_sha256'):raise ValueError('Missing/mutated committed private panel')
 data=json.loads(inventory.read_text());schedule=json.loads(panel.read_text());selected=data['full' if full else 'smoke']
 if len(selected)!=(3 if full else 1) or len(schedule['seeds'])!=(16 if full else 2) or len(set(schedule['seeds']))!=len(schedule['seeds']) or schedule['games_per_seat']!=(4 if full else 1):raise ValueError('Declared panel dimensions differ from contract')
 for name in selected:
  matches=[x for x in data['opponents'] if x['name']==name]
  if len(matches)!=1:raise ValueError('Missing declared opponent')
  item=matches[0];op=(private/item['path']).resolve()
  if not op.is_relative_to(private.resolve()):raise ValueError('Invalid opponent path')
  if any(not (op/f).is_file() or sha(op/f)!=h for f,h in item['files'].items()):raise ValueError('Declared frozen opponent missing or mutated')
 return {'opponents':selected,'seed_groups':len(schedule['seeds']),'games_per_seat':schedule['games_per_seat'],'episodes':len(selected)*len(schedule['seeds'])*PLAYERS*schedule['games_per_seat']}
