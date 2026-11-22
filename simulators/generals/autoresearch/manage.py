"""Host-private evaluation: grade a finalized session and calibrate the reference anchors.

Both take an already constructed provider (`harness/games/generals/provider.py`).
Sessions are run through `python -m harness`; there is no command line here.
"""
from pathlib import Path
import json
from .contract import DEFAULT,TRAIN_SHAPE,BUNDLED,digest,rate,write_json
from .bundle import source_payload,verify
from .ledger import Ledger
from .grading import calibration_gate,multi_seed_gate

def panel_paths(release,which):
 root=Path(release)/'private'
 catalog=json.loads((root/'panel.json').read_text())
 if catalog.get(which) is None:raise ValueError('Official panel is not calibrated')
 return [dict(p,bundle=str((root/p['bundle']).resolve())) if p['kind'] in BUNDLED else p for p in catalog[which]]

def grade_session(release,session,provider,development=False,maps=None,catalog=None,timeout=None,output=None,contract_usd=None):
 # contract_usd: the dollar allowance an official session of this kind must have run at.
 # A network session runs at the contract's research allowance; a recipe's fresh
 # final training runs at the harness's declared final-training allowance.
 release=Path(release);session=Path(session);ledger=Ledger(session);state=ledger.finalize()
 expected=DEFAULT['budget_usd'] if contract_usd is None else contract_usd
 if not development and ledger.config['budget_usd']!=expected:raise ValueError('Smoke sessions cannot receive full-budget grades')
 if not development and maps is not None and maps!=DEFAULT['grade_maps']:raise ValueError('Official grading uses the complete frozen map schedule')
 selected=json.loads(Path(catalog).read_text()) if catalog else None
 panel=selected['official'] if selected else panel_paths(release,'official')
 anchors=selected['anchors'] if selected else json.loads((release/'private/anchors.json').read_text())
 if selected and anchors.get('release_commitment')!=digest(json.loads((release/'release.json').read_text())):raise ValueError('Calibration belongs to another release')
 if selected and anchors.get('grade_seed_commitment')!=digest(json.loads((release/'private/grade-seeds.json').read_text())):raise ValueError('Calibration belongs to another private grading schedule')
 count=maps or DEFAULT['grade_maps']
 actual=len(json.loads((release/'private/grade-seeds.json').read_text())['seeds'])
 if count>actual:raise ValueError('Not enough prepared private maps')
 schedule={'pool_path':str(release/'private/grade-maps.npz'),'map_indices':list(range(count)),'split':'heldout'}
 # Scripted opponents alone graded in 650-1,100 s per network on an L40S (768 games, suite-20261113). Each
 # learned opponent adds 256 games in which two processes are asked every tick; its allowance is unmeasured.
 learned=sum(p['kind']=='policy' for p in panel)
 timeout=timeout or ((600+120*learned) if development else (3600+900*learned))
 destination=Path(output) if output else session
 results={};cached={};private=destination/'grading-private';private.mkdir(mode=0o700,parents=True,exist_ok=True)
 for point in state['points']:
  sub=next((s for s in state['submissions'] if s['id']==point['submission']),None)
  if not sub:results[point['fraction']]={'status':'no_submission','reward':0};continue
  job=next(j for j in ledger.jobs() if j['name']==sub['job']);identity=sub['artifact_sha']
  try:verify(job['artifact'],identity)
  except ValueError:
   results[point['fraction']]={'status':'invalid_submission','reward':0,'artifact_sha256':identity};continue
  if identity not in cached:
   result=provider.evaluate(job['artifact'],panel,schedule,anchors,development,timeout=timeout)
   write_json(private/(identity+'.json'),result)
   cached[identity]=result['public']
  results[point['fraction']]=dict(cached[identity],submission=sub['id'],artifact_sha256=identity)
 write_json(destination/'grades.json',results)
 return results

def calibrate(release,output,provider,pilot=False,seconds=120,maps=2):
 release=Path(release);output=Path(output)
 if output.exists():raise FileExistsError('Use a new calibration output; failed attempts are retained')
 output.mkdir(parents=True);parts={'starter':[],'reference':[]};runs=[]
 plan=json.loads((release/'private/calibration-plan.json').read_text())['jobs']
 if pilot:plan=[next(p for p in plan if p['family']==f) for f in ('starter','reference')]
 for spec in plan:
  grant=seconds if pilot else spec['allocated_seconds']
  config=dict(DEFAULT,budget_usd=grant*rate(TRAIN_SHAPE),min_run_seconds=min(DEFAULT['min_run_seconds'],grant))
  ledger=Ledger(output/spec['name'],config);code=source_payload(release/'public/agent')
  if spec['family']=='reference':code['learner.py']=(release/'private/reference/reference_learner.py').read_text()
  ledger.queue(spec['name'],grant,{'kind':'train','seed':spec['seed'],'config':spec['recipe'],'code':code})
  provider.execute(ledger,ledger.jobs()[0]);job=ledger.jobs()[0]
  if job['status']!='complete':raise RuntimeError('Calibration worker failed; allocation records retained')
  if not pilot and job['charged']<.95*grant:raise RuntimeError('Full-budget calibration ended early')
  schedule={'pool_path':str(release/'private/calibration-maps.npz'),'map_indices':list(range(maps if pilot else DEFAULT['grade_maps'])),'split':'calibration'}
  result=provider.evaluate(job['artifact'],panel_paths(release,'official'),schedule,{},True,timeout=600 if pilot else 3600)
  write_json(output/(spec['name']+'-evaluation.json'),result)
  parts[spec['family']].append(result['private']['rows']);runs.append(dict(job,family=spec['family']))
 gate=calibration_gate(parts['starter'][0],parts['reference'][0]) if pilot else multi_seed_gate(parts['starter'],parts['reference'])
 if pilot:gate.update(qualified=False,status='pilot_only',reason='Short single-seed pilot does not qualify full-budget anchors')
 else:gate.update(status='qualified' if gate['qualified'] else 'reference_gap_failed',release_commitment=digest(json.loads((release/'release.json').read_text())))
 write_json(output/'gate.json',gate)
 if gate['qualified']:
  calibration_gate_result=dict(gate);grade_parts={'starter':[],'reference':[]}
  for job in runs:
   schedule={'pool_path':str(release/'private/grade-maps.npz'),'map_indices':list(range(DEFAULT['grade_maps'])),'split':'heldout'}
   result=provider.evaluate(job['artifact'],panel_paths(release,'official'),schedule,{},True,timeout=3600)
   write_json(output/(job['name']+'-grade-anchor.json'),result)
   grade_parts[job['family']].append(result['private']['rows'])
  gate=multi_seed_gate(grade_parts['starter'],grade_parts['reference'])
  gate.update(release_commitment=digest(json.loads((release/'release.json').read_text())),
   grade_seed_commitment=digest(json.loads((release/'private/grade-seeds.json').read_text())),
   calibration_gate=calibration_gate_result)
  write_json(output/'grade-anchor-gate.json',gate)
 if gate['qualified']:
  # New catalog, never overwrite the initial release/seed manifests or receipts.
  write_json(output/'calibrated-panel.json',{'official':panel_paths(release,'official'),'anchors':gate},exclusive=True)
 return {'gate':gate,'runs':[{k:r[k] for k in ('name','status','charged','artifact_sha')} for r in runs]}
