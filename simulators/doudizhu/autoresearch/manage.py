"""Host-private evaluation: grading a finalized session and calibrating the starter/reference anchors.

Both functions take a provider (harness.games.doudizhu.provider.ModalProvider) and
are called by the host only; nothing here is reachable from a researcher box.
"""
from pathlib import Path
import json
import time

from .contract import DEFAULT,write_json,digest
from .ledger import Ledger
from .bundle import source_payload,verify
from .grading import multi_seed_gate

def panel_paths(release,which):
    root=Path(release)/'private'
    catalog=json.loads((root/'panel.json').read_text())
    if catalog.get(which) is None:raise ValueError('Official panel is not calibrated')
    return [dict(p,bundle=str((root/p['bundle']).resolve())) if p['kind']=='checkpoint' else p for p in catalog[which]]

def grade_session(release,session,provider,development=False,catalog=None,private_deals=None):
    root=Path(release);ledger=Ledger(session);state=ledger.finalize()
    selected=json.loads(Path(catalog).read_text()) if catalog else None
    anchors=selected['anchors'] if selected else json.loads((root/'private/anchors.json').read_text())
    if not development and not anchors.get('qualified'):raise ValueError('Run full-budget calibration before official grading')
    panel=selected['official'] if selected else panel_paths(root,'development' if development else 'official')
    seeds=json.loads((root/'private/grade-seeds.json').read_text())
    full_seed_commitment=digest(seeds)
    if private_deals is not None:
        if not development or not 0<private_deals<=seeds['deals']:raise ValueError('A smaller private prefix is development-only')
        seeds=dict(seeds,deals=private_deals)
    if selected and (anchors.get('grade_seed_commitment')!=digest(seeds) or anchors.get('release_commitment')!=digest(json.loads((root/'release.json').read_text()))):
        raise ValueError('Calibration catalog belongs to another immutable release or grading schedule')
    if not development and ledger.config['gpu_minutes']!=DEFAULT['gpu_minutes']:
        raise ValueError('Smoke-session allocations cannot receive an official full-budget score')
    attempt=Path(session)/'grading-private'/str(time.time_ns())
    attempt.mkdir(parents=True,mode=0o700,exist_ok=False)
    attempt.parent.chmod(0o700)
    results={};cache={}
    write_json(attempt/'schedule-commitment.json',{'full_seed_commitment':full_seed_commitment,
        'executed_schedule_commitment':digest(seeds),'deals':seeds['deals'],'gpu_count':0,'fresh_release':str(root)},exclusive=True)
    for point in state['points']:
        sub=next((s for s in state['submissions'] if s['id']==point['submission']),None)
        if sub is None:results[point['fraction']]={'status':'no_submission','reward':0};continue
        job=next(j for j in ledger.jobs() if j['name']==sub['job'])
        verify(job['artifact'],sub['artifact_sha'],check_tensors=False)
        if sub['artifact_sha'] not in cache:cache[sub['artifact_sha']]=provider.evaluate(job['artifact'],panel,seeds,anchors,development)
        result=cache[sub['artifact_sha']]
        if result.get('allocation'):write_json(attempt/(point['fraction']+'-allocation.json'),result['allocation'],exclusive=True)
        write_json(attempt/(point['fraction']+'.json'),result['private'],exclusive=True)
        results[point['fraction']]=dict(result['public'],submission=sub['id'],artifact_sha256=sub['artifact_sha'])
    write_json(attempt/'grades.json',results,exclusive=True)
    write_json(Path(session)/'latest-grade.json',{'attempt':str(attempt),'results':results})
    return results

def calibrate(release,output,provider):
    release=Path(release);output=Path(output)
    if output.exists():raise FileExistsError('Use a fresh calibration directory; failed attempts are never erased')
    output.mkdir(parents=True)
    plan=json.loads((release/'private/calibration-plan.json').read_text())
    if plan['status']!='not_run':raise ValueError('Calibration plan is not fresh')
    collected={}
    for spec in plan['jobs']:
        ledger=Ledger(output/spec['name'],dict(DEFAULT))
        code=source_payload(release/'public/agent')
        if spec['family']=='reference':code['train.py']=(release/'private/reference/reference_train.py').read_text()
        payload={'kind':'train','seed':spec['seed'],'config':spec['recipe'],'code':code}
        ledger.queue(spec['name'],spec['allocated_seconds'],payload)
        job=provider.execute(ledger,ledger.jobs()[0])
        actual=ledger.jobs()[0]
        if actual['status']!='complete' or actual['charged']<.95*spec['allocated_seconds']:
            raise RuntimeError('Full-budget calibration incomplete; allocation and failure records retained')
        collected[spec['name']]=dict(spec,bundle=actual['artifact'],bundle_sha256=actual['artifact_sha'])
    # The panel and all anchor runs are fixed before seeing candidate submissions.
    simple=panel_paths(release,'development')
    panel=simple+[{'name':n,'kind':'checkpoint','bundle':v['bundle'],'bundle_sha256':v['bundle_sha256']} for n,v in collected.items()]
    def evaluate_anchors(seeds):
        groups={'starter':[],'reference':[]}
        for name,value in collected.items():
            result=provider.evaluate(value['bundle'],panel,seeds,{},True,timeout=3600)
            write_json(output/(name+'-'+digest(seeds)[:12]+'.json'),result)
            groups[value['family']].append(result['private']['rows'])
        return multi_seed_gate(groups['starter'],groups['reference'])
    gate=evaluate_anchors(json.loads((release/'private/calibration-seeds.json').read_text()))
    write_json(output/'calibration-gate.json',gate)
    if not gate['qualified']:raise RuntimeError('Reference did not qualify; no official normalized scores unlocked')
    anchors=evaluate_anchors(json.loads((release/'private/grade-seeds.json').read_text()))
    if not anchors['qualified']:raise RuntimeError('Reference grading-schedule anchor gap did not qualify')
    anchors.update(full_allocated_seconds=DEFAULT['gpu_minutes']*60,seeds_per_family=3,
        panel_commitment=digest(panel),source_calibration=str(output),
        grade_seed_commitment=digest(json.loads((release/'private/grade-seeds.json').read_text())),
        release_commitment=digest(json.loads((release/'release.json').read_text())))
    write_json(output/'anchors.json',anchors)
    # Create a new calibrated release identity instead of modifying original seed,
    # panel or source receipts. Operators select this catalog explicitly.
    catalog={'official':panel,'development':panel,'practice':panel_paths(release,'practice'),'anchors':anchors}
    write_json(output/'calibrated-panel.json',catalog,exclusive=True)
    return {'qualified':True,'catalog':str(output/'calibrated-panel.json'),'anchors':anchors}
