"""Offline closure audit of complete fixed cases, identities and executed source."""
import argparse,json,itertools,math
from pathlib import Path
from .budget import CAMPAIGN,ROOT,EvaluatorBudget
from .policies import admit_bundle
from contract import binding,sha,write

def audit(out):
    root=CAMPAIGN/'tavern/evaluator';selection=json.loads((root/'selection-closed.json').read_text());bank=json.loads((root/'private-final-v1/precommit.json').read_text());panel=json.loads((root/'private-final-v1/panel.json').read_text());registry=root/'registry-v5';reg=json.loads((registry/'registry.json').read_text())
    if Path(out).exists():raise FileExistsError(out)
    if sha(root/'selection-closed.json')!=bank['selection_sha256'] or sha(root/'private-final-v1/panel.json')!=bank['panel_sha256']:raise ValueError('Selection/bank mutated')
    if sha(registry/'manifest.json')!=bank['registry_manifest_sha256']:raise ValueError('Registry changed')
    manifest=json.loads((registry/'manifest.json').read_text())
    if any(sha(registry/f)!=h for f,h in manifest.items()):raise ValueError('Frozen reference source/weights mutated')
    if binding()!=json.loads((root/'initial-binding.json').read_text()):raise ValueError('131-file source/rules/native binding changed')
    for item in reg['opponents']:
        if item['kind']=='learned':admit_bundle(registry/item['path'])
    runtime=['opponents.py','lobby.py','policies.py','screen.py','guest.py'];selected=selection['evaluator_source_files'];records=[];all_valid=True
    expected=set(itertools.product(reg['profiles'],panel['seeds'],range(8)))
    for policy in selection['roster']:
        attempts=[]
        for p in root.glob('cpu-final-*/provider.json'):
            if json.loads(p.read_text())['candidate']==policy:attempts.append(p.parent)
        if len(attempts)!=1:raise ValueError('Required policy has missing or multiple final attempts: '+policy)
        path=attempts[0];result=json.loads((path/'summary/screen/result.json').read_text());rows=json.loads((path/'summary/screen/episodes.json').read_text());allocation=json.loads((path/'allocation.json').read_text());source=path/'source/simulators/tavern/evaluation_readiness'
        for name in runtime:
            key='simulators/tavern/evaluation_readiness/'+name
            if sha(source/name)!=selected[key]:raise ValueError('Executed decision/lifecycle source changed after selection: '+name)
        launch=json.loads((path/'provider.json').read_text())
        if not allocation['provider'].get('finished_at') or allocation['allocated_seconds']>launch['cap'] or launch['cap'] not in (600,900):raise ValueError('Missing/out-of-cap provider lifetime')
        keys=[(r['profile'],r['seed'],r['seat']) for r in rows]
        if len(keys)!=len(set(keys)) or set(keys)!=expected:raise ValueError('Required final case set incomplete/duplicated: '+policy)
        valid=all(r['complete_valid_lobby'] for r in rows);all_valid &= valid
        for row in rows:
            if row['native']!='217577b83fc4525e':raise ValueError('Native result changed')
            if row['lineup'][row['seat']]!=policy:raise ValueError('Candidate routed to wrong seat')
            for offset,identity in enumerate(reg['profiles'][row['profile']],1):
                if row['lineup'][(row['seat']+offset)%8]!=identity:raise ValueError('Opponent roster/routing changed')
            if row['complete_valid_lobby']:
                terminals=row['terminals']
                if len(terminals)!=8 or sorted(t['placement'] for t in terminals.values())!=list(range(1,9)) or not all(t['natural'] and t['stop_reason']=='natural' for t in terminals.values()):raise ValueError('Incomplete native endings were scored')
                if row['candidate_placement']!=terminals[str(row['seat'])]['placement']:raise ValueError('Candidate placement does not match native terminal')
        score=sum(100*(8-r['candidate_placement'])/7 for r in rows)/len(rows) if valid else None
        if (score is None)!=(result['raw_score'] is None) or score is not None and not math.isclose(score,result['raw_score'],abs_tol=1e-9):raise ValueError('Reported aggregate differs from native placements')
        records.append(dict(candidate=policy,attempt=path.name,cases=len(rows),all8_natural=valid,raw_score=score,episodes_sha256=sha(path/'summary/screen/episodes.json'),result_sha256=sha(path/'summary/screen/result.json'),allocation_seconds=allocation['allocated_seconds'],executed_runtime_source_unchanged=True))
    usage=EvaluatorBudget().usage()
    if usage['gpu_seconds'] or usage['model_usd'] or usage['resource_usd']>8:raise ValueError('Evaluator allocation/model/resource boundary violated')
    for name in ('evaluation-readiness-2026-11-09.md','evaluation-readiness-2026-11-09.json'):
        text=(ROOT/'reports/tavern'/name).read_text()
        if any(str(seed) in text for seed in panel['seeds']):raise ValueError('Private evaluation seed exposed in canonical report')
    result=dict(audit_passed=True,complete_case_count=sum(r['cases'] for r in records),all_required_cells_valid=all_valid,selection_and_bank_unchanged=True,registry_source_weights_unchanged=True,exact_131_file_binding_unchanged=True,all_executed_decision_source_unchanged=True,actual_private_seeds_excluded_from_report=True,evaluator_GPU_seconds=usage['gpu_seconds'],model_liability_usd=usage['model_usd'],resource_charged_plus_held_usd=usage['resource_usd'],records=records)
    write(out,result);return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();r=audit(a.out);print(json.dumps({k:v for k,v in r.items() if k!='records'},indent=2))
