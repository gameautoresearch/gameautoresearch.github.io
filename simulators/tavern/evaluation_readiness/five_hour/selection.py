"""Public-only deterministic best/latest roster under the precommitted policy."""
import json,re
from pathlib import Path
from simulators.tavern.autoresearch.evaluation_v2.release import sha


def choose(records,controls):
    arms={}
    for arm in ('quarter','anchor'):
        rows=[r for r in records if re.fullmatch('new-'+arm+r'\d+',r['candidate'])]
        eligible=[r for r in rows if r.get('public_score') is not None]
        if not rows or not eligible:raise ValueError('Each sustained arm needs admitted latest and complete original PUBLIC192 evidence')
        best=min(eligible,key=lambda r:(-r['public_score'],r['update'],r['candidate']))
        latest=max(rows,key=lambda r:(r['update'],r['candidate']))
        arms[arm]=dict(best=best,latest=latest)
    selected=[*controls,*[r[k]['candidate'] for r in arms.values() for k in ('best','latest')]]
    by_id={r['candidate']:r for r in records};seen=set();roster=[]
    for identity in selected:
        row=by_id.get(identity)
        key=(row['checkpoint_sha256'],row['policy_sha256']) if row else identity
        if key not in seen:roster.append(identity);seen.add(key)
    if len(roster)>12:raise ValueError('Precommitted campaign roster limit exceeded')
    if by_id['new-plain266'].get('public_score') is None:raise ValueError('Parent266 needs complete original PUBLIC192 evidence')
    primary=min([a['best'] for a in arms.values()]+[by_id['new-plain266']],key=lambda r:(-r['public_score'],r['update'],r['candidate']))
    return dict(arms=arms,roster=roster,primary=primary,private_results_used=False)


def plan(root):
    root=Path(root);base=root/'public-development-v1';original=json.loads((base/'registry.json').read_text());panel_hash=sha(base/'panel-extended.json');records=[]
    for path in root.glob('admission-*.json'):
        admission=json.loads(path.read_text())
        if not admission.get('admission_qualified'):continue
        match=re.fullmatch(r'admission-(quarter\d+|anchor\d+|plain266)(?:-v\d+)?\.json',path.name)
        if not match:continue
        identity='new-'+match[1];bundle=Path(admission['bundle']);row=dict(candidate=identity,update=admission['update'],checkpoint_sha256=admission['checkpoint_sha256'],policy_sha256=sha(bundle/'policy.py'),admission=str(path.resolve()),bundle=str(bundle),public_score=None)
        for provider in sorted(root.glob('*/provider.json')):
            launch=json.loads(provider.read_text());result_path=provider.parent/'summary/screen/result.json'
            if launch.get('candidate')!=identity or launch.get('purpose','development')!='development' or launch.get('panel_sha256')!=panel_hash or not result_path.exists():continue
            result=json.loads(result_path.read_text());registry=json.loads((provider.parent/'assets/registry.json').read_text())
            if registry['profiles']!=original['profiles'] or registry['binding']!=original['binding']:raise ValueError('Public selection opposition/interface differs')
            items={r['policy_id']:r for r in registry['opponents']}
            for item in original['opponents']:
                if any(item['policy_id'] in p for p in original['profiles'].values()) and item['kind']=='learned' and sha(provider.parent/'assets'/items[item['policy_id']]['path']/'identity.json')!=sha(base/item['path']/'identity.json'):raise ValueError('Public selection reference identity differs')
            entrant=provider.parent/'assets'/items[identity]['path']
            if sha(entrant/'weights.npz')!=admission['weights_sha256'] or sha(entrant/'identity.json')!=admission['export_identity_sha256'] or result['candidate_id']!=identity:raise ValueError('Public selection entrant differs from admission')
            lifecycle=json.loads((provider.parent/'summary/lifecycle.json').read_text())
            if lifecycle.get('status')!='completed':continue
            if launch.get('runtime_protocol')=='private-paths-terminal-v2-all-lobbies':
                terminal=json.loads((provider.parent/'summary/qualification-terminal-replay.json').read_text())
                if not terminal.get('passed') or not terminal.get('all_lobby_actual_terminals_retained') or terminal.get('recorded_lobbies')!=192:continue
            if result.get('complete_valid_schedule') and result.get('observed_cases')==192 and result.get('expected_cases')==192 and result.get('raw_score') is not None:
                if row['public_score'] is not None and row['public_score']!=result['raw_score']:raise ValueError('Repeated frozen public schedule changed its score')
                row.update(public_score=result['raw_score'],public_result=str(result_path.resolve()),public_result_sha256=sha(result_path))
        records.append(row)
    policy=json.loads((root/'final-assessment-policy-v1.json').read_text());result=choose(records,policy['controls']+policy['first_pair_endpoint_controls'])
    return dict(result,selection_policy_sha256=sha(root/'final-assessment-policy-v1.json'),helper_sha256=sha(__file__),original_panel_sha256=panel_hash,selection_closed=False)
