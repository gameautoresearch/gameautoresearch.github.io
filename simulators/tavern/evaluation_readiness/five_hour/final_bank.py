"""Freeze exact campaign roster, then draw one fresh private all8seat schedule."""
import argparse,json,secrets,time
from pathlib import Path
from datetime import datetime
from .budget import Budget,CAMPAIGN
from ..policies import admit_bundle
from simulators.tavern.autoresearch.evaluation_v2.seed_keys import lobby_key,training_overlap
from simulators.tavern.autoresearch.evaluation_v2.release import sha,write
from simulators.tavern.autoresearch.evaluation_v2.seed_exclusions import collect


def freeze(registry,roster,out):
    registry,out=Path(registry),Path(out);budget=Budget();close=datetime.fromisoformat(budget.auth['selection_close_utc'].replace('Z','+00:00')).timestamp()
    if time.time()<close:raise ValueError('Campaign training/selection remains open')
    if time.time()>=datetime.fromisoformat(budget.auth['worker_stop_utc'].replace('Z','+00:00')).timestamp():raise ValueError('Campaign worker window has closed')
    coordination=CAMPAIGN/'tavern/coordination/rl.json';rl_raw=coordination.read_bytes();rl=json.loads(rl_raw);closed=rl.get('selection_closed')
    if not (closed is True or isinstance(closed,dict) and closed.get('closed') is True):raise ValueError('RL owner has not explicitly frozen selection')
    policy=CAMPAIGN/'tavern/evaluator/final-assessment-policy-v1.json';closed_path=policy.parent/'public-selection-closed-v1.json';public_selection=json.loads(closed_path.read_text())
    if public_selection.get('selection_closed') is not True or public_selection.get('selection_policy_sha256')!=sha(policy) or public_selection.get('private_results_used') is not False or public_selection.get('roster')!=roster or len(roster)>12:raise ValueError('Final roster differs from the closed precommitted public-only selection')
    manifest=json.loads((registry/'manifest.json').read_text());reg=json.loads((registry/'registry.json').read_text());items={p['policy_id']:p for p in reg['opponents']}
    if any(sha(registry/f)!=h for f,h in manifest.items()):raise ValueError('Selected inference pool changed')
    if reg['binding']['native_source_hash']!='217577b83fc4525e' or len(reg['profiles'])!=3 or any(len(v)!=7 for v in reg['profiles'].values()):raise ValueError('Pinned native and full7opponent compositions required')
    if len(roster)!=len(set(roster)) or set(roster)-set(items) or not {'deepseek-r3','starter-111901','tempo','economy','synergy'}<=set(roster):raise ValueError('Exact unique roster and retained learned/script controls required')
    identities={};configs=[]
    for identity,item in items.items():
        if item['kind']=='learned':
            ident,_=admit_bundle(registry/item['path'])
            if ident['binding']!=reg['binding']:raise ValueError('Selected reference task differs')
            configs.append(ident['config']);identities[identity]=dict(identity_sha256=sha(registry/item['path']/'identity.json'),weights_sha256=sha(registry/item['path']/'weights.npz'))
    out.mkdir(parents=True,mode=0o700,exist_ok=False)
    (out/'rl-selection-at-freeze.json').write_bytes(rl_raw)
    selection=dict(selection_closed=True,closed_unix=time.time(),rl_selection_receipt_sha256=sha(out/'rl-selection-at-freeze.json'),public_selection_sha256=sha(closed_path),selection_policy_sha256=sha(policy),registry_manifest_sha256=sha(registry/'manifest.json'),roster=roster,identities={i:identities[i] for i in roster if i in identities},profiles=reg['profiles'],native_source_hash=reg['binding']['native_source_hash'],private_results_used_for_selection=False)
    write(out/'selection.json',selection)
    keys,configs,exclusions=collect(configs);seeds=[];rejected=0
    while len(seeds)<12:
        seed=secrets.randbits(64)
        if seed in seeds or lobby_key(seed) in keys or any(training_overlap(seed,c) for c in configs):rejected+=1;continue
        seeds.append(seed)
    write(out/'panel.json',dict(label='five-hour-fresh-private-final-after-global-selection',seeds=seeds))
    receipt=dict(selection_closed=True,selection_sha256=sha(out/'selection.json'),registry_manifest_sha256=sha(registry/'manifest.json'),panel_sha256=sha(out/'panel.json'),bank_generator_sha256=sha(__file__),seed_exclusions=exclusions,groups=12,seats=8,profiles=list(reg['profiles']),required_lobbies_per_policy=288,roster=roster,native_stream_proposals_rejected=rejected,source_weights_profiles_fixed=True,no_private_feedback_for_tuning=True,created_unix=time.time())
    write(out/'precommit.json',receipt)
    for p in out.iterdir():p.chmod(0o600)
    return dict(panel_sha256=receipt['panel_sha256'],precommit_sha256=sha(out/'precommit.json'),roster=roster,required_lobbies_per_policy=288,seed_values_private=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--registry',type=Path,required=True);p.add_argument('--roster',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();print(json.dumps(freeze(a.registry,json.loads(a.roster.read_text()),a.out)))
