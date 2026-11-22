"""Host-private fresh schedule, generated only after exact policy selection closes."""
import argparse,json,secrets,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
from .policies import admit_bundle
from contract import sha,write
MASK=(1<<64)-1
STRIDE=1000003
INV_STRIDE=pow(STRIDE,-1,1<<64)

def sm(x):
    x=(x+0x9e3779b97f4a7c15)&MASK
    x=((x^(x>>30))*0xbf58476d1ce4e5b9)&MASK
    x=((x^(x>>27))*0x94d049bb133111eb)&MASK
    return x^(x>>31)

def lobby_key(seed,game=0):return int(seed)^sm(game)

def training_overlap(seed,config):
    key=lobby_key(seed)
    return any((((key^sm(g))-config['seed'])*INV_STRIDE)&MASK < config['updates'] for g in range(config['n_games']))

def freeze(registry,selection,out,groups=6):
    registry,selection,out=[Path(p).resolve() for p in (registry,selection,out)];closed=json.loads(selection.read_text());reg=json.loads((registry/'registry.json').read_text())
    if closed.get('selection_closed') is not True:raise ValueError('Selection remains open')
    if closed['registry_manifest_sha256']!=sha(registry/'manifest.json'):raise ValueError('Selection registry differs')
    manifest=json.loads((registry/'manifest.json').read_text())
    if any(sha(registry/f)!=h for f,h in manifest.items()):raise ValueError('Selected policy/source mutated')
    configs=[];identities={}
    for item in reg['opponents']:
        if item['kind']=='learned':
            ident,_=admit_bundle(registry/item['path']);configs.append(ident['config'])
            identities[item['policy_id']]=sha(registry/item['path']/'identity.json')
    if set(closed['roster'])-set(p['policy_id'] for p in reg['opponents']):raise ValueError('Missing selected policy')
    # Read only old schedule declarations, never grade or mutate a historical bank.
    roots=[ROOT/'evidence/overnight-20261108-1533Z/tavern',registry.parent.parent/'rl',registry.parent/'panels']
    excluded={lobby_key(s,g) for s in (1,2026,30101,30102,30103,12345) for g in range(16)};receipts=[]
    def declarations(data):
        if isinstance(data,dict):
            if type(data.get('seed')) is int and type(data.get('n_games')) is int and type(data.get('updates')) is int:
                configs.append(dict(seed=data['seed'],n_games=data['n_games'],updates=data['updates']))
            if isinstance(data.get('seeds'),list):
                n=max(1,int(data.get('games_per_seat',data.get('n_games',1))))
                for seed in data['seeds']:
                    if type(seed) is int:
                        for g in range(n):excluded.add(lobby_key(seed,g))
            for value in data.values():declarations(value)
        elif isinstance(data,list):
            for value in data:declarations(value)
    for config in list(configs):declarations(config.get('evaluation',{}))
    for root in roots:
        for path in root.rglob('*.json'):
            if not any(s in path.name.lower() for s in ('panel','seed','recipe','commitment','config','identity')) or path.stat().st_size>2000000:continue
            try:data=json.loads(path.read_text())
            except (ValueError,OSError):continue
            declarations(data)
            if isinstance(data,dict) and isinstance(data.get('seeds'),list):
                seeds=data['seeds'];n=max(1,int(data.get('games_per_seat',1)))
                for seed in seeds:
                    if type(seed) is int:
                        for game in range(n):excluded.add(lobby_key(seed,game))
                receipts.append(dict(path=str(path.relative_to(ROOT)),sha256=sha(path),groups=len(seeds),games_per_group=n))
    configs=list({(c['seed'],c['n_games'],c['updates']):c for c in configs}.values())
    seeds=[];rejected=0
    while len(seeds)<groups:
        seed=secrets.randbits(64)
        if lobby_key(seed) in excluded or seed in seeds or any(training_overlap(seed,c) for c in configs):rejected+=1;continue
        seeds.append(seed)
    out.mkdir(mode=0o700,parents=True,exist_ok=False)
    write(out/'panel.json',dict(label='fresh-private-final-v1-six-lobby-groups-all8seats-three-mixed-profiles',seeds=seeds))
    write(out/'precommit.json',dict(selection_closed=True,selection_sha256=sha(selection),bank_generator_sha256=sha(__file__),registry_manifest_sha256=sha(registry/'manifest.json'),panel_sha256=sha(out/'panel.json'),roster=closed['roster'],identities=identities,groups=groups,seats=8,profiles=list(reg['profiles']),cases_per_policy=groups*8*len(reg['profiles']),profiles_unchanged=True,native_seed_key='sm(sm(base_seed ^ sm(game_index)) ^ episode_index); first episode0; invertible sm excludes matching native streams',excluded_historical_schedule_keys=len(excluded),schedule_receipts=receipts,training_cohorts_excluded=[dict(seed=c['seed'],n_games=c['n_games'],entire_declared_update_ceiling=c['updates'],stride=STRIDE) for c in configs],random_proposals_rejected=rejected,private_feedback_after_selection_only=True))
    for f in out.iterdir():f.chmod(0o600)
    return dict(panel_sha256=sha(out/'panel.json'),precommit_sha256=sha(out/'precommit.json'),cases_per_policy=groups*8*len(reg['profiles']))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('registry','selection','out'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();print(json.dumps(freeze(a.registry,a.selection,a.out)))
