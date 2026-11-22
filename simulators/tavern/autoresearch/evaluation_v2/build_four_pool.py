"""Build the four-player small form's frozen reference pool (private local evidence) and its selection receipt.

Trains three development PPO opponents from the public starter on CPU in the four-player rules (distinct seeds,
increasing wall time), then assembles a registry beside the three public visible scripts: three profiles of
three references each (one per other seat), and writes `selected_registry-four-small-v1.json`.

The opponents are development policies, not strength-qualified anchors. Nothing here launches cloud compute.

    .venv-tavern/bin/python -m simulators.tavern.autoresearch.evaluation_v2.build_four_pool \
        --out evidence/tavern-four-small-20261113/pool-v1
"""
import argparse,json,shutil,subprocess,sys,time
from pathlib import Path
from .release import HERE,AUTO,ROOT,RECEIPT,sha,write

# (policy_id, seed, CPU wall seconds). Seeds are distinct from the starter's 12345 and the public practice seeds.
OPPONENTS=(('ppo-1m',41101,60),('ppo-5m',41102,300),('ppo-15m',41103,900))
PROFILES={
    'scripts':['tempo','economy','synergy'],
    'mixed':['ppo-5m','economy','ppo-1m'],
    'learned':['ppo-15m','ppo-5m','synergy'],
}
N_GAMES=64


def train(out,name,seed,seconds,python):
    """One opponent's training run; a completed run already in OUT is reused, never retrained."""
    folder=out/'training'/name
    if (folder/'artifact/summary.json').is_file():return folder/'artifact'
    recipe=json.loads((AUTO/'recipes/starter.json').read_text());recipe.update(device='cpu',seed=seed,n_games=N_GAMES)
    folder.mkdir(parents=True);write(folder/'recipe.json',recipe)
    with (folder/'train.log').open('w') as log:
        subprocess.run([python,str(AUTO/'train_entry.py'),'--source',str(AUTO/'starter'),'--recipe',str(folder/'recipe.json'),'--out',str(folder/'artifact'),'--seconds',str(seconds),'--device','cpu'],cwd=AUTO,stdout=log,stderr=subprocess.STDOUT,check=True)
    return folder/'artifact'


def build(out,python=sys.executable,parallel=True):
    sys.path.insert(0,str(AUTO))
    from contract import binding,EDITABLE
    from simulators.tavern.evaluation_readiness.policies import admit_bundle
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=True);started=time.time()
    if (out/'registry').exists():raise FileExistsError(out/'registry')
    if parallel:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(len(OPPONENTS)) as pool:artifacts=dict(zip([o[0] for o in OPPONENTS],pool.map(lambda o:train(out,*o,python),OPPONENTS)))
    else:artifacts={o[0]:train(out,*o,python) for o in OPPONENTS}
    pool=out/'registry';items=[]
    for name,seed,seconds in OPPONENTS:
        path=artifacts[name]
        # The trainer exports weights and identity; the editable source it trained is the public starter, whose
        # hashes the identity records (admission below checks them).
        for f in EDITABLE:
            if not (path/f).exists():shutil.copy2(AUTO/'starter'/f,path/f)
        ident,summary=admit_bundle(path);dest=pool/'learned'/name;dest.mkdir(parents=True)
        for f in [*EDITABLE,'weights.npz','identity.json','summary.json']:shutil.copy2(path/f,dest/f);(dest/f).chmod(0o444)
        history=json.loads((path/'metrics.json').read_text())
        items.append(dict(policy_id=name,kind='learned',path='learned/'+name,source_origin=str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
                          files={f.name:sha(f) for f in sorted(dest.iterdir())},training=dict(seed=seed,cpu_seconds=seconds,n_games=N_GAMES,updates=summary['updates'],samples=summary['samples'],
                          last_update_mean_placement=_mean_placement(history[-1])),admission_qualified=True,
                          source_visibility_audit='The public starter policy module, unmodified: visible-only feedforward, no search or state access.',strong_qualified=False,historical_experiment=False))
    for style in ('tempo','economy','synergy'):items.append(dict(policy_id=style,kind='script',style=style,source_sha256=sha(AUTO.parent/'evaluation_readiness/opponents.py'),admission_qualified=True,strong_qualified=False))
    pin=binding()
    registry=dict(schema='tavern-mixed-visible-v1',binding=pin,opponents=items,profiles=PROFILES,profile_assignment=f"{pin['rules']['num_players']-1} slots go clockwise from the candidate seat; every candidate seat required.",
                  difficulty_labels='Profile names describe composition, not an assumed difficulty order.',
                  candidate_constraints=dict(max_parameters=2000000,max_weights_bytes=32000000,forward_deadline_seconds=15,visibility='candidate exact numeric/cards/mask'),
                  reference_constraints='Host-owned public starter module with trained weights, or the public visible scripts; same visible tensors, CPU one thread.',
                  readiness=dict(harness_qualified=False,learned_reference_qualified=False,strong_panel_qualified=False,superhuman_evidence=False),created_at_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    write(pool/'registry.json',registry);write(pool/'manifest.json',{str(f.relative_to(pool)):sha(f) for f in sorted(pool.rglob('*')) if f.is_file()})
    receipt=dict(schema='tavern-selected-mixed-pool-public-v2',path=str(pool.relative_to(ROOT)),manifest_sha256=sha(pool/'manifest.json'),benchmark_id=pin['benchmark_id'],
                 selection_basis='Three development PPO opponents trained from the public starter on CPU in the four-player rules (seeds 41101-41103; 1, 5 and 15 minutes, 64 lobbies per update), plus the three public visible scripts. Built by evaluation_v2/build_four_pool.py; no strength qualification.',
                 opponents={i['policy_id']:i.get('training') for i in items if i['kind']=='learned'},private_final_seed_values=False,strong_human_expert_claim=False,selection_still_open=True)
    write(HERE/RECEIPT,receipt);write(out/'build.json',dict(receipt=receipt,seconds=time.time()-started))
    return receipt


def _mean_placement(row):
    values=[e['placement'] for e in row.get('episodes',[]) if e.get('valid')]
    return sum(values)/len(values) if values else None


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--serial',action='store_true');a=p.parse_args()
    print(json.dumps(build(a.out,parallel=not a.serial),indent=2))
