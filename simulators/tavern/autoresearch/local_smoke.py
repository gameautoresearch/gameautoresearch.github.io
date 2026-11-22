"""Local CPU end-to-end smoke of the four-player small form: train, practise, grade. No cloud, no model calls.

Uses a prepared release (`python -m harness prepare tavern --out RELEASE`) and runs, on this machine:
  1. the trusted training entry (`train_entry.py`) on the release's public starter and smoke config, on CPU;
  2. public practice: the release's 48 public lobbies through the evaluator (`evaluation_readiness.screen`)
     with the terminal-state recording the CPU worker installs, then the independent terminal replay of the
     predeclared cases (all three profiles, first seed group, candidate seat 0);
  3. a grade on a fresh private bank of 24 seed groups (288 lobbies), screened like the harness's final bank
     (`harness/games/tavern/final_bank.py`) and written only under OUT/private.
The near-untrained starter (exactly one PPO update) is practised too, as the zero-training comparison.

The candidate runs in-process here (trusted local code); the cloud evaluator isolates it under another uid.

    .venv-tavern/bin/python simulators/tavern/autoresearch/local_smoke.py --release RELEASE --out NEW_DIR --seconds 60
"""
import argparse,json,secrets,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(HERE))


def train(release,out,seconds,python,updates=None):
    recipe=json.loads((release/'public/recipes/smoke.json').read_text());recipe['device']='cpu'
    if updates:recipe['updates']=updates
    path=out/'recipe.json';path.write_text(json.dumps(recipe,indent=2)+'\n')
    started=time.time()
    done=subprocess.run([python,str(HERE/'train_entry.py'),'--source',str(release/'public/starter'),'--recipe',str(path),'--out',str(out/'artifact'),'--seconds',str(seconds),'--device','cpu'],cwd=HERE,capture_output=True,text=True)
    (out/'train.log').write_text(done.stdout+done.stderr)
    if done.returncode:raise RuntimeError('training failed: '+done.stderr[-2000:])
    import shutil
    for f in ('policy.py','learner.py','train.py'):shutil.copy2(release/'public/starter'/f,out/'artifact'/f)  # as the controller packages a run
    summary=json.loads((out/'artifact/summary.json').read_text());history=json.loads((out/'artifact/metrics.json').read_text())
    def mean(row):
        values=[e['placement'] for e in row['episodes'] if e.get('valid')];return sum(values)/len(values) if values else None
    return dict(wall_seconds=time.time()-started,updates=summary['updates'],samples=summary['samples'],first_update_mean_placement=mean(history[0]),last_update_mean_placement=mean(history[-1]))


def screen(release,bundle,panel,out,replay):
    from simulators.tavern.autoresearch.evaluation_v2.registry import append
    from simulators.tavern.evaluation_readiness.screen import screen as run
    from simulators.tavern.evaluation_readiness.five_hour.recording import install
    from simulators.tavern.evaluation_readiness.five_hour.terminal_replay import audit_dir
    registry=append(release/'private/registry',bundle,out/'registry','entrant-local')
    undo=install()
    try:result=run(registry,panel,'entrant-local',out/'screen',isolated=False,stage_seconds=10**6)
    finally:undo()
    proof=None
    if replay and result['complete_valid_schedule']:
        seed=json.loads(Path(panel).read_text())['seeds'][0];reg=json.loads((registry/'registry.json').read_text())
        proof=[audit_dir(out/'screen/traces'/f'{profile}-{seed}-0') for profile in reg['profiles']]
    keys=('players','expected_cases','observed_cases','complete_valid_schedule','raw_score','mean_placement','top_half_rate','confidence_interval_95','failures','process_seconds')
    return dict({k:result.get(k) for k in keys},per_profile={p:v['raw_score'] for p,v in result['per_profile'].items()},
                frames=result['action_counts'].get('frames'),terminal_replay=[dict(trace=r['trace'],frames=r['frames'],exact=r['exact_true_terminal_game_state']) for r in proof] if proof else None)


def private_bank(release,roots,out,groups):
    from simulators.tavern.autoresearch.evaluation_v2.seed_exclusions import collect
    from simulators.tavern.autoresearch.evaluation_v2.seed_keys import lobby_key,training_overlap
    excluded,configs,_=collect(roots=roots);seeds=[]
    while len(seeds)<groups:
        seed=secrets.randbits(64)
        if seed not in seeds and lobby_key(seed) not in excluded and not any(training_overlap(seed,c) for c in configs):seeds.append(seed)
    private=out/'private';private.mkdir(mode=0o700);panel=private/'panel.json'
    panel.write_text(json.dumps(dict(label='local-smoke-private-bank',seeds=seeds))+'\n');panel.chmod(0o600)
    return panel


def main():
    p=argparse.ArgumentParser();p.add_argument('--release',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--seconds',type=float,default=60);a=p.parse_args()
    from simulators.tavern.autoresearch.rules_enabled.release import verify,FINAL_GROUPS
    release=a.release.resolve();verify(release);out=a.out.resolve();out.mkdir(parents=True,exist_ok=False);report=dict(release=str(release))
    for name,seconds,updates in (('untrained',120,1),('trained',a.seconds,None)):
        folder=out/name;folder.mkdir();report[name]=dict(train=train(release,folder,seconds,sys.executable,updates))
        report[name]['practice']=screen(release,folder/'artifact',release/'public/practice.json',folder/'practice',replay=True)
        print(json.dumps({name:report[name]}),flush=True)
    panel=private_bank(release,[release/'public',release/'private/registry',out],out,FINAL_GROUPS)
    report['trained']['grade']=screen(release,out/'trained/artifact',panel,out/'trained/grade',replay=False)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
