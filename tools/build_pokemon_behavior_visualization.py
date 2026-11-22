"""Read-only analysis of the closed Sol/Kimi pilot; produces offline HTML.

Run: .venv-pokemon/bin/python tools/build_pokemon_behavior_visualization.py
Behavior labels are explicit editorial annotations, not inferred thinking time.
"""
import hashlib
import json
import math
from pathlib import Path
import sqlite3

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'evidence/pokemon-four-model-4h-20261109-1354Z'
OUT = ROOT / 'reports/pokemon/sol-kimi-research-behavior-2026-11-10.html'

BUCKETS = [
    dict(id='initial', name='Initial experiments', color='#edb447',
         definition='Establish the first approach; includes the starter-versus-attention comparison for Kimi.'),
    dict(id='diagnose', name='Diagnose', color='#ed896d',
         definition='Explicit diagnostic training attempts, including a failed attempt. Static checks are shown separately below.'),
    dict(id='revise', name='Revise the approach', color='#43a98e',
         definition='The first training lease after a substantive new architecture or learning-recipe revision, beyond the initial experiments.'),
    dict(id='continue', name='Continue unchanged', color='#5b9dc9',
         definition='Later training leases with unchanged source/config after the initial comparison or a revised approach.'),
    dict(id='evaluate', name='Evaluate candidates', color='#a37cc6',
         definition='All allocated public-practice workers. Includes overhead and incomplete evaluation; excludes private final grading.'),
]

# This is an exhaustive, reviewable stage-purpose annotation for these two runs.
STAGES = {
 'sol61-high': {
  'initial': ('initial', 'Larger learner + a new PPO recipe', 'Width 64 → 192; longer rollouts; more PPO epochs. Trainer checkpoint writes moved to lease end.'),
  'main1': ('initial', 'Establish the first approach', 'Continue the initial recipe before the first substantive revision.'),
  'diagnostic': ('diagnose', 'Probe the training behavior', 'First diagnostic fork failed. Its allocation remains charged.'),
  'diagnostic2': ('diagnose', 'Retry the diagnostic probe', 'Successful diagnostic fork from main1, with expanded training metrics.'),
  'structured1': ('revise', 'Add Pokémon-specific action structure', 'Fork from main1: active friendly/opponent Pokémon, team summaries and switch-target features; lower learning rate.'),
  'structured2': ('continue', 'Train the structured policy longer', 'Strict full-state continuation of structured1.'),
  'horizon1': ('revise', 'Extend the learning horizon', 'Fork from structured2: gamma 0.999, GAE lambda 0.98, lower learning rate and new training streams.'),
  'horizon2': ('continue', 'Train the longer-horizon policy', 'Strict full-state continuation of horizon1.'),
  'averaged_final': ('revise', 'Smooth the final policy weights', 'Fork from horizon2: exponential moving average, decay 0.995, learning rate 0.00005. Explicitly selected as final.'),
 },
 'kimi31pro': {
  'starter_smoke': ('initial', 'Check the supplied starter', 'Short starter training run; retains its learner and trainer.'),
  'attn_smoke': ('initial', 'Try an attention architecture', 'Two four-head attention blocks, positional embeddings and action/state pairing. Width 128; embeddings 16.'),
  'attn_1500': ('initial', 'Extend the attention trial', 'Continue attention to a cumulative requested 1,500 seconds before the comparison.'),
  'starter_1500': ('initial', 'Run the matching starter control', 'Continue the starter to the same cumulative requested training budget. Public result: 30/64 versus attention’s 33/64.'),
  'attn_main1': ('continue', 'Commit to the attention branch', 'First near-hour continuation; unchanged attention source and training configuration.'),
  'attn_main2': ('continue', 'Continue the same policy', 'Second near-hour strict continuation. Public practice improves on the common bank.'),
  'attn_main3': ('continue', 'Continue, then compare checkpoints', 'Third near-hour strict continuation. Additional practice comparisons lead to selecting attn_main3.'),
 },
}


def read(p):
    return json.loads(p.read_text())


def collect():
    db = sqlite3.connect(f'file:{BASE}/campaign.sqlite3?mode=ro', uri=True)
    db.row_factory = sqlite3.Row
    source = ROOT / 'reports/pokemon/sol-kimi-research-analysis-2026-11-10.json'
    report = read(source)
    common_start = float(db.execute("SELECT value FROM pilot_metadata WHERE name='common_start'").fetchone()[0])
    agents = []
    for arm, name, effort in [('sol61-high', 'Sol 6.1', 'high'), ('kimi31pro', 'Kimi K3.1 Pro', 'max')]:
        home = BASE / 'arms' / arm
        stages = {p.stem: read(p) for p in (home / 'stages').glob('*.json')}
        assert stages.keys() == STAGES[arm].keys(), 'Stage annotation needs updating'
        native = read(home / 'native-state.json')
        order = {c['id']: i for i, c in enumerate(native['calls'])}
        events = []
        for p in (home / 'tool-journal').glob('*.json'):
            j = read(p)
            a = json.loads(j['tool']['function']['arguments'])
            events.append(dict(turn=order[j['turn']], args=a, result=j.get('result', {})))
        events.sort(key=lambda x: x['turn'])
        training = {e['args']['stage']: e for e in events if e['args']['operation'] == 'train'}
        jobs = [dict(r) for r in db.execute("SELECT * FROM pilot_jobs WHERE arm=? AND kind='research'", (arm,))]
        totals = {b['id']: 0. for b in BUCKETS}
        trials = []
        allocations = {j['id']: j for j in jobs}
        practice_events = {e['result']['operation']: e for e in events
                           if e['args']['operation']=='practice' and e['result'].get('operation')}
        timeline = []
        def interval(job):
            receipt = json.loads(job['receipt'])
            start, end = receipt['started_at']-common_start, receipt['finished_at']-common_start
            assert math.isclose(end-start, job['actual_seconds'], abs_tol=1e-5)
            return dict(start=start, end=end)
        for stage, s in stages.items():
            bucket, title, explanation = STAGES[arm][stage]
            job = allocations[s['operation']]
            assert job['status'] == 'settled'
            seconds = job['actual_seconds']
            totals[bucket] += seconds
            args = training[stage]['args']
            parent = args.get('resume_stage') or args.get('fork_stage')
            kind = 'Strict continuation' if 'resume_stage' in args else 'Checkpoint fork' if 'fork_stage' in args else 'Fresh start'
            trials.append(dict(stage=stage, bucket=bucket, title=title, explanation=explanation,
                               seconds=seconds, parent=parent, kind=kind,
                               successful=s.get('successful', False), turn=training[stage]['turn']))
            timeline.append(dict(**trials[-1], **interval(job), id=stage, type='training', label=stage))
        evaluation = []
        for j in jobs:
            request = read(BASE / 'jobs' / j['id'] / 'request.json')
            if request['action'] == 'assess':
                totals['evaluate'] += j['actual_seconds']
                evaluation.append(j['id'])
                event = practice_events[j['id']]
                result = event['result'].get('public_practice', {})
                label = event['args']['label']
                score = result.get('raw_score_0_100')
                explanation = ('Incomplete public practice panel.' if score is None else
                               f"Public practice: {score:.2f} points; {result.get('natural_games', 0)} natural games.")
                timeline.append(dict(**interval(j), id=j['id'], bucket='evaluate', type='evaluation',
                                     title=f'Evaluate {label}', label=label, explanation=explanation,
                                     seconds=j['actual_seconds'], successful=score is not None,
                                     score=score, games=result.get('natural_games'),
                                     turn=event['turn']))
        gpu = sum(j['actual_seconds'] for j in jobs)
        assert math.isclose(sum(totals.values()), gpu, abs_tol=1e-5)
        expected = report['arms'][arm]['allocation_seconds_by_action']
        assert math.isclose(gpu, sum(expected.values()), abs_tol=1e-5)
        stats = dict(fresh=sum(t['kind'] == 'Fresh start' for t in trials),
                     forks=sum(t['kind'] == 'Checkpoint fork' for t in trials),
                     resumes=sum(t['kind'] == 'Strict continuation' for t in trials),
                     public_workers=len(evaluation),
                     write_files=sorted({e['args']['file'] for e in events if e['args']['operation'] == 'write'}),
                     rejected_reads=sum(e['args']['operation']=='read' and 'error' in e['result'] for e in events),
                     rejected_practice=sum(e['args']['operation']=='practice' and 'error' in e['result'] for e in events))
        configs = [json.loads(e['args']['content']) for e in events if e['args']['operation']=='write' and e['args']['file']=='config.json']
        # Network sizes and random streams are not a change to the PPO recipe.
        recipes = {json.dumps({k:v for k,v in c.items() if k not in {'network','seed','training_seed_start'}}, sort_keys=True) for c in configs}
        stats['ppo_recipes'] = len(recipes)
        model_intervals = []
        for i, call in enumerate(native['calls']):
            seconds = call['receipt']['seconds']
            start = call['dispatch_at'] - common_start
            own = [e['args'] for e in events if e['turn'] == i]
            actions = []
            for action in own:
                op = action['operation']
                detail = action.get('file') or action.get('stage') or action.get('label')
                actions.append(op + (f' {detail}' if detail else ''))
            model_intervals.append(dict(id=f'model-{i}', bucket='model', type='model',
                                        title=f'Model request {i+1}', start=start, end=start+seconds,
                                        seconds=seconds, actions=actions,
                                        explanation='Returned actions: '+', '.join(actions)+'. Action execution follows the response; edits have no independent duration measurement.'))
        final = read(home / 'final.json')['finalized_at_unix'] - common_start
        timeline.sort(key=lambda t:t['start'])
        assert all(t['start']>=0 and t['end']<=final+0.1 for t in timeline+model_intervals)
        assert all(a['end']<=b['start'] for a,b in zip(timeline,timeline[1:])), 'Overlapping GPU workers'
        assert math.isclose(sum(t['end']-t['start'] for t in timeline), gpu, abs_tol=1e-5)
        markers = ([dict(stage='structured1',label='R1',title='Revision 1 · structured actions'),
                    dict(stage='horizon1',label='R2',title='Revision 2 · longer horizon'),
                    dict(stage='averaged_final',label='R3',title='Revision 3 · averaged weights')]
                   if arm=='sol61-high' else
                   [dict(stage='attn_smoke',label='A',title='Attention trial begins'),
                    dict(stage='attn_main1',label='C',title='Commit to attention')])
        for marker in markers:
            marker['at'] = next(t['start'] for t in timeline if t['id']==marker['stage'])
        agents.append(dict(id=arm, name=name, effort=effort, gpu=gpu, totals=totals,
                           trials=sorted(trials, key=lambda t:t['turn']), stats=stats,
                           timeline=timeline, model_intervals=model_intervals,
                           final_seconds=final, markers=markers))
    db.close()
    return dict(agents=agents, buckets=BUCKETS, common_start_unix=common_start,
                timeline_end=max(a['final_seconds'] for a in agents),
                timeline_methodology='GPU intervals use provider started_at/finished_at receipts, relative to the common research start. Model intervals use recorded dispatch_at plus request latency. Revision markers locate the first training worker using the revision, not the beginning of code authoring. Unallocated gaps remain visible. Later independent private grading is excluded.',
                methodology='Stage-purpose labels are editorial annotations based on tool arguments, source/config changes and the owning analysis. Each research allocation is assigned exactly once. Initial trial continuations stay in Initial experiments; subsequent unchanged continuations are separate. Bars measure full allocated GPU seconds committed to the experiment purpose, not authoring time or model thinking. Failed work remains included.',
                source=source.relative_to(ROOT).as_posix(),
                source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                campaign=BASE.relative_to(ROOT).as_posix(),
                outcomes=dict(sol_gain=14.84375, sol_gain_ci=[10.35,19.53],
                              kimi_diagnostic_gain=2.8645833333, kimi_diagnostic_ci=[-2.60,8.59],
                              crossplay_points=69.140625, crossplay_games=256))


def main():
    data = collect()
    template = (ROOT / 'tools/templates/pokemon-research-behavior.html').read_text()
    OUT.write_text(template.replace('__DATA__', json.dumps(data, ensure_ascii=False).replace('<', '\\u003c')))
    OUT.with_suffix('.json').write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')
    print(OUT)
    for a in data['agents']:
        print(a['name'], {b:round(s/60,2) for b,s in a['totals'].items()}, a['stats'])


if __name__ == '__main__':
    main()
