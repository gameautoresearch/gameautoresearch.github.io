"""Read-only evidence adapter. No simulator imports, model calls or cloud writes.

Produces sanitized contract-v1 data and extends the existing local Sol/Kimi page.
Run: python3 tools/build_pokemon_four_trajectories.py
"""
from pathlib import Path
import copy
import datetime as dt
import hashlib
import importlib.util
import json
import math
import re
import sqlite3

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'reports/pokemon'
ORIGINAL = ROOT / 'evidence/pokemon-four-model-4h-20261109-1354Z'
CONT = ROOT / 'evidence/pokemon-continuation-20261110-0456Z'
SKILL = ROOT / '.agents/skills/trajectory-visualization'
PREFIX = 'four-researcher-trajectories-2026-11-10'
ZONE = dt.timezone(dt.timedelta(hours=8))

def read(p):
    return json.loads(p.read_text())

def stamp(t):
    return dt.datetime.fromtimestamp(t, ZONE).strftime('%b %d %H:%M:%S Shanghai')

def load_builder():
    spec = importlib.util.spec_from_file_location('trajectory_builder', SKILL / 'scripts/build.py')
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

BUCKETS = [
    ('initial', 'Initial experiments', '#edb447', 'First architecture/control experiments. Analyst-assigned purpose, not a measured authoring interval.'),
    ('diagnose', 'Diagnose / verify', '#ed896d', 'Explicit resume, compatibility or diagnostic checks; actual failed allocations remain charged.'),
    ('revise', 'Revise the approach', '#43a98e', 'First trial of an annotated substantive source/recipe revision. A checkpoint fork alone is not classified as a new idea.'),
    ('continue', 'Continue unchanged', '#5b9dc9', 'Strict resume of the same source/config lineage. Initial control extensions can remain in initial experiments.'),
    ('evaluate', 'Public practice', '#a37cc6', 'Full allocated public-practice workers, including incomplete panels. Different banks are not one learning curve.'),
    ('model', 'Model requests', '#899680', 'Recorded request latency only; failed requests with no end timestamp appear as points.'),
    ('actions', 'Edits / submissions', '#bf925c', 'Timestamped submissions, and response-boundary markers for returned file-edit actions. Tool execution duration is unmeasured.'),
    ('infra', 'Infrastructure / gap', '#af939b', 'Recorded interruption and restart gap; no model activity is inferred within gaps. Excluded from allocation totals.'),
]
# Explicit stage annotations, checked exhaustively against the two stage manifests.
STAGES = {
 'opus55-high': {
  's0_starter': ('initial', 'Check the starter'),
  'a1_tf_league': ('initial', 'Transformer + league attempt'),
  'a2_tf_league': ('initial', 'Retry transformer + league'),
  'm1': ('initial', 'Long transformer/league training'),
  'resume-check': ('diagnose', 'Mechanical full-state resume proof'),
  'm2': ('continue', 'Resume m1 lineage; export fails'),
  'm2f': ('infra', 'Upload rejected before allocation'),
  'm2g': ('infra', 'Source guard rejected before allocation'),
  'm2h': ('revise', 'Fork restored parent with trainer revision'),
  'm3': ('continue', 'Continue m2h unchanged'),
 },
 'deepseek-v4-flash': {
  'starter-baseline': ('initial', 'Check the starter'),
  'transformer-test': ('initial', 'Transformer trial'),
  'transformer-test2': ('initial', 'Second transformer trial'),
  'residual-mlp-test': ('revise', 'Try residual MLP'),
  'residual-mlp-v2': ('revise', 'Revise residual MLP'),
  'manual-attn-test': ('revise', 'Manual action/mon attention'),
  'pos-encoding-test': ('revise', 'Add positional encoding'),
  'hidden320-test': ('revise', 'Try width 320'),
  'improved-arch-test': ('revise', 'Revise the architecture'),
  'improved-training-v1': ('revise', 'Revise PPO training recipe'),
  'hidden384-test': ('revise', 'Try width 384'),
  'self-attn-test': ('revise', 'Self-attention trial'),
  'proven-config-test': ('revise', 'Return to working configuration'),
  'resume-check': ('diagnose', 'Mechanical full-state resume proof'),
  'resume-verify': ('continue', 'Verify unchanged continuation'),
  'self-attn-fork-test': ('revise', 'Fork to self-attention variant'),
  'nblocks3-fork-test': ('diagnose', 'Probe fork compatibility'),
  'nblocks3-fresh-test': ('diagnose', 'Check architecture from fresh state'),
  'fork-from-fresh-test': ('diagnose', 'Test fork from fresh checkpoint'),
  'fork-matching-test': ('diagnose', 'Test matching-source fork'),
  'improved-learner-fresh': ('revise', 'Fresh policy + revised learner/trainer'),
  'main-training-run1': ('continue', 'Main training 1'),
  'main-training-run2': ('continue', 'Main training 2'),
  'main-training-run3': ('continue', 'Main training 3'),
  'main-training-run4': ('continue', 'Main training 4'),
 },
}

SUMMARY = {
 'sol61-high': 'Revised structured action features, the learning horizon and final weight averaging; strict continuations connected compatible stages. Original final remains frozen.',
 'kimi31pro': 'Compared attention against the starter, then spent most allocation on three unchanged attention continuations. Its official primary panel remains incomplete.',
 'opus55-high': 'Built a transformer with a historical-policy league, then resumed m1. A long export failure consumed real allocation. Forked the restored parent with trainer changes, continued m2h to m3, and submitted it; the predeclared fallback selected m3 after source-guard failures.',
 'deepseek-v4-flash': 'Tested multiple transformer/residual/attention variants, then checked fork compatibility. Started a fresh improved-learner branch and used four unchanged main continuations. Transport retries interrupted requests; main-run4 was explicitly finalized.',
}

def event(eid, start, end, bucket, title, detail, **extra):
    return dict(id=eid, start=start, end=end, bucket=bucket, title=title, detail=detail,
                include_in_aggregate=False, **extra)


def prior_agents(origin):
    old = read(REPORT / 'sol-kimi-research-behavior-2026-11-10.json')
    assert old['common_start_unix'] == origin
    agents = []
    for a in old['agents']:
        ev = []
        for e in a['timeline'] + a['model_intervals']:
            evidence = f"{ORIGINAL.name}/arms/{a['id']}/" + ('native-state.json' if e['type']=='model' else f"stages/{e['stage']}.json" if e['type']=='training' else f"public practice operation {e['id']}")
            detail = e['explanation'] + f" Actual: {stamp(origin+e['start'])} → {stamp(origin+e['end'])}. Evidence: {evidence}."
            if e.get('kind'):
                detail += ' ' + e['kind'] + (' from '+e['parent'] if e.get('parent') else '') + '.'
            q=event(e['id'],e['start'],e['end'],e['bucket'],e['title'],detail,
                    short_label=e.get('stage',e.get('label','')), status='failed / incomplete' if e.get('successful') is False else 'recorded', actions=e.get('actions',[]))
            q['include_in_aggregate']=e['type']!='model'
            ev.append(q)
        agents.append(dict(id=a['id'],name=a['name'],subtitle=a['effort']+' · original frozen',summary=SUMMARY[a['id']],
                           end_seconds=a['final_seconds'],end_label='Original finalized',events=ev,
                           facts=[dict(label='H100 allocated',value=f"{a['gpu']/3600:.3f} h"),dict(label='training trials',value=len(a['trials']))],
                           markers=[dict(at=m['at'],label=m['label'],title=m['title'],event_id=m['stage']) for m in a['markers']]))
    return agents


def collect_arm(arm, origin):
    events=[]; markers=[]; receipts=[]; segments=[]; untimed=[]; accounted=0; train_count=0; practice_count=0; failures=0
    seen_stages=set(); last_original=None; resume_start=None
    for si,base in enumerate([ORIGINAL,CONT]):
        label='Original' if si==0 else 'Continuation'
        home=base/'arms'/arm; native=read(home/'native-state.json')
        calls={c['id']:c for c in native['calls']}
        actions=[]
        for p in (home/'tool-journal').glob('*.json'):
            j=read(p);args=json.loads(j['tool']['function']['arguments'])
            actions.append((j,args))
        actions.sort(key=lambda pair:calls.get(pair[0]['turn'],{}).get('dispatch_at',float('inf')))
        train={j.get('result',{}).get('operation'):(j,a) for j,a in actions if a['operation']=='train' and j.get('result',{}).get('operation')}
        practice={j.get('result',{}).get('operation'):(j,a) for j,a in actions if a['operation']=='practice' and j.get('result',{}).get('operation')}
        stages={p.stem:read(p) for p in (home/'stages').glob('*.json') if p.stem!='imported-parent'}
        seen_stages.update(stages)
        byop={s['operation']:(name,s) for name,s in stages.items() if s.get('operation')}
        db=sqlite3.connect(f'file:{base}/campaign.sqlite3?mode=ro',uri=True);db.row_factory=sqlite3.Row
        jobs=list(map(dict,db.execute("SELECT * FROM pilot_jobs WHERE arm=? AND kind='research' ORDER BY created",(arm,))))
        db.close(); segment_seconds=0
        for j in jobs:
            assert j['status']=='settled', (arm,j['id'],j['status'])
            rc=json.loads(j['receipt']); secs=j['actual_seconds'];assert secs is not None
            accounted+=secs;segment_seconds+=secs
            req=read(base/'jobs'/j['id']/'request.json')
            eid=f'{si}-{j["id"]}'
            evidence=f'{base.name}/jobs/{j["id"]} · ledger receipt'
            if secs==0:
                assert rc.get('no_provider_created') or rc.get('provider_verified_absent')
                events.append(event(eid,j['created']-origin,j['created']-origin,'infra',label+' · pre-provider rejection',
                    f"Zero allocated GPU seconds, verified before provider creation. {rc.get('cause','')} Evidence: {evidence}. Timestamp is reservation creation, not a measured outage duration.",status='blocked before allocation',segment=label))
                continue
            start,end=rc['started_at'],rc['finished_at'];assert math.isclose(end-start,secs,abs_tol=1e-5)
            if si==1:resume_start=min(resume_start or start,start)
            if si==0:last_original=max(last_original or end,end)
            if req['action']=='train':
                stage,s=byop[j['id']];bucket,title=STAGES[arm][stage]
                args=train.get(j['id'],({},{}))[1]
                if stage=='resume-check':args=dict(resume_stage='imported-parent')
                parent=args.get('resume_stage') or args.get('fork_stage')
                kind='Unchanged full-state resume' if args.get('resume_stage') else 'Checkpoint fork' if args.get('fork_stage') else 'Fresh start'
                if stage=='resume-check':parent='original m1 → imported-parent' if arm=='opus55-high' else 'original proven-config-test → imported-parent'
                detail=f'{kind}'+(f' from {parent}.' if parent else '.')
                if stage=='improved-learner-fresh':detail+=' Adds entropy annealing alongside the retained learning-rate schedule and value clipping; starts fresh rather than inheriting the prior checkpoint.'
                if stage=='m2h':detail+=' Forks resume-check, not failed m2. Trainer now reserves 180 seconds for save/export and imports league snapshot tensors plus the anneal counter; Adam/RNG/native battle state start fresh. Sealed policy/config retained. This is a first-run revision boundary, not timed code authoring.'
                result_path=base/'jobs'/j['id']/'output/stage/result.json'
                metrics={}
                if result_path.exists():
                    r=read(result_path);metrics={k:r[k] for k in ['trained_samples','iterations','optimizer_updates','faults','truncations'] if k in r}
                    detail+=f" New samples in this stage: {r.get('trained_samples','unreported'):,}. Reported final iteration index: {r.get('iterations','unreported')} (may include inherited updates; not a new-sample count). Faults: {r.get('faults','unreported')}; truncations: {r.get('truncations','unreported')}."
                else:detail+=' No sealed result.json retained; do not count partial progress as a successful checkpoint.'
                successful=s.get('successful',False);failures+=not successful;train_count+=1
                detail+=f" Source {s.get('source_sha256','unreported')[:12]}, config {s.get('config_sha256','unreported')[:12]}."
                extra=dict(stage=stage,parent=parent,kind=kind,metrics=metrics)
                if bucket=='revise':markers.append(dict(at=start-origin,label='R',title=stage+' · first revised trial',event_id=eid))
            else:
                assert req['action']=='assess';bucket='evaluate';p,args=practice.get(j['id'],({},{}));stage=args.get('label','practice');title='Practice '+stage
                out=base/'jobs'/j['id']/'output/summary.json';r=read(out) if out.exists() else {};r=p.get('result',{}).get('public_practice',r)
                score=r.get('raw_score_0_100');successful=score is not None;practice_count+=1
                detail=f"Public panel points: {score if score is not None else 'null / incomplete'}. Natural games: {r.get('natural_games','unreported')}. Public practice only; no comparison across different banks is implied."
                extra=dict(stage=stage)
            detail+=f' {label}. Actual: {stamp(start)} → {stamp(end)}. Full allocation {secs:.6f} seconds, including setup/export/failures. Evidence: {evidence}.'
            e=event(eid,start-origin,end-origin,bucket,title,detail,short_label=stage,status='succeeded' if successful else 'failed / incomplete',segment=label,**extra)
            e['include_in_aggregate']=True;events.append(e)
            receipts.append(dict(segment=label,operation=j['id'],stage=stage,seconds=secs,bucket=bucket,**({'metrics':metrics} if req['action']=='train' else {})))
        segments.append(dict(label=label,allocated_seconds=segment_seconds))
        for index,c in enumerate(native['calls']):
            start=c.get('dispatch_at',c['started_at']);seconds=c.get('receipt',{}).get('seconds');failed=c['status']!='complete'
            end=start+seconds if seconds is not None else start
            own=[(j,a)for j,a in actions if j['turn']==c['id']]
            safe_actions=[a['operation']+(f" {a.get('file') or a.get('stage') or a.get('label')}" if a.get('file')or a.get('stage')or a.get('label') else '') for _,a in own]
            events.append(event(f'{si}-request-{index}',start-origin,end-origin,'model',f'{label} request {index+1}'+(' · transport failure' if failed else ''),
                f"{'Failed request; end timestamp unrecorded, so shown as a point.' if seconds is None else 'Recorded request latency; tool execution occurs after the response.'} Not code-authoring or thinking duration. Actual dispatch: {stamp(start)}. Evidence: {base.name}/arms/{arm}/native-state.json call {index+1}.",status='transport failed' if failed else 'complete',actions=safe_actions,segment=label))
            writes=[a['file']for _,a in own if a['operation']=='write']
            if writes and seconds is not None:
                events.append(event(f'{si}-edits-{index}',end-origin,end-origin,'actions','Returned edits · '+', '.join(writes),
                    'Response-completion boundary only. These file-write actions were returned; exact tool execution/authoring duration is not recorded. See the first affected training trial for its source/config identity.',segment=label))
            for ji,(journal,args) in enumerate(own):
                result=journal.get('result',{});op=args['operation'];t=(result.get('finalized_at_unix') or result.get('submitted_at_unix')) if op=='finalize' else result.get('submitted_at_unix')
                if op in ['submit','finalize'] and t:
                    events.append(event(f'{si}-{op}-{index}-{ji}',t-origin,t-origin,'actions',op.title()+' '+args.get('label',''),f"Explicit model tool action. Stage {result.get('stage',args.get('stage','unreported'))}. Actual: {stamp(t)}. Evidence: {base.name}/arms/{arm}/submission-receipts/. No weights or private inputs are embedded.",segment=label))
                if result.get('error'):
                    text=str(result['error'])
                    # Summaries, never raw errors that can contain credentials or payloads.
                    cause=('Source snapshot guard blocked dispatch' if 'Protected source changed' in text else 'Duplicate stage rejected' if 'already attempted' in text else 'Lease bounds rejected' if 'Finite H100 lease' in text else 'Static-workspace command rejected' if 'static checks only' in text else 'Read outside editable scope rejected' if 'Only own editable' in text else 'Unavailable submission rejected' if 'No such file' in text else 'Unsuccessful stage rejected' if 'Only successful settled' in text else 'Tool request rejected; raw message retained privately')
                    if seconds is not None:events.append(event(f'{si}-reject-{index}-{ji}',end-origin,end-origin,'infra',cause,
                        f"{op} {args.get('stage',args.get('label',''))}. Marker is the preceding response boundary, not exact rejection time. No invented allocation. Evidence: {base.name}/arms/{arm}/tool-journal/.",segment=label,status='rejected'))
                    else:untimed.append(cause)
        if si==1 and arm=='deepseek-v4-flash':
            recovery=native.get('transport_recovery_events',[])
            for ri,r in enumerate(recovery):
                last=calls[r['last_failed_call']]; recovered=next((c for c in native['calls'] if c.get('dispatch_at',0)>last['dispatch_at'] and c['status']=='complete'),None)
                if recovered:
                    failed=[c for c in native['calls'] if c['status']!='complete' and c['dispatch_at']<=last['dispatch_at']]
                    start=failed[-r['failures']]['dispatch_at'];end=recovered['dispatch_at']
                    assert not any(e['include_in_aggregate'] and e['start']<end-origin and e['end']>start-origin for e in events)
                    events.append(event('transport-pause-'+str(ri),start-origin,end-origin,'infra','Transport disruption → successful redispatch',f"{r['failures']} failed attempts in one logical turn. Span from first failed dispatch to next successfully completed request’s dispatch, {stamp(start)} → {stamp(end)}. Includes retries/backoff/recovery; exact network-down time is unknown. No researcher GPU allocation overlaps this span. Evidence: native-state.json transport_recovery_events.",status='transport pause',segment=label))
        if si==0:
            status=read(home/'status.json');t=status.get('updated_unix')
            if t:
                events.append(event('original-stop',t-origin,t-origin,'infra','Original session interrupted',
                    'Global unknown-liability guard stopped Opus after another arm’s transport failure; Opus had no failed API request.' if arm=='opus55-high' else 'Incomplete transport response closed the original session. Later billing reconciliation did not restart research.',status='interrupted',segment=label))
                last_original=max(last_original or t,t)
    assert seen_stages==set(STAGES[arm]), (seen_stages^set(STAGES[arm]))
    final=read(CONT/'arms'/arm/'final.json');finish=final['finalized_at_unix']
    assert math.isclose(accounted,final['at_allocated_seconds'],abs_tol=1e-5)
    if last_original and resume_start:
        events.append(event('restart-gap',last_original-origin,resume_start-origin,'infra','Closed original → authorized checkpoint restore',
            f"{stamp(last_original)} → {stamp(resume_start)}. Historical closed session followed by a separately authorized continuation. No researcher GPU allocation in this interval; do not call this thinking time. Same four-hour lifetime allowance; restored parent: {'m1' if arm=='opus55-high' else 'proven-config-test'}.",status='restart gap',segment='Between sessions'))
    events.append(event('final-selection',finish-origin,finish-origin,'actions','Fallback selected m3' if arm=='opus55-high' else 'Explicitly finalized main-run4',
        f"{'Predeclared fallback selected the latest valid explicit submission; researcher did not complete an explicit finalize call.' if arm=='opus55-high' else 'Researcher explicitly finalized its submitted main-run4.'} Actual: {stamp(finish)}. Evidence: {CONT.name}/arms/{arm}/final.json. Lifetime allocated H100: {accounted:.6f} s of 14,400 s."))
    for e in events:assert 0<=e['start']<=e['end']<=finish-origin+0.01,(arm,e)
    return dict(id=arm,name='Opus 5.5' if arm=='opus55-high' else 'DeepSeek V4 Flash',subtitle='high' if arm=='opus55-high' else 'xhigh',summary=SUMMARY[arm],
        events=sorted(events,key=lambda e:e['start']),markers=markers,end_seconds=finish-origin,
        end_label='Fallback locked m3' if arm=='opus55-high' else 'Finalized main-run4',
        facts=[dict(label='lifetime H100',value=f'{accounted/3600:.3f} h'),dict(label='training trials',value=train_count),dict(label='failed training trials',value=failures),dict(label='allocated practice workers',value=practice_count)],
        segments=segments,receipt_audit=receipts,untimed_notes=untimed)


def navigation():
    return f'<nav class="trajectory-nav" aria-label="Research views"><a href="{PREFIX}.html">All four · active time</a><a href="{PREFIX}-continuation.html">Opus + DeepSeek · continuation</a><a href="sol-kimi-research-behavior-2026-11-10.html">Sol + Kimi · original</a></nav>'


def compact_activity_clock(agent, origin):
    """Remove unrecorded idle spans; retain every measured work/request duration.

    The user requested comparable timelines without closure gaps. This clock is
    the union of measured worker and request intervals, not elapsed wall time or
    cumulative GPU allocation. Overlaps count once. Source coordinates survive.
    """
    measured = sorted((e['start'], e['end']) for e in agent['events']
                      if (e['include_in_aggregate'] or e['bucket'] == 'model')
                      and e['end'] > e['start'])
    union = []
    for start, end in measured:
        if union and start <= union[-1][1]:
            union[-1][1] = max(union[-1][1], end)
        else:
            union.append([start, end])

    def position(t):
        return sum(max(0, min(t, end) - start) for start, end in union)

    gap = next((e for e in agent['events'] if e['id'] == 'restart-gap'), None)
    agent['wall_end_seconds'] = agent['end_seconds']
    agent['clock_intervals'] = union
    agent['events'] = [e for e in agent['events']
                       if e['id'] not in {'restart-gap', 'original-stop'}]
    for e in agent['events']:
        start, end = e['start'], e['end']
        e['wall_start_seconds'], e['wall_end_seconds'] = start, end
        e['actual_start_unix'], e['actual_end_unix'] = origin + start, origin + end
        e['start'], e['end'] = position(start), position(end)
        if e['include_in_aggregate'] or e['bucket'] == 'model':
            assert math.isclose(e['end'] - e['start'], end - start, abs_tol=1e-6)
        if e['id'].startswith('transport-pause-'):
            e['title'] = 'Transport retries recovered'
    for m in agent.get('markers', []):
        m['wall_at_seconds'] = m['at']
        m['at'] = position(m['at'])
    if gap:
        t = position(gap['end'])
        marker = event('session-resume', t, t, 'actions', 'Resume saved checkpoint',
                       'Continuation follows the earlier session directly on this active-time clock. '
                       'Shutdown and unrecorded idle time are omitted; actual dates remain in event details.')
        marker['wall_start_seconds'] = marker['wall_end_seconds'] = gap['end']
        marker['actual_start_unix'] = marker['actual_end_unix'] = origin + gap['end']
        agent['events'].append(marker)
    agent['events'].sort(key=lambda e: e['start'])
    agent['end_seconds'] = position(agent['wall_end_seconds'])
    agent['omitted_idle_seconds'] = agent['wall_end_seconds'] - agent['end_seconds']


def main():
    old=read(REPORT/'sol-kimi-research-behavior-2026-11-10.json');origin=old['common_start_unix']
    agents=prior_agents(origin)+[collect_arm(a,origin)for a in ['opus55-high','deepseek-v4-flash']]
    summary=read(CONT/'final-round-robin-summary.json')
    data=dict(schema_version=1,title='Research, from start to finish.',kicker='Pokémon / four researchers / Nov 9–10, 2026',
        subtitle='Four research trajectories. Sol and Kimi retain their original runs; Opus and DeepSeek include the original attempt and its authorized continuation. Read the timelines first, then compare allocation below.',
        time_origin=f'Common real origin: {stamp(origin)}. All x positions are actual elapsed wall seconds from this origin, including the overnight restart gap. No compressed gaps or per-agent renormalization.',
        timing_note='Full allocated H100/4 physical CPU/16 GiB intervals use settled provider receipts; model blocks use dispatch plus recorded request latency. Failed requests without an end time are points. Edit markers use response boundaries, revisions use first trial starts; code-authoring durations were not measured. Four GPU hours are lifetime, not four additional hours.',
        buckets=[dict(id=i,label=l,color=c,description=d)for i,l,c,d in BUCKETS],
        aggregate=dict(label='Allocated H100 time by purpose · lifetime',note='Denominator: all settled researcher H100 allocations in the selected evidence segments, including resume proofs, practice, failed workers and overhead. Excludes request latency, restart gaps, pre-provider rejections and independent final CPU grading. Four-hour lifetime cap per researcher.'),
        agents=agents,comparisons=[dict(label='Demonstrated iteration',cells={a['id']:dict(title={'sol61-high':'Structured actions → horizon → EMA','kimi31pro':'Attention/control → sustained resume','opus55-high':'League transformer → export fix → resume','deepseek-v4-flash':'Architecture probes → revised learner → resume'}[a['id']],detail=SUMMARY[a['id']])for a in agents})],notes=[
            'Behavior categories are analyst annotations from tool arguments and retained source/config identities. A failed fork is a compatibility/experiment outcome, not automatically a strategic revision.',
            'Stage trained_samples are new samples generated in that stage, not inherited totals. Reported final iteration indices may include inherited updates. Branch samples must not be described as one continuous lineage.',
            'Original Sol/Kimi outcomes refer only to the original private panel: Sol 74.02 versus baseline 59.18; Kimi primary null after a reference timeout. They are not the continuation round robin.',
            f"Continuation round-robin summary at build: {summary['status'].upper()}. All original attempted pair scores are null; source-snapshot dispatch failed before grader workers/games. Coordinator’s completed phase is not evidence of a valid tournament. Evaluation-only recovery is tracked separately; no model losses or zero scores are assigned.",
            'No raw native conversation/reasoning, code payloads, checkpoint data, credentials, gateway state, private seeds/banks or game assets are embedded. Evidence paths identify private local originals; the local server only exposes derived report files.',
        ],sources=[dict(label='Original Sol/Kimi view',href='sol-kimi-research-behavior-2026-11-10.html'),dict(label='Sanitized evidence audit',href=PREFIX+'-audit.json'),dict(label='Original pilot readout',href='four-model-4h-pilot-2026-11-09.md')],
        outcomes=[dict(agent_id=a['id'],title='Research endpoint',value=a['end_label'],detail='Original frozen endpoint.' if a['id']in ['sol61-high','kimi31pro'] else 'Selected checkpoint is locked. Original continuation tournament invalid; no valid four-way ranking yet.')for a in agents])
    builder=load_builder();template=(ROOT/'tools/templates/pokemon-four-trajectories.html').read_text()
    variants=[('',data)]
    close=copy.deepcopy(data);close['agents']=close['agents'][2:]
    close_origin=min(e['start']for a in close['agents'] for e in a['events'] if e.get('segment')=='Continuation')
    for a in close['agents']:
        a['events']=[e for e in a['events']if e.get('segment')=='Continuation' or e['id']=='final-selection']
        for e in a['events']:e['start']-=close_origin;e['end']-=close_origin
        a['markers']=[m for m in a['markers']if m['at']>=close_origin]
        for m in a['markers']:m['at']-=close_origin
        a['end_seconds']-=close_origin
        a['summary']+=' Lifetime facts include the original attempt; the bars below include continuation allocations only.'
    close['title']='The continuation, in detail.';close['subtitle']='Opus and DeepSeek from checkpoint restore through final selection. This closer view retains every continuation timestamp and transport gap.'
    close['time_origin']=f'Common continuation display origin: {stamp(origin+close_origin)} (first mechanical resume worker). Research window opened later. Both timelines use actual elapsed time from this same origin; this is not per-agent time since start.'
    close['aggregate']['label']='Allocated H100 time by purpose · continuation only'
    close['aggregate']['note']='Denominator: continuation research allocations only, including mechanical resume proofs and failed workers. Lifetime facts above include the original attempt. Same H100/4 physical CPU/16 GiB. No renewed allowance.'
    close['outcomes']=close['outcomes'][2:]
    for c in close['comparisons']:c['cells']={k:v for k,v in c['cells'].items() if k in ['opus55-high','deepseek-v4-flash']}
    variants.append(('-continuation',close))
    # The main comparison uses the user's requested gap-free activity clock.
    for a in data['agents']:
        compact_activity_clock(a, origin)
    data['subtitle']='Four research trajectories on the same active-time scale. Training and measured model requests run consecutively; shutdowns and empty waiting gaps are removed.'
    data['time_origin']='Active recorded time from 0 for each researcher: GPU-worker intervals plus measured model-request intervals, counting overlaps once. All four share the same hour scale. Actual dates remain in event details.'
    data['timing_note']='Empty gaps and the overnight shutdown are omitted. Every worker and measured request keeps its real duration. The clock is recorded activity time; allocated H100 totals are shown separately below. Untimed edits, failures and submissions remain point markers.'
    data['tick_seconds']=1800
    data['time_domain_seconds']=math.ceil(max(a['end_seconds'] for a in data['agents'])/1800)*1800
    for b in data['buckets']:
        if b['id']=='infra':b['label']='Failures / retries'
    data['notes'].insert(0,'At the user’s request, the main comparison removes all unrecorded waiting gaps and concatenates the original and continuation sessions. Its shared cursor compares recorded activity elapsed, not simultaneous real-world timestamps. Source timestamps and omitted durations remain in the audit; the continuation detail view retains its wall clock.')
    for suffix,d in variants:
        path=REPORT/(PREFIX+suffix+'.json');path.write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n');builder.validate(d)
        encoded=json.dumps(d,ensure_ascii=False).replace('&','\\u0026').replace('<','\\u003c').replace('>','\\u003e')
        (REPORT/(PREFIX+suffix+'.html')).write_text(template.replace('__DATA__',encoded).replace('__NAV__',navigation()))
    audit=dict(origin_unix=origin,continuation_display_origin_unix=origin+close_origin,summary_status=summary['status'],main_clock='union of recorded worker and measured request intervals; unrecorded gaps omitted',
        agents=[dict(id=a['id'],end_seconds=a['end_seconds'],allocated_seconds=sum(e['end']-e['start'] for e in a['events']if e['include_in_aggregate']),
                events=len(a['events']),**{k:a[k]for k in ['segments','receipt_audit','wall_end_seconds','omitted_idle_seconds','clock_intervals']if k in a})for a in agents])
    (REPORT/(PREFIX+'-audit.json')).write_text(json.dumps(audit,indent=2)+'\n')
    # Navigation only: original content and embedded two-run dataset remain unchanged.
    for p in [REPORT/'sol-kimi-research-behavior-2026-11-10.html',ROOT/'tools/templates/pokemon-research-behavior.html']:
        text=p.read_text();nav=navigation()
        text=re.sub(r'<nav class="trajectory-nav".*?</nav>',nav,text,count=1)
        if 'class="trajectory-nav"' not in text:text=text.replace('<main>','<main>'+nav,1)
        if 'nav.trajectory-nav' not in text:text=text.replace('</style>', 'nav.trajectory-nav{display:flex;flex-wrap:wrap;gap:10px 20px;padding:0 0 22px;font-size:12px}nav.trajectory-nav a{color:#4b7c6b}</style>',1)
        p.write_text(text)
    print(json.dumps(audit,indent=2)[:1300])

if __name__=='__main__':main()
