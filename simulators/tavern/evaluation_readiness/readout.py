"""Rebuild the evaluator readout from retained results without exposing private seeds."""
import json,datetime,argparse,sys
from pathlib import Path
import numpy as np
from .budget import CAMPAIGN,ROOT,EvaluatorBudget
sys.path.insert(0,str(ROOT/'simulators/tavern/autoresearch'))
from contract import sha,write

def paired(initial,final):
    key=lambda r:(r['profile'],r['seed'],r['seat'])
    a={key(r):r for r in initial};b={key(r):r for r in final}
    if set(a)!=set(b) or not all(r['complete_valid_lobby'] for r in initial+final):return dict(qualified=False,reason='Missing/invalid paired cells')
    groups=sorted({k[1] for k in a});d=np.array([np.mean([a[k]['candidate_placement']-b[k]['candidate_placement'] for k in a if k[1]==g]) for g in groups]);rng=np.random.default_rng(92317)
    interval=np.quantile(rng.choice(d,(10000,len(groups))).mean(1),[.025,.975]).tolist()
    return dict(qualified=True,cases=len(a),lobby_seed_groups=len(groups),initial_minus_final_mean_placement=float(d.mean()),positive_means_improvement=True,paired_placement_gain_95=interval,raw_score_gain=float(d.mean()*100/7),raw_score_gain_95=[x*100/7 for x in interval],method='Paired lobby-seed-group bootstrap preserving all seats and profiles',limits=f'One RL training seed; {len(groups)} environment groups; conditional on these frozen lineups, not population or human-level strength')

def generate():
    root=CAMPAIGN/'tavern/evaluator';attempts=[];public={};fresh={};details={};episodes={}
    for path in sorted(root.glob('cpu-*')):
        if not path.is_dir():continue
        launch=path/'provider.json';status=path/'last-status.json';allocation=path/'allocation.json'
        if not launch.exists():continue
        l=json.loads(launch.read_text());s=json.loads(status.read_text()) if status.exists() else {};r=s.get('screen/result.json',{})
        item=dict(attempt=path.name,candidate=l['candidate'],cap_seconds=l['cap'],CPU=4,GiB=16,GPU=False,source_manifest_sha256=sha(path/'source-files.json'),provider=l['sandbox_id'],status=s.get('lifecycle.json',{}).get('status','unresolved'),allocated_seconds=json.loads(allocation.read_text())['allocated_seconds'] if allocation.exists() else None,provider_app=l['app_id'])
        if r.get('candidate_id'):
            # Results carry aggregates only. Raw native states/seeds stay ignored/private.
            safe=dict(r)
            if path.name.startswith('cpu-final-'):
                safe['pending']=[{k:v for k,v in cell.items() if k!='seed'} for cell in r.get('pending',[])]
            details[path.name]=safe;item.update(cases=r['observed_cases'],expected_cases=r['expected_cases'],complete=r['complete_valid_schedule'],raw_score=r['raw_score'])
            dest=fresh if path.name.startswith('cpu-final-') else public
            dest[r['candidate_id']]={k:r[k] for k in ('raw_score','mean_placement','top4_rate','complete_valid_schedule','observed_cases','confidence_interval_95')}
            for candidate in (path/'summary/screen/episodes.json',path/'collected/screen/episodes.json'):
                if candidate.exists():episodes[path.name]=json.loads(candidate.read_text());break
        attempts.append(item)
    pairs={}
    for label,prefix in [('public','cpu-screen-'),('fresh','cpu-final-')]:
        initial=[v for k,v in episodes.items() if k.startswith(prefix+'initial-')];final=[v for k,v in episodes.items() if k.startswith(prefix+'selfplay-final-') or k.startswith(prefix+'final-')]
        if initial and final:pairs[label]=paired(initial[-1],final[-1])
    if 'cpu-screen-initial-v2' in episodes and 'cpu-screen-pilot43-v1' in episodes:pairs['public_pilot43']=paired(episodes['cpu-screen-initial-v2'],episodes['cpu-screen-pilot43-v1'])
    usage=EvaluatorBudget().usage();ops=usage.pop('operations');held=[r for r in ops if r['status']!='settled'];usage.update(CPU_allocated_seconds=sum(r['allocated_seconds'] or 0 for r in attempts),unpriced_resource_holds_usd=sum(r['usd_hold'] for r in held if r['bucket']=='resource'),settled_resource_estimate_usd=sum(r['usd_charged'] or 0 for r in ops if r['status']=='settled' and r['bucket']=='resource'),actual_invoice_verified=False,all_failed_attempts_retained=True)
    for attempt in attempts:attempt['booked_resource_usd']=sum((o['usd_charged'] if o['status']=='settled' else o['usd_hold']) or 0 for o in ops if o['bucket']=='resource' and o['kind'].endswith('-'+attempt['attempt']))
    cleanup_path=root/'cleanup-audit.json';cleanup=json.loads(cleanup_path.read_text()) if cleanup_path.exists() else dict(complete=False,reason='Final stages still open')
    proof_counts=[]
    for path in root.glob('cpu-*/summary/qualification.xml'):
        import xml.etree.ElementTree as ET
        tree=ET.parse(path);suites=tree.getroot().iter('testsuite')
        proof_counts.append(dict(attempt=path.parents[1].name,**{k:sum(int(s.get(k,'0')) for s in tree.getroot().iter('testsuite')) for k in ('tests','errors','failures','skipped')}))
    final_complete=len(fresh)==7 and all(r['complete_valid_schedule'] for r in fresh.values())
    audit_path=root/'admission-selfplay-final.json';audit=json.loads(audit_path.read_text()) if audit_path.exists() else None
    readiness=dict(harness_qualified=True,learned_reference_qualified=bool(final_complete and audit and audit['admission_qualified'] and audit['changed_weights'] and audit['finite_learning_metrics'] and audit['update']>0),strong_panel_qualified=False,superhuman_evidence=False)
    status=('complete' if final_complete else 'complete_with_unresolved_cells') if len(fresh)==7 and cleanup.get('complete') else 'in_progress'
    data=dict(status=status,updated_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),campaign=str(CAMPAIGN.relative_to(ROOT)),native_fingerprint='217577b83fc4525e',binding_sha256='4524a967fc0bbff73ba3ab49b3de347344a89c68647825245527ef87073e0598',readiness=readiness,readiness_scope='Harness correctness; learned-reference flag means exact trained export plus complete fixed assessment, not strong play.',Linux_qualification=proof_counts,public_screen=public,fresh_final=fresh,paired_learning=pairs,budget=usage,attempts=attempts,result_details=details,cleanup=cleanup,normalization_qualified=False,normalized_reference_score=None,limitations=['Difficulty profile names are hypotheses; no universal ordering across policies.','Two public seed groups and six final groups provide limited environmental uncertainty; only one new RL training seed.','Exact candidate encoder omits option-specific Choose-One descriptions, some target locations and action-use counters. References receive the same arrays.','Synthetic legal-mask coverage is separate from natural action/choice coverage.','Pinned AutoTavern emulator tests do not establish Blizzard-game parity.'],no_paid_model_calls=True,no_evaluator_GPU=True)
    bank=root/'private-final-v1/precommit.json'
    if bank.exists():
        precommit=json.loads(bank.read_text())
        data['final_commitment']={k:precommit[k] for k in ('selection_closed','selection_sha256','bank_generator_sha256','registry_manifest_sha256','panel_sha256','roster','identities','groups','seats','profiles','cases_per_policy','profiles_unchanged','excluded_historical_schedule_keys','private_feedback_after_selection_only')}
        data['final_commitment']['precommit_sha256']=sha(bank)
        data['final_commitment']['excluded_training_configurations']=len(precommit['training_cohorts_excluded'])
    replay=root/'starter-public-native-replay.json'
    if replay.exists():data['native_recorded_replay']={k:v for k,v in json.loads(replay.read_text()).items() if k!='records'}
    data['fresh_recorded_replay']={p.stem:{k:v for k,v in json.loads(p.read_text()).items() if k!='records'} for p in (root/'native-final-replay').glob('*-proof.json')}
    closure=root/'final-closure-audit.json'
    if closure.exists():data['closure_audit']={k:v for k,v in json.loads(closure.read_text()).items() if k!='records'}
    write(ROOT/'reports/tavern/evaluation-readiness-2026-11-09.json',data)
    rows=lambda results:'\n'.join(f"| {k} | {v['observed_cases']} | {v['raw_score']:.2f} | {v['mean_placement']:.3f} | {100*v['top4_rate']:.1f}% |" if v['raw_score'] is not None else f"| {k} | {v['observed_cases']} | unresolved | unresolved | unresolved |" for k,v in results.items())
    profile_rows='\n'.join('| '+d['candidate_id']+' | '+' | '.join(f"{d['per_profile'][p]['mean_placement']:.3f}" if d['per_profile'][p]['mean_placement'] is not None else 'unresolved' for p in ('easy','competent-mixed','harder-mixed'))+' |' for name,d in details.items() if name.startswith('cpu-screen-'))
    cost_rows='\n'.join(f"| {a['attempt']} | {a['status']} | {a['allocated_seconds']:.3f} | ${a['booked_resource_usd']:.4f} |" if a['allocated_seconds'] is not None else f"| {a['attempt']} | {a['status']} | unresolved | ${a['booked_resource_usd']:.4f} |" for a in attempts)
    text=f'''# Battlegrounds evaluation readiness — November 9

Status: **{data['status']}**, updated {data['updated_at_utc']}. Harness qualified: **true**. Learned reference technically qualified: **{str(readiness['learned_reference_qualified']).lower()}**. Strong panel qualified: **false**. Superhuman evidence: **false**.

The new evaluator controls all eight seats until the entire lobby ends naturally. A candidate dying early cannot salvage a later fault. Natural placements must form 1–8; an incomplete/faulted required cell makes the primary aggregate null. Illegal actions, inference deadlines, visible engine rejection, simulator faults, infrastructure failures and administrative caps are reported separately, with no fallback action or invented placement.

## Frozen interface and opposition

Native AutoTavern 37.0.2 fingerprint `217577b83fc4525e`, admission-v2, 131 bound files and binding SHA `{data['binding_sha256']}` are unchanged. The task remains eight players, five races, two hero choices, seven board slots, ten hand slots, 30 recruit actions and the 100-turn fault guard. Native Rust CPU workers and the L4 learner remain supported; the GPU simulator rewrite stays paused.

Three independently configured public scripts cover tempo, economy and synergy using the current static card catalog plus the exact 3,227 numeric features, 124 card indices and 245-action mask. Buy, sell, play, refresh, freeze, level, beneficial positioning, powers, activations, targets, dark gifts and generic pending choices are handled. No hidden game, future RNG, search or model service is available to these policies. The immutable script source SHA is `e8d987f90678fa605bc1e76cc9b984dda406fb13984c0c8575a43165106cd62e`; the full-lobby driver SHA is `7488060b651dc0606c2789615c043f129be1048175dd6de61b6265ef5939e9c7`.

The registry includes six exact historical inference exports and the new corrected-lineage initial/pilot/final exports when admitted. Every neural artifact retains its own policy source, strict source/rules binding, finite NPZ state and digest; the new RL exports additionally match their full archived training source and exact checkpoint arrays. No old weights are rebound to new source. Seven opponents are assigned clockwise around every candidate seat. Each of the three profiles mixes all three scripts and several retained learned identities; at most two opponent copies of an identity appear (a candidate may share that identity). Profiles remain identical across all candidate assessments.

The candidate runs in a separate unprivileged Linux process with an empty credential environment, only its own readable bundle, visible-array pipe, 2M-parameter/32MB limit and 15-second forward deadline covering partial input/output. Private registry, schedules and native traces are root-private. Host-owned reference inference is reviewed finite feedforward CPU code using the same visible tensors and a single Torch thread. This is a declared benchmark isolation boundary, not a general hostile-code sandbox claim.

## Qualification evidence

The retained Linux qualification passed 48 tests plus explicit old-source rejection. The expanded local evaluator suite passes 16 tests, including blocked/partial policy replies, native training-stream exclusion and early death followed by a later fault; later Linux proof counts are recorded here: `{json.dumps(proof_counts,sort_keys=True)}`. Earlier-seat death, stale Move, replacement pending menus sharing a Choose index, illegal preframe actions, deferred training credit, exact save/resume and source rejection remain covered. Eight native Rust batch tests pass. The synthetic no-damage continuation reaches the unchanged turn-limit fault without a placement or reward. Synthetic mask tests exercise all 245 response IDs and eight pending kinds; natural runs exercise pending kinds 1–7, with Timewarp disabled by the frozen rules.

Complete native initial/final states, action arrays, all-seat terminals, action/pending counts and per-call latency persist privately for every assessed lobby. Result JSON includes per-seat, per-profile and per-opponent diagnostics; pairwise placement order within a lobby is explicitly not a duel win rate. Seed-group bootstrap intervals preserve all seats and profiles.

## Public development screen

Two public lobby seed groups × eight seats × three profiles = 48 required lobbies per policy. These are development outcomes; difficulty names are provisional and are not consistently ordered across policies.

| Policy | Lobbies | Raw /100 | Mean placement | Top four |
|---|---:|---:|---:|---:|
{rows(public)}

Public mean placement by profile (lower is better):

| Policy | Easy hypothesis | Competent mixed | Harder mixed |
|---|---:|---:|---:|
{profile_rows}

The JSON retains all eight physical-seat breakdowns, opponent-specific placement-order fractions, action/pending counts and CPU inference latency. Earlier public runs combine latency when candidate and reference share a policy identity; later runs label candidate latency separately. None of these measurements is a GPU-throughput comparison.

## Fresh final assessment

The predeclared roster is the retained starter, DeepSeek r3, all three scripts, matched corrected-lineage random initialization and the last complete update of the single corrected RL continuation. The pilot remains a separate development diagnostic. Only after exact selection closes may six fresh private lobby groups be generated, with all eight seats and unchanged three profiles: 144 lobbies per policy. The commitment pins policy/source/weight identities before outcomes. Seed exclusion checks native stream identity against historical schedules and the entire declared training-update ceiling. Private seeds and raw traces stay outside public source/report payloads.

Fresh bank panel SHA: `{data.get('final_commitment',{}).get('panel_sha256','not yet created')}`. Precommit SHA: `{data.get('final_commitment',{}).get('precommit_sha256','not yet created')}`. The private audit excludes {data.get('final_commitment',{}).get('excluded_historical_schedule_keys',0)} historical/fixture native stream keys and {data.get('final_commitment',{}).get('excluded_training_configurations',0)} complete declared training configurations. Recorded native replay: `{json.dumps(data.get('native_recorded_replay',{}),sort_keys=True)}`.

| Policy | Lobbies | Raw /100 | Mean placement | Top four |
|---|---:|---:|---:|---:|
{rows(fresh) if fresh else '| Pending selection closure | — | — | — | — |'}

Matched initial/final results: `{json.dumps(pairs,sort_keys=True)}`. These comparisons use one RL training seed and a small number of fresh environment groups. Normalized reference scores remain null. No human, expert or commercial-ladder strength is inferred.

## Cost, failures and cleanup

Evaluator allocation: **{usage['CPU_allocated_seconds']:.3f} CPU-worker seconds**, **{usage['gpu_seconds']:.0f} GPU seconds**, **${usage['model_usd']:.2f} model liability**. Conservative settled resource estimate plus retained holds: **${usage['resource_usd']:.4f} / $8**; unpriced build/observer/storage holds are **${usage['unpriced_resource_holds_usd']:.4f}**. Provider start/end intervals cover complete allocated lifetimes. This is not a final invoice.

| Attempt | Status | Allocated CPU-worker seconds | Resource estimate + holds |
|---|---|---:|---:|
{cost_rows}

Every worker has four fixed CPU cores and 16 GiB RAM. The complete stage cost includes all seven opponent policies; individual inference wall time is reported without inventing separate billing attribution. Fresh first-group/seat-zero native replay summaries: `{json.dumps(data['fresh_recorded_replay'],sort_keys=True)}`.

Earlier CPU stages have a 600-second provider cap. The three final script stages use a 900-second aggregate buffer after slower fresh-lobby progress was observed; the 120-second per-lobby and 15-second per-forward limits, all cases, weights, rules and lineups remain fixed. These are CPU allocations, with no evaluator GPU use. The final stage runs sequentially when needed to keep charged-plus-held liability below $8.

Retained failures: proof-v1 host interpreter preflight, zero allocations; proof-v2 missing tests, 2.121 CPU seconds; proof-v3 missing public fixture files, 43 tests passed and five packaging failures, 25.880 seconds; proof-v4 48 tests passed, then generated-cache artifact-copy failure, 30.488 seconds. A partial registry-v3 builder and initial-screen-v1 failed before paid reservation. Final-public-v1 passed 53/54 Linux tests but the seed helper imported a host-only budget dependency; it failed before policy scoring. Final-public-v2 isolates that path helper from the ledger dependency. The bank generator's earlier host relative-path failure occurred before seed generation; the bank/roster stayed fixed after creation. Each paid repair used a new output directory; none was an engine or learning failure, and all failed source/receipts remain preserved. Unknown charges remain held. No allowance was reset and no automatic retry controller ran.

Cleanup: `{json.dumps(cleanup,sort_keys=True)}`. Exact per-attempt allocation/cost and all result details are in the companion JSON. Canonical implementation is `simulators/tavern/evaluation_readiness/`; private evidence is `{root.relative_to(ROOT)}`. Historical campaigns and private schedules are unchanged. No commit, push, publication, external-model call or evaluator GPU allocation was performed.

Closure audit: `{json.dumps(data.get('closure_audit',{}),sort_keys=True)}`. The final private release manifest is `qualified-release-v1.json` under the owning evidence directory; it pins the verified source, pool, results and cleanup receipts for explicit selection in a new campaign.
'''
    (ROOT/'reports/tavern/evaluation-readiness-2026-11-09.md').write_text(text)
    return {k:data[k] for k in ('status','readiness','budget')}

if __name__=='__main__':print(json.dumps(generate(),indent=2))
