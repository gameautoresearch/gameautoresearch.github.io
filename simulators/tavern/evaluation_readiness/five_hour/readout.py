"""Live new-campaign readout; never exposes private final seed values."""
import datetime,json,hashlib
from pathlib import Path
from .budget import ROOT,CAMPAIGN,Budget
from simulators.tavern.autoresearch.evaluation_v2.release import write
from .compare import compare

def generate():
    root=CAMPAIGN/'tavern/evaluator';stages=[];public={};final={}
    for p in sorted(root.glob('*/provider.json')):
        launch=json.loads(p.read_text());path=p.parent;s=json.loads((path/'last-status.json').read_text()) if (path/'last-status.json').exists() else {};allocation=json.loads((path/'allocation.json').read_text()) if (path/'allocation.json').exists() else {}
        row=dict(attempt=path.name,kind=launch.get('kind','screen'),candidate=launch.get('candidate'),app_id=launch['app_id'],sandbox_id=launch['sandbox_id'],cap_seconds=launch['cap'],allocated_seconds=allocation.get('allocated_seconds'),lifecycle=s.get('lifecycle.json',{}).get('status','unknown'))
        result=s.get('screen/result.json',{})
        if result.get('candidate_id'):
            clean=dict(result);clean['pending']=[{k:v for k,v in cell.items() if k!='seed'} for cell in result.get('pending',[])]
            row.update(observed_cases=result['observed_cases'],expected_cases=result['expected_cases'],complete_valid_schedule=result['complete_valid_schedule'],raw_score=result['raw_score'],mean_placement=result['mean_placement'],failures=result['failures'],panel_sha256=launch['panel_sha256'])
            (final if launch.get('purpose')=='final' or path.name.startswith('final-') or '-grade-' in path.name and '-qualification-grade-' not in path.name else public)[path.name]=clean
        stages.append(row)
    build_only=[]
    for p in sorted(root.glob('*/image-build.json')):
        if not (p.parent/'provider.json').exists():build_only.append(dict(attempt=p.parent.name,image_build=json.loads(p.read_text()),worker_receipt_recorded=False,no_allocation_proof=json.loads((p.parent/'no-allocation-proof.json').read_text()) if (p.parent/'no-allocation-proof.json').exists() else None))
    def comparison_group(name):
        path=root/name;launch=json.loads((path/'provider.json').read_text());reg=json.loads((path/'assets/registry.json').read_text());items={r['policy_id']:r for r in reg['opponents']};refs={}
        for identity in {x for v in reg['profiles'].values() for x in v}:
            item=items[identity];refs[identity]=hashlib.sha256((path/'assets'/item['path']/'identity.json').read_bytes()).hexdigest() if item['kind']=='learned' else {'style':item['style']}
        value=dict(profiles=reg['profiles'],reference_identities=refs,binding=reg['binding'],panel_sha256=launch['panel_sha256'])
        return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()
    baselines={comparison_group(name):name for name,result in public.items() if result['candidate_id']=='deepseek-r3' and (root/name/'summary/screen/episodes.json').exists()}
    comparisons={}
    for name,result in public.items():
        if not str(result['candidate_id']).startswith('new-'):continue
        base=baselines.get(comparison_group(name))
        if base is None:continue
        paths=[root/n/'summary/screen/episodes.json' for n in (base,name)]
        if all(p.is_file() for p in paths):comparisons[name]=dict(baseline_attempt=base,**compare(*paths,expected_groups=result['expected_cases']//24))
    final_baselines={comparison_group(name):name for name,result in final.items() if result['candidate_id']=='deepseek-r3' and (root/name/'summary/screen/episodes.json').exists()}
    final_comparisons={}
    for name,result in final.items():
        base=final_baselines.get(comparison_group(name))
        if base is None or name==base:continue
        paths=[root/n/'summary/screen/episodes.json' for n in (base,name)]
        if all(p.is_file() for p in paths):final_comparisons[name]=dict(baseline_attempt=base,**compare(*paths,expected_groups=12))
    terminal_audits={}
    for p in sorted(root.glob('*/summary/qualification-terminal-replay.json')):
        record=json.loads(p.read_text())
        terminal_audits[p.parent.parent.name]={k:record.get(k) for k in ('passed','lobbies','all_lobby_actual_terminals_retained','recorded_lobbies')}
    final_precommits={str(p.parent.relative_to(root)):json.loads(p.read_text()) for p in root.glob('*/precommit.json') if json.loads(p.read_text()).get('required_lobbies_per_policy')==288}
    admissions={p.name:{k:r.get(k) for k in ('admission_qualified','export_identity_sha256','weights_sha256','checkpoint_sha256','update','samples','total_known_lineage_samples','parameters')} for p in root.glob('admission-*.json') if (r:=json.loads(p.read_text())).get('admission_qualified')}
    usage=Budget().usage();operations=usage.pop('operations');usage.update(unknown_resource_holds_usd=sum(r['usd_hold'] for r in operations if r['bucket']=='resource' and r['status']!='settled'),settled_estimated_resource_usd=sum(r['usd_charged'] or 0 for r in operations if r['bucket']=='resource' and r['status']=='settled'),actual_invoice_verified=False)
    workflow=root/'workflow-qualification-v1';state=json.loads((workflow/'state.json').read_text()) if (workflow/'state.json').exists() else {};grades=list(workflow.glob('qualification-grade-*/result.json'));workflow_qualified=bool(state.get('finalized') and grades and all(json.loads(p.read_text())['complete_valid_schedule'] for p in grades))
    cleanup=json.loads((root/'cleanup-final.json').read_text()) if (root/'cleanup-final.json').exists() else dict(complete=False)
    data=dict(status='complete' if cleanup.get('complete') else 'in_progress',updated_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),campaign=str(CAMPAIGN.relative_to(ROOT)),deadlines={k:Budget().auth[k] for k in ('selection_close_utc','worker_stop_utc','deadline_utc')},readiness=dict(native_harness_qualified=True,mixed_v2_workflow_qualified=workflow_qualified,strong_panel_qualified=False,superhuman_evidence=False),native_fingerprint='217577b83fc4525e',artifact_task_binding='tavern-eight-implemented-v1',active_evaluation_release='tavern-eight-mixed-v2',budget=usage,stages=stages,public_development=public,private_final=final,workflow=dict(finalized=bool(state.get('finalized')),qualified=workflow_qualified,operator_orchestrated=True,external_model_calls=0,qualification_uses_PUBLIC_cases=True,private_final_still_required=True),cleanup=cleanup,normalization_qualified=False,reference_normalized_score=None,limitations=['Public two-group screens are provisional; extended public and fresh private results remain separate.','Original strong incumbent is retained unless a new policy beats it in independent matched cross-play.','Warm start is explicit original r3 policy/optimizer/RNG import, not a relabeled original-source resume.','No human/expert or commercial-ladder calibration; superhuman remains a target, not an achieved certificate.'])
    selected=json.loads((ROOT/'simulators/tavern/autoresearch/evaluation_v2/selected_registry.json').read_text());catalog=json.loads((ROOT/'simulators/catalog.json').read_text())['games']['tavern']
    current_workflow={};current_pointer=root/'workflow-selected-current.json'
    if current_pointer.exists():
        current_workflow=json.loads(current_pointer.read_text());check=json.loads(Path(current_workflow['qualification_check']).read_text());current_workflow.update(public_workflow_qualified=bool(check.get('workflow_qualified') and current_workflow['registry_manifest_sha256']==selected['manifest_sha256']),training_reuse_is_not_independent_replicate=True,private_full_entrypoint_qualified=False)
        current_session=Path(current_workflow['session']);current_state=json.loads((current_session/'state.json').read_text());grade=current_session/('grade-'+current_state['final_submission'])/'result.json'
        if grade.exists():
            result=json.loads(grade.read_text());current_workflow['private_full_entrypoint_qualified']=bool(result.get('complete_valid_schedule') and result.get('observed_cases')==288 and result.get('qualification_public_cases') is False and result.get('runtime_protocol')=='private-paths-terminal-v2-all-lobbies' and result.get('actual_terminal_game_lobbies')==288 and result.get('terminal_replay_qualified') and result.get('candidate_privacy_qualified'))
    data.update(selected_reference_pool=selected,shared_catalog_contract=catalog['benchmark_contract'],shared_catalog_selects_mixed_v2='evaluation_v2/' in catalog['benchmark_contract'],build_only_attempts=build_only,paired_public_comparisons=comparisons,local_integration_tests=dict(passed=85,log=str((root/'local-integration-v7.log').relative_to(ROOT)),additional_current_release_suite_passed=16),post_step_state_boundary='BatchEnv automatically resets after all8 terminals. Recorded final_state_sha256 is the exact post-step next-episode state; terminal stop codes/placements define natural scores. Viewing exporter independently shadows pinned admission steps to retain and replay the actual terminal Game state.')
    data.update(paired_private_comparisons=final_comparisons,private_final_precommits=final_precommits,terminal_state_audits=terminal_audits,source_admitted_checkpoints=admissions)
    closed_selection=root/'public-selection-closed-v1.json'
    if closed_selection.exists():data['frozen_public_selection']=json.loads(closed_selection.read_text())
    data['current_selected_pool_workflow']=current_workflow;data['readiness']['current_selected_pool_public_workflow_qualified']=current_workflow.get('public_workflow_qualified',False);data['readiness']['current_selected_pool_private_entrypoint_qualified']=current_workflow.get('private_full_entrypoint_qualified',False)
    stop_path=root/'stop-record.json';stop=json.loads(stop_path.read_text()) if stop_path.exists() else None
    if stop:
        data.update(status='stopped_by_user' if cleanup.get('complete') else 'stopping_by_user',user_stop=stop)
        data['workflow'].update(private_final_still_required=False,private_final_cancelled_by_user=True,scientific_private_qualification_unfulfilled=True)
        for row in stages:
            if row['attempt'] in stop.get('interrupted_attempts',[]):row.update(lifecycle='stopped_by_user',administrative_user_stop=True,primary_score=None)
    out=ROOT/'reports/tavern/strong-bots-five-hour-2026-11-09-evaluator.json';write(out,data)
    rows=[]
    for name,r in public.items():
        pool=json.loads((root/name/'provider.json').read_text())['registry_manifest_sha256'][:8]
        rows.append(f"| {name} | {r['candidate_id']} | {pool} | {r['observed_cases']}/{r['expected_cases']} | {r['raw_score']:.2f} | {r['mean_placement']:.3f} |" if r['raw_score'] is not None else f"| {name} | {r['candidate_id']} | {pool} | {r['observed_cases']}/{r['expected_cases']} | unresolved | unresolved |")
    paired_rows=[]
    for name,r in comparisons.items():
        ci=r.get('raw_score_gain_95');gain=r.get('raw_score_gain');paired_rows.append(f"| {name} | {r['baseline_attempt']} | {r.get('seed_groups','?')} | {gain:+.2f} | [{ci[0]:+.2f}, {ci[1]:+.2f}] |" if gain is not None and ci else f"| {name} | {r['baseline_attempt']} | unresolved | unresolved | unresolved |")
    final_rows=[]
    for name,r in final.items():
        paired=final_comparisons.get(name,{});gain=paired.get('raw_score_gain');ci=paired.get('raw_score_gain_95');score=r.get('raw_score')
        final_rows.append(f"| {name} | {r['candidate_id']} | {r['observed_cases']}/{r['expected_cases']} | "+(f"{score:.2f}" if score is not None else 'unresolved')+' | '+(f"{gain:+.2f} [{ci[0]:+.2f}, {ci[1]:+.2f}]" if gain is not None and ci else '—')+' |')
    text=f'''# Tavern strong-bots five-hour campaign — evaluator

Status: **{data['status']}**, {data['updated_at_utc']}. Native harness qualified; mixed-v2 workflow qualified: **{str(workflow_qualified).lower()}**. Strong-panel/superhuman evidence remain **false**.

This new campaign uses `tools.open_credit_campaign.OpenCreditBudget` and the explicit ledger `{CAMPAIGN.relative_to(ROOT)}/campaign.sqlite3`. Modal dollars/aggregate GPU seconds are uncapped; each worker, concurrency and absolute deadlines remain bounded. External paid model API allowance is zero. Historical ledgers, exports, banks, failures and cancelled/paused work stay unchanged.

## Incumbent, interface and new public development

The old DeepSeek-r3 incumbent is update257 /11.30M decisions, with exact original policy/optimizer/RNG/native-state checkpoint. Its exported NPZ exactly matches that full checkpoint. Its two-Tanh architecture has no LayerNorm; the old canonical reference class is not a compatible direct load. An explicit new warm-start class independently matches old logits, values, masked distributions and pre-encoded numerics on1,796 NEW PUBLIC rows. Original identity/checkpoint remain unchanged; new campaign lineage declares the import and reset domain. Strict new exports must match their exact archived policy/learner/trainer source and full checkpoint arrays.

Native `217577b83fc4525e` /admission-v2 and the 3,227/124/245 visible ABI remain fixed. Reference inference is neural-only or reviewed visible scripts; no hidden state, test-time search or LLM rescue. Full8-seat natural endings and placement permutation1–8 are required even after candidate death. Illegal moves, visible rejections, simulator faults, inference/process deadlines, infrastructure failures and administrative caps stay distinct; any required unresolved cell makes the primary aggregate null.

New public source/weights/API release: `{(root/'public-development-v1').relative_to(ROOT)}`. Public seeds50101/50102, extended50111–50118, all8candidate seats, three neutral7-opponent compositions: scripts, incumbent-mixed, diverse-history. These are development schedules, not private final evidence. Difficulty ordering and expert strength are not assumed.

| Stage | Candidate | Pool manifest | Lobbies | Raw /100 | Mean placement |
|---|---|---|---:|---:|---:|
{chr(10).join(rows)}

Per-profile, physical-seat, individual opponent placement-order distributions, full-lobby faults, action/pending counts and observed inference latency are in the companion JSON. Pairwise placement order is not a duel win rate. Compare only matched panel/source compositions; do not mix the historical or RL-owner assessment with this panel. Two public groups do not establish population-level strength.

| Challenger | Matched incumbent stage | Seed groups | Gain /100 | Paired95%interval |
|---|---|---:|---:|---|
{chr(10).join(paired_rows)}

Each comparison requires identical seed/profile/seat cells and opponent identities. An incomplete or faulted required cell makes the difference null. Intervals resample complete environment seed groups, not independent training seeds. Several checkpoints were screened on the same public schedule; the selected public maximum can be optimistic. Fresh private assessment stays separate.

## New benchmark release integration

`autoresearch/benchmark-current.json` selects `evaluation_v2/benchmark.json`. Default `manage.py prepare` now selects mixed-v2; explicit/registered legacy-v1 releases and sessions remain available. The new public instruction/brief, source-bound broker, random entrant starter, persistent L4 run, public practice, immutable submit/finalize and isolated mixed-profile grading share the qualified evaluator. The host never imports entrant source/pickle for grading. Selected trusted policy source is preserved, not substituted into the starter class.

The current default reference receipt selects `{selected['path']}`, manifest `{selected['manifest_sha256']}`. New learned references participate in the actual7-opponent compositions, with the original incumbent retained. Pool changes require a new immutable version and public qualification; old releases stay intact. Shared catalog currently selects `{catalog['benchmark_contract']}`; parent-owned catalog migration is tracked separately from the working game entrypoint.

Current selected-pool public workflow qualified: **{str(current_workflow.get('public_workflow_qualified',False)).lower()}**; fresh full-private entrypoint qualified: **{str(current_workflow.get('private_full_entrypoint_qualified',False)).lower()}**. Opposition-only qualification carries the original42.057allocated GPU seconds and preserves its closed ledger/source/identity; it creates no new GPU allocation and is not an independently trained replicate. Its new public practice and isolated PUBLIC qualification each contain48natural lobbies.

The entrant contract stays7200 full allocated L4 seconds (smoke600),4physical CPUs/16GiB,<=2Mparameters/32MBNPZ and15-second full input/response deadline. References are independently declared. Training containers contain only public source and the entrant's own inputs/checkpoints; private registry/panel data live in separate root-protected grading containers and Volumes. Private final banks require both global and entrant selection closure. Qualification uses explicitly PUBLIC cases through the same isolation, not a leaked scientific final bank.

Current workflow qualification: `{json.dumps(data['workflow'],sort_keys=True)}`. The weak smoke policy proves workflow behavior, not strength. The updated local integration suite passes85checks; earlier plan-field failure remains recorded. Sixteen release/admission/selection tests cover routing, pool mutation, no host import, immutable submission, selection closure, historical/fork stream exclusion, rejected ancestry/history claims, required-cell pairing and disjoint public/private grade caches. The retained54Linux admission/core checks and old-source rejection passed in the new campaign runtime. Practice and isolated public qualification produced identical all8terminals, action counts and native post-step hashes in48lobbies.

`BatchEnv` automatically resets after all8terminals: its recorded post-step hash identifies the next episode state, while explicit natural stop codes and placements define the score. The qualified v2 wrapper retains the actual terminal `Game` state for every naturally complete lobby, and independently replays three predeclared cases per policy. The viewing exporter also verifies projected combat frames. A locked12-case local comparison preserved actions/terminals/scores/post-reset state with about3.3% measured local overhead; this is not a cloud/GPU throughput claim. The first batch-only film failed viewer validation and is retained as rejected; its corrected v2 film passed the viewer codec and all8seat timelines.

## Independent private assessment

{('The user cancelled private final grading before any private bank was drawn.' if stop else 'No private results yet; training selection remains independent of this final bank.' if not final else 'These results use frozen identities/opposition and fresh private environment groups drawn after selection closure. They do not change the preselected checkpoint or opponent pool.')}

| Stage | Candidate | Lobbies | Raw /100 | Gain vs matched old incumbent [paired95%] |
|---|---|---:|---:|---|
{chr(10).join(final_rows)}

Private intervals resample12 complete environment groups. They measure this fixed policy and opposition panel, not variation across independent training seeds. Every required lobby must end naturally; incomplete/faulted schedules remain null. The closed smoke artifact grades the benchmark entrypoint and is not a separate trained replicate. Exact source/weights/checkpoint hashes, public selection records, terminal-state coverage and frozen-bank hash receipts are retained in the companion JSON; private seed values are omitted.

## Finalization and accounting

{('The user explicitly stopped the campaign at09:04UTC. All evaluator-owned workers/apps were terminated and provider-verified inactive. Planned final/private grading and further snapshots are cancelled; no private final bank was drawn. Completed public bests, admitted latest checkpoints, original sources, logs and partial episodes are retained. Interrupted required schedules have no primary aggregate score.' if stop else 'Training/selection closes09:36:36UTC; workers stop10:11:36; report/cleanup deadline10:16:36. Final identities/roster/source/pool will close before any fresh private bank is drawn. Independent evaluator grading will include oldbest, starter/scripts and all precommitted trained controls/finals. No private final feedback returns to training or selection. A few viewing replays use PUBLIC seeds only.')}

Evaluator GPU allocation charged+held: **{usage['gpu_seconds']:.3f}s**; resource estimate+holds: **${usage['resource_usd']:.4f}**; unknown resource holds: **${usage['unknown_resource_holds_usd']:.4f}**; paid model calls/liability:0/${usage['model_usd']:.2f}. Full authoritative provider intervals include setup, idle, failures, simulation, transfers/training and teardown. Unknown builds/observers/storage remain held; this is not a final invoice.

Cleanup: `{json.dumps(cleanup,sort_keys=True)}`. No commit, push, public publication or commercial ladder operation is part of this work. The supported native Rust CPU simulator and L4 learner are retained.
'''
    out.with_suffix('.md').write_text(text);return dict(status=data['status'],readiness=data['readiness'],public_stages=len(public),final_stages=len(final),budget=usage)

if __name__=='__main__':print(json.dumps(generate(),indent=2))
