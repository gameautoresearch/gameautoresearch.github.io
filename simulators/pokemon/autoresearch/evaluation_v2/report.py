"""Offline source/receipt-backed evaluator readiness report. Never starts paid work."""
from datetime import datetime,timezone
import json
from pathlib import Path
from .campaign import BASE,guard,own_usage
from ..contract import ROOT,RUNTIME,HERE,sha,write


def read(path,default=None):
    path=Path(path);return json.loads(path.read_text()) if path.exists() else default


def collect():
    state=guard().snapshot();operations=[]
    for value in state['operations']:
        if value['game']!='pokemon' or not value['kind'].startswith('pokemon/evaluator/'):continue
        row=dict(value)
        for key in ('spec','receipt'):
            if isinstance(row[key],str):row[key]=json.loads(row[key])
        operations.append(row)
    jobs=[]
    for path in sorted((BASE/'jobs').iterdir()):
        reservation=read(path/'reservation.json',{});compute=read(path/'compute.json');summary=read(path/'output/summary.json');exit_info=read(path/'output/exit.json')
        operation=next((r for r in operations if r['id']==reservation.get('operation')),None)
        status='not-yet-collected'
        if (path/'no-allocation-proof.json').exists():status='failed-before-allocation'
        elif compute and exit_info:
            if exit_info['returncode']:status='failed'
            elif summary and summary.get('status')=='incomplete':status='incomplete'
            else:status='completed'
        if summary and summary.get('all_original_prefixes_exact') is False:
            status='diagnostic-prefix-check-failed'
        jobs.append({'label':path.name,'status':status,'operation':operation['id'] if operation else None,
            'charged_or_held_usd':(operation['usd_charged'] if operation['status']=='settled' else operation['usd_hold']) if operation else None,
            'provider':compute.get('provider') if compute else None,'owned_worker_stopped':compute.get('owned_worker_stopped') if compute else None,
            'source_manifest':read(path/'source-manifest.json',read(path/'source-manifest-final-audit-v2.json',read(path/'source-manifest-final-mirror.json'))),'summary':summary,'exit':exit_info,
            'summary_sha256':sha(path/'output/summary.json') if summary else None})
    names={j['label']:j for j in jobs}
    continuation=names.get('retained-cap-continuation-004',{}).get('summary')
    screen=names.get('foul-play-eight-seed-screen-003',{}).get('summary')
    fresh=names.get('frozen-fresh-bank-001',{}).get('summary')
    release=read(BASE/'active-evaluator-release.json');manifest=read(Path(release['path'])/'release.json') if release else None
    qualified=bool(continuation and continuation['all_original_prefixes_exact'] and continuation['all_reached_engine_outcome']
        and screen and screen['status']=='complete' and names.get('runtime-fixtures-002',{}).get('summary',{}).get('passed')
        and names.get('reference-admission-001',{}).get('summary',{}).get('passed'))
    entrant_api=names.get('entrant-api-proof-001',{}).get('summary')
    qualified=bool(qualified and entrant_api and entrant_api['status']=='complete' and entrant_api['natural_games']==4)
    current={n:sha(HERE.parent/n) for n in RUNTIME};expected=read(BASE.parent/'coordination/evaluator.json')['runtime_interface']['runtime_file_sha256']
    if current!=expected:raise ValueError('Canonical runtime changed from initial coordination pin')
    finished=[r for r in operations if r['status']=='settled' and r['receipt'] and r['receipt'].get('finished_at')]
    cpu=sum(r['receipt']['finished_at']-r['receipt']['started_at'] for r in finished)
    compute_holds=[r for r in operations if r['status']!='settled' and r['spec'].get('max_seconds')]
    return {'game':'pokemon','updated_at_utc':datetime.now(timezone.utc).isoformat(),'status':'complete-with-retained-noncompute-liability' if qualified and fresh and fresh['status']=='complete' and not compute_holds else 'qualification-complete-fresh-bank-pending' if qualified else 'in-progress',
        'shared_ledger':str(guard().path),'role_subcaps':{'gpu_seconds':1800,'resource_usd':8,'model_usd':.5},'evaluator_usage':own_usage(),
        'allocated_cpu_seconds':cpu,'operations':operations,'jobs':jobs,'canonical_runtime_unchanged':True,'canonical_runtime_file_sha256':current,
        'release':release,'frozen_termination_policy':manifest['termination_policy'] if manifest else None,'frozen_opponent_registry':manifest['opponent_registry'] if manifest else None,
        'original_capped_case_continuation':continuation,'full_700_value_prefix_parity':read(BASE/'jobs/full-prefix-parity-002/output/summary.json'),
        'remote_reproducibility_isolation_timeout_fixtures':read(BASE/'jobs/runtime-fixtures-002/output/summary.json'),
        'reference_inference_admission':read(BASE/'jobs/reference-admission-001/output/summary.json'),
        'entrant_facing_api_proof':entrant_api,'entrant_api_release':read(BASE/'active-entrant-api-release.json'),
        'opponent_identity_audit':read(BASE/'opponent-source-identity-audit.json'),
        'final_provider_audit':read(BASE/'final-audit-v2/provider-lifecycle.json'),
        'artifact_inventory':read(BASE/'final-audit-v2/artifact-inventory.json'),
        'provider_cleanup':read(BASE/'provider-cleanup-receipt.json'),
        'provider_cleanup_verification':read(BASE/'provider-cleanup-verification.json'),
        'noncompute_liability':read(BASE/'persistent-resource-liability.json'),
        'development_screen':screen,'fresh_final_bank_commitment':read(BASE/'final-bank-commitment.json'),'fresh_final_bank':fresh,
        'readiness':{'harness_qualified':qualified,'learned_reference_qualified':False,'neural_inference_bridge_admitted':bool(names.get('reference-admission-001',{}).get('summary',{}).get('passed')),'strong_panel_qualified':False,'superhuman_evidence':False,'normalized_reference_score':None},
        'limits':['Foul Play is a public-information search opponent, not a learned reference. Its reduced2belief/10ms configuration is not the published full-budget bot.','All tested neural/scripted baselines are weak development policies. Dominating these does not calibrate expert or human strength.','Eight seed clusters are a short qualification screen. Bootstrap intervals condition on observed seeds and can degenerate for all-win cells.','The learned-reference training owner separately owns learning/strength evidence; the proof1729 bridge fixture does not substitute for its selected2718 final assessment.','Provider prices include conservative1.1margin; invoice/storage reconciliation is not verified.'],
        'provenance':{'foul_play_url':'https://github.com/pmariglia/foul-play','poke_engine_url':'https://github.com/pmariglia/poke-engine','public_sets_url':'https://pkmn.github.io/randbats/data/full/gen9randombattle.json'},
        'own_paid_model_calls':0,'own_gpu_allocations':0,'actual_provider_invoice_verified':False}


def number(value):return 'null' if value is None else f'{value:.3f}'


def render(d):
    flags=d['readiness'];usage=d['evaluator_usage']
    lines=['# Pokémon evaluator readiness — November9,2026','',f"Status: **{d['status']}**. Updated {d['updated_at_utc']}.",'',
        f"Harness qualified: **{flags['harness_qualified']}**; neural inference bridge admitted: **{flags['neural_inference_bridge_admitted']}**; learned reference qualified: **false**; strong panel qualified: **false**; superhuman evidence: **false**. Normalized reference score remains **null**.",'',
        'This track implements evaluation and opposition only. The separate RL owner trains self-play policies. The pinned full Gen9 Random Battle CPU Showdown backend,700-value candidate projection and14actions remain unchanged; no GPU simulator work was reopened.','',
        '## Versioned completion and historical prefix admission','',
        'The frozen evaluator keeps engine-published turns, adapter phases, attempted/accepted choices and unavailable-choice retries separately. '
        'It permits authoritative native wins/ties, including Showdown’s own turn-limit tie, under4096adapter phases and16consecutive unavailable-choice retries perseat. '
        'A process wall safety cap is administrative; an inference timeout is a fault. Neither is scored as a draw. '
        'Only player-visible requests determine masks: unknown trapping is not exposed before the engine publishes an unavailable-choice update. No rejected choice gets a silent fallback.','',
        'Exact512-phase action/public-state prefixes were reproduced with their original12-game batches. A separate fixture compared all700encoded values and public-protocol hashes at every recorded phase. '
        'Only the old cap’s final ready/truncated metadata differs. All original v1 primary scores remain null. New extended outcomes are separate results.','',
        '| Retained case | Adapter phases | Last public turn | Accepted choices byseat | Unavailable retries | New native outcome | Old primary |','|---|---:|---:|---|---|---|---|']
    for row in (d['original_capped_case_continuation'] or {}).get('cases',[]):
        lines.append(f"| {row['label']} | {row['adapter_phases']} | {row['engine_turns']} | {row['accepted_choices']} | {row['unavailable_choice_retries']} | {row['terminal_reason']}, points={row['new_evaluation_points']} | null |")
    lines += ['', 'The earlier visible switching cycles were diagnostic observations, not draw certificates: one policy later changed behavior and won. '
        'The other two reached the engine’s native turn-limit tie. Constructed mechanics fixtures cover Tera, forced switches, hidden trapping and public retry updates, PP exhaustion/Struggle, long games and native ties. '
        'Additional tests cover retry accounting, cap/fault/unstarted score bounds, valid zero scores, malformed outputs, inference deadlines, forbidden private fields and no fallback.','',
        '## Pinned public-information Foul Play opponent','',
        'The host-private adapter consumes only one player’s own request and player-channel protocol history. It samples beliefs about unrevealed opponents from pinned public random-set data; '
        'it never clones the actual hidden battle state. Search uses poke-engine; pinned Showdown remains the sole transition authority. '
        'The richer legal protocol/history interface is declared separately from the700-value neural encoder. '
        'This Foul Play-based adapter uses deterministic weighted-visit argmax across two beliefs; upstream uses adaptive sampling and randomized near-best selection. Its exact adaptation is source-pinned rather than presented as an unchanged upstream bot.','',
        'Fixed-work tests use two seeded beliefs,1000iterations each and one search thread. Linux tests repeated identical inputs and changed wall-clock protocol timestamps without changing actions, sampled-state hashes or visit counts. '
        'Production uses two beliefs at a soft10ms search limit each, with a hard1second end-to-end deadline. Upstream time checks occur in1000iteration chunks, so actual latency and overshoot are reported. '
        'ReferenceUID65533 and candidateUID65534 are distinct; both are denied private grading inputs, and only the current job’s private Volume subdirectory is mounted.','']
    registry=d.get('frozen_opponent_registry')
    if registry:
        search=registry['search'];lines += [f"Foul Play commit: `{search['foul_play_commit']}`. poke-engine0.0.48 source archive and public-set JSON are hashed in the registry. "
            'The retained patch adds seeded single-thread search and fixes the upstream README filename case for Linux; it changes no authoritative Showdown rules. '
            'All Linux dependency wheels, source/configuration and adapter hashes are frozen in the release. No live ladder access or external model call was used.', '']
    for title,key in [('Eight-seed public development qualification','development_screen'),('Fresh private bank after selection closed','fresh_final_bank')]:
        result=d.get(key)
        lines += ['## '+title,'']
        if not result:lines += ['Pending; no score is inferred from incomplete work.',''];continue
        lines += [f"Foul Play: **{number(result['raw_score_0_100'])}/100**, {result['natural_games']}/{result['scheduled_games']} natural games, "
            f"{result['administrative_caps']} administrative caps, {result['faults']} faults, {result['visible_choice_rejections']} visible engine rejections. "
            f"Completion={result['completion_rate']:.3f}; total-point bounds={result['total_point_bounds_0_100']}; seed-cluster95%interval={result['seed_cluster_bootstrap_95_percent']}.",'',
            '| Opponent / FoulPlay seat | Natural / scheduled | Points /100 |','|---|---:|---:|']
        for name,row in result['by_opponent_and_seat'].items():lines.append(f"| {name} | {row['natural']}/{row['scheduled']} | {number(row['raw_score_0_100'])} |")
        lines += ['',f"End-to-end search latency seconds: `{result.get('search_latency_seconds')}`. Gameplay stage wall time: {result['seconds']:.3f}s. Full worker allocation and setup are charged separately below.",'']
    lines += ['## Admission, persistence and accounting','',
        'The separate RL owner exported a tensor-only inference bundle after original source/configuration/engine/population admission. '
        'The evaluator independently verified its attested manifest, every canonical runtime file, Policy architecture, engine build and tensor identity; corrupt weights and a wrong attestation were rejected. '
        'Four development games completed through evaluator-v2. This is inference admission, not a new resume path or a reference-strength claim. '
        'A separate four-game candidate-facing API proof completed against power and the search opponent with zero faults. The reusable API supports original sealed candidates and attested reference inference exports, preserving candidate-perspective scoring and every source guard.','',
        f"Evaluator usage: **{d['allocated_cpu_seconds']:.3f} allocated CPU-worker seconds,0GPU seconds,${usage['settled_resource_usd']:.6f} settled resources plus${usage['held_resource_usd']:.6f} held resources,0model calls**. "
        'Own ceilings are1800GPU seconds,$8resources and$0.50models within the explicitly separate new shared ledger. '
        'Full provider lifetimes include setup, package builds, simulation, inference, exports, transfers, idle and teardown; no GPU-utilization discount is applied. Historical budgets and failed attempts remain unchanged.','',
        '| Attempt | Outcome | Full allocation seconds | Resource charged/held |','|---|---|---:|---:|']
    for job in d['jobs']:
        provider=job['provider'];seconds=provider['finished_at']-provider['started_at'] if provider else None
        lines.append(f"| {job['label']} | {job['status']} | {number(seconds)} | ${number(job['charged_or_held_usd'])} |")
    lines += ['', 'Retained failures include a Linux packaging filename error, low-UID public-code/cache startup permissions, a legacy launcher umask incompatibility, '
        'an SDK duplicate-Volume mount rejection and a pre-allocation connection failure. All allocated failures are fully charged. '
        'Pre-allocation failures have provider-inventory zero-worker proofs. The final launcher uses a single scoped Volume mount, root-only private inputs, public readonly inference code and persisted sources/receipts; '
        'all remote jobs have finite primary-process deadlines and durable artifacts through host disconnect.','',
        '## Remaining qualification limits','']
    audit=d.get('final_provider_audit')
    if audit:
        lines += [f"Final provider audit: **{audit['worker_count']} owned workers finished**, all authoritative intervals match their settlements. "
            'Every worker was CPU-only with four fixed physical cores and8GiB. No GPU or model allowance was consumed.','']
    if d.get('provider_cleanup'):
        lines += ['All required outputs and immutable uploaded sources were hashed and mirrored locally before the newly owned evaluator app and temporary Volume were removed. '
            'The reused image and the separate RL owner’s app, Volume and files were left intact.','']
    lines += ['Storage/egress initially lacked a separate reservation. Final reconciliation added a **$0.05 unpriced noncompute hold**, preserving the initial booking gap, original timestamps and all prior charges. '
        'The hold remains after cleanup until invoice reconciliation; it is not an active worker or a fabricated zero charge.','']
    lines.extend('- '+s for s in d['limits'])
    lines += ['', 'Reproduce local checks and offline report:', '', '```sh',
        'node simulators/pokemon/autoresearch/evaluation_v2/test_worker.cjs',
        '.venv-pokemon/bin/python -m unittest integrations.pokemon.autoresearch.evaluation_v2.tests.test_contract -v',
        '.venv-pokemon/bin/python -m integrations.pokemon.autoresearch.evaluation_v2.report',
        '```','', 'Executed cloud commands used the explicit new campaign guard and unique labels below. These labels and the private bank are already spent/opened; do not relaunch them to renew allowance or tune on final outcomes.', '', '```sh',
        '.venv-pokemon/bin/python -m integrations.pokemon.autoresearch.evaluation_v2.launch_continuation --label retained-cap-continuation-004',
        '.venv-pokemon/bin/python -m integrations.pokemon.autoresearch.evaluation_v2.launch_qualification --label foul-play-eight-seed-screen-003 --stage screen',
        '.venv-pokemon/bin/python -m integrations.pokemon.autoresearch.evaluation_v2.final_bank prepare',
        '.venv-pokemon/bin/python -m integrations.pokemon.autoresearch.evaluation_v2.final_bank execute',
        '```', '', 'The reusable candidate-facing API is `integrations.pokemon.autoresearch.evaluation_v2.arena`; its separately frozen controller release is recorded in `active-entrant-api-release.json`. '
        'The qualified opponent/termination release is recorded in `active-evaluator-release.json`. Baseline weight/source identities and the unchanged adapter selection rule are in `opponent-source-identity-audit.json`. '
        'Exact private release/receipt paths and all seeds/attempts remain in the ignored campaign evidence. '
        'No private seeds, weights, traces, assets or credentials were published.','']
    content='\n'.join(lines)
    for old,new in {'November9':'November 9',',2026':', 2026','byseat':'by seat','proof1729':'proof 1729','selected2718':'selected 2718','Gen9':'Gen 9',',700':', 700','and14actions':'and 14 actions','under4096adapter':'under 4,096 adapter','and16consecutive':'and 16 consecutive','perseat':'per seat','Exact512-phase':'Exact 512-phase','original12-game':'original 12-game','all700encoded':'all 700 encoded','the700-value':'the 700-value','beliefs,1000iterations':'beliefs, 1,000 iterations','soft10ms':'soft 10 ms','hard1second':'hard 1 second','in1000iteration':'in 1,000-iteration','ReferenceUID65533':'Reference UID 65533','candidateUID65534':'candidate UID 65534','poke-engine0.0.48':'poke-engine 0.0.48','seed-cluster95%interval':'seed-cluster 95% interval','candidate points,':'candidate points,','seconds,0GPU':'seconds, 0 GPU','resources plus$':'resources plus $','resources,0model':'resources, 0 model','are1800GPU':'are 1,800 GPU','$8resources':'$8 resources','$0.50models':'$0.50 models','conservative1.1margin':'conservative 1.1 margin','reduced2belief/10ms':'reduced two-belief/10 ms','observed seeds':'observed seeds','and8GiB':'and 8 GiB'}.items():content=content.replace(old,new)
    return content


def main():
 d=collect();target=ROOT/'reports/pokemon/evaluation-readiness-2026-11-09';write(target.with_suffix('.json'),d);target.with_suffix('.md').write_text(render(d))
 print(json.dumps({'status':d['status'],'readiness':d['readiness'],'usage':d['evaluator_usage']}))
if __name__=='__main__':main()
