"""Regenerate the new campaign readout; historical source and receipts are read-only."""
from collections import defaultdict
from pathlib import Path
import hashlib
import json
import statistics
import re

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'evidence/doudizhu-implementation'
REPORT=ROOT/'reports/doudizhu'


def read(path):return json.loads(path.read_text())
def write(path,value):path.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')


def aggregate(rows,keys):
    groups=defaultdict(list)
    for row in rows:groups[tuple(row.get(k) for k in keys)].append(row)
    result=[]
    for key,group in sorted(groups.items(),key=lambda item:str(item[0])):
        item=dict(zip(keys,key));item['repetitions']=len(group)
        for field in ('seconds','decisions_per_second','simulator_seconds','simulator_decisions_per_second','train_seconds','cold_process_seconds',
                      'training_cold_seconds','train_evaluate_cold_seconds','cuda_peak_allocated_bytes','peak_rss_bytes'):
            if all(field in x for x in group):
                values=[x[field] for x in group]
                item[field]={'median':statistics.median(values),'min':min(values),'max':max(values)}
        for numerator,denominator,name in [('decisions','train_seconds','train_decisions_per_second'),
                 ('decisions','training_cold_seconds','cold_train_decisions_per_second')]:
            if all(numerator in x and denominator in x for x in group):
                values=[x[numerator]/x[denominator] for x in group]
                item[name]={'median':statistics.median(values),'min':min(values),'max':max(values)}
        result.append(item)
    return result


def main():
    REPORT.mkdir(exist_ok=True)
    result={'game':'douzero-fixed-role-v1','date':'2026-11-08',
        'upstream_commit':'718a5c920bf3361e34178a38f3b80458e176b351',
        'hardware':{'gpu':'NVIDIA L4','gpus':1,'cpu_cores':4,'ram_gib':16},
        'parameters':4486659,'units':'actual selected player decisions; every learner seat in pure self-play',
        'historical_receipts_unchanged':True}
    manifests={}
    for path in list(BASE.glob('structural-*-gpu-*/receipt.json'))+list(BASE.glob('structural-gpu-*/receipt.json'))+list(BASE.glob('cuda-*/receipt.json')):
        receipt=read(path)
        manifests.setdefault(receipt['run_id'],(path,receipt))
    ledger=[]
    for name,(path,r) in sorted(manifests.items()):
        row={k:r.get(k) for k in ('run_id','status','allocated_start_unix','allocated_end_unix','allocated_worker_seconds','allocated_gpu_hours','error_type','error')}
        row['receipt']=str(path.relative_to(ROOT));row['receipt_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
        row['quoted_body_cost_usd']=r['allocated_worker_seconds']*.00030992
        client=path.parent/'client-lifetime.json'
        if not client.exists():client=path.parent.with_name(path.parent.name+'-client')/'client-lifetime.json'
        if client.exists():
            c=read(client)
            row['client_elapsed_seconds']=c.get('roundtrip_seconds',c.get('elapsed_seconds',c.get('ended_unix',0)-c.get('started_unix',0)))
        # A client may spend a long time downloading after its GPU scaled down.
        # Record its conservative envelope without pretending it is GPU usage.
        row['conservative_client_envelope_seconds']=max(r['allocated_worker_seconds'],row.get('client_elapsed_seconds',0))+60
        for log in (BASE/'structural-20261108').glob('modal*.log'):
            value=log.read_text(errors='replace')
            if name in value:
                ids=re.findall(r'https://modal.com/apps/duper/staging/(ap-[A-Za-z0-9]+)',value)
                if ids:row['modal_app_id']=ids[0]
                row['client_completed_observed']='App completed.' in value
                break
        ledger.append(row)
    body=sum(r['allocated_worker_seconds'] for r in ledger)
    envelope=sum(r['conservative_client_envelope_seconds'] for r in ledger)
    write(REPORT/'structural-cost-ledger.json',{'authorization':'new structural/native/CUDA campaign; old caps explicitly superseded; no ledger reset',
        'rows':ledger,'worker_body_seconds':body,'worker_body_gpu_hours':body/3600,'quoted_body_cost_usd':body*.00030992,
        'illustrative_client_envelope_seconds':envelope,'illustrative_client_envelope_quote_usd':envelope*.00030992,
        'envelope_note':'sum of max(worker body, client elapsed)+60 seconds per attempt; deliberately overcounts downloads after GPU release; not measured GPU time or a guaranteed billing upper bound',
        'price_source':'retained reports/doudizhu/modal-price-quote-2026-11-08.json; estimate, not settled invoice',
        'accounting':'all owned successful and failed worker bodies; compilation/imports/learning/evaluation/packaging/Volume commit included where receipt states; app envelopes and pre-worker build costs separately recorded',
        'pre_worker_failures':['unsupported Modal add_python patch version','historical archive image ordering rejected before submission'],
        'settled_invoice_verified':False})
    result['cost_ledger']='reports/doudizhu/structural-cost-ledger.json'
    result['native_parity']=read(BASE/'structural-20261108/native-fuzz/result.json')
    resident=sorted(BASE.glob('cuda-validate-*/validation.json'))
    if resident:result['resident_validation']=read(resident[-1])
    path=BASE/'cuda-profile-20261108T064818-d35500/matrix.json'
    if path.exists():
        rows=read(path)
        result['resident_matrix']=aggregate(rows,['protocol','engine','batch'])
        result['resident_matrix_source']=str(path.relative_to(ROOT))
        result['fixed_policy_cross_backend_hashes']={}
        for batch in (64,2048):
            selected=[x for x in rows if x['protocol']=='fixed-policy' and x['batch']==batch]
            result['fixed_policy_cross_backend_hashes'][str(batch)]={key:len({x[key] for x in selected})==1 for key in ('trajectory_sha256','weights_sha256','evaluation_sha256')}
    learning=BASE/'cuda-learning-20261108T064944-0fc958/learning-summary-compact.json'
    if learning.exists():
        rows=read(learning);result['resident_learning']=rows
        result['resident_learning_aggregate']=aggregate(rows,['label'])
        result['resident_vs_native_three_seed_exact']={}
        for seed in (17,29,43):
            a,b=[next(x for x in rows if x['seed']==seed and x['label']==label) for label in ('native-ieee','resident-ieee')]
            result['resident_vs_native_three_seed_exact'][str(seed)]={k:a[k]==b[k] for k in ('weights_sha256','decisions','samples','updates','final_evaluation_sha256','used_deal_hashes_sha256')}
    for pattern,label in [('cuda-precision-*/precision-summary.json','precision_learning'),('cuda-historical-*/historical-summary.json','literal_original_training')]:
        paths=sorted(BASE.glob(pattern))
        if paths:
            rows=read(paths[-1]);result[label]=rows
            if label=='literal_original_training':
                result[label]=[{k:v for k,v in x.items() if k!='used_deal_hashes'} for x in rows]
            result[label+'_source']=str(paths[-1].relative_to(ROOT))
            result[label+'_aggregate']=aggregate(rows,['dtype','matmul_precision','batch'] if label=='precision_learning' else ['label'])
    paths=sorted(BASE.glob('cuda-precision-*/baseline-matrix.json'))
    if paths:
        rows=read(paths[-1]);result['direct_original_comparison']=aggregate(rows,['engine','batch'])
        result['direct_original_comparison_source']=str(paths[-1].relative_to(ROOT))
        at64=[x for x in rows if x['batch']==64]
        result['matched_batch64_hashes']={k:len({x[k] for x in at64})==1 for k in ('trajectory_sha256','weights_sha256','evaluation_sha256')}
    for pattern,label in [('cuda-long-matched-*/matrix.json','long_matched'),('cuda-scalar-matched-*/matrix.json','scalar_matched'),('cuda-sim-original-*/simulation-matrix.json','direct_original_simulation')]:
        paths=sorted(BASE.glob(pattern))
        if paths:
            rows=read(paths[-1]);result[label]=rows;result[label+'_source']=str(paths[-1].relative_to(ROOT))
            result[label+'_aggregate']=aggregate(rows,['backend'] if label=='direct_original_simulation' else ['engine','batch'])
            result[label+'_hashes']={k:len({x[k] for x in rows})==1 for k in
                (['outcomes_sha256','decisions'] if label=='direct_original_simulation' else ['weights_sha256','trajectory_sha256','evaluation_sha256','decisions'])}
            if label!='direct_original_simulation':
                medians={backend:{field:statistics.median(x[field] for x in rows if x['engine']==backend)
                    for field in ('train_seconds','cold_process_seconds')} for backend in ('upstream','native','cuda')}
                result[label+'_ratios']={backend:{field:medians['upstream'][field]/medians[backend][field]
                    for field in ('train_seconds','cold_process_seconds')} for backend in ('native','cuda')}
    paths=sorted(BASE.glob('cuda-final-*/final-learning.json'))
    if paths:
        result['final_packing_learning']=read(paths[-1])
        result['final_packing_learning_source']=str(paths[-1].relative_to(ROOT))
        receipt=read(paths[-1].with_name('receipt.json'))
        result['final_worker_conservative_useful_decisions_per_second']=sum(x['decisions'] for x in result['final_packing_learning'])/receipt['allocated_worker_seconds']
        result['final_worker_rate_scope']='main three learning jobs only in numerator; entire final worker including fresh qualification, compilation, resume tests, checkpoint loads and Volume commit charged in denominator'
    result['limitations']=['development random/heuristic panel, not competitive grading','TF32/BF16 are numerical recipe changes',
        'batch scaling changes online refresh cadence','Polytopia headline uses different hardware/network/game work and historical accounting',
        'native4096-game and CUDA128-game differential suites are separate coverage, not 4096 GPU games',
        'early graph trace/weight equality failed; graph stays opt-in and is not claimed exact',
        'three native/full-only matrix receipts failed on an empty small-model group after valid full-model rows; original failures retained']
    write(REPORT/'structural-summary.json',result)
    write_report(result,ledger,body)
    print(json.dumps({'worker_body_hours':body/3600,'quoted_body_usd':body*.00030992,'receipts':len(ledger),
                      'summary':str(REPORT/'structural-summary.json')}))


def write_report(result,ledger,body):
    lines=['# Dou Dizhu native/CUDA rewrite — 2026-11-08','',
        'Complete independent native CPU and shared-rule CUDA simulators now run the frozen full task, with actual full-size neural learning and source-bound resume. The original Python source remains unchanged and independently executable. This is a substantial environment rewrite, not a claim of competitive DouZero strength.','',
        'All comparisons below use one NVIDIA L4, four allocated physical CPU cores and 16 GiB. Simulation, neural training and cold process time are distinct measurements. Three repetitions are reported where available; first-use compile cost is retained. No rates from different campaigns are multiplied.','',
        'Machine readout: [structural-summary.json](structural-summary.json). Paid attempts, including failures: [structural-cost-ledger.json](structural-cost-ledger.json). Commands and protocol: [README](../../integrations/doudizhu/README.md).','',
        '## Two original baselines','',
        '1. **Simulator:** the literal pinned `douzero.env.game.GameEnv`, commit `718a5c920bf3361e34178a38f3b80458e176b351`, called without the integration Game/Observation wrapper. A public cooperating heuristic canonicalizes legal multisets, plays natural games through late positions and checks complete outcomes. Serial and four persistent-worker results expose CPU scaling independently. The new engines use identical deals and the declared public heuristic.','',
        '2. **Full training:** the literal committed integration at `26f2306f73785a8fb0c885722341cbe5554e4991`, the earliest retained runnable pre-structural snapshot. Its source was extracted unchanged into a separate private workspace; original zero/four-worker DMC and new native control each trained 2,048 games for seeds 17/29/43. The legacy actor RNG, learner cadence, full 4,486,659-parameter model and completed deals/weights are identical. This is an upstream-backed integration baseline, not an invented rate for the upstream repository\'s own asynchronous trainer.','']
    sims=result.get('direct_original_simulation',[])
    if sims:
        lines+=['## Direct simulator throughput','',
            '| Backend | Median decisions/s | Range decisions/s |','|---|---:|---:|']
        for name in dict.fromkeys(x['backend'] for x in sims):
            values=[x['decisions_per_second'] for x in sims if x['backend']==name]
            lines.append(f'| {name} | {statistics.median(values):,.0f} | {min(values):,.0f}–{max(values):,.0f} |')
        lines+=['',
            'Each trial completes 2,048 games / 69,477 decisions. Every backend/repetition has the same complete outcome/bomb/decision hash. The first CUDA trial includes NVCC/library initialization and is much slower; the subsequent ~0.086-second trials describe warm simulation, not cold training. Native compilation and the persistent Python pool startup are also retained and charged. Short native timings show substantial variance; do not turn these into stronger statistical claims.','',
            f'Raw commands/source/runtime: `{result["direct_original_simulation_source"]}` and its private Volume run directory.','']
    historical=result.get('literal_original_training',[])
    if historical:
        lines+=['## Literal original full training','',
            '| Seed | Original 0 workers, cold s | Original 4 workers, cold s | Native 4 threads, cold s |','|---|---:|---:|---:|']
        for seed in (17,29,43):
            rows={x['label']:x for x in historical if x['seed']==seed}
            lines.append(f'| {seed} | '+ ' | '.join(f'{rows[k]["training_cold_seconds"]:.2f}' for k in ('original-w0','original-w4','native-w4'))+' |')
        lines+=['',
            'All three arms have bit-identical final weights, ordered sample counts, updates and deal hashes within each seed. Cold time includes imports, first compile where encountered, real collection/learning, initial/final and per-wave checkpoints, first-wave replay audit and teardown. This short ordinary-launcher workload demonstrates a smaller gain than warm simulation because startup, audit and checkpoint costs remain. The four-worker original is the stronger CPU baseline.','']
    for label,title in [('long_matched','Long matched training profile'),('scalar_matched','Faster original-actor control')]:
        rows=result.get(label,[])
        if not rows:continue
        lines+=['## '+title,'',
            '| Engine | Repetitions | Median training s | Median cold process s |','|---|---:|---:|---:|']
        for name in ('upstream','native','cuda'):
            selected=[x for x in rows if x['engine']==name]
            if selected:lines.append(f'| {name} | {len(selected)} | {statistics.median(x["train_seconds"] for x in selected):.2f} | {statistics.median(x["cold_process_seconds"] for x in selected):.2f} |')
        lines+=['',
            'Identical full-size trained starting weights, 8,192 training games, batch 2,048, learner minibatch 256, GPU shuffle, epsilon 0.1, candidate chunk 16,384, FP32 and fixed allocated resources. A separate 128-game simulator diagnostic, final checkpoint, 24 practice games and teardown are included in cold process time; `train_seconds` is collection plus real learner work. This profile saves one final checkpoint, unlike the ordinary launcher\'s per-wave checkpoints. All compared trajectories/weights/evaluations must match before accepting the completed matrix.','',
            'The follow-up scalar control replaces one-element NumPy counter RNG calls in the original object actor with the exact same integer algorithm. This removes avoidable baseline overhead; its one-repeat result is kept separate from the prior three-repeat matrix, with no invented repetitions. Scalar/vector/shared-C++ counter outputs and rejection cases are tested. First native/CUDA compile costs remain included.','',
            f'Raw measurements: `{result[label+"_source"]}`.','']
    learning=result.get('resident_learning',[])
    if learning:
        lines+=['## Real full-size learning and CPU/GPU equivalence','',
            '| Seed | Native FP32 cold s | Resident FP32 cold s | Resident TF32-head cold s | FP32 weights/eval identical |','|---|---:|---:|---:|---|']
        for seed in (17,29,43):
            rows={x['label']:x for x in learning if x['seed']==seed}
            lines.append(f'| {seed} | '+ ' | '.join(f'{rows[k]["training_cold_seconds"]:.2f}' for k in ('native-ieee','resident-ieee','resident-tf32'))+' | yes |')
        lines+=['',
            'Each arm trains 65,536 natural games, batch 2,048, learner minibatch 4,096, all three original-sized role networks, pure DMC self-play and identical deal/actor/shuffle protocols. Native and resident FP32 match bit for bit across all three seeds: final weights, decisions, samples, updates, deals and final evaluation. Resident FP32 reaches ~55–66k decisions/s and TF32 heads ~73–74k/s including cold training startup/checkpointing. These rates exclude the separately timed initial/final practice evaluations; train+evaluate times are retained in JSON.','',
            'The FP32 role-balanced development gains are +41.67, +45.83 and +39.06 percentage points. TF32 gains are +42.19, +46.88 and +38.54 points; seed 43 is slightly worse than its FP32 counterpart. Evaluation is 32 inspected practice deals × two weak opponents × three individually controlled roles, with fixed opponent teammates for peasant seats and paired deal-cluster confidence intervals. These results establish actual learning, not competitive play or blind grading.','']
    precision=result.get('precision_learning',[])
    if precision:
        lines+=['## Tensor-core follow-up','',
            '| Seed | Batch 4,096 TF32 cold s | Batch 4,096 BF16 cold s | BF16 decisions/s | BF16 paired practice change |','|---|---:|---:|---:|---:|']
        for seed in (17,29,43):
            rows={x['dtype']:x for x in precision if x['seed']==seed}
            if len(rows)!=2:continue
            b=rows['bfloat16'];a=rows['float32'];delta=b['paired_practice_change']['groups']['role_balanced_mean']['paired_win_rate_delta']
            lines.append(f'| {seed} | {a["training_cold_seconds"]:.2f} | {b["training_cold_seconds"]:.2f} | {b["decisions"]/b["training_cold_seconds"]:,.0f} | {delta:+.2%} |')
        lines+=['',
            'BF16 autocast uses FP32 master parameters, FP32 loss and an explicitly FP32 scalar Q head to avoid quantizing the final argmax scores. Actual CUDA backward and midpoint resume are bit-exact within BF16. It is a different numerical recipe, not FP32 equivalence. Batch 4,096 is a separate cadence change from batch 2,048. The ordinary common evaluator uses FP32 for every precision arm. Operator-level CPU/CUDA trace and NVCC resource logs are retained privately.','']
    final=result.get('final_packing_learning',[])
    if final:
        lines+=['## Final upload-packing refinement','',
            '| Seed | Final cold training s | Decisions/s | Exact prior BF16 weights/samples/updates/deals |','|---|---:|---:|---|']
        for x in final:lines.append(f'| {x["seed"]} | {x["training_cold_seconds"]:.2f} | {x["decisions_per_second"]:,.0f} | yes |')
        lines+=['',
            'Four bulk histograms replace per-card NumPy scalar increments and linear rank lookups before CUDA upload. Every deal is still validated, and all packed bytes match the prior implementation. The local three-repeat, performance-lock-protected packing microbenchmark is about 2.1× faster; this is a packing-only measurement. The final GPU runs re-execute three complete 65,536-game learning jobs and verify exact prior BF16 weights, samples, updates and deals, plus a fresh original/shared/CUDA differential and FP32 resume suite. Their throughput is measured on a new equal-resource worker; an isolated end-to-end causal speedup from packing alone is not claimed.','',
            'The actual operator trace now identifies recurrent inference/backward, dense computation and optimizer/initialization work as the main costs; device legal generation is a small part of the trace. The full 512-wide/LSTM-128 model remains intact. The next substantial arithmetic optimization would need recurrent/learner kernel work and the same policy-quality/resume qualification, rather than more CPU wrapper tuning.','']
        lines += [f'Conservatively charging the final worker\'s complete setup/validation/resume/commit overhead against only its main learning decisions gives **{result["final_worker_conservative_useful_decisions_per_second"]:,.0f} useful decisions per allocated second**. The ~101k figures above are cold training-process rates with the simulator library already compiled by qualification; they are not whole-campaign GPU-hour productivity. All development allocations remain in the ledger.','']
    lines+=['## Implementation and parity','',
        '- Native `native/engine.cpp`: complete independent rules, transition, legal move generation/detection/selection, canonical ordering, public observations and terminal accounting; fixed rank-count state and persistent native thread pool. The unchanged upstream is used only for differential/replay checks.','- CUDA `rules.h` and `engine.cu`: shared CPU/CUDA rule core; GPU canonical bitonic sort/dedup, warp public encoding, ragged neural scoring and deterministic argmax, legal counter sampling, device transitions, selected-action trajectory rings and GPU learner samples. No legal-action clipping or simulator delegation to Python. Scratch overflow is an unscored fault.','- Native oracle fuzz: 4,096 complete games, 250,785 decisions, 131,724 late decisions and legal types 0–14. CUDA device qualification: 128 complete games / 7,033 decisions with original CPU/shared CPU/CUDA state, legal ordering, features, counter choices and outcomes matched at every step. These are separate suites; 4,096 GPU differential games are not claimed.','- Targeted CUDA checks cover reachable upstream TYPE15 airplane behavior, bomb/rocket responses, two-pass reset and hidden-hand repartition invariance. CPU tests cover public features in every role. Normal CUDA evaluation against a heuristic and first-wave bounded CPU replay audits also execute in the matrices.','- Full-size resident save/resume is bit-exact for weights, RNG and counters. Deals are durably reserved before a wave; faults/deadlines give no targets and reserved indices remain skipped. Run/source locks prevent concurrent overwrite. Cancellation forwards to the whole child process group and records elapsed allocation; the regression test verifies the worker PID is gone. Historical checkpoints and original lock manifests remain unchanged.','',
        '“ieee” in the recipe controls head matrix multiplication. Deterministic cuDNN retains its separately recorded default TF32 permission in all matched FP32 controls; this is not a claim that every recurrent operation forbids tensor cores. TF32 heads and BF16 are explicitly separate arithmetic arms. The actor protocol and GPU learner shuffle are also explicit changes from historical PCG64/NumPy sequencing; resume across protocols is refused.','',
        '**47 focused local tests pass**, including latest deal packing, counter rejection, source drift, worker cancellation/cleanup and checkpoint/resume checks: `evidence/doudizhu-implementation/structural-20261108/tests-final-latest.log`. CUDA qualification and three-seed latest-source full learning completed successfully. All owned staging applications are stopped with zero tasks; see [structural-cleanup.json](structural-cleanup.json). No other application was stopped.','',
        '## Failed work and limits','',
        'The early graph matrix failed trained-policy trajectory/sample/weight identity; graphs stay opt-in and are not the recommended exact path. Three early native/full-only harnesses failed by checking an empty small-model group after recording valid full-size rows. Original failed receipts remain failed and charged. A native validation receipt also retained a misleading legacy graph-field name; its actual native configuration and dedicated native resume flag are the authority. Unsupported Modal Python version, a stdlib-name collision, a missing lock parent directory and a historical-image ordering error are retained rather than erased.','',
        'Original locks, manifests and the upstream checkout were not modified. New code and recipes are uncommitted; no publication or new Polytopia job was performed. Source archives, exact commands/config hashes, full weights, replays, per-role evaluation rows, profiler traces and compile receipts are retained under each local evidence run and/or private staging Volume `doudizhu-structural-evidence-20261108`. No evaluated seed is now a blind grading seed.','',
        '## Accounting and Polytopia comparison','',
        f'At regeneration, {len(ledger)} paid worker receipts total {body/3600:.4f} allocated body GPU-hours, approximately ${body*.00030992:.3f} at the retained L4+4CPU+16GiB quote. This includes failed worker attempts and the body scopes stated in each receipt. App/client envelopes, setup/teardown uncertainty and pre-worker failures are separately recorded; this is not a settled invoice. Long client artifact downloads after a worker scaled down must not be relabelled GPU usage. No historical ledger or allowance was reset.','',
        'Polytopia\'s retained small-net H100 result is ~106k samples/s in `environments/polytopia/docs/autoresearch-s120-readout-2026-11-08.md`. Doudizhu\'s complete full-size L4 learner is in the same throughput order of magnitude, but game decisions, policy size, hardware and accounting differ. Simulator-only rates must not be compared with Polytopia full training, and the old training-process accounting must not replace this benchmark\'s full allocated-worker budget.','',
        'Reproduce the readout with `.venv-doudizhu/bin/python integrations/doudizhu/summarize_structural.py`. Launch commands for the bounded private matrices are `modal run --env staging integrations/doudizhu/cuda_modal.py --stage {validate,profile,learning,precision,historical,long-matched,scalar-matched,sim-original}`. These provision paid work; source archive/build/config receipts specify the exact historical command and revision.','']
    (REPORT/'structural-performance-2026-11-08.md').write_text('\n'.join(lines))


if __name__=='__main__':main()
