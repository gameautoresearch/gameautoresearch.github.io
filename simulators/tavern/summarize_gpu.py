"""Assemble local/CUDA performance, paired learning and provider receipts without editing raw runs."""
from datetime import datetime,timezone
from pathlib import Path
import csv
import json
import statistics

ROOT=Path(__file__).resolve().parents[2]
E=ROOT/'evidence/tavern-implementation'


def read(path):return json.loads(path.read_text())


def main():
    paired=read(E/'cuda/cuda-paired-v2/cuda-pairs/paired-summary.json')
    cpu=read(E/'cuda/cuda-paired-v2/same-host-cpu-pairs/paired-summary.json')
    local=read(E/'optimized-cpu-pairs/paired-summary.json')
    result={'schema':'tavern-cuda-validation-v1','date':'2026-11-08','actual_cuda_executed':True,
      'hardware':read(E/'cuda/cuda-paired-v2/device.json'),'cuda_performance':paired['summary'],
      'same_host_cpu_performance':cpu['summary'],'local_cpu_performance':local['summary'],
      'cuda_parity_checks':paired['correctness'],'same_host_cpu_parity_checks':cpu['correctness'],
      'local_cpu_parity_checks':local['correctness'],'budget':read(E/'gpu-budget.json')}
    cases=[]
    for workload in ['initial','busy-trained']:
        row={'workload':workload}
        for optimized in [False,True]:
            runs=[r for r in paired['rows'] if r['workload']==workload and r['optimized']==optimized]
            row['optimized' if optimized else 'baseline']={
              'median_full_process_seconds':statistics.median(r['full_process_seconds'] for r in runs),
              'median_warmed_update_seconds':statistics.median(r['warmed_update_seconds'] for r in runs),
              'median_warmed_samples_per_second':statistics.median(r['warmed_samples_per_second'] for r in runs),
              'median_stage_seconds':{k:statistics.median(r['profile']['stage_seconds'][k] for r in runs) for k in runs[0]['profile']['stage_seconds']},
              'completed_placement_episodes':sum(r['training']['counts']['completed_learning_episodes'] for r in runs),
              'samples_per_trial':runs[0]['training']['counts']['accepted_learning_samples'],
              'faults':sum(r['training']['counts']['invalid_lobbies'] for r in runs),
              'rejected':sum(r['training']['counts']['rejected'] for r in runs),
              'peak_cuda_allocated_bytes':max(r['profile']['peak_cuda_allocated_bytes'] for r in runs)}
            if workload=='busy-trained':
                c=runs[0]['training']['counts'];states=[s for ctx in runs[0]['profile']['workload_context'] for s in ctx['states']]
                row['busy_workload']={'initial_controlled_alive':sum(s['learning_alive'] for s in states),
                  'continuation_count':len(states),'min_starting_board_size':min(s['board_sizes'][0] for s in states),
                  'mean_ready_board_minions':c['ready_board_count_sum']/c['ready_learning_decisions'],
                  'turn10plus_ready_fraction':c['ready_turn10_plus']/c['ready_learning_decisions'],
                  'cohort_tail_fraction':c['cohort_tail_game_steps']/c['total_game_step_opportunities']}
        cases.append(row)
    result['performance_details']=cases
    seeds={}
    allocations=[]
    for folder in sorted((E/'cuda').glob('*')):
        allocation=folder/'allocation.json'
        if allocation.exists(): allocations.append({'run_id':folder.name,**read(allocation)})
        learning=folder/'learning/learning-summary.json'
        if not learning.exists():continue
        for row in read(learning)['results']:
            row=dict(row,run_id=folder.name)
            row['full_process_speedup']=row['baseline_seconds']/row['optimized_seconds']
            row['baseline_samples_per_allocated_gpu_hour']=3600*row['training_samples']/row['baseline_seconds']
            row['optimized_samples_per_allocated_gpu_hour']=3600*row['training_samples']/row['optimized_seconds']
            seeds[row['seed']]=row
    result['learning']=list(seeds.values())
    if seeds:
        result['learning_aggregate']={'seeds':len(seeds),'mean_initial_placement':statistics.mean(r['initial_mean_placement'] for r in seeds.values()),
          'mean_final_placement':statistics.mean(r['final_mean_placement'] for r in seeds.values()),
          'mean_improvement':statistics.mean(r['placement_improvement'] for r in seeds.values()),
          'all_optimization_weights_identical':all(r['weights_identical'] and r['counts_identical'] for r in seeds.values()),
          'median_full_process_speedup':statistics.median(r['full_process_speedup'] for r in seeds.values()),
          'all_placements_valid':all(r['invalid_evaluation_games']==0 for r in seeds.values())}
    sampling=E/'cuda/cuda-sampling-v6/sampling/sampling-summary.json'
    if sampling.exists():
        raw=read(sampling);result['sampler']={'summary':raw['summary'],'checks':raw['checks'],'valid':raw['valid'],'learning64_full_process_speedup':raw['learning64_full_process_speedup'],'selected_as_default':False,'reason':'No consistent busy-continuation improvement; keep optional on qualified PyTorch 2.9.1 CUDA'}
    resume=E/'cuda/cuda-seed78901-v5/cuda-resume-check/summary.json'
    if resume.exists():result['cuda_resume']=read(resume)
    result['allocations']=allocations
    result['measured_function_worker_seconds']=sum(r['worker_seconds'] for r in allocations)
    if (E/'modal-billing.json').exists():result['provider_billing']=read(E/'modal-billing.json')
    if (E/'remote-cleanup.json').exists():result['remote_cleanup']=read(E/'remote-cleanup.json')
    # NVML observations are diagnostic intervals, not active-kernel accounting or a score.
    trace=E/'cuda/cuda-paired-v2/nvidia-smi.csv'
    windows=[]
    for row in paired['rows']:
        run=E/'cuda/cuda-paired-v2/cuda-pairs'/row['run'];m=read(run/'manifest.json')
        windows.append((m['started_unix'],m['started_unix']+row['accounting']['session_allocated_worker_seconds']))
    utilization=[]
    with trace.open() as stream:
        for row in csv.DictReader(stream):
            row={k.strip():v.strip() for k,v in row.items()}
            try:
                t=datetime.strptime(row['timestamp'],'%Y/%m/%d %H:%M:%S.%f').replace(tzinfo=timezone.utc).timestamp()
                if any(a<=t<=b for a,b in windows): utilization.append(float(row['utilization.gpu [%]'].replace('%','').strip()))
            except (ValueError,KeyError):pass
    result['cuda_utilization_diagnostic']={'samples':len(utilization),'mean_percent':statistics.mean(utilization) if utilization else None,
        'max_percent':max(utilization) if utilization else None,'note':'Coarse nvidia-smi utilization; GPU idle periods remain charged. CUDA event spans include host dispatch gaps.'}
    result['limits']=['Policies remain weak; no strong reference claim','Commercial-game fidelity not independently established',
      'Small three-seed pilot, not a long reference campaign','Busy profile uses explicit scripted turn-eight continuations; strength evaluation starts from turn one',
      'Completed games metric counts terminal evaluated-seat placements in eight-seat lobbies, not necessarily all eight seats surviving to final lobby resolution']
    (E/'gpu-summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k in ['cuda_performance','learning','learning_aggregate','provider_billing']},indent=2))


if __name__=='__main__':main()
