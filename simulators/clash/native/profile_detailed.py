"""Actual-device profile of the qualified native pipeline; launches no allocation.

A diagnostic profile is separate from uninstrumented end-to-end comparisons.
All setup, warmup, profiling and output finalization remain allocation time.
"""
from pathlib import Path
import argparse, json, sys, time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def run(config, out, warmup_updates=24, measured_updates=4, chrome_trace=False):
    import torch
    from train import Trainer, archive_owned_source
    if torch.cuda.device_count() != 1:
        raise ValueError('exactly one actual CUDA device required')
    if warmup_updates < 1 or measured_updates < 1:
        raise ValueError('positive explicit warmup and measured updates required')
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    config = json.loads(Path(config).read_text())
    if config.get('backend') not in ('native-cuda', 'native-cpu-cuda'):
        raise ValueError('profile requires native simulation plus CUDA neural training')
    started = time.perf_counter(); trainer = Trainer(config, HERE / 'generated-v2/task-data.json')
    source = archive_owned_source(out, trainer.binding['source'])
    rows = []; restored = []
    def update(label):
        torch.cuda.synchronize()
        events = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
        start = time.perf_counter(); events[0].record()
        data = trainer.collect(); events[1].record()
        metrics, steps = trainer.learn(data); events[2].record()
        torch.cuda.synchronize()
        row = dict(phase=label, update=trainer.update, rows=data['action'].numel(),
                   optimizer_steps=steps, wall_seconds=time.perf_counter()-start,
                   collect_timeline_ms=events[0].elapsed_time(events[1]),
                   learn_timeline_ms=events[1].elapsed_time(events[2]),
                   metrics=metrics.cpu().tolist(), counters=trainer.counters.cpu().tolist(),
                   bucket_usage=trainer.bucket_usage.cpu().tolist())
        rows.append(row)
        print(json.dumps({k: row[k] for k in ('phase','update','rows','wall_seconds','collect_timeline_ms','learn_timeline_ms')}), flush=True)
        return row
    try:
        for _ in range(warmup_updates): update('warmup')
        for _ in range(measured_updates): update('uninstrumented-diagnostic')
        # Labels affect profiling only, never the uninstrumented phase above.
        def label_method(obj, name, label):
            original = getattr(obj, name)
            def labelled(*args, **kwargs):
                with torch.profiler.record_function(label):
                    return original(*args, **kwargs)
            setattr(obj, name, labelled); restored.append((obj, name, original))
        for method in ('step','read','reset'):
            label_method(trainer.env, method, 'clash_sim_'+method)
        if hasattr(trainer, '_bucket_actor'):
            label_method(trainer._bucket_actor, 'sample', 'clash_actor_sample')
        else:
            label_method(trainer.model, 'sample_batch', 'clash_actor_sample')
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                               torch.profiler.ProfilerActivity.CUDA],
                                    record_shapes=True, profile_memory=False) as prof:
            with torch.profiler.record_function('clash_collect'):
                data = trainer.collect()
            with torch.profiler.record_function('clash_learn'):
                trainer.learn(data)
            torch.cuda.synchronize()
        for obj, name, original in reversed(restored): setattr(obj, name, original)
        restored.clear()
        if chrome_trace: prof.export_chrome_trace(str(out/'trace.json'))
        def operator(event):
            return {key: getattr(event,key,None) for key in
                    ('key','input_shapes','count','self_cpu_time_total','self_device_time_total',
                     'cpu_time_total','device_time_total')}
        operators = [operator(e) for e in prof.key_averages()]
        shapes = [operator(e) for e in prof.key_averages(group_by_input_shape=True)]
        kernels = {}
        for event in prof.events():
            if str(event.device_type).endswith('CUDA'):
                key = event.name
                entry = kernels.setdefault(key, dict(name=key, count=0, device_microseconds=0.))
                entry['count'] += 1
                entry['device_microseconds'] += float(event.device_time_total)
        # Reporting-only host reads occur after all profiled/timed phases.
        counts=data['obs'][:,trainer.schema.header['entity_count'][0]].to(torch.int64)
        active=data['mask'][:,1:].bool().any(1)
        population=dict(rows=counts.numel(), mean_entities=counts.float().mean().item(),
                        maximum_entities=counts.max().item(),
                        entity_histogram=torch.bincount(counts,minlength=201).cpu().tolist(),
                        nonforced_rows=active.sum().item(),
                        bounds=[32,64,96,200],
                        full_public_entity_capacity_retained=True)
        (out/'operators-device.txt').write_text(prof.key_averages().table(sort_by='self_device_time_total', row_limit=100))
        (out/'operators-shapes.txt').write_text(prof.key_averages(group_by_input_shape=True).table(sort_by='self_device_time_total', row_limit=120))
        result = dict(status='passed', config=trainer.c, source_binding=trainer.binding,
                      source_archive=source, backend_build=trainer.backend_build,
                      auxiliary_builds=trainer.auxiliary_builds, execution_runtime=trainer.execution_runtime,
                      gpu=torch.cuda.get_device_name(), phase_timings=rows,
                      operators=operators, operators_by_shape=shapes,
                      population=population,
                      kernels=sorted(kernels.values(), key=lambda x:x['device_microseconds'], reverse=True),
                      profiled_extra_updates=1, peak_cuda_bytes=torch.cuda.max_memory_allocated(),
                      process_seconds=time.perf_counter()-started,
                      timing_scope='separate diagnostic warm/phase/operator profile; not a matched throughput claim',
                      external_allocation_receipt_authoritative=True)
        (out/'result.json').write_text(json.dumps(result, indent=2)+'\n')
        return result
    finally:
        for obj, name, original in reversed(restored): setattr(obj, name, original)
        trainer.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--warmup-updates', type=int, default=24)
    p.add_argument('--measured-updates', type=int, default=4)
    p.add_argument('--chrome-trace', action='store_true')
    a=p.parse_args(); run(a.config,a.out,a.warmup_updates,a.measured_updates,a.chrome_trace)

if __name__=='__main__': main()
