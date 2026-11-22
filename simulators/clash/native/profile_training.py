"""Actual-device phase profile; full worker wall time remains charged externally."""
from pathlib import Path
import argparse,json,time,sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from train import Trainer

def main():
    import torch
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    c=json.loads(a.config.read_text());started=time.perf_counter();t=Trainer(c,Path(__file__).parent/'generated-v2/task-data.json');rows=[]
    try:
        for update in range(24):
            torch.cuda.synchronize();start=time.perf_counter();events=[torch.cuda.Event(enable_timing=True) for _ in range(3)];events[0].record()
            data=t.collect();events[1].record();metrics,steps=t.learn(data);events[2].record();torch.cuda.synchronize()
            row={'update':update+1,'wall_seconds':time.perf_counter()-start,'collect_device_timeline_ms':events[0].elapsed_time(events[1]),'learn_device_timeline_ms':events[1].elapsed_time(events[2]),'samples':data['action'].numel(),'losses':metrics.cpu().tolist(),'counters':t.counters.cpu().tolist()};rows.append(row);print(json.dumps(row),flush=True)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=True) as prof:
            with torch.profiler.record_function('native_collect'):data=t.collect()
            with torch.profiler.record_function('native_learn'):t.learn(data)
            torch.cuda.synchronize()
        prof.export_chrome_trace(str(a.out/'trace.json'))
        (a.out/'operators-cuda.txt').write_text(prof.key_averages().table(sort_by='self_cuda_time_total',row_limit=100))
        (a.out/'operators-cpu.txt').write_text(prof.key_averages().table(sort_by='self_cpu_time_total',row_limit=100))
        operators=[{key:getattr(event,key,None) for key in ('key','count','self_cpu_time_total','self_device_time_total','cpu_time_total','device_time_total')} for event in prof.key_averages()]
        result={'config':t.c,'source_binding':t.binding,'phase_timings':rows,'profile_operators':operators,'process_seconds':time.perf_counter()-started,'gpu':torch.cuda.get_device_name(),'peak_cuda_bytes':torch.cuda.max_memory_allocated(),'profiler_timing_separate':True};(a.out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    finally:t.close()
if __name__=='__main__':main()
