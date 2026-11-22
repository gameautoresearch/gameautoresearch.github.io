"""Analyze retained private receipts/checkpoints close to storage, returning small summaries."""
from pathlib import Path
import json,statistics

def analyze(folder):
    import torch
    folder=Path(folder)
    summary=json.loads((folder/'summary.json').read_text())
    result={'directory':folder.name,'comparisons':{},'artifact_files':sum(1 for x in folder.rglob('*') if x.is_file())}
    # Fixed-size backend comparisons. Scaling trajectories intentionally differ.
    pairs=[]
    for prefix in ('sparse','deferred','both','graph','graph-fused'):
        for trial in range(3):
            if f'baseline-{trial}' in summary and f'{prefix}-{trial}' in summary:pairs.append((f'baseline-{trial}',f'{prefix}-{trial}'))
    for left,right in pairs:
        a=torch.load(folder/left/'checkpoint.pt',map_location='cpu',weights_only=False)
        b=torch.load(folder/right/'checkpoint.pt',map_location='cpu',weights_only=False)
        comparison={'engine_digest_equal':a['session']['observation_sha256']==b['session']['observation_sha256'],'histories_equal':a['session']['histories']==b['session']['histories'],'cpu_rng_equal':torch.equal(a['torch_rng'],b['torch_rng']),'numpy_rng_equal':a['session']['rng']==b['session']['rng'],'cuda_rng_equal':all(torch.equal(x,y) for x,y in zip(a.get('cuda_rng') or [],b.get('cuda_rng') or [])),'model_bit_identical':all(torch.equal(a['model'][k],b['model'][k]) for k in a['model']),'max_model_error':max(float((a['model'][k]-b['model'][k]).abs().max()) for k in a['model'] if a['model'][k].is_floating_point())}
        result['comparisons'][right]=comparison
    for name in ('cuda-parity/result.json','native-parity/result.json','cuda-resume/result.json','cgroup-limits.json','function-receipt.json','function-settlement.json','simulator/result.json'):
        if (folder/name).exists():result[name]=json.loads((folder/name).read_text())
    result['cpu_model_lines']=[line for line in (folder/'cpuinfo.txt').read_text().splitlines() if line.startswith('model name')][:1] if (folder/'cpuinfo.txt').exists() else []
    result['memory_peak_kib']={}
    for path in folder.glob('*-memory.txt'):
        for line in path.read_text().splitlines():
            if 'Maximum resident set size' in line:result['memory_peak_kib'][path.stem]=int(line.rsplit(':',1)[-1])
    return result
