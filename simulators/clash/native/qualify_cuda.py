"""Qualify actual CUDA transitions against independent native CPU, not timings."""
from pathlib import Path
import hashlib,json,sys,time
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
def main(out):
    import numpy as np,torch
    from runtime import NativeCPU,NativeCUDA
    out=Path(out);out.mkdir(parents=True,exist_ok=False);started=time.perf_counter()
    seeds=[0,1,-1,771];cpu=NativeCPU(seeds);gpu=NativeCUDA(seeds);torch.cuda.synchronize()
    rng=np.random.default_rng(4451);frames=0
    def compare(label):
        nonlocal frames
        arrays=(gpu.observations,gpu.masks,gpu.infos,gpu.rewards)
        refs=(cpu.observations,cpu.masks,cpu.infos,cpu.rewards)
        for name,a,b in zip(('observations','masks','infos','rewards'),arrays,refs):
            v=a.cpu().numpy()
            if not np.array_equal(v,b):
                idx=np.argwhere(v!=b);torch.save(gpu.states.cpu(),out/'state-divergence.pt');np.savez_compressed(out/'divergence.npz',native=b,cuda=v)
                (out/'divergence.json').write_text(json.dumps({'frame':frames,'label':label,'array':name,'differences':idx[:50].tolist(),'native_values':[b[tuple(i)].item() for i in idx[:50]],'cuda_values':[v[tuple(i)].item() for i in idx[:50]]},indent=2)+'\n');raise AssertionError(f'CUDA {name} diverged at {label}')
        frames+=len(seeds)
    compare('reset')
    natural_games=0
    for step in range(1300):
        a=np.zeros((len(seeds),2),np.int32)
        if step>=18:
            for lane in range(len(seeds)):
                for seat in range(2):
                    legal=np.flatnonzero(cpu.masks[lane,seat]);a[lane,seat]=int(rng.choice(legal)) if rng.random()<.2 else 0
        cpu.step(a);gpu.step(torch.as_tensor(a,device='cuda'));compare(str(step))
        done=cpu.infos[:,2]>=0;natural_games+=int(done.sum())
        if done.any():
            next_seeds=rng.integers(0,1000000,len(seeds),dtype=np.int32);cpu.reset(next_seeds,done);gpu.reset(torch.as_tensor(next_seeds,device='cuda'),torch.as_tensor(done,device='cuda'));compare('reset-selected')
    # Rebind raw snapshot pointers, replay subsequent actions, and verify exact
    # encoded state after CPU/GPU-local restore. All copies are charged setup/test.
    cs=cpu.state();gs=gpu.state();rngstate=rng.bit_generator.state
    actions=[rng.integers(0,1,size=(len(seeds),2),dtype=np.int32) for _ in range(8)]
    for a in actions:cpu.step(a);gpu.step(torch.as_tensor(a,device='cuda'));compare('resume-before')
    expected=[a.clone() for a in (gpu.observations,gpu.masks,gpu.infos,gpu.rewards)]
    cpu.load(cs);gpu.load(gs)
    for a in actions:cpu.step(a);gpu.step(torch.as_tensor(a,device='cuda'));compare('resume-after')
    for a,b in zip((gpu.observations,gpu.masks,gpu.infos,gpu.rewards),expected):assert torch.equal(a,b)
    engine_result={'status':'passed','frames':frames,'natural_games':natural_games,'cuda_name':torch.cuda.get_device_name(),'cuda_library_sha256':hashlib.sha256(gpu.library.read_bytes()).hexdigest(),'world_bytes':int(gpu.states.shape[1]),'raw_snapshot_resume':True,'scope':'complete native CPU/CUDA outputs and deterministic native resume; Java differential separately retained'}
    (out/'engine-result.json').write_text(json.dumps(engine_result,indent=2)+'\n');print(json.dumps(engine_result),flush=True)
    from train import schema
    from model import Policy
    from gpu_features import qualify_cuda_features
    data=json.loads((HERE/'generated-v2/task-data.json').read_text());policy=Policy(schema(data),len(data['vocabulary']),len(data['buff_names']),width=16,embedding=4).cuda()
    feature_result=qualify_cuda_features(policy,gpu.observations,gpu.masks,output_dir=out/'features')
    result={'status':'passed','natural_games':natural_games,'feature_qualification':feature_result,'frames':frames,'cuda_name':torch.cuda.get_device_name(),'cuda_library':str(gpu.library),'cuda_library_sha256':hashlib.sha256(gpu.library.read_bytes()).hexdigest(),'world_bytes':int(gpu.states.shape[1]),'peak_cuda_bytes':torch.cuda.max_memory_allocated(),'elapsed_setup_compile_qualification_seconds':time.perf_counter()-started,'scope':'CPU/CUDA shared native transitions, complete output parity and raw native resume; Java full-world parity separately required'}
    gpu.close();cpu.close();(out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main(sys.argv[1])
