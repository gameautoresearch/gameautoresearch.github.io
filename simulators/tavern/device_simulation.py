"""Complete GPU replay of the SAME frozen original-simulator trace.

Native and GPU timings are separate from neural PPO timings. Validation transfers
and hashes every player-visible row; timed replay keeps simulation/view/action data
on CUDA, includes input transfer, initialization and final full checkpoint IO.
"""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from device_env import DeviceEnv
from device_state_schema import native_episode


def replay(trace,verify=False,state_out=None):
    t=json.loads(trace.read_text());digest=hashlib.sha256();start=time.perf_counter()
    env=DeviceEnv(t['n_games'],seed=t['seed'],controlled=(0,),config=t['rules'])
    initialized=time.perf_counter()-start
    at=time.perf_counter();actions=torch.tensor(t['actions'],dtype=torch.int64,device='cuda')
    torch.cuda.synchronize();transferred=time.perf_counter()-at
    totals=torch.zeros(3,dtype=torch.int64,device='cuda');events=[]
    try:
        for a in actions:
            before=torch.cuda.Event(enable_timing=True);middle=torch.cuda.Event(enable_timing=True);after=torch.cuda.Event(enable_timing=True)
            before.record();o=env.observe();middle.record()
            if verify:
                h={k:v.cpu().numpy() for k,v in o.items()};ready=h['ready'];ids=np.flatnonzero(ready)
                digest.update(np.asarray(ready,dtype=np.uint8).tobytes())
                for key in ['features','cards','action_mask']:digest.update(h[key][ids].tobytes())
            r,d,info=env.step(a);after.record();events.append((before,middle,after))
            totals+=torch.stack((d.sum(),info['rejected'].sum(),(d&info['truncated']).sum()))
            if verify:
                for v in [r,d,info['placement'],info['rejected']]:digest.update(v.cpu().numpy().tobytes())
        torch.cuda.synchronize();at=time.perf_counter()
        states=[native_episode(json.loads(s),seed=t['seed'],controlled=(0,)) for s in env.save()]
        checkpoint=time.perf_counter()-at;protocol=time.perf_counter()-start
        natural,rejected,faults=totals.cpu().tolist()
        if rejected or faults:raise AssertionError(('invalid GPU scripted trace',rejected,faults))
        if natural!=t['natural_episodes']:raise AssertionError(('natural outcome count changed',natural,t['natural_episodes']))
        state_hash=hashlib.sha256(json.dumps(states,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        if state_out is not None:
            state_out.write_text(json.dumps(states,sort_keys=True,separators=(',',':'))+'\n')
        return {'trace_sha256':hashlib.sha256(trace.read_bytes()).hexdigest(),'module_receipts':env.module_receipts,
            'gpu':torch.cuda.get_device_name(),'n_games':t['n_games'],'verify':verify,'initialization_seconds':initialized,
            'input_transfer_seconds':transferred,'checkpoint_seconds':checkpoint,'complete_protocol_seconds':protocol,
            'observation_cuda_seconds':sum(a.elapsed_time(b) for a,b,c in events)/1000,
            'simulation_cuda_seconds':sum(b.elapsed_time(c) for a,b,c in events)/1000,
            'decisions':t['decisions'],'decisions_per_complete_protocol_second':t['decisions']/protocol,
            'natural_episodes':natural,'rejected':rejected,'faults':faults,
            'views_masks_results_sha256':digest.hexdigest() if verify else None,'final_state_rng_sha256':state_hash,
            'allocator':env.allocator_stats(),'wall_seconds':time.perf_counter()-start,
            'scope':'Frozen original full-eight-player/all-active scripted trace; seven internal scripted bots remain in GPU rules. Timing includes startup/initialization/input H2D/final checkpoint; neural learning is separately measured.'}
    finally:env.close()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--trace',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--verify',action='store_true');a=p.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    a.out.write_text(json.dumps(replay(a.trace,a.verify,a.out.with_suffix('.states.json') if a.verify else None),indent=2)+'\n')


if __name__=='__main__':main()
