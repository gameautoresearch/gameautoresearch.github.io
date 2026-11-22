"""Actual CUDA complete-rule differential probe against the pinned CPU oracle."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import traceback

import numpy as np
import torch
import taverngym as oracle
from device_env import DeviceEnv

RULES={'num_players':8,'content':'Implemented','turn_action_caps':[30],
       'max_actions_per_turn':30,'max_turns':100,'record_log':False}


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--managed-out',action='store_true');a=p.parse_args()
    if a.managed_out and not (a.out/'worker-start.json').is_file():raise RuntimeError('missing worker claim')
    a.out.mkdir(parents=True,exist_ok=a.managed_out);started=time.perf_counter();checks=[]
    try:
        import shutil
        for module in sorted(Path('/workspace/integrations/tavern/device').glob('tavern*.ptx'))+sorted(Path('/workspace/integrations/tavern/device').glob('tavern*.cubin')):
            shutil.copyfile(module,a.out/module.name)
        torch.set_num_threads(1)
        print(json.dumps({'stage':'complete-rule CUDA module initialization'}),flush=True)
        from device_divergence_probe import qualify as divergence
        divergent_control=divergence(a.out)
        for seats in [[0],[3,0],list(range(8))]:
            cpu=oracle.BatchEnv(2,seed=12345,controlled=seats,config=RULES,threads=1)
            env=DeviceEnv(2,seed=12345,controlled=seats,config=RULES)
            try:
                import ctypes as ct
                buf=torch.tensor([11,29,47],dtype=torch.uint8,device='cuda')
                output=torch.empty(2,dtype=torch.int64,device='cuda')
                env.launch('tavern_abi_test',1,[buf,ct.c_uint32(3),output])
                assert output.cpu().tolist()==[2,29], 'opaque-pointer ABI regression'
                print(json.dumps({'limits':env.limits,'ptx_sha256':env.ptx_sha256,'modules':env.module_receipts,'free_total_cuda_bytes':torch.cuda.mem_get_info()}),flush=True)
                if seats==[0]:
                    print(json.dumps({'kernel_resources':env.kernel_resources(),
                        'streaming_multiprocessors':torch.cuda.get_device_properties(0).multi_processor_count}),flush=True)
                initial=env.save()
                for gi,state in enumerate(initial):
                    gpu_state=json.loads(state);cpu_state=json.loads(cpu.save(gi))
                    assert gpu_state['game']==cpu_state['game'],('initial full state/RNG',seats,gi)
                print(json.dumps({'initial_full_state_rng_identical':True,'seats':seats}),flush=True)
                if env.split:
                    baseline=env.save()
                    for corruption in ['parse_type','card_table','implicit_rule']:
                        damaged=list(baseline);item=json.loads(damaged[-1])
                        if corruption=='parse_type':item['seed']='invalid integer type'
                        elif corruption=='card_table':item['game']['card_table']=int(item['game']['card_table'])^1
                        else:item['game']['cfg']['buy_cost']+=1
                        damaged[-1]=json.dumps(item)
                        try:env.load(damaged)
                        except ValueError:pass
                        else:raise AssertionError(('invalid checkpoint accepted',corruption))
                        assert [json.loads(s) for s in env.save()]==[json.loads(s) for s in baseline],corruption
                    print(json.dumps({'transactional_invalid_import_preserved':True,'seats':seats}),flush=True)
                natural=0;digest=hashlib.sha256()
                for step in range(180):
                    active=np.array([True,step%11!=0])
                    env.active.copy_(torch.from_numpy(active.astype(np.uint8)).cuda())
                    x=cpu.observe(copy=False,active_games=active);yg=env.observe()
                    y={k:v.cpu().numpy() for k,v in yg.items()}
                    assert np.array_equal(x['ready'],y['ready']),('ready',seats,step)
                    ids=np.flatnonzero(x['ready'])
                    for key in ['action_mask','cards','features']:
                        if not np.array_equal(x[key][ids],y[key][ids]):
                            maximum=float(np.max(np.abs(x[key][ids].astype(float)-y[key][ids].astype(float))))
                            raise AssertionError((key,seats,step,'max difference',maximum))
                    actions=np.zeros(cpu.n_slots,np.int64)
                    for s in ids:
                        g,j=divmod(int(s),len(seats))
                        actions[s]=cpu.game(g).bot_action(seats[j])
                    ar,ad,ai=cpu.step(actions,active_games=active)
                    if step==0:print(json.dumps({'first_step_actions':actions.tolist(),'seats':seats}),flush=True)
                    br,bd,bi=env.step(torch.from_numpy(actions).cuda())
                    br=br.cpu().numpy();bd=bd.cpu().numpy();bi={k:v.cpu().numpy() for k,v in bi.items()}
                    assert np.array_equal(ar,br) and np.array_equal(ad,bd),('reward/done',seats,step)
                    assert all(np.array_equal(ai[k],bi[k]) for k in ai),('terminal metadata',seats,step)
                    natural+=int((bi['stop_reason']==1).sum())
                    digest.update(ar.tobytes());digest.update(actions.tobytes())
                    if step%31==0:
                        states=env.save()
                        for g,s in enumerate(states):
                            device=json.loads(s);reference=json.loads(cpu.save(g))
                            assert device['game']==reference['game'],('full state/RNG',seats,step,g)
                            assert device['index']==reference['index']
                            assert device['finished'][:len(seats)]==reference['finished']
                        env.load(states)
                        assert [json.loads(s) for s in env.save()]==[json.loads(s) for s in states]
                        print(json.dumps({'seats':seats,'step':step,'natural_slot_episodes':natural}),flush=True)
                checks.append({'seats':seats,'steps':180,'natural_slot_episodes':natural,
                    'actions_results_sha256':digest.hexdigest(),'views_masks_states_rng_resume_identical':True,
                    'allocator':env.allocator_stats()})
                (a.out/'progress.json').write_text(json.dumps(checks,indent=2)+'\n')
            finally:env.close()
        from device_fixtures import qualify as fixtures
        hard=fixtures(RULES)
        (a.out/'hard-mechanics.json').write_text(json.dumps(hard,indent=2)+'\n')
        from device_learning_probe import qualify as neural
        learning=neural(a.out)
        from device_wide_probe import qualify as wide_replay
        wide=wide_replay(a.out,RULES)
        from device_training import runtime_options
        result={'passed':True,'checks':checks,'wide_replay':wide,'divergent_control':divergent_control,'hard_mechanics':hard,'neural':learning,'runtime_options':runtime_options(),'gpu':torch.cuda.get_device_name(),
            'wall_seconds':time.perf_counter()-started,'scope':'Complete shared 37.0.2 rules; initial/busy scripted actions, resets, exact state/RNG and view/mask differential, adversarial mechanics, CUDA neural updates and source-bound resume. Performance qualification is separate.'}
        (a.out/'differential.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result),flush=True)
    except BaseException:
        (a.out/'failure.txt').write_text(traceback.format_exc());raise


if __name__=='__main__':main()
