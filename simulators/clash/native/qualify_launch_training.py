"""Exact training qualification across one-world/thread CUDA launch policies.

Fresh strict-FP32 Policy64/e12 compact/bucket arms, launch1 vs4/8/16/32 and render4.
No cloud/worker provisioning, GPU performance claims or checkpoint-policy bypass.
Run only on an already allocated single GPU. --self-test is CPU-only comparator/
layout qualification, not evidence that actual CUDA launch policies are equal.
"""
from __future__ import annotations
from pathlib import Path
import argparse,copy,hashlib,json,os,shutil,signal,subprocess,sys,time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE));sys.path.insert(0,str(HERE.parent))
from qualify_backend_training import assert_exact,ComparisonError,worker as capture_worker
POLICIES=(1,4,8,16,32)
POINTER_FIELDS=('NativeWorld.spec','NativeWorld.arenaGrid.tiles','NativeWorld.arenaGrid.current','NativeWorld.arenaGrid.previous')

def dump(path,value):Path(path).write_text(json.dumps(value,indent=2)+'\n')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def pointer_layout(out):
    """Independent sizeof/offsetof only; no native world step or byte projection."""
    from build_receipt import compiler_identity
    compiler=shutil.which('clang++') or shutil.which('g++')
    if not compiler:raise RuntimeError('host C++ compiler required for independent four-pointer layout')
    identity=compiler_identity(compiler);source=out/'pointer-layout.cpp';binary=out/'pointer-layout'
    source.write_text('#include <cstddef>\n#include <cstdio>\n#include <type_traits>\n#include "'+str(HERE/'world.hpp')+'"\nint main(){using namespace clash_native;static_assert(std::is_standard_layout_v<NativeWorld>);std::printf("%zu %zu %zu %zu %zu %zu",sizeof(NativeWorld),sizeof(void*),offsetof(NativeWorld,spec),offsetof(NativeWorld,arenaGrid)+offsetof(GridView,tiles),offsetof(NativeWorld,arenaGrid)+offsetof(GridView,current),offsetof(NativeWorld,arenaGrid)+offsetof(GridView,previous));}\n')
    command=[identity['path'],'-std=c++20',str(source),'-o',str(binary)];r=subprocess.run(command,capture_output=True,text=True,timeout=60)
    (out/'pointer-layout.stdout').write_text(r.stdout);(out/'pointer-layout.stderr').write_text(r.stderr);r.check_returncode()
    parts=list(map(int,subprocess.check_output([str(binary.resolve())],text=True,timeout=5).split()))
    if len(parts)!=6:raise ValueError('independent layout did not produce exactly four pointers')
    size,pointer,*offsets=parts
    if pointer not in (4,8) or len(set(offsets))!=4 or offsets!=sorted(offsets) or any(a<0 or a+pointer>size for a in offsets) or any(a+pointer>b for a,b in zip(offsets,offsets[1:])):raise ValueError('invalid or overlapping documented pointer layout')
    result={'world_bytes':size,'pointer_bytes':pointer,'pointer_fields':dict(zip(POINTER_FIELDS,offsets)),'compiler':identity,'command':command,'layout_source_sha256':sha(source),'layout_binary_sha256':sha(binary),'owning_world_header_sha256':sha(HERE/'world.hpp'),'normalization':'only these four complete address spans in comparison copies; raw artifacts/checkpoints unchanged'}
    dump(out/'pointer-layout.json',result);return result

def normalized_image(image,layout):
    if len(image)!=layout['world_bytes']:raise ComparisonError({'path':'engine_state','reason':'raw World bytes differ from independently compiled current ABI','actual':len(image),'expected':layout['world_bytes']})
    result=bytearray(image)
    for offset in layout['pointer_fields'].values():result[offset:offset+layout['pointer_bytes']]=b'\0'*layout['pointer_bytes']
    return result

def raw_equal(actual,expected,layout,label):
    """Losslessly decode every byte; exactly four explicit address exclusions."""
    import numpy as np
    from snapshots import decode_engine_state
    a=decode_engine_state(actual);b=decode_engine_state(expected)
    if len(a)!=len(b):raise ComparisonError({'path':label,'reason':'raw lane count differs','actual':len(a),'expected':len(b)})
    raw_counts=[];raw_total=0;excluded=[]
    allowed=set(i for offset in layout['pointer_fields'].values() for i in range(offset,offset+layout['pointer_bytes']))
    for lane,(aa,bb) in enumerate(zip(a,b)):
        na=normalized_image(aa,layout);nb=normalized_image(bb,layout)
        av=np.frombuffer(aa,dtype=np.uint8);bv=np.frombuffer(bb,dtype=np.uint8);where=np.flatnonzero(av!=bv)
        raw_counts.append(len(where));raw_total+=len(where)
        if na!=nb:
            unknown=[int(i) for i in where if int(i) not in allowed]
            raise ComparisonError({'path':label,'lane':lane,'reason':'raw semantic/unused-capacity/RNG/padding bytes differ outside four documented pointers','different_raw_bytes':len(where),'different_nonpointer_bytes':len(unknown),'first_nonpointer_offsets':unknown[:24],'actual_bytes':[int(av[i]) for i in unknown[:24]],'expected_bytes':[int(bv[i]) for i in unknown[:24]],'four_pointer_fields':layout['pointer_fields']})
        if len(where):excluded.append({'lane':lane,'different_address_bytes':len(where),'fields':[name for name,off in layout['pointer_fields'].items() if any(off<=int(i)<off+layout['pointer_bytes'] for i in where)]})
    return {'status':'passed','lanes':len(a),'world_bytes':layout['world_bytes'],'raw_differing_bytes_per_lane':raw_counts,'raw_differing_bytes_total':raw_total,'documented_pointer_differences':excluded,'four_pointer_fields':layout['pointer_fields'],'every_other_byte_exact':True,'unknown_or_padding_normalization':False,'stored_raw_images_changed':False}

def manifest_equal(actual,expected,actual_policy,expected_policy=1):
    """Only explicit launch-thread metadata differences are expected/accepted."""
    for value,policy in ((actual,actual_policy),(expected,expected_policy)):
        config=value['config'];runtime=value['execution_runtime'];p=runtime.get('engine_policy')
        assert_exact(config['backend'],'native-cuda','backend stays native-cuda')
        assert_exact(config['cuda_launch_threads'],policy,'declared launch policy')
        assert_exact(config['cuda_render_mode'],4,'render mode stays4')
        assert_exact(p,{'cuda_launch_threads':policy,'cuda_render_mode':4},'runtime launch/render policy matches exact config')
        for name,want in {'precision':'fp32','compact_embedding_grad':True,'bucket_actor':True,'bucket_learner':True,'width':64,'embedding':12,'packed_head':False,'compile_loss':False,'rollout_graph':False}.items():assert_exact(config[name],want,'fixed recipe.'+name)
    # Same native binary and neural helpers, not merely same rule/source pin.
    for key in ('format','source_binding','metadata','backend_build','auxiliary_builds','gpu','initial_seed_observation'):assert_exact(actual[key],expected[key],'shared manifest.'+key)
    ac=copy.deepcopy(actual['config']);bc=copy.deepcopy(expected['config']);a_launch=ac.pop('cuda_launch_threads');b_launch=bc.pop('cuda_launch_threads');assert_exact(ac,bc,'entire PPO recipe except explicit launch threads')
    ar=copy.deepcopy(actual['execution_runtime']);br=copy.deepcopy(expected['execution_runtime']);a_policy=ar.pop('engine_policy');b_policy=br.pop('engine_policy');assert_exact(ar,br,'entire execution runtime except recorded launch policy')
    extra=set(actual)-{'config','execution_runtime','format','source_binding','metadata','backend_build','auxiliary_builds','gpu','initial_seed_observation'}
    assert_exact(extra,set(expected)-{'config','execution_runtime','format','source_binding','metadata','backend_build','auxiliary_builds','gpu','initial_seed_observation'},'manifest key schema')
    for key in extra:assert_exact(actual[key],expected[key],'shared extra manifest.'+key)
    return {'baseline_launch_threads':b_launch,'candidate_launch_threads':a_launch,'render_mode':4,'expected_config_differences':{'cuda_launch_threads':[b_launch,a_launch]},'expected_runtime_differences':{'engine_policy':[b_policy,a_policy]},'same_actual_native_binary_and_helper_receipts':True,'no_checkpoint_binding_rewrite_or_cross_policy_resume':True}

def compare_control(folder,updates,policies,layout):
    import torch
    baseline=folder/'launch1';reference=json.loads((baseline/'manifest.json').read_text());comparisons=[]
    for policy in policies:
        arm=folder/('launch'+str(policy));manifest=json.loads((arm/'manifest.json').read_text());meta=manifest_equal(manifest,reference,policy);records=[]
        for stage in ['initial']+[f'update-{i+1:02d}' for i in range(updates)]:
            a=torch.load(arm/(stage+'.pt'),map_location='cpu',weights_only=False);b=torch.load(baseline/(stage+'.pt'),map_location='cpu',weights_only=False)
            aa=dict(a);bb=dict(b);image_a=aa.pop('engine_state');image_b=bb.pop('engine_state');assert_exact(aa,bb,f'launch{policy}.{stage}')
            records.append({'stage':stage,'all_rollout_PPO_model_Adam_RNG_public_and_counters_exact':True,'raw_world':raw_equal(image_a,image_b,layout,f'launch{policy}.{stage}.raw_world')})
        a=torch.load(arm/'continuation.pt',map_location='cpu',weights_only=False);b=torch.load(baseline/'continuation.pt',map_location='cpu',weights_only=False);aa=dict(a);bb=dict(b);ia=aa.pop('final_engine_state');ib=bb.pop('final_engine_state');assert_exact(aa,bb,f'launch{policy}.32step continuation')
        summary=json.loads((arm/'worker-result.json').read_text());base=json.loads((baseline/'worker-result.json').read_text());assert_exact(summary,base,'identical completed counters/worker summary')
        comparisons.append({'status':'passed','manifest':meta,'training_records':records,'continuation':{'fixed_actions':32,'public_actions_done_resets_allRNG_exact':True,'final_raw_world':raw_equal(ia,ib,layout,f'launch{policy}.continuation.final_raw_world')},'actual_counters':summary['counters'],'actual_bucket_usage':summary['bucket_usage']})
    return {'status':'passed','policies':policies,'baseline':1,'comparisons':comparisons,'raw_normalization':'only four documented rebound address spans; every other byte exact','all_raw_artifacts_retained':True}

def configuration(base,policy,args,gpu,limit):
    c=copy.deepcopy(base)
    c.update(backend='native-cuda',device='cuda',seed=args.seed,envs=args.envs,threads=args.cpu_threads,torch_threads=1,width=64,embedding=12,updates=args.updates,rollout_steps=args.rollout_steps,minibatch=args.minibatch,epochs=2,budget_seconds=args.budget_seconds,max_ticks=limit,opponent_mode='selfplay',warmstart_steps=0,reward_shaping=0.,target_kl=0.,precision='fp32',compact_embedding_grad=True,bucket_actor=True,bucket_learner=True,learner_graph=False,rollout_graph=False,graph_sampling=True,optimizer_fused=True,gae_kernel='triton',compile_loss=False,packed_head=False,cuda_launch_threads=policy,cuda_render_mode=4)
    c['hardware_allocation']={'gpu_model':gpu,'gpu_count':1,'cpu_threads':args.cpu_threads,'ram_gib':int(os.getenv('CLASH_RAM_GIB','64')),'qualification':'same GPU/CPU/RAM, fresh sequential launch-policy correctness arms; no timing comparison'}
    return c

def main(args):
    from train import binding
    from snapshots import available_cpu_workers
    import torch
    started=time.monotonic()
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise RuntimeError('exactly one already allocated CUDA GPU required; no GPU/cloud job is provisioned')
    if not 1<=args.cpu_threads<=available_cpu_workers() or min(args.envs,args.updates,args.minibatch,args.budget_seconds,args.subprocess_timeout)<1 or args.rollout_steps not in (16,32):raise ValueError('invalid bounded workload/attached CPU allocation')
    policies=tuple(args.policies)
    if len(policies)<2 or len(set(policies))!=len(policies) or policies[0]!=1 or any(p not in POLICIES for p in policies):raise ValueError('unique launch policies must start with baseline1 and use1/4/8/16/32')
    args.out=args.out.resolve();args.data=args.data.resolve()
    if args.out.is_relative_to(HERE.parent):raise ValueError('outputs must be outside integration source to preserve strict source binding')
    args.out.mkdir(parents=True,exist_ok=False);source=binding(args.data);results=[];failure=None
    try:
        layout=pointer_layout(args.out);assert binding(args.data)==source,'source changed while compiling independent ABI layout'
        gpu=torch.cuda.get_device_name();base=json.loads(args.base_config.read_text())
        for name,limit in [('live',0),('truncated',125)]:
            folder=args.out/name;folder.mkdir()
            for policy in policies:
                c=configuration(base,policy,args,gpu,limit);config=folder/f'launch{policy}-config.json';dump(config,c);target=folder/f'launch{policy}'
                command=[sys.executable,str(Path(__file__).resolve()),'--worker','--config',str(config),'--data',str(args.data),'--out',str(target)]
                remaining=args.budget_seconds-(time.monotonic()-started)
                if remaining<1:raise TimeoutError('whole bounded qualification budget exhausted')
                with (folder/f'launch{policy}.log').open('w') as log:
                    proc=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONDONTWRITEBYTECODE='1',CLASH_CUDA_LAUNCH_THREADS=str(policy),CLASH_CUDA_RENDER_MODE='4'))
                    try:proc.wait(timeout=min(args.subprocess_timeout,remaining))
                    except subprocess.TimeoutExpired:
                        try:os.killpg(proc.pid,signal.SIGKILL)
                        except ProcessLookupError:pass
                        proc.wait();raise TimeoutError(f'launch{policy} child/process group exceeded bounded remaining budget')
                if proc.returncode:raise RuntimeError(f'{name} launch{policy} worker failed; retained log and artifacts preserved')
                assert binding(args.data)==source,'source/data/rules changed between launch policy workers'
            result=compare_control(folder,args.updates,policies,layout);result['control']=name;results.append(result);dump(folder/'comparison.json',result)
        assert binding(args.data)==source,'source/data/rules changed during launch-policy qualifier'
    except BaseException as exc:failure={'type':type(exc).__name__,'message':str(exc),'details':exc.details if isinstance(exc,ComparisonError) else None}
    report={'status':'failed' if failure else 'passed','error':failure,'controls':results,'source_binding':source,'policies':policies,'workload':{'envs':args.envs,'updates':args.updates,'rollout_steps':args.rollout_steps,'minibatch':args.minibatch,'width':64,'embedding':12,'cpu_threads':args.cpu_threads,'precision':'strict FP32','render_mode':4,'controls':['live','max_ticks125']},'cloud_or_GPU_worker_launched':False,'timing_comparison':False,'checkpoint_policy_bypass':False,'scope':'fresh same-source same-binary CUDA arms; exact shared training and public continuation; only four documented raw addresses excluded in comparison copies'}
    dump(args.out/'result.json',report);print(json.dumps({k:v for k,v in report.items() if k not in ('source_binding','controls')}),flush=True)
    if failure:raise RuntimeError(failure['message'])


def self_test(out):
    """CPU static/negative controls ONLY, not launch equivalence evidence."""
    import torch
    from snapshots import encode_engine_state
    out=out.resolve()
    if out.is_relative_to(HERE.parent):raise ValueError('self-test outputs must stay outside integration source')
    out.mkdir(parents=True,exist_ok=False);layout=pointer_layout(out);checks=[]
    def manifest(policy):
        return {'format':1,'config':{'backend':'native-cuda','cuda_launch_threads':policy,'cuda_render_mode':4,'precision':'fp32','compact_embedding_grad':True,'bucket_actor':True,'bucket_learner':True,'width':64,'embedding':12,'packed_head':False,'compile_loss':False,'rollout_graph':False,'seed':81108},'source_binding':{'source':'same'},'metadata':{'schema':'complete'},'backend_build':{'sha':'same exact native binary'},'auxiliary_builds':{'device_branch':'same'},'gpu':'H100 placeholder static fixture','initial_seed_observation':'same observer','execution_runtime':{'engine_policy':{'cuda_launch_threads':policy,'cuda_render_mode':4},'precision':{'mode':'fp32','tf32':False},'torch':'same'}}
    for policy in POLICIES:manifest_equal(manifest(policy),manifest(1),policy)
    checks.append('only explicit launch-thread config/runtime differences accepted across all five policies')
    for label,change in [
        ('render',lambda x:x['config'].__setitem__('cuda_render_mode',3)),
        ('runtime_render',lambda x:x['execution_runtime']['engine_policy'].__setitem__('cuda_render_mode',3)),
        ('precision',lambda x:x['execution_runtime']['precision'].__setitem__('tf32',True)),
        ('source',lambda x:x.__setitem__('source_binding',{'source':'changed'})),
        ('native_binary',lambda x:x.__setitem__('backend_build',{'sha':'different'})),
        ('neural_helper',lambda x:x.__setitem__('auxiliary_builds',{'device_branch':'different'})),
        ('recipe',lambda x:x['config'].__setitem__('seed',9)),
        ('wrong_declared_launch',lambda x:x['config'].__setitem__('cuda_launch_threads',8)),
        ('wrong_runtime_launch',lambda x:x['execution_runtime']['engine_policy'].__setitem__('cuda_launch_threads',8)),
        ('metadata',lambda x:x.__setitem__('metadata',{'schema':'changed'})),
    ]:
        value=manifest(4);change(value)
        try:manifest_equal(value,manifest(1),4)
        except ComparisonError:checks.append('rejected '+label)
        else:raise AssertionError('manifest comparator accepted '+label)
    original=bytes(layout['world_bytes']);different=bytearray(original)
    for off in layout['pointer_fields'].values():different[off:off+layout['pointer_bytes']]=b'\xff'*layout['pointer_bytes']
    encoded=encode_engine_state([original],workers=1);addresses=encode_engine_state([bytes(different)],workers=1);raw_equal(addresses,encoded,layout,'only addresses');assert bytes(different)!=original
    checks.append('four documented address spans accepted in comparison copies while original raw bytes retained')
    permitted=set(i for off in layout['pointer_fields'].values() for i in range(off,off+layout['pointer_bytes']))
    for offset in sorted({i for off in layout['pointer_fields'].values() for i in (off-1,off+layout['pointer_bytes']) if 0<=i<layout['world_bytes'] and i not in permitted}|{layout['world_bytes']-1}):
        mutated=bytearray(different);mutated[offset]^=1
        try:raw_equal(encode_engine_state([bytes(mutated)],workers=1),encoded,layout,'unknown/padding byte')
        except ComparisonError:checks.append('rejected nonpointer raw byte '+str(offset))
        else:raise AssertionError('raw comparator hid semantic/padding byte '+str(offset))
    shared={'rollout':torch.arange(12,dtype=torch.float32),'rng':{'cuda':torch.arange(5,dtype=torch.uint8)},'counter':torch.tensor([2,0],dtype=torch.int64)}
    for key in shared:
        changed=copy.deepcopy(shared)
        if key=='rng':changed[key]['cuda'][0]+=1
        else:changed[key][0]+=1
        try:assert_exact(changed,shared,'static training record')
        except ComparisonError:checks.append('rejected changed '+key)
        else:raise AssertionError('training comparator hid '+key)
    # End-to-end exact comparison path using synthetic files, never GPU data.
    fixture=out/'synthetic-control';fixture.mkdir()
    from qualify_backend_training import save
    for policy in POLICIES:
        folder=fixture/f'launch{policy}';folder.mkdir();dump(folder/'manifest.json',manifest(policy))
        image=encoded if policy==1 else addresses
        for stage in ('initial','update-01'):save(folder/(stage+'.pt'),{'model':{'weight':torch.tensor([1.])},'shared_training':shared,'engine_state':image})
        save(folder/'continuation.pt',{'frames':[{'actions':torch.zeros((1,2),dtype=torch.int32),'public':torch.tensor([1,2,3])} for _ in range(32)],'rng':shared['rng'],'final_engine_state':image})
        dump(folder/'worker-result.json',{'backend':'native-cuda','updates':1,'samples':256,'counters':[0]*11,'bucket_usage':[0]*8})
    report=compare_control(fixture,1,POLICIES,layout);dump(out/'synthetic-comparison.json',report);checks.append('complete five-policy file/continuation/raw comparison path passed with known address differences only')
    target=fixture/'launch8/update-01.pt';original_record=torch.load(target,weights_only=False);changed=copy.deepcopy(original_record);changed['model']['weight'][0]+=torch.finfo(torch.float32).eps;save(target,changed)
    try:compare_control(fixture,1,POLICIES,layout)
    except ComparisonError:checks.append('complete comparison rejects single-FP32-ULP model drift')
    else:raise AssertionError('complete comparison accepted tiny model drift')
    save(fixture/'launch8/rejected-model-update-01.pt',changed)
    changed=copy.deepcopy(original_record);unknown=bytearray(different);unknown[-1]^=1;changed['engine_state']=encode_engine_state([bytes(unknown)],workers=1);save(target,changed)
    try:compare_control(fixture,1,POLICIES,layout)
    except ComparisonError:checks.append('complete comparison rejects one unknown tail/padding byte')
    else:raise AssertionError('complete comparison accepted unknown raw byte')
    save(fixture/'launch8/rejected-raw-update-01.pt',changed);save(target,original_record)
    report={'status':'passed','CPU_static_audit_only':True,'CUDA_executed':False,'cloud_launched':False,'timing_comparison':False,'checks':checks,'pointer_layout':layout,'source_sha256':sha(Path(__file__)),'normalization':'only four known address spans; no unknown byte exemptions'}
    dump(out/'result.json',report);print(json.dumps({'status':'passed','checks':len(checks),'CUDA_executed':False}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);p.add_argument('--data',type=Path,default=HERE/'generated-v2/task-data.json');p.add_argument('--base-config',type=Path,default=HERE.parent/'configs/cuda-learning.json');p.add_argument('--cpu-threads',type=int,default=16);p.add_argument('--envs',type=int,default=4);p.add_argument('--rollout-steps',type=int,choices=(16,32),default=32);p.add_argument('--updates',type=int,default=4);p.add_argument('--minibatch',type=int,default=32);p.add_argument('--policies',type=int,nargs='+',default=POLICIES);p.add_argument('--seed',type=int,default=81108);p.add_argument('--budget-seconds',type=int,default=1200);p.add_argument('--subprocess-timeout',type=int,default=300);p.add_argument('--worker',action='store_true');p.add_argument('--config',type=Path);p.add_argument('--self-test',action='store_true');args=p.parse_args()
    if args.worker:
        if args.config is None:p.error('--worker requires --config')
        c=json.loads(args.config.read_text())
        if c.get('backend')!='native-cuda' or c.get('cuda_launch_threads') not in POLICIES or c.get('cuda_render_mode')!=4:raise ValueError('worker must use explicit nativeCUDA launch1/4/8/16/32 render4')
        capture_worker(args)
    elif args.self_test:self_test(args.out)
    else:main(args)
