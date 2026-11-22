"""Freeze a verified stock-NVPTX build and execute one owned bounded attempt."""
import argparse,json,os,shutil,subprocess,time,hashlib,gzip,lzma
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]


def main():
    preparation_started=time.perf_counter();started_unix=time.time()
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-id',required=True);p.add_argument('--build',type=Path,required=True)
    p.add_argument('--asm-opt',choices=['1','2','3'],default='1')
    p.add_argument('--state-asm-opt',choices=['1','2','3'])
    p.add_argument('--max-registers',choices=['32','48','64','80','96','128','192','255'])
    p.add_argument('--compression',choices=['gzip','xz'],default='gzip')
    p.add_argument('--build-only',action='store_true');a=p.parse_args()
    build=a.build.resolve();receipt=json.loads((build/'build.json').read_text())
    if receipt['returncode'] or not receipt['inputs_unchanged']:raise ValueError('unverified compiler inputs')
    for name,row in receipt['modules'].items():
        if hashlib.sha256((build/name).read_bytes()).hexdigest()!=row['sha256']:raise ValueError('compiled PTX changed')
    inputs=json.loads((build/'inputs.json').read_text())
    for name,sha in inputs.items():
        if hashlib.sha256((ROOT/'integrations/tavern/device'/name).read_bytes()).hexdigest()!=sha:raise ValueError(('source changed since compile',name))
    out=ROOT/'evidence/tavern-implementation/device-port'/a.run_id;out.mkdir(parents=True,exist_ok=False);payload=out/'payload'
    shutil.copytree(ROOT/'integrations/tavern/device',payload/'integrations/tavern/device',ignore=shutil.ignore_patterns('target','target-*','__pycache__','*.ptx','*.cubin'))
    (payload/'compiled').mkdir()
    for name in receipt['modules']:
        if a.compression=='xz':
            with lzma.open(payload/'compiled'/(name+'.xz'),'xb',preset=6) as packed:packed.write((build/name).read_bytes())
        else:
            with (payload/'compiled'/(name+'.gz')).open('xb') as raw:
                with gzip.GzipFile(filename='',fileobj=raw,mode='wb',mtime=0) as packed:packed.write((build/name).read_bytes())
    (payload/'compiled/build.json').write_text(json.dumps(receipt,indent=2)+'\n')
    shutil.copyfile(ROOT/'integrations/tavern/device/native/target-frozen/nvptx64-nvidia-cuda/release/deps/tavern_native_abi.ptx',payload/'compiled/abi.ptx')
    for name in ['native_modal.py','device_probe.py','device_wide_probe.py','device_divergence_probe.py','device_state_schema.py','device_env.py','device_fixtures.py','device_training.py','device_learning_probe.py','model.py','learner_graph.py','rollout.py']:
        shutil.copyfile(ROOT/'integrations/tavern'/name,payload/'integrations/tavern'/name)
    shutil.copytree(build,out/'compiler-receipt')
    record={'run_id':a.run_id,'start_unix':started_unix,'local_freeze_compress_seconds':time.perf_counter()-preparation_started,'compression':a.compression,'status':'started','compiler':receipt,'assembler_opt':a.asm_opt,'state_assembler_opt':a.state_asm_opt or a.asm_opt,'max_registers':a.max_registers,'build_only':a.build_only,'resources':{'gpu':'B200','cpu_cores':4,'ram_gib':16,'timeout_seconds':1800,'retries':0,'warm_containers':0},
        'authorization':'Explicit complete GPU rewrite request; earlier caps superseded; setup/build/failures fully retained.'}
    (out/'attempt.json').write_text(json.dumps(record,indent=2)+'\n');start=time.perf_counter()
    with (out/'modal.log').open('x') as log:
        command=['modal','run','--detach','--env','staging',str(payload/'integrations/tavern/native_modal.py'),'--run-id',a.run_id]
        if a.build_only:command.append('--build-only')
        proc=subprocess.run(command,env=dict(os.environ,TAVERN_NATIVE_PAYLOAD=str(payload),TAVERN_NATIVE_ASM_OPT=a.asm_opt,TAVERN_NATIVE_STATE_ASM_OPT=a.state_asm_opt or a.asm_opt,TAVERN_NATIVE_MAX_REGISTERS=a.max_registers or ''),stdout=log,stderr=subprocess.STDOUT,timeout=3600)
    effective=proc.returncode
    if proc.returncode and not a.build_only:
        from modal_call_wait import reattach
        result=reattach(out/'modal.log',out/'call-recovery.json',a.run_id)
        if result is not None:effective=int(result['returncode']);record['reattached_existing_call']=True
    record.update(returncode=effective,initial_client_returncode=proc.returncode,detached=True,status='succeeded' if not effective else 'failed',client_build_and_call_seconds=time.perf_counter()-start)
    if a.build_only:record.update(status='succeeded_build_only' if not proc.returncode else 'failed',gpu_executed=False)
    (out/'attempt.json').write_text(json.dumps(record,indent=2)+'\n')
    if a.build_only:
        print(json.dumps(record),flush=True)
        if proc.returncode:raise SystemExit(proc.returncode)
        return
    (out/'retained').mkdir()
    result=subprocess.run(['modal','volume','get','autoresearch-tavern-device-evidence-20261108',a.run_id,str(out/'retained'),'--env','staging'],capture_output=True,text=True)
    (out/'retrieval.json').write_text(json.dumps({'returncode':result.returncode,'stderr':result.stderr},indent=2)+'\n');print(json.dumps(record),flush=True)
    if effective:raise SystemExit(effective)


if __name__=='__main__':main()
