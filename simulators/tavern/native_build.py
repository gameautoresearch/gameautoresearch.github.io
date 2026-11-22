"""Freeze source-bound stock-NVPTX compilation in a fresh evidence directory."""
import argparse,hashlib,json,os,shutil,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
DEVICE=ROOT/'integrations/tavern/device'
TOOLCHAIN='nightly-2026-04-02-aarch64-apple-darwin'
LINKER=Path('/Users/K.Z./.rustup/toolchains')/TOOLCHAIN/'lib/rustlib/aarch64-apple-darwin/bin/self-contained/llvm-bitcode-linker'


def inputs():
    rows={}
    for folder in ['core','kernels','state-kernels','wire','native']:
        for p in sorted((DEVICE/folder).rglob('*')):
            if not p.is_file() or any(part.startswith('target') or part=='__pycache__' for part in p.parts):continue
            rows[str(p.relative_to(DEVICE))]=hashlib.sha256(p.read_bytes()).hexdigest()
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--inline-threshold',type=int,default=5)
    p.add_argument('--rust-opt',choices=['2','3'],default='2')
    p.add_argument('--ordered-tables',action='store_true')
    p.add_argument('--state-inline-threshold',type=int)
    p.add_argument('--state-rust-opt',choices=['2','3']);a=p.parse_args()
    if any(value is not None and not 0<=value<=1000 for value in [a.inline_threshold,a.state_inline_threshold]):
        raise ValueError('bounded compiler inline threshold required')
    a.out.mkdir(parents=True,exist_ok=False);out=a.out.resolve();before=inputs()
    (out/'inputs.json').write_text(json.dumps(before,indent=2)+'\n')
    def flags_for(threshold):
        return f'-Ctarget-cpu=sm_100 -Cpanic=abort -Zcrate-attr=feature(abi_ptx) -Cllvm-args=-inline-threshold={threshold} -Cllvm-args=-inlinehint-threshold={threshold} -Clinker={DEVICE}/native/linker.py -Clink-arg=--lto'
    main=('tavern-device-kernels','tavern_device_kernels.ptx','tavern.ptx')
    state=('tavern-device-state-kernels','tavern_device_state_kernels.ptx','tavern-state.ptx')
    phases=[([main,state],a.rust_opt,a.inline_threshold)]
    if a.state_inline_threshold is not None or a.state_rust_opt is not None or a.ordered_tables:
        phases=[([main],a.rust_opt,a.inline_threshold),
                ([state],a.state_rust_opt or a.rust_opt,a.state_inline_threshold if a.state_inline_threshold is not None else a.inline_threshold)]
    start=time.perf_counter();rc=0;commands=[];modules={};error=None
    with (out/'build.log').open('x') as log:
        for selected,opt,threshold in phases:
            flags=flags_for(threshold)
            command=['cargo','+'+TOOLCHAIN,'build','--manifest-path',str(DEVICE/'native/Cargo.toml')]
            for package,_,_ in selected:command+=['-p',package]
            features='arena,frozen_rules' if main in selected else 'arena'
            # Only simulation requires constructor-free table reads. Checkpoint
            # IO retains canonical once getters and never imports script hooks.
            if a.ordered_tables and main in selected:features+=',ordered_tables'
            command+=['--features',features,'--target','nvptx64-nvidia-cuda','-Zbuild-std=core,alloc','--release']
            commands.append(command);phase_start=time.perf_counter()
            try:
                proc=subprocess.run(command,cwd=ROOT,env=dict(os.environ,CARGO_TARGET_DIR=str(DEVICE/'native/target-frozen'),CARGO_PROFILE_RELEASE_OPT_LEVEL=opt,
                    TAVERN_NATIVE_REAL_LINKER=str(LINKER),TAVERN_NATIVE_LINK_LOG=str(out/'link.jsonl'),RUSTFLAGS=flags),stdout=log,stderr=subprocess.STDOUT,timeout=600)
                rc=proc.returncode
            except subprocess.TimeoutExpired as exc:rc=124;error=str(exc)
            if rc:break
            for _,src,dst in selected:
                source=DEVICE/'native/target-frozen/nvptx64-nvidia-cuda/release/deps'/src
                shutil.copyfile(source,out/dst);raw=(out/dst).read_bytes()
                modules[dst]={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),
                    'rustflags':flags,'rust_opt':opt,'inline_threshold':threshold,
                    'command':command,'compilation_phase_seconds':time.perf_counter()-phase_start}
    result={'command':commands[0] if len(commands)==1 else commands,'rustflags':flags_for(a.inline_threshold),'returncode':rc,'local_setup_compile_seconds':time.perf_counter()-start,
        'rust_opt':a.rust_opt,'inline_threshold':a.inline_threshold,'ordered_tables':a.ordered_tables,
        'toolchain':subprocess.check_output(['rustc','+'+TOOLCHAIN,'--version','--verbose'],text=True),'inputs_unchanged':inputs()==before,
        'gpu_runtime_qualified':False,'backend':'stock LLVM NVPTX; complete shared rule source, frozen benchmark configuration'}
    result['modules']=modules
    if error:result['error']=error
    (out/'build.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
    if rc or not result['inputs_unchanged']:raise SystemExit(1)


if __name__=='__main__':main()
