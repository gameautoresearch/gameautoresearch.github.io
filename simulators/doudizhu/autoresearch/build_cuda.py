"""Build the supplied frozen L40S (sm_89) engine before leasing a research GPU."""
from pathlib import Path
import hashlib
import json
import subprocess
import time
from .contract import ROOT,GAME,sha,write_json

def main():
    from integrations.doudizhu.cuda.backend import source_hash
    from integrations.doudizhu.native_backend import NativeVectorGames
    # Exact same key, flags, source and checked receipt as cuda.backend.library().
    arch='sm_89';key=hashlib.sha256((source_hash()+arch+'nvcc-O3-cxx17-v2').encode()).hexdigest()
    out=ROOT/'evidence/doudizhu-implementation/cuda-build'/key
    out.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter()
    for name in ('rules.h','engine.cu'):(out/name).write_bytes((GAME/'cuda'/name).read_bytes())
    command=['nvcc','-O3','-std=c++17','--ptxas-options=-v','-shared','-Xcompiler=-fPIC','-arch='+arch,str(out/'engine.cu'),'-o',str(out/'engine.so')]
    result=subprocess.run(command,capture_output=True,text=True)
    (out/'compile.log').write_text(result.stdout+result.stderr)
    if result.returncode:raise RuntimeError('Supplied L40S engine failed to compile')
    write_json(out/'receipt.json',{'source_sha256':source_hash(),'architecture':arch,
        'compiler':subprocess.check_output(['nvcc','--version'],text=True),'command':command,
        'binary_sha256':sha(out/'engine.so'),'compile_seconds':time.perf_counter()-started,
        'scope':'operator-provided frozen environment image build; no GPU attached'})
    (out/'build.lock').touch();(out/'build.lock').chmod(0o666)
    native=NativeVectorGames(1);native.close()
    for lock in (ROOT/'evidence/doudizhu-implementation/native-build').glob('*/build.lock'):
        lock.chmod(0o666)
    print('Frozen L40S CUDA and native engines built with source/binary receipts; no GPU allocated.')

if __name__=='__main__':main()
