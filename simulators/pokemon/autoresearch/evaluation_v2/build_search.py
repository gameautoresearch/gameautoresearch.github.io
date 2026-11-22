"""Bounded CPU build of pinned public search source; wheel outputs persist on Volume."""
import argparse,hashlib,json,os,pathlib,subprocess,tarfile,urllib.request


def main():
 p=argparse.ArgumentParser();p.add_argument('--request',required=True);p.add_argument('--output',required=True);a=p.parse_args()
 spec=json.loads(pathlib.Path(a.request).read_text());out=pathlib.Path(a.output);work=pathlib.Path('/tmp/search-build');work.mkdir();log=(out/'build.log').open('w')
 def run(cmd,timeout=120):
  subprocess.run(cmd,check=True,stdout=log,stderr=subprocess.STDOUT,timeout=timeout)
 run(['apt-get','update'],60);run(['apt-get','install','-y','--no-install-recommends','build-essential','patchelf'],90)
 url='https://static.rust-lang.org/rustup/dist/x86_64-unknown-linux-gnu/rustup-init'
 installer=work/'rustup-init'
 for suffix,target in [('',installer),('.sha256',work/'rustup-init.sha256')]:
  with urllib.request.urlopen(url+suffix,timeout=30) as response:target.write_bytes(response.read())
 expected=(work/'rustup-init.sha256').read_text().split()[0];assert hashlib.sha256(installer.read_bytes()).hexdigest()==expected
 installer.chmod(0o700);os.environ['RUSTUP_HOME']=str(work/'rustup');os.environ['CARGO_HOME']=str(work/'cargo')
 run([str(installer),'-y','--no-modify-path','--profile','minimal','--default-toolchain','1.91.0'],120)
 os.environ['PATH']=str(work/'cargo/bin')+':'+os.environ['PATH'];os.environ['CARGO_BUILD_JOBS']='4'
 source=work/'engine';source.mkdir()
 with tarfile.open(spec['engine_archive']) as tar:tar.extractall(source,filter='data')
 source=next(source.iterdir())
 run(['python','-m','pip','install','maturin==1.9.6'],60)
 wheels=out/'wheels';wheels.mkdir()
 run(['python','-m','pip','wheel','--no-deps','--no-build-isolation','--wheel-dir',str(wheels),str(source),
      '--config-settings','build-args=--locked --features poke-engine/terastallization --no-default-features'],200)
 run(['python','-m','pip','download','--dest',str(wheels),'--only-binary=:all:','requests==2.33.0','websockets==14.1','python-dateutil==2.8.0'],60)
 manifest={'rustup_init_sha256':expected,'rust_toolchain':'1.91.0','maturin':'1.9.6','engine_archive_sha256':hashlib.sha256(pathlib.Path(spec['engine_archive']).read_bytes()).hexdigest(),
  'wheel_sha256':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in wheels.iterdir()}}
 (out/'summary.json').write_text(json.dumps(manifest,indent=2));log.close()
 # Compiler caches have no evidentiary value; retain logs, pinned archive and wheels.
 import shutil;shutil.rmtree(work)
if __name__=='__main__':main()
