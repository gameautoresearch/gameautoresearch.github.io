"""Compare independent holder callback order with pinned Java EntityHolder."""
from pathlib import Path
import hashlib,json,os,shutil,subprocess,sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]))
from game import HERE as GAME,resolve

def main():
    out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=False)
    config=json.loads((GAME/'configs/profile-trimmed.json').read_text());jars=sorted((resolve(config['install'])/'lib').glob('*.jar'));classpath=os.pathsep.join(map(str,jars));classes=out/'classes';classes.mkdir()
    javac=Path(os.environ['JAVA_HOME'])/'bin/javac';java=Path(os.environ['JAVA_HOME'])/'bin/java'
    subprocess.run([str(javac),'-cp',classpath,'-d',str(classes),str(HERE/'HolderOracle.java')],check=True)
    native=out/'holder';compiler=shutil.which('clang++') or shutil.which('g++');command=[compiler,'-std=c++20','-O2','-Wall','-Wextra','-Werror',str(HERE/'test_holder_bridge.cpp'),'-o',str(native)];subprocess.run(command,check=True)
    cases=[]
    for scenario in range(3):
        expected=subprocess.check_output([str(java),'-cp',str(classes)+os.pathsep+classpath,'org.crforge.core.battle.HolderOracle',str(scenario)],text=True)
        actual=subprocess.check_output([str(native.resolve()),str(scenario)],text=True)
        (out/f'java-{scenario}.csv').write_text(expected);(out/f'native-{scenario}.csv').write_text(actual)
        if actual!=expected:
            for i,(a,b) in enumerate(zip(actual.splitlines(),expected.splitlines())):
                if a!=b:raise AssertionError(f'scenario{scenario} first differing event{i}: native{a} Java{b}')
            raise AssertionError(f'scenario{scenario} count native{len(actual.splitlines())} Java{len(expected.splitlines())}')
        cases.append({'scenario':scenario,'events':len(expected.splitlines()),'trace_sha256':hashlib.sha256(expected.encode()).hexdigest()})
    result=dict(status='passed',cases=cases,holder_sha256=hashlib.sha256((HERE.parent/'holder.hpp').read_bytes()).hexdigest(),upstream_commit='f0659dafabfb9d2a4a8d3b22a9d14de306f5e4de',complete_task_parity=False)
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
