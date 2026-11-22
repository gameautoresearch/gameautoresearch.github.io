"""CPU-only public replay games from the exact frozen overnight submissions."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
import numpy as np

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--runtime-root',type=Path,default=ROOT);a=p.parse_args()
    runtime=a.runtime_root.resolve();sys.path.insert(0,str(runtime))
    from integrations.pokemon.autoresearch.arena import PolicyProcess
    from integrations.pokemon.autoresearch.bundle import verify
    from integrations.pokemon.autoresearch.contract import runtime_signature
    from integrations.pokemon.network import scripted
    output=Path(a.output);output.mkdir(parents=True,exist_ok=False)
    campaign=ROOT/'evidence/overnight-20261108-1533Z/pokemon'
    roster=json.loads((campaign/'final-roster-20261108T1932Z.json').read_text())['candidates']
    entries=[];receipts=[]
    for label,display,public_seed in [('research-deepseek','DeepSeek',2026110901),('research-glm','GLM',2026110902)]:
        bundle=Path(next(r['bundle'] for r in roster if r['label']==label))
        manifest=verify(bundle)
        binding=json.loads((bundle/'binding.json').read_text())
        if binding['runtime_sha256']!=runtime_signature():raise ValueError('Frozen runtime binding differs')
        worker=runtime/'integrations/pokemon/worker.cjs'
        process=subprocess.Popen(['node',str(Path(__file__).with_name('replay_capture.cjs')),str(worker)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True)
        def call(value):
            process.stdin.write(json.dumps(value)+'\n');process.stdin.flush()
            result=json.loads(process.stdout.readline())
            if not result['ok']:raise RuntimeError(result['error'])
            return result
        try:
            data=call({'op':'reset','seed':public_seed});frames=[data['frame']];actions=[]
            rng=np.random.default_rng(np.random.SeedSequence([public_seed,1,8675309]))
            with PolicyProcess(bundle,development=True,timeout=30) as candidate:
                while not (data['state']['terminated'] or data['state']['truncated']):
                    state=data['state'];joint=[None,None]
                    if state['ready'][0]:joint[0]=candidate.choose([state['observations'][0]])[0]
                    if state['ready'][1]:joint[1]=scripted(state['observations'][1],rng,'power')
                    data=call({'op':'step','actions':joint});actions.append(joint);frames.append(data['frame'])
            result=data['state'];checked=call({'op':'verify','seed':public_seed,'actions':actions})
            if not checked['matched'] or checked['frames']!=frames:raise ValueError('Independent replay differs')
            rid='pokemon-'+label+'-power';title=display+' vs Public Power'
            ending=('AI wins' if result['winner']=='p1' else 'Bot wins' if result['winner']=='p2' else 'Natural tie') if result['terminated'] else 'Decision cap · unscored'
            movie={'schemaVersion':1,'gameId':'pokemon','id':rid,'title':title,
                'players':[display+' final submission','Public Power bot'],'frames':frames,
                'outcome':{'natural':result['terminated'],'truncated':result['truncated'],'winner':result['winner'],'fault':None,'phases':result['steps'],'turns':result['turn']},
                'provenance':{'kind':'fresh-public-demo','campaign':'November 8–9, 2026','bundleSha256':manifest['bundle_sha256'],
                    'sourceSha256':binding['source_sha256'],'publicSeed':public_seed,'runtimeSha256':binding['runtime_sha256'],
                    'format':'gen9randombattle','opponent':'Same public-power heuristic used in evaluation panel',
                    'verification':'Exact independent action replay, full spectator frames and player-stream hash',
                    'traceSha256':checked['trace'],'note':'New public-seed demonstration of the frozen final submission; not a historical scored evaluation game. Original incomplete/capped panel scores remain unchanged.'}}
            file=output/(rid+'.json');file.write_text(json.dumps(movie,separators=(',',':'))+'\n')
            entries.append({'gameId':'pokemon','id':rid,'title':title,'subtitle':f"{ending} · {result['turn']} turns · fresh public demo",'url':'replays/pokemon/overnight-20261109/'+file.name,'model':display,'opponent':'Public Power','ending':ending})
            receipts.append({'id':rid,'fileSha256':hashlib.sha256(file.read_bytes()).hexdigest(),'bundle':str(bundle),'bundleSha256':manifest['bundle_sha256'],'traceSha256':checked['trace'],'exactFrames':True,'phases':result['steps'],'natural':result['terminated'],'truncated':result['truncated'],'rejections':result['rejections']})
            print(rid,ending,result['steps'],flush=True)
        finally:
            process.stdin.close();process.wait(timeout=10)
    (output/'index.json').write_text(json.dumps(entries,indent=2)+'\n')
    (output/'verification.json').write_text(json.dumps(receipts,indent=2)+'\n')

if __name__=='__main__':main()
