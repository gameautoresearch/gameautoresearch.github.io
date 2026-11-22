"""CPU-only viewing games of the sealed November 8–9 researcher submissions.

Fresh public seeds are deliberately distinct from private evaluation schedules.
Only projected replay movies enter the local viewer; bundles stay in evidence.
"""
from pathlib import Path
import argparse,json,sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from simulators.clash.export_demo_replays import Recorder,NativeCPU,PublicActor,DATA,no_jvm,public_digest,public_outputs,digest,write,sha
from simulators.clash.autoresearch.arena import PolicyProcess
from simulators.clash.autoresearch.bundle import verify

CAMPAIGN=ROOT/'evidence/overnight-20261108-1533Z/clash'

def generate(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    rows=json.loads((CAMPAIGN/'researcher-outcomes.json').read_text())['rows']
    entries=[];receipts=[];data=json.loads(DATA.read_text());rec=Recorder(data)
    labels=['DeepSeek V4.1 Flash','GLM 5.3 Flash','Qwen 3.8 27B']
    for i,(row,label) in enumerate(zip(rows,labels)):
        bundle=Path(row['final_source_or_bundle_path']);identity=verify(bundle)['identity']
        identifier=['clash-overnight-deepseek','clash-overnight-glm','clash-overnight-qwen'][i]
        seed=2026110911+i;seat=i%2;title=f'{label} vs Standard bot'
        actor=PublicActor('scripted',seed+100,data=data)
        candidate=PolicyProcess(bundle,development=True)
        env=NativeCPU([seed]);frames=[];hashes=[];actions=[]
        try:
            frames.append(rec.frame(env,[]));hashes.append(public_digest(public_outputs(env)))
            for _ in range(3000):
                obs=env.observations[0].copy();mask=env.masks[0].copy();tick=int(env.infos[0,0])
                a=[0,0];a[seat]=candidate.choose(obs[seat:seat+1],mask[seat:seat+1])[0]
                a[1-seat]=actor(obs[1-seat],mask[1-seat])
                if any(not mask[p,a[p]] for p in (0,1)):raise AssertionError('Illegal action')
                commands=[c for p in (0,1) if (c:=rec.command(a[p],obs[p],p,tick)) is not None]
                env.step([a]);actions.append(a);f=rec.frame(env,commands);frames.append(f);hashes.append(public_digest(public_outputs(env)))
                if any(int(env.infos[0,rec.s.info[k]]) for k in ('fault','not_modelled','illegal_0','illegal_1','truncated')):raise AssertionError('Invalid episode')
                if f['ended']:break
            else:raise AssertionError('Natural horizon exceeded')
            final=public_digest(public_outputs(env))
        finally:env.close();candidate.close()
        env=NativeCPU([seed])
        try:
            assert public_digest(public_outputs(env))==hashes[0] and rec.frame(env,[])==frames[0]
            for n,a in enumerate(actions,1):
                assert all(env.masks[0,p,a[p]] for p in (0,1));env.step([a])
                assert public_digest(public_outputs(env))==hashes[n]
                assert rec.frame(env,frames[n]['commands'])==frames[n]
            assert public_digest(public_outputs(env))==final
        finally:env.close()
        last=frames[-1];verify(bundle,identity);no_jvm()
        result='Draw' if last['winner'] is None else ('AI wins' if last['winner']==seat else 'Standard bot wins')
        players=[{'name':'Standard scripted bot','kind':'scripted'} for _ in range(2)]
        players[seat]={'name':label+' · sealed final','kind':'neural'}
        movie={'schemaVersion':1,'gameId':'clash','id':identifier,'title':title,
            'subtitle':f'Sealed overnight AI · fresh public evaluation · AI {"Blue" if seat==0 else "Red"}',
            'players':players,'arena':{'width':18000,'height':32000,'decisionTicks':5,'tickMs':50,'deck':data['rules']['deck'],'towers':[e for e in frames[0]['entities'] if e['kind']=='tower']},
            'provenance':{'kind':'fresh-public-demo','engineCommit':'f0659dafabfb9d2a4a8d3b22a9d14de306f5e4de','ruleset':data['task'],'dataVersion':'16.635.8','demoSeed':seed,
                'verification':'All legal actions, complete public outputs and visual frames independently replayed; natural finish; no JVM or paid worker',
                'verifiedFrames':len(frames),'trajectorySha256':digest(actions),'finalPublicSha256':final,
                'policy':f'{label} sealed final {identity[:12]}; greedy public-only inference vs frozen Standard scripted panel bot. New viewing game; not an original private grade.',
                'submissionIdentity':identity,'researcher':row['model'],'candidateSeat':seat,'evaluationScore':row['raw_score_0_100'],'evaluationGames':row['private_games']},
            'frames':frames,'outcome':{'kind':'natural','winner':last['winner'],'draw':last['winner'] is None,'tick':last['tick'],'decisions':len(actions),'crowns':last['crowns']}}
        write(out/(identifier+'.json'),movie)
        entries.append({'gameId':'clash','id':identifier,'title':title,'subtitle':f'{result} · {last["elapsedMs"]/1000:.1f}s · overnight panel {row["raw_score_0_100"]:.2f}/100','moves':len(actions),'url':f'replays/clash/overnight-20261109/{identifier}.json'})
        receipts.append({'id':identifier,'bundle':str(bundle),'identity':identity,'seed':seed,'candidateSeat':seat,'actions':len(actions),'frames':len(frames),'sha256':sha(out/(identifier+'.json')),'outcome':movie['outcome']})
        print(json.dumps(receipts[-1]),flush=True)
    write(out/'index.json',entries);write(out/'verification.json',{'status':'passed','cpuOnly':True,'privateScheduleUsed':False,'recordings':receipts})
    return entries

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);generate(p.parse_args().output)
