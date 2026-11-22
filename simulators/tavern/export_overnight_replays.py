"""Record frozen overnight AI submissions against the B2 anchor on public seeds.

CPU-only viewing examples, distinct from original private grading trajectories.
"""
from pathlib import Path
import argparse,json,sys,hashlib
import numpy as np
import taverngym as tg

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(Path(__file__).resolve().parent/'autoresearch'))
from simulators.tavern.export_demo_replays import film_round,valid,digest,write,name,RACES
from contract import binding,validate,sha,EDITABLE,RULES
from grade import Remote

FIXED=ROOT/'evidence/overnight-20261108-1533Z/tavern/fresh-final48/fixed-bundles'

def verify(bundle):
    ident=json.loads((bundle/'identity.json').read_text());summary=json.loads((bundle/'summary.json').read_text())
    if ident['binding']!=binding():raise ValueError('Source/rules checkpoint binding differs')
    validate(ident['config'])
    if any(sha(bundle/f)!=ident['source'][f] for f in EDITABLE):raise ValueError('Editable source changed')
    if sha(bundle/'identity.json')!=summary['identity_sha256'] or sha(bundle/'weights.npz')!=summary['weights_sha256']:raise ValueError('Identity/weights changed')
    return summary

def choose(remote,g,seat):
    f,c=g.observe(seat)
    obs={'features':np.frombuffer(f,dtype='<f4')[None,:],'cards':np.frombuffer(c,dtype='<i4')[None,:],
         'action_mask':np.frombuffer(g.legal_mask(seat),dtype=np.bool_)[None,:]}
    return int(remote.remote_act(obs,np.array([0]))[0][0])

def generate(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False);entries=[];receipts=[]
    anchor=FIXED/'B2-starter-111901';anchor_id=verify(anchor)['weights_sha256']
    rules={**RULES,'record_log':True}
    for i,(slug,label,seat,score) in enumerate([('deepseek','DeepSeek V4.1 Flash',0,73.81),('glm','GLM 5.3 Flash',3,32.14),('deepseek','DeepSeek V4.1 Flash',6,73.81)]):
        bundle=FIXED/f'C-{slug}';ident=verify(bundle);seed=2026110921+i
        candidate=Remote(bundle);opponent=Remote(anchor);g=tg.Game(seed,json.dumps(rules))
        actions=[];rounds=[];boundaries=[]
        try:
            for _ in range(30000):
                if g.is_over():break
                acted=False
                for p in range(8):
                    while g.needs_action(p):
                        acted=True;a=choose(candidate if p==seat else opponent,g,p)
                        if not g.legal_mask(p)[a]:raise AssertionError('Illegal action')
                        before=json.loads(g.state_json());turn=g.turn;g.step(p,a);actions.append([p,a])
                        if before['phase']!='HeroSelect' and (g.turn!=turn or g.is_over()):
                            after=valid(g);rounds.append(film_round(before,after));boundaries.append([len(actions),digest(after)])
                        if len(actions)>30000:raise RuntimeError('Action guard')
                if not acted and not g.is_over():raise RuntimeError('No progress')
            else:raise RuntimeError('Driver guard')
            final=valid(g);places=list(g.placements())
            if sorted(places)!=list(range(1,9)):raise AssertionError('Incomplete natural placements')
        finally:candidate.close();opponent.close()
        check=tg.Game(seed,json.dumps(rules));boundary=dict(boundaries);checked=0
        for step,(p,a) in enumerate(actions,1):
            assert check.needs_action(p) and check.legal_mask(p)[a]
            before=json.loads(check.state_json());turn=check.turn;check.step(p,a)
            if before['phase']!='HeroSelect' and (check.turn!=turn or check.is_over()):
                after=valid(check);assert digest(after)==boundary[step];assert film_round(before,after)==rounds[checked];checked+=1
        assert check.is_over() and digest(valid(check))==digest(final) and checked==len(rounds)
        verify(bundle);verify(anchor);winner=places.index(1);identifier=f'tavern-overnight-{slug}-{i+1:02}'
        title=f'{label} vs seven PPO bots';subtitle=f'AI seat {seat+1} · #{places[seat]} of 8 · {len(rounds)} rounds'
        movie=dict(schemaVersion=1,gameId='tavern',id=identifier,title=title,subtitle=subtitle,
            provenance=dict(kind='fresh-public-demo',demoSeed=seed,engineSource=tg.build_info()['source_hash'],ruleset='AutoTavern-37.0.2-eight-player',
                policy=f'{label} sealed final vs seven frozen B2 starter-anchor PPO bots; greedy public-only inference. Fresh viewing game, not an original private grade.',
                verification='Every legal action independently replayed; exact full state/RNG and projected combat frames at every round and finish; natural outcome',
                actionCount=len(actions),trajectorySha256=digest(actions),finalStateSha256=digest(final),verifiedRounds=len(rounds),
                submissionWeightsSha256=ident['weights_sha256'],anchorWeightsSha256=anchor_id,candidateSeat=seat,researcher=label,evaluationScore=score,evaluationGames=48,
                assisted=slug=='deepseek',recordingOnlyRule='record_log=true captures combat snapshots; all gameplay rules unchanged'),
            races=[s for j,s in enumerate(RACES) if final['lobby_races']&(1<<j)],
            players=[dict(seat=p['id'],hero=name(p['hero']),policy=label if p['id']==seat else 'Frozen B2 PPO anchor') for p in final['players']],
            rounds=rounds,outcome=dict(kind='natural',winner=winner,placements=places))
        write(out/f'{identifier}.json',movie)
        entries.append(dict(gameId='tavern',id=identifier,title=title,subtitle=subtitle+f' · overnight panel {score:.2f}/100',moves=len(rounds),url=f'replays/tavern/overnight-20261109/{identifier}.json'))
        receipts.append(dict(id=identifier,bundle=str(bundle),weightsSha256=ident['weights_sha256'],seed=seed,candidateSeat=seat,actions=len(actions),rounds=len(rounds),placements=places,sha256=sha(out/f'{identifier}.json')))
        print(json.dumps(receipts[-1]),flush=True)
    write(out/'index.json',entries);write(out/'verification.json',dict(status='passed',cpuOnly=True,privateScheduleUsed=False,recordings=receipts))
    return entries

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);generate(p.parse_args().output)
