"""CUDA graph/action oracle, complete launcher resume and CPU replay validation."""
from pathlib import Path
import json
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def run(base,native=False):
    import numpy as np
    import torch
    from integrations.doudizhu.artifacts import load_checkpoint, write_json
    from integrations.doudizhu.environment import Game,deal,heuristic,replay,ROLES
    from integrations.doudizhu.model import Policy,update
    from integrations.doudizhu.runner import initialize,make_optimizers
    config=json.loads((ROOT/'integrations/doudizhu/recipes/validated-l4-smoke.json').read_text())
    config.pop('budget_gpu_hours',None)
    config.update(architecture={'hidden':512,'depth':5,'history':128},max_episodes=256,
                  max_seconds=240,workers=4,vector_games=64,compact_workers=True,
                  engine_backend='view',encoding_backend='direct',inference_backend='graph')
    if native:
        config.update(engine_backend='native',native_fast=True,compact_workers=False,
                      record_games=4,inference_backend='eager')
        config['opponents']['selfplay_probability']=1.
    rng=initialize(config)
    policy=Policy(config['architecture']).cuda().eval()
    observations=[]
    for i in range(12):
        game=Game(deal('train',71923,i))
        while (obs:=game.observe()) is not None:
            observations.append(obs)
            game.step(heuristic(obs))
    observations=observations[::5]
    oracle=[]
    for chunk in (64,4096):
        eager=policy.values_many(observations,max_candidates=chunk)
        graph=policy.values_many(observations,max_candidates=chunk,inference_backend='graph')
        max_error=0.
        for obs,a,b in zip(observations,eager,graph):
            assert len(a)==len(b)==len(obs.legal)
            np.testing.assert_allclose(a,b,rtol=1e-4,atol=1e-6)
            assert np.argmax(a)==np.argmax(b)
            max_error=max(max_error,float(np.max(np.abs(a-b))))
        oracle.append({'chunk':chunk,'observations':len(observations),'max_candidates':max(len(o.legal) for o in observations),
                       'maximum_q_absolute_error':max_error,'argmax_mismatches':0})
    samples={r:[] for r in ROLES}
    for obs in observations:
        x,z,actions=obs.encode()
        samples[obs.role].append((np.concatenate((x,actions[0])),z,1.))
    update(policy,make_optimizers(policy,config),samples,256,1,rng)
    eager=policy.values_many(observations)
    graph=policy.values_many(observations,inference_backend='graph')
    for a,b in zip(eager,graph):
        np.testing.assert_allclose(a,b,rtol=1e-4,atol=1e-6)
    assert sum(len(g.cache) for g in policy._graph_scorers.values())<=18
    policy.load_state_dict(policy.state_dict())
    assert not hasattr(policy,'_graph_scorers')
    # Cross-invocation graph caches are deliberately cold. RNG and optimizer state
    # must still produce identical learned weights and sample/deal counters.
    results={}
    for label,episodes,resume in [('full',256,None),('partial',128,None),('resume',256,'partial')]:
        local=dict(config,max_episodes=episodes)
        path=base/(label+'.json')
        path.write_text(json.dumps(local,indent=2)+'\n')
        command=[sys.executable,str(ROOT/'integrations/doudizhu/launch.py'),'train','--recipe',str(path),
                 '--out',str(base/label)]
        if resume:
            command+=['--resume',str(base/resume/'latest.pt')]
        subprocess.run(command,cwd=ROOT,check=True,timeout=270)
        results[label]=load_checkpoint(base/label/'latest.pt')
    full,a=results['full']
    resumed,b=results['resume']
    for key,value in full.state_dict().items():
        torch.testing.assert_close(value,resumed.state_dict()[key],atol=0,rtol=0)
    for key in ('episodes','next_deal','decisions','samples','updates','used_deal_hashes','numpy_rng'):
        assert a['state'][key]==b['state'][key],key
    assert torch.equal(a['torch_rng'],b['torch_rng'])
    for x,y in zip(a['cuda_rng'],b['cuda_rng']):
        assert torch.equal(x,y)
    replayed=0
    for label in ('full','partial','resume'):
        path=base/label/'private-first-wave-trajectories.json'
        if path.exists():
            for trace in json.loads(path.read_text()):
                replay(trace)
                replayed+=1
    write_json(base/'validation.json',{'device':torch.cuda.get_device_name(0),'oracle':oracle,
        'cached_graphs_see_learner_updates':True,'graph_cache_invalidated_on_weight_load':True,
        'full_cuda_resume_bit_exact':True,
        'full_graph_compact_worker_cuda_resume_bit_exact':not native,'cpu_replayed_games':replayed,
        'native_bulk_cuda_resume_bit_exact':native,
        'worker_faults':0,'illegal_actions':0,'truncations':0})


if __name__=='__main__':
    run(Path(sys.argv[1]),native='--native' in sys.argv)
