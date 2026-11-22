"""Small genuine PPO, three seeds, paired control/optimized and matched placement evaluation."""
import copy
import json
from pathlib import Path
import time
from gpu_suite import HERE,run_command


def experiment(out,device='cuda:0',seeds=(12345,)):
    import torch
    out=Path(out);out.mkdir(parents=True,exist_ok=False);started=time.perf_counter();rows=[];results=[]
    base=json.loads((HERE/'recipes/profile-t4-copy1.json').read_text())
    base.update(device=device,updates=64,profile_updates=2,n_games=32,torch_threads=1,
                minibatch_size=1024,snapshot_every=0,gamma=1.0,save_initial=True,start_turn=1)
    base['evaluation'].update(seeds=[1000000007,1000000009],games_per_seat=4)
    base['machine'].update(label='Modal-L4-fixed-4core-16GiB',attached_cpu_threads=4,attached_ram_gib=16,gpu_model='NVIDIA L4')
    for seed in seeds:
        train=[]
        for optimized in [False,True]:
            c=copy.deepcopy(base);c.update(seed=seed,worker_backend='pooled' if optimized else 'scoped',
                fast_inference=optimized,skip_finished_lobbies=optimized,defer_learner_metrics=optimized)
            name=f'seed{seed}-'+('optimized' if optimized else 'baseline')
            row=run_command(out/name,c,'train');row.update(seed=seed,optimized=optimized);rows.append(row);train.append(row)
            print(json.dumps({'trained':name,'seconds':row['full_process_seconds'],'samples':row['training']['counts']['accepted_learning_samples']}),flush=True)
        a=torch.load(out/train[0]['run']/'latest.pt',weights_only=False,map_location='cpu')
        b=torch.load(out/train[1]['run']/'latest.pt',weights_only=False,map_location='cpu')
        identical=all(torch.equal(a['policy'][k],b['policy'][k]) for k in a['policy'])
        counts_equal=all(train[0]['training']['counts'][k]==train[1]['training']['counts'][k] for k in ['accepted_learning_samples','completed_learning_episodes','invalid_lobbies','rejected'])
        evals=[]
        ec=copy.deepcopy(base);ec.update(seed=seed,worker_backend='pooled',fast_inference=True,skip_finished_lobbies=True,defer_learner_metrics=True)
        for label,path in [('initial',out/train[0]['run']/'initial.pt'),('final',out/train[1]['run']/'latest.pt')]:
            row=run_command(out/f'seed{seed}-eval-{label}',ec,'evaluate',path);rows.append(row);evals.append(row)
        improvement=evals[0]['evaluation']['mean_placement']-evals[1]['evaluation']['mean_placement']
        samples=train[0]['training']['counts']['accepted_learning_samples']
        results.append({'seed':seed,'weights_identical':identical,'counts_identical':counts_equal,
            'baseline_seconds':train[0]['full_process_seconds'],'optimized_seconds':train[1]['full_process_seconds'],
            'baseline_loop_seconds':sum(train[0]['profile']['stage_seconds'].values()),
            'optimized_loop_seconds':sum(train[1]['profile']['stage_seconds'].values()),
            'training_samples':samples,'training_episodes':train[0]['training']['counts']['completed_learning_episodes'],
            'initial_mean_placement':evals[0]['evaluation']['mean_placement'],'final_mean_placement':evals[1]['evaluation']['mean_placement'],
            'placement_improvement':improvement,'final_top4_rate':evals[1]['evaluation']['top4_rate'],
            'evaluation_games_per_policy':evals[1]['evaluation']['valid_episodes'],
            'invalid_evaluation_games':evals[0]['evaluation']['invalid_episodes']+evals[1]['evaluation']['invalid_episodes']})
        (out/'learning-summary.json').write_text(json.dumps({'results':results,'rows':rows,'worker_seconds':time.perf_counter()-started,'complete':len(results)==len(seeds), 'requested_seeds':list(seeds)},indent=2)+'\n')
        if not identical or not counts_equal:raise RuntimeError('optimization changed paired learning trajectory')
    return results
