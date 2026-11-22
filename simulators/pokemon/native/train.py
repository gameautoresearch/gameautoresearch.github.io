"""Ordinary terminal-reward PPO CLI on the admitted native Vector backend.

Uses the original editable recipe and learner without changing the frozen task or
benchmark ledgers. Qualification mode is explicitly marked and cannot be exported
as an admitted production run.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from ..autoresearch.contract import runtime_config, identity, write
from ..autoresearch.starter.learner import update
from ..network import Policy
from ..rl import rollout, Inference, Timings
from .vector import NativeVector


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--runtime',required=True);p.add_argument('--out',required=True);p.add_argument('--seconds',type=float,required=True)
    p.add_argument('--device',default='cpu');p.add_argument('--resume');p.add_argument('--updates',type=int);p.add_argument('--qualification',action='store_true');a=p.parse_args()
    config=json.loads(Path(a.config).read_bytes());cfg=runtime_config(config,a.device)
    runtime_path=Path(a.runtime);runtime=json.loads(runtime_path.read_bytes());paths={}
    for key in('library','generators','constructors','chart'):
        path=Path(runtime[key]);paths[key]=path if path.is_absolute()else runtime_path.parent/path
        if hashlib.sha256(paths[key].read_bytes()).hexdigest()!=runtime['sha256'][key]:raise ValueError(f'Native runtime {key} artifact changed')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    random.seed(config['seed']);np.random.seed(config['seed']);torch.manual_seed(config['seed'])
    if a.device.startswith('cuda'):torch.cuda.manual_seed_all(config['seed'])
    model=Policy(**config['network']).to(a.device);optimizer=torch.optim.Adam(model.parameters(),lr=config['learning_rate'],eps=1e-5)
    rng=np.random.default_rng(config['seed']);iteration=0;rows=[];timings=Timings()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False);started=time.monotonic();deadline=started+a.seconds
    scope='Ordinary CLI differential qualification; not an admitted production run'if a.qualification else None
    with NativeVector(**paths,workers=cfg['hardware']['workers'],per_worker=config['environments_per_worker'],developer_scope=scope,team_size=runtime.get('team_size',6))as vector:
        binding=dict(source_signature=vector.arena.source_signature,binary_sha256=vector.arena.binary_sha256,rules_sha256=vector.rules_sha256,
            recipe_sha256=identity(config),runtime_manifest_sha256=hashlib.sha256(runtime_path.read_bytes()).hexdigest(),
            learner_sha256=hashlib.sha256((Path(__file__).parents[1]/'autoresearch/starter/learner.py').read_bytes()).hexdigest(),device=a.device,qualification=a.qualification)
        if a.resume:
            cp=torch.load(a.resume,map_location=a.device,weights_only=False)
            if cp['binding']!=binding:raise ValueError('Native source/rules/recipe/runtime resume mismatch')
            model.load_state_dict(cp['model']);optimizer.load_state_dict(cp['optimizer']);rng.bit_generator.state=cp['numpy_rng'];random.setstate(cp['python_rng']);torch.set_rng_state(cp['torch_rng'].cpu())
            if cp['cuda_rng']is not None:torch.cuda.set_rng_state_all([v.cpu()for v in cp['cuda_rng']])
            iteration=cp['iteration'];vector.restore(cp['vector'])
        initial={k:v.detach().cpu().clone()for k,v in model.state_dict().items()}
        stop=min(config['iterations'],iteration+a.updates)if a.updates else config['iterations']
        while iteration<stop and time.monotonic()<deadline:
            seeds=list(range(config['training_seed_start']+iteration*vector.size,config['training_seed_start']+(iteration+1)*vector.size))
            samples,report=rollout(vector,model,[[model,model]for _ in seeds],seeds,cfg,rng,Inference(a.device,timings),timings,time.perf_counter()+max(0,deadline-time.monotonic()))
            stats=update(model,optimizer,samples,config,a.device)if time.monotonic()<deadline else dict(samples=0,updates=0,skipped='deadline')
            iteration+=1;row=dict(iteration=iteration,ppo=stats,natural_games=report['natural_terminals'],truncations=report['truncations'],visible_choice_rejections=report['rejected_visible_choices']);rows.append(row)
            with(out/'progress.jsonl').open('a')as stream:stream.write(json.dumps(row)+'\n')
            cp=dict(binding=binding,model=model.state_dict(),optimizer=optimizer.state_dict(),iteration=iteration,numpy_rng=rng.bit_generator.state,python_rng=random.getstate(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()if a.device.startswith('cuda')else None,vector=vector.snapshot())
            torch.save(cp,out/'checkpoint.tmp');(out/'checkpoint.tmp').replace(out/'checkpoint.pt')
        result=dict(full_gen9_admission=bool(vector.arena.lib.pkn_gen9_complete())and not a.qualification,qualification=a.qualification,
            iterations=iteration,iterations_this_process=len(rows),trained_samples=sum(r['ppo']['samples']for r in rows),optimizer_updates=sum(r['ppo']['updates']for r in rows),
            natural_games=sum(r['natural_games']for r in rows),truncations=sum(r['truncations']for r in rows),
            parameter_change_l2=sum(float((v.detach().cpu()-initial[k]).square().sum())for k,v in model.state_dict().items())**.5,
            reward='original terminal reward only; faint_reward=0',device=a.device,process_seconds=time.monotonic()-started,binding=binding)
        write(out/'result.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
