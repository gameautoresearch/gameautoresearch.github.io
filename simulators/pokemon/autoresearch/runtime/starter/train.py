"""Editable weak random PPO starter, exact source-bound state between leases."""
import argparse
import copy
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
import policy
import learner
from integrations.pokemon.autoresearch.runtime.sdk import (BENCHMARK_ID,identity,write,
    source_signature,runtime_signature,validate_recipe,vector_for)


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--out',required=True)
    p.add_argument('--seconds',type=float,required=True);p.add_argument('--device',default='cuda:0')
    p.add_argument('--resume');p.add_argument('--fork');p.add_argument('--updates',type=int);a=p.parse_args()
    cfg=validate_recipe(json.loads(Path(a.config).read_bytes()));out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    random.seed(cfg['seed']);np.random.seed(cfg['seed']);torch.manual_seed(cfg['seed'])
    if a.device.startswith('cuda'):torch.cuda.manual_seed_all(cfg['seed'])
    rng=np.random.default_rng(cfg['seed'])
    binding=dict(benchmark_id=BENCHMARK_ID,source_sha256=source_signature(Path(__file__).parent),
        runtime_sha256=runtime_signature(),config_sha256=identity(cfg),device=a.device)
    env=vector_for(cfg,a.device)
    try:
        binding.update(native_source_sha256=env.source_signature,runtime_manifest_sha256=env.runtime_manifest_sha256,
            native_binary_sha256=env.binary_sha256,native_rules_sha256=env.rules_sha256)
        model=policy.create(cfg['network']).to(a.device)
        # The graph is the architecture; the weights it is scored with come from the checkpoint. Writing it now
        # means a run that is stopped or reaches its limit is still packaged, from the last update it finished.
        # The exporter moves what it is given to the CPU in eval mode, so it gets a copy. This executes under the
        # same unprivileged researcher identity; the trusted sealer later admits only JSON pure tensor operators.
        from integrations.pokemon.autoresearch.runtime.graph_export import export
        export(copy.deepcopy(model),out/'graph.json')
        optimizer=torch.optim.Adam(model.parameters(),lr=cfg['learning_rate'],eps=1e-5)
        trainer=learner.Trainer(env,model,optimizer,cfg)
        iteration=0
        if a.resume:
            cp=torch.load(a.resume,map_location=a.device,weights_only=True)
            if cp['binding']!=binding:raise ValueError('Resume source/config/native runtime changed; explicit fork required')
            model.load_state_dict(cp['model']);optimizer.load_state_dict(cp['optimizer']);env.restore(cp['environment'])
            torch.set_rng_state(cp['torch_rng'].cpu());random.setstate(cp['python_rng']);rng.bit_generator.state=cp['numpy_rng']
            if cp['cuda_rng'] is not None:torch.cuda.set_rng_state_all([v.cpu()for v in cp['cuda_rng']])
            iteration=cp['iteration']
        elif a.fork:
            cp=torch.load(a.fork,map_location=a.device,weights_only=True)
            # Explicit source fork imports tensors only, never relabels Adam/RNG/native state.
            model.load_state_dict(cp['model'],strict=True)
        initial={k:v.detach().cpu().clone()for k,v in model.state_dict().items()}
        # Initialization and state restoration do not restart the learning clock.
        stop=min(time.monotonic()+a.seconds,float(os.environ.get('POKEMON_TRAINING_DEADLINE_MONOTONIC','inf')))
        updates=0;totals=dict(trained_samples=0,optimizer_updates=0,natural_games=0,official_draws=0,truncations=0)
        def summary():
            # Running totals of this process's updates, written after every one and once more at the end.
            return dict(**binding,iterations=iteration,**totals,faults=0,
                parameter_change_l2=sum(float((v.detach().cpu()-initial[k]).square().sum())for k,v in model.state_dict().items())**.5,
                full_gen9_admission=True,source_fork_parent=a.fork)
        while time.monotonic()<stop and (a.updates is None or updates<a.updates):
            tick=time.monotonic();row=trainer.collect_update();iteration+=1;row.update(iteration=iteration,seconds=time.monotonic()-tick)
            updates+=1
            for total,key in (('trained_samples','samples'),('optimizer_updates','optimizer_updates'),('natural_games','natural'),
                    ('official_draws','official_draws'),('truncations','truncations')):totals[total]+=row.get(key,0)
            with(out/'progress.jsonl').open('a')as f:f.write(json.dumps(row)+'\n')
            # Printed output is what `ar log` can show while this run is still going; flush or it sits in a buffer.
            print(json.dumps(row),flush=True)
            checkpoint=dict(binding=binding,**binding,model=model.state_dict(),optimizer=optimizer.state_dict(),
                environment=env.snapshot(),torch_rng=torch.get_rng_state(),python_rng=random.getstate(),numpy_rng=rng.bit_generator.state,
                cuda_rng=torch.cuda.get_rng_state_all()if a.device.startswith('cuda')else None,iteration=iteration)
            # Both files are replaced whole, checkpoint first: whenever this process is ended, the directory holds
            # a complete checkpoint.pt, graph.json, result.json and progress.jsonl (result.json can trail the
            # checkpoint by the one update whose two writes were interrupted).
            torch.save(checkpoint,out/'checkpoint.tmp');(out/'checkpoint.tmp').replace(out/'checkpoint.pt')
            write(out/'result.json',summary())
        result=summary()
        # The final graph replaces the first one whole; for a fixed architecture the two are identical.
        export(model,out/'graph.tmp');(out/'graph.tmp').replace(out/'graph.json')
        write(out/'result.json',result);print(json.dumps(result),flush=True)
    finally:env.close()


if __name__=='__main__':main()
