"""Run with: python -m unittest integrations.pokemon.tests.test_rl -v"""
import copy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from integrations.pokemon.env import Vector, HERE
from integrations.pokemon.network import Policy, batch
from integrations.pokemon.rl import Inference, Population, Timings, rollout, update
from integrations.pokemon.cli import initialize, checkpoint, signature
from integrations.pokemon.provenance import Ledger
import time

class LearningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.cfg=json.loads((HERE/'recipes/smoke.json').read_text())
    def test_mask_and_encoder_finite(self):
        with Vector(1,2) as v:
            states=v.reset([1,2],3)
            observations=[s['observations'][seat] for s in states for seat in (0,1)]
            x=batch(observations,'cpu')
            self.assertTrue(all(torch.isfinite(t).all() for t in x.values()))
            model=Policy(hidden=32,embedding=4)
            dist,value=model.distribution(x)
            self.assertTrue(torch.equal(dist.probs[~x['mask']],torch.zeros_like(dist.probs[~x['mask']])))
            for _ in range(10):
                actions=dist.sample()
                self.assertTrue(x['mask'].gather(1,actions[:,None]).all())
            bad={k:t.clone() for k,t in x.items()};bad['mask'][0]=False
            with self.assertRaises(ValueError):model.distribution(bad)
    def test_truncation_bootstrap_gae_and_update(self):
        cfg=dict(self.cfg,max_decision_phases=1,gamma=.9,gae_lambda=.95,ppo_epochs=1)
        model=initialize(cfg)
        # Fixed V=.25 makes the expected cutoff bootstrap unambiguous.
        with torch.no_grad():
            model.value.weight.zero_();model.value.bias.fill_(.25)
        timing=Timings()
        with Vector(1,2) as v:
            samples,report=rollout(v,model,[[model,'power'],['power',model]],[11,12],cfg,
                                   np.random.default_rng(1),Inference('cpu',timing),timing)
        self.assertEqual(report['truncations'],2)
        self.assertEqual(report['natural_terminals'],0)
        for t in samples:
            self.assertAlmostEqual(t['target'],t['reward']+.9*.25,places=5)
        optimizer=torch.optim.Adam(model.parameters(),lr=.001)
        before={k:v.clone() for k,v in model.state_dict().items()}
        stats=update(model,optimizer,samples,cfg,'cpu',timing)
        self.assertGreater(stats['minibatch_updates'],0)
        self.assertTrue(any(not torch.equal(before[k],v) for k,v in model.state_dict().items()))
    def test_persistent_workers_and_checkpoint_resume(self):
        cfg=dict(self.cfg,max_decision_phases=120,ppo_epochs=1)
        model=initialize(cfg);optimizer=torch.optim.Adam(model.parameters(),lr=.001)
        rng=np.random.default_rng(5);population=Population(cfg,model,'cpu');population.add(model)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'checkpoint.pt'
            checkpoint(p,model,optimizer,cfg,1,rng,population,Ledger(time.perf_counter()),'test')
            cp=torch.load(p,weights_only=False)
            self.assertEqual(cp['config_signature'],signature(cfg))
            expected=torch.rand(5);expected_np=rng.random(5)
            restored=Policy(**cfg['network']);restored.load_state_dict(cp['model'])
            restored_rng=np.random.default_rng();restored_rng.bit_generator.state=cp['numpy_rng']
            torch.set_rng_state(cp['torch_rng'])
            self.assertTrue(torch.equal(torch.rand(5),expected))
            self.assertTrue(np.array_equal(restored_rng.random(5),expected_np))
            timing=Timings()
            with Vector(2,1) as v:
                pids=[w.process.pid for w in v.workers]
                a,ra=rollout(v,model,[[model,'power'],['power',model]],[71,72],cfg,np.random.default_rng(5),Inference('cpu',timing),timing,deterministic=True)
                hashes_a=[x['sha256'] for x in v.trace()]
                b,rb=rollout(v,restored,[[restored,'power'],['power',restored]],[71,72],cfg,np.random.default_rng(5),Inference('cpu',timing),timing,deterministic=True)
                hashes_b=[x['sha256'] for x in v.trace()]
                self.assertEqual(pids,[w.process.pid for w in v.workers])
            self.assertEqual(hashes_a,hashes_b)
            self.assertEqual([t['action'] for t in a],[t['action'] for t in b])
            for t in a:
                self.assertTrue(np.isfinite(t['target']))
    def test_scripted_opponents_batch_invariant(self):
        # Schedule's scripted RNG belongs to each battle/seat, not vector order.
        cfg=dict(self.cfg,max_decision_phases=100)
        m=initialize(cfg)
        def play(workers,n):
            timing=Timings()
            with Vector(workers,n) as v:
                rollout(v,m,[[m,'random'],['random',m]],[91,92],cfg,np.random.default_rng(0),Inference('cpu',timing),timing,deterministic=True)
                return [t['sha256'] for t in v.trace()]
        self.assertEqual(play(1,2),play(2,1))
    def test_population_frozen(self):
        m=initialize(self.cfg);p=Population(self.cfg,m,'cpu');p.add(m)
        frozen={k:v.clone() for k,v in p.history[0].state_dict().items()}
        with torch.no_grad():next(m.parameters()).add_(1)
        self.assertTrue(all(torch.equal(v,frozen[k]) for k,v in p.history[0].state_dict().items()))
        cfg=dict(self.cfg,opponents=[{'kind':'historical','weight':1}])
        p.config=cfg
        pairs=p.assignments(2,0,np.random.default_rng(1),m)
        self.assertIs(pairs[0][0],m);self.assertIs(pairs[1][1],m)
        self.assertIs(pairs[0][1],p.history[0])

if __name__=='__main__':unittest.main()
