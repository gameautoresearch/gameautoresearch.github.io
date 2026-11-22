"""Public correctness fixture: refill and interrupted resume preserve every byte."""
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[5]
from simulators.pokemon.autoresearch.runtime.fastenv import GPUCPU


class Window(unittest.TestCase):
    def test_refills_equal_full_window_and_exact_interrupted_resume(self):
        torch.set_num_threads(1)
        runtime=ROOT/'evidence/pokemon-implementation/native-cpu-completion-20261109/admitted-runtime-1/runtime.json'
        cfg=dict(environments=2,training_seed_start=440100,reset_schedule_length=4096,reset_window=2)
        with patch.dict(os.environ,{'POKEMON_PILOT_RUNTIME':str(runtime)}):
            a=GPUCPU(cfg,'cpu');b=GPUCPU(dict(cfg,reset_window=64),'cpu')
            try:
                a.allocate_rollout(1);b.allocate_rollout(1);rng=np.random.default_rng(1729)
                refills=0
                for phase in range(384):
                    np.testing.assert_array_equal(a.cpu.obs.numpy(),b.cpu.obs.numpy())
                    np.testing.assert_array_equal(np.asarray(a.offsets)+a.cpu.episodes,np.asarray(b.offsets)+b.cpu.episodes)
                    self.assertEqual(a.cpu.arena.snapshot(),b.cpu.arena.snapshot())
                    actions=[]
                    for observation in a.cpu.obs.reshape(-1,700):
                        legal=np.flatnonzero(observation[-14:].numpy());actions.append(int(rng.choice(legal))if len(legal)else -1)
                    tensor=torch.tensor(actions,dtype=torch.int32).reshape(2,2)
                    old=list(a.offsets);a.step(tensor,0);b.step(tensor,0)
                    self.assertFalse(bool(a.flags[...,2].any()))
                    if a.offsets!=old:
                        refills+=1;state=a.snapshot();a.restore(state)
                        self.assertEqual(state,a.snapshot())
                    if refills>=5:break
                self.assertGreaterEqual(refills,5)
            finally:a.close();b.close()


if __name__=='__main__':unittest.main()
