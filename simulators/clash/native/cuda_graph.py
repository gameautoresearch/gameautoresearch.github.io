"""Capture complete native device rollouts without consuming warmup game/RNG state."""
from __future__ import annotations

class RolloutGraph:
    def __init__(self,trainer):
        import torch
        if trainer.c['backend']!='native-cuda':raise ValueError('rollout graph requires device-resident native worlds')
        self.trainer=trainer
        # The outer graph owns sampling; nested actor capture would introduce a
        # second warmup/RNG boundary. Both paths use the same dense policy.
        trainer.model.graph_sampling=False
        state=trainer.env.state();outputs=[x.clone() for x in trainer.outputs()];counters=trainer.counters.clone();rng=torch.cuda.get_rng_state()
        def restore():
            trainer.env.load(state)
            for target,saved in zip(trainer.outputs(),outputs):target.copy_(saved)
            trainer.counters.copy_(counters);trainer.refresh_actor();torch.cuda.set_rng_state(rng)
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):trainer._collect_eager()
        torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize();restore();torch.cuda.synchronize()
        self.graph=torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(self.graph):self.data=trainer._collect_eager()
        finally:
            restore()
        torch.cuda.synchronize()
    def __call__(self):
        self.graph.replay()
        return self.data
