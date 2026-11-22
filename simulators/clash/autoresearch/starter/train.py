"""Editable rollout adapter to the canonical, qualified native PPO trainer.
The simulator/observations/masks and native runtime remain supervisor-owned.
"""
import os,random,time
from pathlib import Path
from simulators.clash.autoresearch.fast_adapter import Trainer as NativeTrainer

class Trainer(NativeTrainer):
    def checkpoint(self,path,elapsed,*,entrant_binding,entrant_state):
        """Model, optimizer, counters, RNG and bindings only, written atomically.

        The native checkpoint also serializes every environment's full engine state, which can take longer than
        a long run's finalization margin. This one is small and fast; it is not an exact-resume checkpoint (the
        world is not saved). The policy itself is exported separately as weights.npz, and the host still commits
        this file's hash when it seals the run. Delete this method to get the native exact-resume checkpoint.
        """
        import numpy
        torch=self.torch;started=time.perf_counter()
        path=Path(path);temporary=path.with_suffix('.tmp')
        saved={'format':'clash-fast-model-optimizer-v1','config':self.c,'binding':self.binding,
               'model':self.model.state_dict(),'optimizer':self.optimizer.state_dict(),'update':self.update,
               'samples':self.samples,'allocated_seconds':self.prior_seconds+elapsed,'counters':self.counters.cpu(),
               'bucket_usage':self.bucket_usage.cpu(),'python_rng':random.getstate(),
               'numpy_rng':numpy.random.get_state(),'torch_rng':torch.get_rng_state(),
               'cuda_rng':torch.cuda.get_rng_state_all() if self.gpu else None,
               'entrant_binding':entrant_binding,'entrant_state':entrant_state}
        try:
            with temporary.open('wb') as stream:
                torch.save(saved,stream)
                stream.flush();os.fsync(stream.fileno())
            temporary.replace(path)
        finally:
            if temporary.exists():temporary.unlink()
        return {'format':saved['format'],'codec':'model-optimizer-only','engine_state_saved':False,
                'exact_resume':False,'serialization_passes':1,'file_fsynced':True,
                'total_seconds':time.perf_counter()-started,'file_bytes':path.stat().st_size}
