"""Reuse native Trainer unchanged, with explicit editable policy/learner hooks.

The qualified native constructor imports model.Policy locally. A construction-only
patch supplies the entrant factory, then restores that symbol before any rollout.
This keeps historical engine source/binary bindings unchanged and avoids a second
rollout/graph/PPO implementation. Workers execute one trainer in one process.
"""
from unittest.mock import patch
from simulators.clash.native.train import Trainer as NativeTrainer

class Trainer(NativeTrainer):
    def __init__(self,c,data_path,resume=None):
        import policy,learner
        self.editable_learner=learner
        with patch('model.Policy',lambda *args,**kwargs:policy.create(c)):
            super().__init__(c,data_path,resume)

    def learn(self,data):
        return self.editable_learner.update(self,data)

    def checkpoint(self,path,elapsed,*,entrant_binding,entrant_state):
        from .checkpoint_writer import checkpoint
        return checkpoint(self,path,elapsed,entrant_binding=entrant_binding,entrant_state=entrant_state)
