"""Editable policy. create(model_config) and act(...) are the evaluation API."""
import numpy as np
from integrations.doudizhu.model import Policy

def create(config):return Policy(config)

def act(policy,role,static,history,actions,counts):
    q=policy.values_arrays(role,static,history,actions,counts,16384)
    starts=np.concatenate(([0],np.cumsum(counts)))
    return [int(np.argmax(q[starts[i]:starts[i+1]])) for i in range(len(counts))]
