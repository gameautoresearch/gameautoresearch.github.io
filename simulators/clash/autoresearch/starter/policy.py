"""Small random public-observation network; edit create/Policy to research it.
The trusted dependency is hash-bound in each exported bundle. No pretrained weights.
"""
from simulators.clash.autoresearch.fast_policy import Policy,create

def act(model,observation,mask):
    return model(observation,mask)[0].argmax(-1)
