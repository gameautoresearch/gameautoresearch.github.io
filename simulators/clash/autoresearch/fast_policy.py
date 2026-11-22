"""Stateless public-only adapter for the qualified Policy64/e12 network.

No weights, teacher, opponent state or simulator state are imported. The constructor
reads only frozen schema/vocabulary metadata. Both training and exported inference
use the same model parameters and the same full 200-row public encoder.
"""
import json
from .common import DATA,setup_native
setup_native()
from simulators.clash.model import Policy as NativePolicy
from simulators.clash.native.train import schema
from gpu_features import DeviceBatchEncoder

class Policy(NativePolicy):
    def forward(self,observation,mask=None):
        batch=observation if mask is None else DeviceBatchEncoder(self.s,copy=False)(observation,mask)
        return super().forward(batch)

def create(config):
    data=json.loads(DATA.read_text())
    return Policy(schema(data),len(data['vocabulary']),len(data['buff_names']),config.get('width',64),config.get('embedding',12))
