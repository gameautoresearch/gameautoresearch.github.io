"""Retain already-transferred actor batches; assemble accepted slot trajectories on device.

No fixed horizon or invented boundary: only naturally completed valid lobby trajectories
are gathered, in exactly the legacy slot/decision order. Faulted lobby rows are omitted.
This removes per-decision feature/card/mask copies and their second host-to-device trip.
"""
import torch


class DeviceEpisodes(list):
    def __init__(self, episodes, chunks):
        super().__init__(episodes)
        self.chunks=chunks
        self._packed=None

    def pack(self, rows, device):
        if self._packed is None:
            # Raw chunk rows are in actor-call order; PPO is in slot/decision order.
            ids=torch.tensor([r['_device_row'] for r in rows],device=device,dtype=torch.long)
            packed=[]
            for column in range(3):
                data=torch.cat([chunk[column].to(device) for chunk in self.chunks],dim=0)
                packed.append(data.index_select(0,ids))
                del data
            self._packed=tuple(packed)
            self.chunks.clear()
        return self._packed
