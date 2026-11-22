"""Bounded CUDA inference graphs; legal candidates are padded internally, never dropped.

Graphs retain parameter storage, so ordinary in-place learner updates are visible on
next replay. They contain no optimizer/RNG state and are rebuilt after checkpoint
loading. Eager inference remains the portable/default path.
"""
from collections import OrderedDict

import numpy as np
import torch


class GraphScorer:
    def __init__(self, net, capacity=6):
        self.net = net
        self.capacity = capacity
        self.cache = OrderedDict()
        self.captures = 0

    def _get(self, histories, candidates, width):
        hsize = max(64, 1 << (histories-1).bit_length())
        csize = max(128, 1 << (candidates-1).bit_length())
        key = (hsize,csize,width)
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        device = next(self.net.parameters()).device
        xhost = torch.zeros((csize,width),dtype=torch.float32,pin_memory=True)
        owners_host = torch.zeros(csize,dtype=torch.int64,pin_memory=True)
        x = torch.zeros((csize,width),device=device)
        z = torch.zeros((hsize,5,162),device=device)
        owners = torch.zeros(csize,dtype=torch.int64,device=device)
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream):
            for _ in range(3):
                self.net.head(self.net.encode_history(z)[owners],x)
        torch.cuda.current_stream(device).wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = self.net.head(self.net.encode_history(z)[owners],x)
        item = (graph,x,z,owners,output,xhost,owners_host)
        self.cache[key] = item
        self.captures += 1
        while len(self.cache) > self.capacity:
            self.cache.popitem(last=False)
        return item

    def values(self, static, z, actions, owners, max_candidates):
        values = []
        for offset in range(0,len(actions),max_candidates):
            end = min(offset+max_candidates,len(actions))
            n = end-offset
            graph,x,zz,oi,output,xhost,ohost = self._get(len(static),n,static.shape[1]+54)
            host = xhost.numpy()
            # Casting directly into reusable pinned storage avoids concatenate +
            # float-conversion allocations for the expanded candidate matrix.
            host[:n,:static.shape[1]] = static[owners[offset:end]]
            host[:n,static.shape[1]:] = actions[offset:end]
            ohost.numpy()[:n] = owners[offset:end]
            x.copy_(xhost,non_blocking=True)
            oi.copy_(ohost,non_blocking=True)
            zz[:len(static)].copy_(z)
            graph.replay()
            # Actions depend on these values now; one required host boundary/chunk.
            values.append(output[:n].cpu().numpy())
        return np.concatenate(values)
