"""Public projected-observation client. No simulator or native state is imported."""
import json
import os
import socket
import struct

import numpy as np
import torch

MAX_HEADER_BYTES=65536
MAX_FRAME_BYTES=256*1024*1024


def _read(sock,length):
    if type(length)is not int or not 0<=length<=MAX_FRAME_BYTES:
        raise ValueError('Invalid bounded environment frame length')
    result=bytearray(length);view=memoryview(result)
    while view:
        count=sock.recv_into(view)
        if not count:raise EOFError('Environment service disconnected')
        view=view[count:]
    return result


def receive(sock):
    size=struct.unpack('>I',_read(sock,4))[0]
    if not 0<size<=MAX_HEADER_BYTES:raise ValueError('Invalid environment header size')
    header=json.loads(_read(sock,size))
    if not isinstance(header,dict):raise ValueError('Environment header must be an object')
    lengths=header.pop('_blob_lengths',None)
    if (not isinstance(lengths,list)or len(lengths)>4
            or any(type(n)is not int or n<0 for n in lengths)or sum(lengths)>MAX_FRAME_BYTES):
        raise ValueError('Invalid bounded environment payload')
    return header,[_read(sock,n)for n in lengths]


def send(sock,header,blobs=()):
    blobs=[memoryview(blob)for blob in blobs]
    blobs=[blob.cast('B')if blob.nbytes else memoryview(b'')for blob in blobs]
    if len(blobs)>4 or sum(len(blob)for blob in blobs)>MAX_FRAME_BYTES:
        raise ValueError('Oversized environment payload')
    data=json.dumps(dict(header,_blob_lengths=[len(blob)for blob in blobs]),allow_nan=False,separators=(',',':')).encode()
    if len(data)>MAX_HEADER_BYTES:raise ValueError('Oversized environment header')
    sock.sendall(struct.pack('>I',len(data)));sock.sendall(data)
    for blob in blobs:sock.sendall(blob)


def _identity(config):
    import hashlib
    return hashlib.sha256(json.dumps(config,sort_keys=True,separators=(',',':')).encode()).hexdigest()


class Client:
    backend='admitted_native_cpu_gpu_learning'
    def __init__(self,config,device):
        self.n=config['environments'];self.device=torch.device(device);self.steps=None
        self.socket=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM);self.socket.settimeout(120)
        self.closed=False
        try:
            self.socket.connect(os.environ['POKEMON_PILOT_ENV_SOCKET'])
            response,blobs=self._request('hello',config_sha256=_identity(config))
            if response['n']!=self.n or len(blobs)!=1:raise ValueError('Environment public schema differs')
            for name in ('source_signature','binary_sha256','rules_sha256','initial_sha256',
                         'integration_signature','runtime_manifest_sha256'):
                value=response[name]
                if not isinstance(value,str)or len(value)!=64:raise ValueError('Invalid environment identity')
                setattr(self,name,value)
            self.obs=self._tensor(blobs[0],np.float32,(self.n,2,700)).to(self.device)
        except BaseException:
            self.socket.close();self.closed=True;raise

    def _request(self,operation,blobs=(),**fields):
        send(self.socket,dict(op=operation,**fields),blobs)
        response,payloads=receive(self.socket)
        if response.get('ok')is not True:
            raise RuntimeError('Trusted environment fault: '+str(response.get('error','unspecified')))
        return response,payloads

    @staticmethod
    def _tensor(blob,dtype,shape):
        if len(blob)!=int(np.prod(shape))*np.dtype(dtype).itemsize:
            raise ValueError('Environment tensor byte count differs')
        return torch.from_numpy(np.frombuffer(blob,dtype=dtype).reshape(shape).copy())

    def allocate_rollout(self,steps):
        if type(steps)is not int or not 1<=steps<=256:raise ValueError('Bounded rollout required')
        _,blobs=self._request('allocate',steps=steps)
        if blobs:raise ValueError('Unexpected allocation payload')
        self.steps=steps
        self.flags=torch.zeros((steps,self.n,3),dtype=torch.int32,device=self.device)
        self.rewards=torch.zeros((steps,self.n,2),dtype=torch.float32,device=self.device)

    def step(self,actions,index,faint_reward=0):
        if faint_reward!=0:raise ValueError('Frozen terminal rewards only')
        if self.steps is None or type(index)is not int or not 0<=index<self.steps:
            raise ValueError('Rollout index outside allocation')
        if tuple(actions.shape)!=(self.n,2):raise ValueError('Expected actions [environments,2]')
        host=actions.detach().to(device='cpu',dtype=torch.int32).contiguous().numpy()
        _,blobs=self._request('step',index=index,blobs=(host,))
        if len(blobs)!=3:raise ValueError('Environment step schema differs')
        self.obs.copy_(self._tensor(blobs[0],np.float32,(self.n,2,700)))
        self.flags[index].copy_(self._tensor(blobs[1],np.int32,(self.n,3)))
        self.rewards[index].copy_(self._tensor(blobs[2],np.float32,(self.n,2)))

    def truncated_views(self):
        response,blobs=self._request('truncated_views')
        count=response['count']
        if type(count)is not int or count<0 or len(blobs)!=1:raise ValueError('Invalid truncation schema')
        views=self._tensor(blobs[0],np.float32,(count,700)).to(self.device)
        selected=self.flags[...,1].bool()[...,None].expand(-1,-1,2).reshape(-1)
        if int(selected.sum())!=count:raise ValueError('Truncation boundary selection differs')
        return views,selected

    def snapshot(self):
        _,blobs=self._request('snapshot')
        if len(blobs)!=1:raise ValueError('Invalid sealed state payload')
        return bytes(blobs[0])

    def restore(self,state):
        if not isinstance(state,bytes):raise ValueError('Authenticated opaque state required')
        _,blobs=self._request('restore',blobs=(state,))
        if len(blobs)!=1:raise ValueError('Invalid restore observation payload')
        self.obs.copy_(self._tensor(blobs[0],np.float32,(self.n,2,700)))

    def close(self):
        if self.closed:return
        try:self._request('close')
        except (OSError,EOFError,RuntimeError):pass
        finally:self.closed=True;self.socket.close()
