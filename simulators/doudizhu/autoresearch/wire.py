"""Length-prefixed public arrays. No pickle, deals, RNG, seeds or engine handles."""
import io
import json
import struct
import numpy as np

MAX_BYTES=16*1024*1024

def encode(role,static,history,actions,counts):
    if role not in ('landlord','landlord_down','landlord_up'):raise ValueError('Unknown role')
    f=io.BytesIO()
    np.savez(f,role=np.asarray(role),static=np.asarray(static,dtype=np.int8),history=np.asarray(history,dtype=np.int8),
             actions=np.asarray(actions,dtype=np.int8),counts=np.asarray(counts,dtype=np.int32))
    value=f.getvalue()
    if len(value)>MAX_BYTES:raise ValueError('Public inference batch too large')
    return struct.pack('>I',len(value))+value

def decode(value):
    if len(value)>MAX_BYTES:raise ValueError('Oversized request')
    with np.load(io.BytesIO(value),allow_pickle=False) as data:
        if set(data.files)!={'role','static','history','actions','counts'}:raise ValueError('Unexpected public fields')
        role=str(data['role']);x=data['static'];z=data['history'];a=data['actions'];counts=data['counts']
    width=319 if role=='landlord' else 430
    if role not in ('landlord','landlord_down','landlord_up') or x.ndim!=2 or x.shape[1]!=width:raise ValueError('Invalid static shape')
    if z.shape!=(len(x),5,162) or a.ndim!=2 or a.shape[1]!=54 or counts.shape!=(len(x),):raise ValueError('Invalid public shapes')
    if counts.dtype!=np.int32 or np.any(counts<=0) or int(counts.sum())!=len(a):raise ValueError('Invalid ragged legal counts')
    return role,x,z,a,counts

def read_exact(stream,n):
    parts=[]
    while n:
        part=stream.read(n)
        if not part:raise EOFError('Policy pipe closed')
        parts.append(part);n-=len(part)
    return b''.join(parts)

def read_frame(stream):
    head=stream.read(4)
    if not head:return None
    if len(head)!=4:head+=read_exact(stream,4-len(head))
    size=struct.unpack('>I',head)[0]
    if size>MAX_BYTES:raise ValueError('Policy message exceeds cap')
    return read_exact(stream,size)

def reply(indices):
    raw=json.dumps([int(x) for x in indices],separators=(',',':')).encode()
    return struct.pack('>I',len(raw))+raw
