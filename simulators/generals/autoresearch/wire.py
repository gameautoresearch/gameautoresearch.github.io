"""Bounded binary public observation protocol. No seeds, engine state or pickle."""
import io
import json
import struct
import numpy as np
MAX_BYTES=8*1024*1024

def encode(x,mask):
 f=io.BytesIO();np.savez(f,x=np.asarray(x,dtype=np.float32),mask=np.packbits(mask,axis=-1,bitorder='little'))
 value=f.getvalue()
 if len(value)>MAX_BYTES:raise ValueError('Public batch exceeds cap')
 return struct.pack('>I',len(value))+value

def decode(value):
 if len(value)>MAX_BYTES:raise ValueError('Public request exceeds cap')
 with np.load(io.BytesIO(value),allow_pickle=False) as a:
  if set(a.files)!={'x','mask'}:raise ValueError('Unexpected public fields')
  x=a['x'];packed=a['mask']
 # The ruleset's board and action count; imported here so the host-side encoder stays free of the engine.
 from integrations.generals.environment import BOARD_SHAPE,ACTION_COUNT
 if x.dtype!=np.float32 or x.ndim!=4 or x.shape[1:]!=(*BOARD_SHAPE,14) or not 1<=len(x)<=256:raise ValueError('Wrong public observation shape')
 if packed.dtype!=np.uint8 or packed.shape!=(len(x),-(-ACTION_COUNT//8)) or not np.isfinite(x).all():raise ValueError('Wrong legal mask or nonfinite observations')
 mask=np.unpackbits(packed,axis=-1,count=ACTION_COUNT,bitorder='little').astype(bool)
 if not mask[:,0].all():raise ValueError('Pass must be legal')
 return x,mask

def read_exact(stream,n):
 parts=[]
 while n:
  part=stream.read(n)
  if not part:raise EOFError('Inference pipe closed')
  parts.append(part);n-=len(part)
 return b''.join(parts)

def read_frame(stream):
 head=stream.read(4)
 if not head:return None
 if len(head)<4:head+=read_exact(stream,4-len(head))
 size=struct.unpack('>I',head)[0]
 if size>MAX_BYTES:raise ValueError('Frame cap exceeded')
 return read_exact(stream,size)

def reply(indices):
 raw=json.dumps([int(i) for i in indices],separators=(',',':')).encode()
 return struct.pack('>I',len(raw))+raw
