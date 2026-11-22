"""Root-only admitted simulator service; bounded JSON/raw-byte IPC, never pickle.

Snapshot encryption is encrypt-then-MAC with distinct HMAC-SHA256-derived keys,
a fresh 256-bit nonce, SHAKE256 keystream and HMAC-SHA256 authentication. Source,
rules, release, exact recipe and arm scope bind every opaque checkpoint.
"""
import argparse
import hashlib
import hmac
import json
import os
from pathlib import Path
import secrets
import socket
import struct

import numpy as np

from .env_client import receive,send,MAX_FRAME_BYTES

MAGIC=b'PKNSTATE1'
KEY_DEFAULT='/pilotjob/input/env-state-key.bin'


def identity(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def _key(path=None):
    path=Path(path or os.environ.get('POKEMON_PILOT_ENV_KEY',KEY_DEFAULT))
    if path.is_symlink()or not path.is_file()or path.stat().st_size!=32:
        raise ValueError('Exact private environment key required')
    if path.stat().st_mode&0o077:raise ValueError('Environment key must be private')
    return path.read_bytes()


def _xor(raw,key,nonce):
    stream=hashlib.shake_256(b'pokemon-pilot-state-stream-v1\0'+key+nonce).digest(len(raw))
    return np.bitwise_xor(np.frombuffer(raw,dtype=np.uint8),np.frombuffer(stream,dtype=np.uint8)).tobytes()


def seal_snapshot(state,binding,key_path=None):
    context=bytes.fromhex(identity(binding))
    meta=dict(state);states=meta.pop('states')
    if not isinstance(states,bytes):raise ValueError('Exact native bytes required')
    encoded=json.dumps(meta,allow_nan=False,separators=(',',':'),sort_keys=True).encode()
    raw=struct.pack('>I',len(encoded))+encoded+states
    if len(encoded)>1024*1024 or len(raw)>MAX_FRAME_BYTES-128:raise ValueError('Bounded native state required')
    master=_key(key_path)
    enc=hmac.digest(master,b'pokemon-pilot-state-encryption-v1\0'+context,'sha256')
    mac=hmac.digest(master,b'pokemon-pilot-state-authentication-v1\0'+context,'sha256')
    nonce=secrets.token_bytes(32);cipher=_xor(raw,enc,nonce)
    header=MAGIC+context+nonce
    return header+hmac.digest(mac,header+cipher,'sha256')+cipher


def decrypt_snapshot(blob,key_path=None,expected_binding=None):
    """Trusted root audit/resume helper. Never expose its result to a learner."""
    if not isinstance(blob,bytes)or not len(MAGIC)+96+4<=len(blob)<=MAX_FRAME_BYTES:
        raise ValueError('Invalid bounded sealed native state')
    if blob[:len(MAGIC)]!=MAGIC:raise ValueError('Wrong sealed state version')
    offset=len(MAGIC);context=blob[offset:offset+32];nonce=blob[offset+32:offset+64]
    tag=blob[offset+64:offset+96];cipher=blob[offset+96:]
    if expected_binding is not None and context!=bytes.fromhex(identity(expected_binding)):
        raise ValueError('Native state source/rules/recipe binding differs')
    master=_key(key_path)
    mac=hmac.digest(master,b'pokemon-pilot-state-authentication-v1\0'+context,'sha256')
    if not hmac.compare_digest(tag,hmac.digest(mac,blob[:offset+64]+cipher,'sha256')):
        raise ValueError('Native state authentication failed')
    enc=hmac.digest(master,b'pokemon-pilot-state-encryption-v1\0'+context,'sha256')
    raw=_xor(cipher,enc,nonce);size=struct.unpack('>I',raw[:4])[0]
    if not 0<size<=1024*1024 or size>len(raw)-4:raise ValueError('Invalid native state metadata')
    state=json.loads(raw[4:4+size])
    if not isinstance(state,dict)or'states'in state:raise ValueError('Invalid native state schema')
    state['states']=raw[4+size:]
    return state


def snapshots_equal(left,right,key_path=None):
    """Compare randomized encrypted checkpoints without releasing raw state."""
    a=decrypt_snapshot(left,key_path);b=decrypt_snapshot(right,key_path)
    return left[len(MAGIC):len(MAGIC)+32]==right[len(MAGIC):len(MAGIC)+32]and a==b


def environment_binding(env,config):
    return dict(config_sha256=identity(config),scope=os.environ.get('POKEMON_PILOT_ENV_SCOPE','qualification'),
        candidate_source_sha256=os.environ.get('POKEMON_PILOT_ENV_SOURCE_SHA256',''),
        release_sha256=os.environ.get('POKEMON_PILOT_RELEASE_SHA256',''),
        **{name:getattr(env,name)for name in ('source_signature','binary_sha256','rules_sha256',
            'initial_sha256','integration_signature','runtime_manifest_sha256')})


def serve_connection(connection,env,config,key_path=None):
    """Only projected observations, rewards, boundary flags and hashes cross IPC."""
    import torch
    binding=environment_binding(env,config);allocated=None;hello=False
    while True:
        try:request,blobs=receive(connection)
        except EOFError:return
        try:
            operation=request.get('op')
            if operation=='hello':
                if hello or blobs or request.get('config_sha256')!=identity(config):raise ValueError('Exact one-time environment recipe handshake required')
                hello=True
                metadata=dict(ok=True,n=env.n,**{name:getattr(env,name)for name in (
                    'source_signature','binary_sha256','rules_sha256','initial_sha256','integration_signature','runtime_manifest_sha256')})
                send(connection,metadata,(env.obs.contiguous().numpy(),));continue
            if not hello:raise ValueError('Environment handshake required')
            if operation=='allocate':
                steps=request.get('steps')
                if blobs or type(steps)is not int or not 1<=steps<=256:raise ValueError('Bounded rollout allocation required')
                env.allocate_rollout(steps);allocated=steps
                # A malformed collector must never read uninitialized host data.
                env.flags.zero_();env.rewards.zero_();env.cpu.flags.zero_();env.cpu.rewards.zero_();env.cpu.final_obs.zero_()
                send(connection,dict(ok=True));continue
            if operation=='step':
                index=request.get('index')
                if allocated is None or type(index)is not int or not 0<=index<allocated or len(blobs)!=1 or len(blobs[0])!=env.n*2*4:
                    raise ValueError('Exact bounded step schema required')
                actions=torch.from_numpy(np.frombuffer(blobs[0],dtype=np.int32).reshape(env.n,2).copy())
                env.step(actions,index,0)
                send(connection,dict(ok=True),(env.obs.contiguous().numpy(),env.flags[index].contiguous().numpy(),env.rewards[index].contiguous().numpy()));continue
            if operation=='truncated_views':
                if blobs or allocated is None:raise ValueError('Allocated rollout required')
                views,_=env.truncated_views();views=views.contiguous()
                send(connection,dict(ok=True,count=len(views)),(views.numpy(),));continue
            if operation=='snapshot':
                if blobs:raise ValueError('Snapshot has no caller payload')
                send(connection,dict(ok=True),(seal_snapshot(env.snapshot(),binding,key_path),));continue
            if operation=='restore':
                if len(blobs)!=1:raise ValueError('One authenticated state required')
                env.restore(decrypt_snapshot(bytes(blobs[0]),key_path,expected_binding=binding))
                send(connection,dict(ok=True),(env.obs.contiguous().numpy(),));continue
            if operation=='close':
                if blobs:raise ValueError('Close has no payload')
                send(connection,dict(ok=True));return
            raise ValueError('Unsupported public environment operation')
        except BaseException as exc:
            # No deserialized candidate object or arbitrary callable reaches root.
            send(connection,dict(ok=False,error=type(exc).__name__+': '+str(exc)[:1000]));raise


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--socket',required=True);parser.add_argument('--config',required=True)
    args=parser.parse_args()
    if os.geteuid()!=0:raise RuntimeError('Native environment service requires trusted root identity')
    import torch
    from .sdk import validate_recipe
    from .fastenv import GPUCPU
    torch.set_num_threads(1)
    config=validate_recipe(json.loads(Path(args.config).read_bytes()));_key()
    env=GPUCPU(config,'cpu');server=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM)
    path=Path(args.socket)
    try:
        server.bind(str(path));path.chmod(0o666);server.listen(1);server.settimeout(60)
        connection,_=server.accept()
        with connection:
            connection.settimeout(120);serve_connection(connection,env,config)
    finally:
        server.close();env.close()
        if path.exists():path.unlink()


if __name__=='__main__':main()
