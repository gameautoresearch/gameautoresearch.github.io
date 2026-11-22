"""Offline wire/state boundary tests with a fake environment; no game compute."""
import json
import os
from pathlib import Path
import socket
import struct
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from simulators.pokemon.autoresearch.runtime.env_client import Client,receive,send,MAX_FRAME_BYTES
from simulators.pokemon.autoresearch.runtime.env_service import (
    seal_snapshot,decrypt_snapshot,snapshots_equal,serve_connection,environment_binding)


class FakeEnvironment:
    def __init__(self):
        self.n=2;self.obs=torch.zeros((2,2,700));self.counter=0
        for name in ('source_signature','binary_sha256','rules_sha256','initial_sha256',
                     'integration_signature','runtime_manifest_sha256'):
            setattr(self,name,'a'*64)
    def allocate_rollout(self,steps):
        self.flags=torch.zeros((steps,2,3),dtype=torch.int32)
        self.rewards=torch.zeros((steps,2,2),dtype=torch.float32)
        self.cpu=SimpleNamespace(flags=self.flags,rewards=self.rewards,
            final_obs=torch.zeros((steps,2,2,700)))
    def step(self,actions,index,reward):
        if tuple(actions.shape)!=(2,2)or reward!=0:raise ValueError('Bad public action schema')
        self.counter+=1;self.obs.fill_(self.counter);self.rewards[index].fill_(self.counter)
    def truncated_views(self):
        mask=self.flags[...,1].bool()[...,None].expand(-1,-1,2).reshape(-1)
        return self.cpu.final_obs.reshape(-1,700)[mask],mask
    def snapshot(self):
        return dict(states=b'PRIVATE_NATIVE_SECRET'+bytes([self.counter]),episodes=[self.counter,0],
            offsets=[0,0],stream=dict(start=1,length=64,window=2,n=2))
    def restore(self,state):
        self.counter=state['episodes'][0];self.obs.fill_(self.counter)


class EnvironmentBoundary(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory();self.root=Path(self.temporary.name)
        self.key=self.root/'key';self.key.write_bytes(b'K'*32);self.key.chmod(0o600)
    def tearDown(self):self.temporary.cleanup()

    def test_encrypted_authenticated_opaque_state_and_exact_comparison(self):
        state=FakeEnvironment().snapshot();binding={'source':'a','recipe':{'x':1}}
        first=seal_snapshot(state,binding,self.key);second=seal_snapshot(state,binding,self.key)
        self.assertNotEqual(first,second);self.assertNotIn(state['states'],first)
        self.assertEqual(decrypt_snapshot(first,self.key,expected_binding=binding),state)
        self.assertTrue(snapshots_equal(first,second,self.key))
        changed=dict(state,episodes=[9,0]);other=seal_snapshot(changed,binding,self.key)
        self.assertFalse(snapshots_equal(first,other,self.key))
        with self.assertRaisesRegex(ValueError,'binding'):decrypt_snapshot(first,self.key,expected_binding={'source':'b'})
        for index in (10,45,80,len(first)-1):
            corrupt=bytearray(first);corrupt[index]^=1
            with self.assertRaises(ValueError):decrypt_snapshot(bytes(corrupt),self.key)
        self.key.chmod(0o644)
        with self.assertRaisesRegex(ValueError,'private'):decrypt_snapshot(first,self.key)

    def test_wire_rejects_nonobject_oversize_and_negative_lengths(self):
        for header in ([],{'_blob_lengths':[-1]},{'_blob_lengths':[MAX_FRAME_BYTES+1]}):
            left,right=socket.socketpair()
            try:
                raw=json.dumps(header).encode();left.sendall(struct.pack('>I',len(raw))+raw)
                with self.assertRaises(ValueError):receive(right)
            finally:left.close();right.close()
        left,right=socket.socketpair()
        try:
            left.sendall(struct.pack('>I',65537))
            with self.assertRaises(ValueError):receive(right)
        finally:left.close();right.close()

    def test_client_only_public_views_and_opaque_resume(self):
        path=str(self.root/'env.sock');server=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM)
        server.bind(path);server.listen(1);env=FakeEnvironment();config={'environments':2};errors=[]
        def run():
            try:
                connection,_=server.accept()
                with connection:serve_connection(connection,env,config,self.key)
            except BaseException as error:errors.append(error)
            finally:server.close()
        thread=threading.Thread(target=run,daemon=True);thread.start()
        with patch.dict(os.environ,{'POKEMON_PILOT_ENV_SOCKET':path}):
            client=Client(config,'cpu')
            try:
                self.assertFalse(hasattr(client,'cpu'));self.assertFalse(hasattr(client,'arena'))
                client.allocate_rollout(2);state=client.snapshot()
                client.step(torch.zeros((2,2),dtype=torch.int64),0)
                self.assertTrue(torch.all(client.obs==1));self.assertTrue(torch.all(client.rewards[0]==1))
                client.restore(state);self.assertTrue(torch.all(client.obs==0))
                self.assertTrue(snapshots_equal(state,client.snapshot(),self.key))
                views,mask=client.truncated_views();self.assertEqual(tuple(views.shape),(0,700));self.assertEqual(int(mask.sum()),0)
            finally:client.close()
        thread.join(5);self.assertFalse(thread.is_alive());self.assertEqual(errors,[])

    def test_service_refuses_wrong_recipe_before_observations(self):
        left,right=socket.socketpair();errors=[]
        def run():
            try:serve_connection(right,FakeEnvironment(),{'environments':2},self.key)
            except ValueError as error:errors.append(error)
            finally:right.close()
        thread=threading.Thread(target=run,daemon=True);thread.start()
        try:
            send(left,{'op':'hello','config_sha256':'wrong'})
            response,blobs=receive(left);self.assertFalse(response['ok']);self.assertEqual(blobs,[])
        finally:left.close()
        thread.join(5);self.assertEqual(len(errors),1)


if __name__=='__main__':unittest.main()
