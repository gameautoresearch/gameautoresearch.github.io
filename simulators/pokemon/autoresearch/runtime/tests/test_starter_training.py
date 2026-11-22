"""The native starter really learns and exports through this runtime, on the admitted engine.

Everything is the production path except privilege separation: the trusted
environment service runs in this process instead of as root, and the trainer
is an ordinary child process instead of uid 65534.
"""
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

import torch

from simulators.pokemon.autoresearch.runtime import bundle, sdk
from simulators.pokemon.autoresearch.runtime.env_service import serve_connection
from simulators.pokemon.autoresearch.runtime.fastenv import GPUCPU

ROOT=Path(__file__).resolve().parents[5]
RUNTIME=ROOT/'evidence/pokemon-implementation/native-cpu-completion-20261109/admitted-runtime-1/runtime.json'


@unittest.skipUnless(RUNTIME.is_file(),'admitted native runtime is not in this checkout')
class StarterTraining(unittest.TestCase):
    def train(self,root,name,config,*arguments):
        key=root/'key.bin'
        if not key.exists():key.write_bytes(os.urandom(32));key.chmod(0o600)
        path=root/(name+'.sock');out=root/name
        scope=dict(POKEMON_PILOT_RUNTIME=str(RUNTIME),POKEMON_PILOT_ENV_SCOPE='test',
            POKEMON_PILOT_RELEASE_SHA256=sdk.runtime_signature(),
            POKEMON_PILOT_ENV_SOURCE_SHA256=sdk.source_signature(sdk.STARTER))
        with patch.dict(os.environ,scope):
            environment=GPUCPU(config,'cpu');server=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM)
            try:
                server.bind(str(path));server.listen(1);server.settimeout(60)
                def serve():
                    connection,_=server.accept()
                    with connection:
                        connection.settimeout(120)
                        with patch.dict(os.environ,scope):serve_connection(connection,environment,config,key)
                service=threading.Thread(target=serve,daemon=True);service.start()
                (root/'config.json').write_text(json.dumps(config))
                done=subprocess.run([sys.executable,str(sdk.STARTER/'train.py'),'--config',str(root/'config.json'),
                    '--out',str(out),'--seconds','120','--device','cpu',*arguments],cwd=sdk.STARTER,capture_output=True,text=True,timeout=240,
                    env=dict(os.environ,PYTHONPATH=str(ROOT),POKEMON_PILOT_ENV_SOCKET=str(path),
                        POKEMON_PILOT_RELEASE_SHA256=sdk.runtime_signature(),OMP_NUM_THREADS='1'))
                service.join(timeout=30)
            finally:
                server.close();environment.close()
        self.assertEqual(done.returncode,0,done.stderr[-3000:])
        return out

    def test_smoke_config_learns_resumes_exactly_and_seals_a_verified_bundle(self):
        config=sdk.validate_recipe(json.loads((sdk.HERE.parent/'recipes/smoke.json').read_bytes()))
        sdk.validate_recipe(json.loads(sdk.STARTER_CONFIG.read_bytes()))
        with tempfile.TemporaryDirectory()as temporary:
            root=Path(temporary)
            whole=self.train(root,'whole',config,'--updates','2')
            result=json.loads((whole/'result.json').read_bytes())
            self.assertEqual((result['iterations'],result['faults'],result['runtime_sha256']),(2,0,sdk.runtime_signature()))
            self.assertGreater(result['trained_samples'],0);self.assertGreater(result['parameter_change_l2'],0)
            first=self.train(root,'first',config,'--updates','1')
            resumed=self.train(root,'resumed',config,'--updates','1','--resume',str(first/'checkpoint.pt'))
            a=torch.load(whole/'checkpoint.pt',weights_only=True);b=torch.load(resumed/'checkpoint.pt',weights_only=True)
            self.assertEqual(a['iteration'],b['iteration'])
            self.assertTrue(all(torch.equal(value,b['model'][name])for name,value in a['model'].items()))
            manifest=bundle.seal(root/'bundle',sdk.STARTER,config,whole/'checkpoint.pt',{'test':True})
            self.assertEqual(bundle.verify(root/'bundle',manifest['bundle_sha256'])['files'].keys(),set(bundle.FILES))
            binding=json.loads((root/'bundle/binding.json').read_bytes())
            self.assertEqual((binding['benchmark_id'],binding['source_sha256']),(sdk.BENCHMARK_ID,sdk.source_signature(sdk.STARTER)))


if __name__=='__main__':unittest.main()
