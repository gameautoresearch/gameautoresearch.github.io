import json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from integrations.pokemon.reference_training.remote import download_tree

class CollectionRecovery(unittest.TestCase):
 def test_interruption_preserves_old_file_then_recovers_atomically(self):
  class Volume:
   interrupted=True
   def listdir(self,*args,**kwargs):return [SimpleNamespace(path='jobs/test/checkpoint.pt',type=SimpleNamespace(name='FILE'))]
   def read_file(self,path):
    yield b'valid-checkpoint-'
    if self.interrupted:raise ConnectionError('simulated host disconnect')
    yield b'complete'
  v=Volume()
  with tempfile.TemporaryDirectory() as directory:
   out=Path(directory);old=out/'checkpoint.pt';old.write_bytes(b'incomplete-prior-copy')
   with self.assertRaises(ConnectionError):download_tree(v,'/jobs/test',out)
   self.assertEqual(old.read_bytes(),b'incomplete-prior-copy');self.assertFalse((out/'collection-complete.json').exists())
   v.interrupted=False;result=download_tree(v,'/jobs/test',out)
   self.assertEqual(old.read_bytes(),b'valid-checkpoint-complete');self.assertTrue(result['complete']);self.assertEqual(len(list(out.glob('checkpoint.pt.incomplete-*'))),1)
   def forbidden(*a,**k):raise AssertionError('Completed files must not be re-downloaded')
   v.listdir=forbidden;self.assertEqual(download_tree(v,'/jobs/test',out),result)
if __name__=='__main__':unittest.main()
