import copy,json,tempfile,unittest
from pathlib import Path
import numpy as np
import torch
from integrations.pokemon import cli
from integrations.pokemon.reference_training import trainer as t
from integrations.pokemon.rl import Population

class ReferenceTests(unittest.TestCase):
 def config(self):return cli.config_load(t.HERE/'recipes/local.json')
 def test_current_history_identity_and_both_seats(self):
  c=self.config();m=cli.initialize(c);p=Population(c,m,'cpu');p.add(m)
  self.assertFalse(any(x.requires_grad for x in p.history[0].parameters()))
  pairs=p.assignments(200,0,np.random.default_rng(1729),m)
  self.assertTrue(any(a is m and b is m for a,b in pairs))
  self.assertTrue(any(a is m and b is p.history[0] for a,b in pairs))
  self.assertTrue(any(b is m and a is p.history[0] for a,b in pairs))
 def test_exact_resume_and_finite_learning(self):
  c=self.config()
  with tempfile.TemporaryDirectory() as d:
   out=Path(d)/'proof';result=t.resume_proof(c,out)
   self.assertTrue(result['valid'])
   learned=json.loads((out/'first/results.json').read_text());self.assertGreater(learned['trained_samples'],0);self.assertGreater(learned['parameter_change_l2'],0)
   cp=t.admit(out/'first/checkpoint.pt',c);cp['executable_signature']='wrong';bad=Path(d)/'bad.pt';torch.save(cp,bad)
   with self.assertRaisesRegex(ValueError,'Runtime source'):t.admit(bad,c)
   cp['executable_signature']=cli.executable_signature();cp['reference_source_signature']='wrong';torch.save(cp,bad)
   with self.assertRaisesRegex(ValueError,'Reference source'):t.admit(bad,c)
if __name__=='__main__':unittest.main()
