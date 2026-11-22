import json,tempfile,unittest
from pathlib import Path
import numpy as np
import torch
from integrations.pokemon.network import Policy
from integrations.pokemon.reference_training.strong_runtime import teacher_step,validate_teacher,load_data
from integrations.pokemon.reference_training.strong_campaign import sha

class StrongTraining(unittest.TestCase):
 def data(self):
  n=12;rng=np.random.default_rng(9)
  return {'tokens':np.zeros((n,12,12),np.int16),'mons':np.zeros((n,12,24),np.float32),'global':np.zeros((n,40),np.float32),'global_tokens':np.zeros((n,4),np.int16),'action_tokens':np.zeros((n,14,3),np.int16),'actions':rng.normal(size=(n,14,12)).astype(np.float32),'mask':np.tile(np.asarray([True,False,False,True]+[False]*10),(n,1)),'action':np.full(n,3,np.int8),'seat':np.zeros(n,np.int8),'value':np.ones(n,np.float32)}
 def test_legal_teacher_learning_has_finite_gradients(self):
  torch.set_num_threads(1);torch.manual_seed(7);m=Policy(16,2);opt=torch.optim.Adam(m.parameters(),lr=.01);data=self.data();before=validate_teacher(m,data,'cpu')['cross_entropy']
  for _ in range(15):stats=teacher_step(m,opt,data,np.arange(12),'cpu')
  self.assertTrue(np.isfinite(stats['grad_norm']));self.assertLess(validate_teacher(m,data,'cpu')['cross_entropy'],before)
 def test_corpus_games_disjoint_and_faults_excluded(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);data=self.data();rows=[]
   for i in (0,1,2):
    file=p/f'g{i}.npz';np.savez_compressed(file,**data);rows.append({'index':i,'natural':i!=2,'truncated':False,'fault':'retained fault' if i==2 else None,'samples':12,'shard':file.name,'sha256':sha(file)})
   (p/'index.json').write_text(json.dumps(rows));result,manifest=load_data(p)
   self.assertEqual(len(manifest),2);self.assertEqual(len(result['train']['action']),12);self.assertEqual(len(result['validation']['action']),12)
   (p/'g1.npz').write_bytes(b'corrupt')
   with self.assertRaisesRegex(ValueError,'shard changed'):load_data(p)
 def test_parent_retention_limits_drift_under_conflicting_teacher_labels(self):
  import copy
  torch.set_num_threads(1);torch.manual_seed(19);parent=Policy(16,2)
  with torch.no_grad():parent.logits.bias[0]=6.
  parent.eval();parent.requires_grad_(False);plain=copy.deepcopy(parent);retained=copy.deepcopy(parent)
  plain.requires_grad_(True);retained.requires_grad_(True);data=self.data();indices=np.arange(12)
  a=torch.optim.Adam(plain.parameters(),lr=.01);b=torch.optim.Adam(retained.parameters(),lr=.01)
  for _ in range(40):
   teacher_step(plain,a,data,indices,'cpu');stats=teacher_step(retained,b,data,indices,'cpu',parent_anchor=parent,parent_kl_coef=4.)
  x=validate_teacher(plain,data,'cpu',parent,4.);y=validate_teacher(retained,data,'cpu',parent,4.)
  self.assertTrue(np.isfinite(stats['grad_norm']));self.assertLess(y['parent_kl'],x['parent_kl']);self.assertTrue(all(p.grad is None for p in parent.parameters()))
 def test_frozen_greedy_peer_preserves_rng_and_learner_stays_stochastic(self):
  from integrations.pokemon.reference_training.strong_runtime import SearchInference
  from integrations.pokemon.rl import Inference,Timings
  torch.set_num_threads(1);torch.manual_seed(31);learner=Policy(16,2);peer=Policy(16,2);d=self.data();observations=[{k:d[k][i] for k in ('tokens','mons','global','global_tokens','action_tokens','actions','mask')} for i in range(12)]
  inference=SearchInference(Inference('cpu',Timings()),greedy_policy=peer);before=torch.get_rng_state().clone();actions,_,_=inference(peer,observations)
  self.assertTrue(torch.equal(before,torch.get_rng_state()));self.assertTrue(d['mask'][np.arange(12),actions].all())
  inference(learner,observations);self.assertFalse(torch.equal(before,torch.get_rng_state()))
if __name__=='__main__':unittest.main()

class SparseTransport(unittest.TestCase):
 def test_exact_qualified_states_and_requested_player_view(self):
  import re
  from integrations.pokemon.autoresearch.evaluation_v2.env import EvaluationVector
  from integrations.pokemon.reference_training.strong_runtime import ReferenceVector
  from integrations.pokemon.network import scripted
  def normalized(value):
   if isinstance(value,str):return re.sub(r'^\|t:\|[0-9]+$', '|t:|0',value,flags=re.MULTILINE)
   if isinstance(value,list):return [normalized(x) for x in value]
   if isinstance(value,dict):return {k:normalized(v) for k,v in value.items()}
   return value
  with EvaluationVector(4) as original,ReferenceVector(1,4) as sparse:
   seeds=list(range(930300000,930300004));a=original.reset(seeds,4096);b=sparse.reset(seeds,4096)
   rngs=[[np.random.default_rng(i*10+s) for s in (0,1)] for i in range(4)]
   for _ in range(20):
    self.assertEqual(normalized(a),normalized(b))
    old=original.visible([0,1,0,1],[0]*4);new=sparse.visible([None,1,None,None],[0]*4)
    self.assertEqual(normalized(old[1]),normalized(new[1]));self.assertIsNone(new[0]);self.assertIsNone(new[2]);self.assertIsNone(new[3])
    actions=[[scripted(x['observations'][s],rngs[i][s],'power') if x['ready'][s] else None for s in (0,1)] for i,x in enumerate(a)]
    a=original.step(actions);b=sparse.step(actions)
