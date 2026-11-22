import copy,json,time,shutil,hashlib
from pathlib import Path
import numpy as np
import pytest
import torch
from simulators.tavern.reference_training.strong.common import ROOT,validate_config
from simulators.tavern.reference_training.strong.population import schedule
from simulators.tavern.reference_training.strong.lineage import import_parent,module
from simulators.tavern.reference_training.strong.train import rng_state,restore_rng
from simulators.tavern.reference_training import collection
from simulators.tavern.reference_training.validate import equal

HERE=Path(__file__).resolve().parents[1]
def test_all_seat_on_policy_and_three_independent_scripts():
 a=schedule(531901,0,48,3)
 assert np.array_equal(a,schedule(531901,0,48,3))
 assert ((a==-1).sum(0)==24).all()
 for row in a:
  assert (row==-1).sum()==4 and sum(row==4)==sum(row==5)==sum(row==6)==1
  assert ((row>=1)&(row<=3)).sum()==1

def test_frozen_rules_are_enforced():
 c=json.loads((HERE/'curriculum.json').read_text());validate_config(c)
 c['rules']['max_turns']=99
 with pytest.raises(ValueError):validate_config(c)

def test_frozen_source_import_does_not_write_bytecode(tmp_path):
 source=tmp_path/'source.py';source.write_text('VALUE = 7\n')
 assert module(source,'frozen_input_cache_fixture').VALUE==7
 assert {p.name for p in tmp_path.iterdir()}=={'source.py'}

def test_strong_dense_compact_collection_exact():
 inputs=ROOT/'evidence/strong-bots-20261109-0516Z/tavern/rl/inputs-v1'
 if not inputs.exists():pytest.skip('Private incumbent import fixtures are local')
 torch.set_num_threads(1);model,_,pop,_=import_parent(inputs,torch.device('cpu'))
 c=json.loads((HERE/'curriculum.json').read_text());c.update(device='cpu',n_games=2,sparse_observations=False)
 scripts=module(inputs/'scripts/opponents.py','strong_script_contract');actors=pop.actors(torch.device('cpu'))+[scripts.VisibleScript(s) for s in ('tempo','economy','synergy')]
 assignment=schedule(c['seed'],0,2,len(pop.entries));torch.manual_seed(8317);rng=rng_state()
 a=collection.collect(model,c,531301,torch.device('cpu'),collection.Timer('cpu'),time.perf_counter()+120,assignment=assignment,population=actors)
 restore_rng(rng);c['sparse_observations']=True
 b=collection.collect(model,c,531301,torch.device('cpu'),collection.Timer('cpu'),time.perf_counter()+120,assignment=assignment,population=actors)
 assert equal(a[0],b[0]) and equal(a[1],b[1]) and equal(a[3],b[3])
 assert a[2]['accepted_learning_samples']==b[2]['accepted_learning_samples']
 assert {r['seat'] for r in a[1]}=={0,1,2,3,4,5,6,7}

def test_explicit_own_parent_keeps_policy_optimizer_rng_and_lineage():
 inputs=ROOT/'evidence/strong-bots-20261109-0516Z/tavern/rl/fork-fixture-v1'
 if not inputs.exists():pytest.skip('Private full-state migration fixture is local')
 original=torch.load(inputs/'parent-resume.pt',map_location='cpu',weights_only=False)
 model,parent,pop,receipt=import_parent(inputs,torch.device('cpu'))
 assert equal(original['policy'],model.state_dict()) and equal(original['optimizer'],parent['optimizer'])
 assert torch.equal(original['rng']['torch'],parent['torch_rng'])
 assert receipt['parent_lineage']==original['lineage'] and receipt['not_an_exact_old_source_resume']
 assert receipt['parent_total_known_samples']==original['lineage']['parent_samples']+sum(r['learning']['samples'] for r in original['history'])
 assert {e['sha256'] for e in original['population']['entries']}<={e['sha256'] for e in pop.entries}

def test_explicit_fork_rejects_parent_source_even_if_input_manifest_is_resealed(tmp_path):
 inputs=ROOT/'evidence/strong-bots-20261109-0516Z/tavern/rl/fork-fixture-v1'
 if not inputs.exists():pytest.skip('Private full-state migration fixture is local')
 copied=tmp_path/'inputs';shutil.copytree(inputs,copied)
 relative='parent-source/simulators/tavern/reference_training/strong/policy.py'
 target=copied/relative;target.chmod(0o644);target.write_text(target.read_text()+'\n# changed parent source\n')
 manifest=json.loads((copied/'manifest.json').read_text());manifest[relative]=hashlib.sha256(target.read_bytes()).hexdigest();(copied/'manifest.json').write_text(json.dumps(manifest))
 with pytest.raises(ValueError,match='Archived strong parent source mutated'):import_parent(copied,torch.device('cpu'))

def test_incumbent_kl_moves_toward_teacher_without_changing_teacher():
 import taverngym as tg
 from simulators.tavern.reference_training.strong.policy import Policy
 from simulators.tavern.reference_training.strong.learner import update
 torch.set_num_threads(1);torch.manual_seed(582);teacher=Policy(hidden=16,embedding=2).eval().requires_grad_(False);model=copy.deepcopy(teacher).requires_grad_(True)
 c=json.loads((HERE/'curriculum-anchor.json').read_text());c.update(ppo_epochs=1,value_coef=0.,entropy_coef=0.,anchor_kl_coef=1.,target_kl=0.)
 obs=tg.BatchEnv(1,seed=531303,controlled=[0,1],config=c['rules'],threads=1).observe();f=torch.from_numpy(obs['features']);cards=torch.from_numpy(obs['cards'].astype(np.int64));mask=torch.from_numpy(obs['action_mask']).bool()
 action=int(np.flatnonzero(obs['action_mask'][0])[0])
 with torch.no_grad():model.actor.bias[action]+=.7
 def divergence():
  with torch.no_grad():
   a,_=teacher.distribution(f,cards,mask);b,_=model.distribution(f,cards,mask);return float((a.probs*(a.logits-b.logits)).sum(-1).mean())
 before=divergence();saved=copy.deepcopy(teacher.state_dict())
 with torch.no_grad():dist,_=model.distribution(f,cards,mask)
 episodes=[[dict(features=obs['features'][i],cards=obs['cards'][i],mask=obs['action_mask'][i],action=int(dist.logits[i].argmax()),logp=float(dist.logits[i].max()),value=0.,reward=0.)] for i in range(2)]
 result=update(model,torch.optim.Adam(model.parameters(),lr=1e-5),episodes,c,torch.device('cpu'),anchor=teacher)
 assert result['incumbent_kl']>0 and divergence()<before
 assert equal(saved,teacher.state_dict()) and all(p.grad is None for p in teacher.parameters())
