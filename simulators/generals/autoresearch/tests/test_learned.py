"""A submitted policy must be learned (learned.py): its moves have to come from its weights. Found 2026-11-19, when
both smoke sessions of the empty starter submitted hand-written players as `policy.py` (one with a single unused
parameter, one with fixed move scores under a 15-weight correction) and practice accepted them."""
import importlib.util
import shutil
import sys
import numpy as np
import pytest
from integrations.generals.environment import BOARD_SHAPE,ACTION_COUNT
from integrations.generals_autoresearch import arena,learned
from integrations.generals_autoresearch.bundle import seal,save_weights
from integrations.generals_autoresearch.contract import HERE,write_json

HAND_WRITTEN='''"""A player written by hand: fixed scores for moves, and one parameter nothing reads."""
import jax.numpy as jnp
from flax import linen as nn
class Policy(nn.Module):
    @nn.compact
    def __call__(self, x, mask):
        self.param('unused', lambda key, shape: jnp.zeros(shape, jnp.float32), (1,))
        army = jnp.expm1(6.0 * x[..., 0]) * x[..., 5]
        moves = jnp.broadcast_to(army[..., None], x.shape[:3] + (8,)) + jnp.arange(8) * 0.01
        scores = jnp.concatenate([jnp.zeros((x.shape[0], 1)), moves.reshape((x.shape[0], -1))], -1)
        return jnp.where(mask, scores, -1e9), jnp.zeros(x.shape[0])
def create(config):
    return Policy()
'''
# The same fixed scores with a small learned correction on top, as the second smoke submission was built.
HYBRID=HAND_WRITTEN.replace("        self.param('unused', lambda key, shape: jnp.zeros(shape, jnp.float32), (1,))\n",'').replace(
 "+ jnp.arange(8) * 0.01","+ jnp.arange(8) * 0.01 + nn.Dense(8, kernel_init=nn.initializers.normal(.02))(x)")

def module(tmp_path,name,source):
 path=tmp_path/(name+'.py');path.write_text(source)
 spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 return mod

@pytest.fixture(scope='module')
def boards():
 """Mid-game visible boards of both seats, from uniformly random legal play."""
 import jax,jax.numpy as jnp
 from integrations.generals.environment import build_pool,observations,encode,action_mask,decode_action,transition
 states=build_pool(list(range(900,932)));n=32;key=jax.random.PRNGKey(3);xs=[];masks=[]
 @jax.jit
 def tick(states,key):
  obs=observations(states);x=encode(obs);mask=action_mask(obs)
  ids=jax.random.categorical(key,jnp.where(mask,0.,-jnp.inf)).astype(jnp.int32)
  return transition(states,decode_action(ids)).final_state,x,mask
 for t in range(140):
  key,k=jax.random.split(key);states,x,mask=tick(states,k)
  if t>=40:xs.append(np.asarray(x).reshape((2*n,*BOARD_SHAPE,14)));masks.append(np.asarray(mask).reshape((2*n,ACTION_COUNT)))
 x=np.concatenate(xs);mask=np.concatenate(masks);real=mask.sum(-1)>=learned.MIN_LEGAL
 assert real.sum()>=3000
 return x[real][:3200],mask[real][:3200]

def play(probe,model,params,boards):
 x,mask=boards
 for start in range(0,len(x),32):
  part=slice(start,start+32);probe.see(x[part],mask[part],np.asarray(probe.choose(params,x[part],mask[part])))
 return probe

def fresh(model,key=7):
 import jax,jax.numpy as jnp
 return model.init(jax.random.PRNGKey(key),jnp.zeros((1,*BOARD_SHAPE,14)),jnp.ones((1,ACTION_COUNT),bool))['params']

def test_a_network_passes_and_a_hand_written_player_or_a_correction_on_one_does_not(tmp_path,boards):
 from integrations.generals_autoresearch.starter.policy import create
 for net in (create({'width':16,'compute_dtype':'float32'}),module(tmp_path,'placeholder',(HERE/'starter_empty/policy.py').read_text()).create({})):
  probe=play(learned.Probe(net,fresh(net)),net,fresh(net),boards)
  assert probe.done and probe.share()<.1
 for name,source in (('written',HAND_WRITTEN),('hybrid',HYBRID)):
  model=module(tmp_path,name,source).create({});params=fresh(model)
  with pytest.raises(learned.NotLearned,match='not learned'):play(learned.Probe(model,params),model,params,boards)
 # Nothing is decided on boards without a real choice, or before enough boards were seen.
 model=module(tmp_path,'written2',HAND_WRITTEN).create({});params=fresh(model);probe=learned.Probe(model,params)
 x,mask=boards;forced=np.zeros_like(mask[:64]);forced[:,0]=True
 probe.see(x[:64],forced,np.zeros(64,np.int64));probe.see(x[:640],mask[:640],np.asarray(probe.choose(params,x[:640],mask[:640])))
 assert probe.counted==640 and not probe.done and probe.share()==1.

def test_weights_of_another_shape_than_the_policy_initializes_are_refused(tmp_path):
 from integrations.generals_autoresearch.starter.policy import create
 with pytest.raises(learned.NotLearned,match='shape'):
  learned.Probe(create({'width':16,'compute_dtype':'float32'}),fresh(create({'width':8,'compute_dtype':'float32'})))

def test_the_evaluator_ends_a_hand_written_submission_as_a_policy_fault(tmp_path,boards):
 work=tmp_path/'source';work.mkdir()
 for n in ('learner.py','train.py'):shutil.copy2(HERE/'starter_empty'/n,work/n)
 (work/'policy.py').write_text(HAND_WRITTEN);write_json(work/'model.json',{})
 save_weights(fresh(module(tmp_path,'sealed',HAND_WRITTEN).create({})),work/'weights.npz');seal(work,tmp_path/'bundle')
 x,mask=boards
 with arena.PolicyProcess(tmp_path/'bundle',development=True,timeout=120) as policy:
  with pytest.raises(arena.PolicyFailure,match='not learned'):
   for start in range(0,len(x),32):policy.choose(x[start:start+32],mask[start:start+32])


SLOW_NETWORK='''"""A real network whose module and initialization are slow, as a large one is."""
import time
import jax.numpy as jnp
from flax import linen as nn
time.sleep(SLEEP)
class Policy(nn.Module):
    @nn.compact
    def __call__(self, x, mask):
        if self.is_initializing():
            time.sleep(1.0)  # each initialization takes this long, and the probe makes two
        moves = nn.Conv(8, (1, 1))(x).reshape((x.shape[0], -1))
        passed = nn.Dense(1)(jnp.mean(x, axis=(1, 2)))
        return jnp.where(mask, jnp.concatenate([passed, moves], -1), -1e9), jnp.zeros(x.shape[0])
def create(config):
    return Policy()
'''

def slow_bundle(tmp_path,sleep):
 source=SLOW_NETWORK.replace('SLEEP',str(sleep));work=tmp_path/'slow-source';work.mkdir()
 for n in ('learner.py','train.py'):shutil.copy2(HERE/'starter_empty'/n,work/n)
 (work/'policy.py').write_text(source);write_json(work/'model.json',{})
 save_weights(fresh(module(tmp_path,'slow',SLOW_NETWORK.replace('SLEEP','0')).create({})),work/'weights.npz')
 seal(work,tmp_path/'slow-bundle')
 return tmp_path/'slow-bundle'

def test_loading_a_policy_and_the_host_checks_are_not_on_its_answer_deadline(tmp_path,boards):
 """Live, 2026-11-19: the learned check's two re-initializations took 13 s for a large network, inside the 15 s a
 policy has for its first answer, so every evaluation of that network ended as an invalid policy."""
 import time
 x,mask=boards;started=time.monotonic()
 with arena.PolicyProcess(slow_bundle(tmp_path,1.5),development=True,timeout=3) as policy:
  assert time.monotonic()-started>3  # loading alone took longer than the answer deadline
  for start in range(0,96,32):
   ids=policy.choose(x[start:start+32],mask[start:start+32])
   assert mask[np.arange(start,start+32),ids].all()

def test_a_policy_that_does_not_finish_loading_is_a_start_up_fault_and_every_fault_is_named(tmp_path,monkeypatch):
 from integrations.generals_autoresearch.evaluate_entry import fault
 monkeypatch.setattr(arena,'STARTUP_SECONDS',2)
 with pytest.raises(arena.PolicyFailure,match='did not finish loading') as caught:
  arena.PolicyProcess(slow_bundle(tmp_path,30),development=True,timeout=3)
 assert fault(caught.value)[0]=='startup'
 assert fault(arena.PolicyFailure('Policy inference deadline exceeded'))[0]=='deadline'
 assert fault(arena.PolicyFailure('Policy request deadline exceeded'))[0]=='deadline'
 assert fault(arena.PolicyFailure('Illegal policy response'))[0]=='illegal_action'
 assert fault(arena.PolicyFailure('Policy process exited or broke its response protocol; returncode=1; stderr=Traceback ... NotLearned: Policy is not learned: with its weights'))[0]=='not_learned'
 assert fault(arena.PolicyFailure('Policy process exited; stderr=ValueError: Policy forward pass is 2,000,000,000 FLOPs'))[0]=='compute_cap'
 kind,sentence=fault(arena.PolicyFailure('Policy process exited or broke its response protocol; returncode=1; resource_diagnostics={}; stderr=Traceback\\n  File "policy.py"\\nZeroDivisionError: division by zero'))
 assert kind=='crashed' and 'ZeroDivisionError' in sentence and 'resource_diagnostics' not in sentence
