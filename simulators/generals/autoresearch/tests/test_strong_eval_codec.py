import io
import numpy as np
import pytest
import jax,jax.numpy as jnp
from simulators.generals.autoresearch.strong_eval.codec import raw_observation,entrant_observation,entrant_mask,to_entrant,encode,decode


def test_exact_public_codec_matches_both_existing_interfaces():
 from simulators.generals.autoresearch.rollout import load_pool
 from simulators.generals.environment import observations,encode as original,action_mask
 from simulators.generals.averagejoe.bootstrap import activate
 activate()
 from networks.common import obs_to_array
 from pathlib import Path
 path=Path('evidence/overnight-20261108-1533Z/generals/release-main/public/train-maps.npz')
 if not path.exists():pytest.skip('Private local retained correctness fixture unavailable')
 states=jax.tree.map(lambda v:v[:2],load_pool(path));obs=jax.tree.map(lambda v:v[:,0],observations(states))
 raw=raw_observation(obs)
 np.testing.assert_array_equal(raw,jax.vmap(obs_to_array)(obs))
 np.testing.assert_array_equal(entrant_observation(raw),original(obs))
 from generals.core.action import compute_valid_move_mask
 directions=jax.vmap(compute_valid_move_mask)(obs.armies,obs.owned_cells,obs.mountains)
 np.testing.assert_array_equal(entrant_mask(directions),action_mask(obs))
 rr,mm=decode(encode(np.asarray(raw),np.asarray(directions))[4:])
 np.testing.assert_array_equal(rr,raw);np.testing.assert_array_equal(mm,directions)
 with pytest.raises(AttributeError):raw_observation(states)


def test_direction_major_all_actions_map_exactly_to_entrant_codec():
 from simulators.generals.averagejoe.bootstrap import activate
 activate()
 from networks.common import decode_action
 from simulators.generals.environment import encode_action
 actions=jax.vmap(lambda i:decode_action(i,24))(jnp.arange(9*24*24))
 # This evaluator is the archived full-size one (24x24, 4,609 IDs); the active ruleset's default width differs.
 np.testing.assert_array_equal(to_entrant(actions),encode_action(actions,24))
 assert len(np.unique(np.asarray(to_entrant(actions))))==4609
 assert (np.asarray(to_entrant(actions))[-576:]==0).all()


def test_public_wire_rejects_extra_fields_and_wrong_shapes():
 f=io.BytesIO();np.savez(f,raw=np.zeros((1,14,24,24),np.float32),directions=np.zeros((1,288),np.uint8),private_seed=np.array([1]))
 with pytest.raises(ValueError,match='Only public'):decode(f.getvalue())
 with pytest.raises(ValueError,match='Bad raw'):decode(encode(np.zeros((1,38,24,24)),np.zeros((1,24,24,4),bool))[4:])


def test_reference_memory_stays_bounded_and_resets_on_new_game():
 from simulators.generals.averagejoe.bootstrap import activate
 activate()
 from networks.common import init_obs_state,augment_obs,reset_done_envs
 m=init_obs_state(24,24);assert sum(v.nbytes for v in jax.tree.leaves(m))<=65536
 raw=jnp.zeros((14,24,24)).at[5,5,5].set(1).at[0,5,5].set(9).at[12].set(40)
 augmented,m=augment_obs(raw,m);assert augmented.shape==(38,24,24)
 assert float(m.opponent_army_history[-1])==40
 batched=jax.tree.map(lambda v:v[None],m);reset=reset_done_envs(batched,jnp.array([True]))
 assert all(np.count_nonzero(v)==0 for v in jax.tree.leaves(reset))


def test_harvester_uses_only_visible_observation_and_emits_legal_action():
 from simulators.generals.autoresearch.rollout import load_pool
 from simulators.generals.environment import observations,action_mask,encode_action
 from generals.agents.harvester_agent import HarvesterAgent
 from pathlib import Path
 path=Path('evidence/overnight-20261108-1533Z/generals/release-main/public/train-maps.npz')
 if not path.exists():pytest.skip('Retained correctness fixture unavailable')
 obs=jax.tree.map(lambda v:v[:,0],observations(jax.tree.map(lambda v:v[:2],load_pool(path))))
 agent=HarvesterAgent();actions=jax.vmap(lambda o:agent.act(o,jax.random.PRNGKey(1)))(obs);ids=np.asarray(encode_action(actions));mask=np.asarray(action_mask(obs));assert mask[np.arange(2),ids].all()
