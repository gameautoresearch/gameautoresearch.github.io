"""Algorithmic optimization must preserve actions, masks, rewards, states and GAE."""
import json
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from integrations.generals.runtime import HERE
from integrations.generals.model import Policy
from integrations.generals.learning import create_train_state,initial_carry
from integrations.generals.environment import build_pool,RULESET
from integrations.generals.efficiency import parity,make_collect_optimized


@pytest.mark.parametrize('time_value',[0,RULESET['truncation']-10,RULESET['truncation']-1])
def test_shifted_values_keep_visible_trajectory_and_truncation_bootstrap(time_value):
    cfg=json.loads((HERE/'recipes/full-smoke.json').read_text());cfg.update(num_envs=4,rollout_steps=16,network_width=8)
    model=Policy(8);train=create_train_state(model,jax.random.PRNGKey(7),cfg)
    bank=jax.tree.map(lambda p:jnp.repeat(p[None],4,0),train.params)
    pool=build_pool([10001,10002,10003,10004]);c=initial_carry(pool,jax.random.PRNGKey(8),cfg,bank)
    c=c._replace(states=c.states._replace(time=jnp.full((4,),time_value,jnp.int32)))
    result=parity(model,cfg,train,bank,c,pool)
    assert result['exact_final_state_rng_opponent_cache'] and not result['invalid']
    if time_value>=RULESET['truncation']-10:assert result['caps']==4


def test_long_horizon_rejected_instead_of_using_wrong_cap_cache():
    cfg=json.loads((HERE/'recipes/full-smoke.json').read_text());cfg['rollout_steps']=RULESET['truncation']+1
    with pytest.raises(ValueError,match='cap'):make_collect_optimized(Policy(8),cfg)
