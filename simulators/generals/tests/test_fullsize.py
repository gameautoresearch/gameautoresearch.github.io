"""The current standard (mid-size since 2026-11-18; the file keeps the name it got with the full-size form):
maps, padding, information and boundary compatibility."""
import json
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from generals.core import game
from integrations.generals.environment import (RULESET,BOARD_SHAPE,ACTION_COUNT,
    build_pool,map_dimensions,make_env,encode,observations,action_mask,decode_action,
    encode_action,transition)
from integrations.generals.runtime import HERE


def test_full_standard_rejects_old_recipe_and_codec_covers_padded_tensor():
    from integrations.generals.__main__ import validate_config
    from simulators.generals.autoresearch.contract import seed_manifest
    seeds=seed_manifest()
    # The retained recipes are bound to the full-size and 8x8 rules: each needs its own archived source.
    for name in ('full-smoke.json','smoke.json'):
        with pytest.raises(ValueError,match='environment_id='+RULESET['id']):
            validate_config(json.loads((HERE/'recipes'/name).read_text()),seeds)
    assert BOARD_SHAPE==(16,16) and ACTION_COUNT==2049 and RULESET['truncation']==1000
    ids=jnp.arange(ACTION_COUNT)
    np.testing.assert_array_equal(encode_action(decode_action(ids)),ids)


def test_mixed_sizes_seed_order_and_upstream_map_oracle():
    seeds=[101,102,103,104]
    dims=map_dimensions(seeds)
    assert len(set(map(tuple,dims)))>1 and ((12<=dims)&(dims<=15)).all()
    pool=build_pool(seeds)
    assert ((pool.castles.sum((1,2))>=4)&(pool.castles.sum((1,2))<=6)).all()
    reverse=build_pool(seeds[::-1])
    for a,b in zip(jax.tree.leaves(pool),jax.tree.leaves(reverse)):
        np.testing.assert_array_equal(a,b[::-1])
    env=make_env()
    for i,(h,w) in enumerate(dims):
        raw=env._make_single_state_fixed(jax.random.PRNGKey(seeds[i]),int(h),int(w))
        for a,b in zip(jax.tree.leaves(pool),jax.tree.leaves(raw)):
            np.testing.assert_array_equal(a[i],b)
    mask=action_mask(observations(pool))[...,1:].reshape(4,2,16,16,4,2)
    # Padded mountains can never become policy sources. Production validates
    # every dimension pair and every per-tick state field on CUDA as well.
    for i,(h,w) in enumerate(dims):
        assert not mask[i,:,h:,:,:,:].any() and not mask[i,:,:,w:,:,:].any()


def test_full_fog_inputs_and_joint_masks_ignore_hidden_army_distribution():
    s=jax.tree.map(lambda a:a[0],build_pool([101]))
    o=game.get_observation(s,0)
    hidden=np.argwhere(np.asarray(o.fog_cells & s.ownership_neutral & s.passable))
    assert len(hidden)>1
    a,b=map(tuple,hidden[:2])
    changed=s._replace(armies=s.armies.at[a].add(3).at[b].add(5))
    p=game.get_observation(changed,0)
    for x,y in zip(jax.tree.leaves(o),jax.tree.leaves(p)):
        np.testing.assert_array_equal(x,y)
    np.testing.assert_array_equal(encode(o),encode(p))
    np.testing.assert_array_equal(action_mask(o),action_mask(p))


def test_capture_at_full_cap_and_no_land_adjudication():
    grid=jnp.zeros(BOARD_SHAPE,jnp.int32).at[0,0].set(1).at[14,14].set(2)
    s=game.create_initial_state(grid)
    s=s._replace(armies=s.armies.at[13,14].set(10),
        ownership=s.ownership.at[0,13,14].set(True),
        ownership_neutral=s.ownership_neutral.at[13,14].set(False),time=jnp.int32(999))
    actions=jnp.array([[[0,13,14,1,0],[1,0,0,0,0]]])
    result=transition(jax.tree.map(lambda a:a[None],s),actions)
    assert bool(result.terminated[0]) and not bool(result.truncated[0])
    np.testing.assert_array_equal(result.reward,[[1.,-1.]])
    cap=transition(jax.tree.map(lambda a:a[None],s),jnp.ones((1,2,5),jnp.int32))
    assert cap.truncated[0] and not cap.terminated[0] and not cap.reward.any()


def _late_board():
    grid=jnp.zeros(BOARD_SHAPE,jnp.int32).at[15,:].set(-2).at[:,15].set(-2)
    grid=grid.at[0,0].set(1).at[0,14].set(2)
    return game.create_initial_state(grid)._replace(time=jnp.int32(10))


def _give(s,player,ij,army):
    return s._replace(armies=s.armies.at[ij].set(army),
        ownership=s.ownership.at[player,ij[0],ij[1]].set(True),
        ownership_neutral=s.ownership_neutral.at[ij].set(False))


def _trade_fixture():
    s=_late_board();s=s._replace(armies=s.armies.at[0,0].set(5).at[0,14].set(7))
    for player,ij,army in [(0,(0,13),40),(1,(1,0),30),(0,(10,10),9),(1,(12,12),4)]:
        s=_give(s,player,ij,army)
    return s,jnp.array([[0,0,13,3,0],[0,1,0,0,0]],jnp.int32)


def test_full_board_general_trade_swaps_without_elimination_and_rounds_up():
    s,actions=_trade_fixture()
    r=transition(jax.tree.map(lambda a:a[None],s),actions[None]);ns=jax.tree.map(lambda a:a[0],r.final_state)
    assert not r.terminated.any() and not r.truncated.any() and not ns.eliminated.any()
    assert ns.generals[0,0] and ns.generals[0,14]
    np.testing.assert_array_equal(ns.general_positions,[[0,14],[0,0]])
    assert ns.armies[0,14]==32 and ns.armies[0,0]==24
    assert ns.ownership[1,10,10] and ns.armies[10,10]==5
    assert ns.ownership[0,12,12] and ns.armies[12,12]==2


def test_full_board_capture_transfers_territory_and_converts_general_to_castle():
    s,actions=_trade_fixture();actions=actions.at[1].set(jnp.array([1,0,0,0,0]))
    r=transition(jax.tree.map(lambda a:a[None],s),actions[None]);ns=jax.tree.map(lambda a:a[0],r.final_state)
    assert r.terminated[0] and not r.truncated[0]
    np.testing.assert_array_equal(r.reward,[[1.,-1.]])
    assert ns.eliminated[1] and ns.winner==0
    assert ns.castles[0,14] and not ns.generals[0,14]
    assert ns.ownership[0,12,12] and ns.armies[12,12]==2


@pytest.mark.parametrize('split,remaining,moved',[(0,1,4),(1,3,2)])
def test_full_board_joint_split_reserves_and_neutral_capture(split,remaining,moved):
    s=_give(_late_board(),0,(0,1),5)
    actions=jnp.array([[[0,0,1,3,split],[1,0,0,0,0]]])
    r=transition(jax.tree.map(lambda a:a[None],s),actions)
    assert r.final_state.armies[0,0,1]==remaining
    assert r.final_state.armies[0,0,2]==moved and r.final_state.ownership[0,0,0,2]


def test_full_board_structure_and_land_income_do_not_grow_neutral_castles():
    s=_late_board()._replace(time=jnp.int32(49))
    s=s._replace(castles=s.castles.at[10,10].set(True).at[11,10].set(True),
        armies=s.armies.at[10,10].set(40))
    s=_give(_give(s,0,(10,1),5),0,(11,10),7)
    r=transition(jax.tree.map(lambda a:a[None],s),jnp.ones((1,2,5),jnp.int32))
    assert r.final_state.time[0]==50
    assert r.final_state.armies[0,0,0]==3
    assert r.final_state.armies[0,10,1]==6 and r.final_state.armies[0,11,10]==9
    assert r.final_state.armies[0,10,10]==40
