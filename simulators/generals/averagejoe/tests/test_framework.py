from dataclasses import replace
import jax
import jax.numpy as jnp
import numpy as np
from simulators.generals.averagejoe.framework import recipe,initial,make_env,memories,make_kernels,next_stage
from simulators.generals.averagejoe.bootstrap import activate
activate()
from generals.core.game import create_initial_state
from networks.common import prepare_action_mask,decode_action,encode_action,obs_to_array,augment_obs,init_obs_state
from generals.core.game import get_observation


def tiny():
    return replace(recipe(),pad_to=6,min_grid_size=6,max_grid_size=6,depth=1,embed_dim=16,n_head=2,ff_factor=2,
                   patch_size=3,num_envs=2,num_steps=4,minibatch_size=4,pool_size=2,use_bf16=False)


def fixture():
    grid=jnp.zeros((6,6),jnp.int32).at[2,1].set(1).at[2,3].set(2).at[5,5].set(40)
    s=create_initial_state(grid)._replace(armies=create_initial_state(grid).armies.at[2,1].set(7).at[2,3].set(7))
    return jax.tree.map(lambda x:jnp.stack([x,x]),s)


def test_full_published_model_and_curriculum_are_preserved():
    c=recipe()
    assert (c.depth,c.embed_dim,c.n_head,c.patch_size,c.ema_decay,c.adv_top_frac,c.num_envs,c.num_steps,c.pool_size)==(7,448,8,3,.999,.25,512,512,200000)
    assert [(s.min_generals_distance,s.max_generals_distance) for s in c.curriculum_stages]==[(2,6),(4,8),(6,13),(11,17),(17,28)]
    assert next_stage(c,0,.6,evaluation_complete=True)==1
    assert next_stage(c,0,1,evaluation_complete=False)==0
    assert next_stage(c,4,1,evaluation_complete=True)==4


def test_direction_major_pass_codec_and_visible_memory():
    c=tiny();s=jax.tree.map(lambda x:x[0],fixture());o=get_observation(s,0);a,m=augment_obs(obs_to_array(o),init_obs_state(6))
    assert a.shape==(38,6,6)
    # Enemy general is visible here; remembered terrain comes only from observed fields.
    assert bool(m.generals[2,1]);assert not bool(m.generals[5,5])
    for row in range(6):
        for col in range(6):
            for direction in range(4):
                for half in range(2):
                    act=jnp.array([0,row,col,direction,half]);np.testing.assert_array_equal(decode_action(encode_action(act,6),6),act)
    assert all(int(decode_action(8*36+k,6)[0])==1 for k in range(36))


def test_rollout_update_and_checkpoint_preserve_finite_learning(tmp_path):
    import equinox as eqx
    cfg=tiny();net,opt,os,key=initial(cfg);env=make_env(cfg);pool=fixture();m=memories(cfg)
    roll,update,ema=make_kernels(cfg,env,net)
    states,data,key,m0,m1=roll(net,pool,key,m,m,pool)
    before=[np.asarray(x).copy() for x in jax.tree.leaves(eqx.filter(net,eqx.is_array))]
    new,os,key,stats=update(net,os,key,data,jnp.int32(0));jax.block_until_ready(stats)
    assert np.isfinite(np.asarray(stats['total_loss']))
    assert any(not np.array_equal(x,y) for x,y in zip(before,jax.tree.leaves(eqx.filter(new,eqx.is_array))))
    target=(new,os,key,states,m0,m1,ema(net,new))
    eqx.tree_serialise_leaves(tmp_path/'state.eqx',target)
    restored=eqx.tree_deserialise_leaves(tmp_path/'state.eqx',target)
    # Exact next collection+update from restored optimizer, environment, memory and RNG.
    def advance(value):
        n,o,k,s,a,b,e=value;s,d,k,a,b=roll(n,s,k,a,b,pool);n,o,k,st=update(n,o,k,d,jnp.int32(1));return n,o,k,s,a,b,ema(e,n),st
    one,two=advance(target),advance(restored)
    for a,b in zip(jax.tree.leaves(one),jax.tree.leaves(two)):
        if eqx.is_array(a):np.testing.assert_array_equal(a,b)


def test_paired_evaluation_completes_each_game_once_with_legal_actions():
    from simulators.generals.averagejoe.evaluation import make_evaluator
    cfg=replace(tiny(),truncation=16)
    net,_,_,key=initial(cfg)
    result,_=make_evaluator(cfg)(net,fixture(),key)
    result={k:int(v) for k,v in jax.device_get(result).items()}
    assert result['games']==result['finished']==4
    assert result['wins']+result['losses']+result['caps']==4
    assert result['illegal_actions']==0
    assert result['ticks_executed']<=16
