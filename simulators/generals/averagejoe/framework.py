"""Single-device optimized orchestration with the actual upstream model/PPO.

No entrant tensor/memory cap applies to this separately qualified reference.
The upstream's direction-major 9-plane action codec is retained, including its
576 pass logits. This must never be confused with our entrant's 4609-ID codec.
"""
from .bootstrap import activate
activate()
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from config import Config
from networks import build_network, get_network_bundle
from train.ppo import compute_gae, make_value_loss_fn, ppo_update
from train.magnet import get_magnet_fn
from train.rollout_selfplay import collect_rollout
from train.rewards import win_lose_reward
from generals.core.env import GeneralsEnv


def recipe():
    from .bootstrap import UPSTREAM
    return Config.from_yaml(str(UPSTREAM/'configs/custom/L_7d_gae90.yaml'))


def make_env(cfg,stage=0,pool_size=None):
    s=cfg.curriculum_stages[stage]
    return GeneralsEnv(min_grid_size=cfg.min_grid_size,max_grid_size=cfg.max_grid_size,pad_to=cfg.pad_to,
        min_generals_distance=s.min_generals_distance,max_generals_distance=s.max_generals_distance,
        truncation=cfg.truncation,castle_val_range=(s.castle_val_min or cfg.castle_val_min,s.castle_val_max or cfg.castle_val_max),
        num_castles_range=(s.num_cities_min or cfg.num_cities_min,s.num_cities_max or cfg.num_cities_max),
        mountain_density_range=(cfg.mountain_density_min,cfg.mountain_density_max),pool_size=pool_size or cfg.pool_size)


def optimizer_for(cfg):
    # Deliberately preserve upstream's schedule denominator: it uses ALL rollout
    # samples, not the filtered quarter. Correcting it would change the recipe.
    per_iteration=cfg.num_epochs*(2*cfg.num_envs*cfg.num_steps//cfg.minibatch_size)
    def lr(step):
        iteration=step/per_iteration+1.
        return jnp.clip(cfg.lr_power_law_numerator/iteration**cfg.lr_power_law_exponent,cfg.lr_power_law_min,cfg.lr_power_law_max)
    return optax.chain(optax.clip_by_global_norm(cfg.max_grad_norm),optax.adam(lr))


def memories(cfg):
    one=get_network_bundle(cfg.network)['init_obs_state'](cfg.pad_to,cfg.pad_to)
    return jax.tree.map(lambda x:jnp.broadcast_to(x,(cfg.num_envs,)+x.shape),one)


def initial(cfg):
    key,net_key=jax.random.split(jax.random.PRNGKey(cfg.seed))
    net=build_network(cfg,net_key);opt=optimizer_for(cfg)
    return net,opt,opt.init(eqx.filter(net,eqx.is_array)),key


def make_kernels(cfg,env,net):
    if cfg.num_epochs!=1:raise ValueError('Published AverageJoe recipe requires one PPO epoch')
    bundle=get_network_bundle(cfg.network)
    static=eqx.partition(net,eqx.is_array)[1]
    opt=optimizer_for(cfg);value_loss=make_value_loss_fn(cfg);magnet=get_magnet_fn(cfg.magnet_policy)
    keep=int(2*cfg.num_envs*cfg.num_steps*cfg.adv_top_frac)//cfg.minibatch_size*cfg.minibatch_size
    if not keep:raise ValueError('Top-advantage filter keeps no minibatch')

    @eqx.filter_jit
    def rollout(net,states,key,m0,m1,pool):
        return collect_rollout(states,env,net,key,cfg.num_steps,m0,m1,cfg.pad_to,win_lose_reward,bundle['augment_obs'],bundle['reset_obs_state'],cfg.gamma,pool=pool)

    @eqx.filter_jit
    def update(net,opt_state,key,data,iteration):
        obs,masks,temporal,actions,lps,vals,next_vals,rews,terminated,truncated,winners,cities=data
        adv=compute_gae(rews,vals,next_vals,terminated,truncated,cfg.gamma,cfg.gae_lambda)
        ret=adv+vals
        # Same mean-square variance and global normalization as upstream pmap.
        variance=jnp.maximum(jnp.mean(adv**2)-jnp.mean(adv)**2,0.)
        adv=(adv-jnp.mean(adv))/(jnp.sqrt(variance)+1e-8)
        _,ix=jax.lax.top_k(jnp.abs(adv).reshape(-1),keep)
        batch=(obs,masks,temporal,actions,lps,adv,ret,1.-truncated.astype(jnp.float32))
        entropy=jnp.maximum(cfg.ent_coef_start/(iteration+1.)**cfg.ent_power,cfg.ent_coef_min)
        key,epoch_key=jax.random.split(key)
        # A one-element named axis preserves upstream pmean exactly, while XLA
        # eliminates multi-device replication and needless host synchronization.
        def epoch(_):return ppo_update(net,opt_state,batch,opt,epoch_key,cfg.clip_eps,cfg.vf_coef,entropy,cfg.minibatch_size,value_loss,ix,magnet_fn=magnet)
        networks,states,metrics=jax.vmap(epoch,axis_name='devices')(jnp.arange(1))
        net=jax.tree.map(lambda x:x[0] if eqx.is_array(x) else x,networks)
        opt_state=jax.tree.map(lambda x:x[0],states)
        metrics=jax.tree.map(lambda x:x[0],metrics)
        tt,nn=jnp.indices(actions.shape[:2])
        chosen=masks[tt,nn,jnp.clip(actions[...,1],0,cfg.pad_to-1),jnp.clip(actions[...,2],0,cfg.pad_to-1),actions[...,3]]
        invalid=jnp.sum((actions[...,0]==0)&~chosen)
        metrics.update(invalid_actions=invalid,terminated=jnp.sum(terminated[:,:cfg.num_envs]),caps=jnp.sum(truncated[:,:cfg.num_envs]&~terminated[:,:cfg.num_envs]),
            p0_wins=jnp.sum(terminated[:,:cfg.num_envs]&(winners[:,:cfg.num_envs]==0)),p1_wins=jnp.sum(terminated[:,:cfg.num_envs]&(winners[:,:cfg.num_envs]==1)),mean_owned_cities=jnp.mean(cities)/cfg.num_envs,
            nonzero_rewards=jnp.count_nonzero(rews),ent_coef=entropy)
        return net,opt_state,key,metrics

    @eqx.filter_jit
    def ema(old,net):return jax.tree.map(lambda a,b:cfg.ema_decay*a+(1.-cfg.ema_decay)*b if eqx.is_array(a) else a,old,net)
    return rollout,update,ema


def next_stage(cfg,current,rate,*,evaluation_complete):
    stages=cfg.curriculum_stages
    if evaluation_complete and current+1<len(stages) and rate>=stages[current+1].win_rate_threshold:return current+1
    return current
