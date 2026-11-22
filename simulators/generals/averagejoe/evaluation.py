"""Greedy paired-seat evaluation, freezing completed games and stopping early."""
from .bootstrap import activate
activate()
import equinox as eqx
import jax
import jax.numpy as jnp
from generals.core.game import get_observation,step
from generals.core.action import compute_valid_move_mask
from networks.common import obs_to_array,augment_obs,init_obs_state,random_action
from networks.transformer import greedy_action_transformer


def make_evaluator(cfg):
    @eqx.filter_jit
    def evaluate(net,starts,key):
        maps=len(starts.time);count=2*maps;rr=jnp.arange(count);seats=jnp.repeat(jnp.arange(2),maps)
        states=jax.tree.map(lambda x:jnp.concatenate([x,x]),starts)
        memory=jax.tree.map(lambda x:jnp.broadcast_to(x,(count,)+x.shape),init_obs_state(cfg.pad_to,cfg.pad_to))
        z=jnp.zeros(count,bool)
        def condition(c):return (c[0]<cfg.truncation)&~jnp.all(c[4])
        def tick(c):
            t,s,k,m,finished,won,lost,caps,illegal=c
            obs=jax.vmap(lambda s,p:get_observation(s,p))(s,seats)
            opp=jax.vmap(lambda s,p:get_observation(s,1-p))(s,seats)
            x=jax.vmap(obs_to_array)(obs);mask=jax.vmap(lambda o:compute_valid_move_mask(o.armies,o.owned_cells,o.mountains))(obs)
            x,mem=jax.vmap(augment_obs)(x,m);temporal=jnp.stack([mem.opponent_army_history,mem.opponent_land_history],1)
            actions=jax.vmap(greedy_action_transformer,(None,0,0,0))(net,x,mask,temporal)
            k,ak=jax.random.split(k);others=jax.vmap(random_action)(jax.random.split(ak,count),opp)
            joint=jnp.zeros((count,2,5),jnp.int32).at[rr,seats].set(actions).at[rr,1-seats].set(others)
            valid=actions[:,0].astype(bool)|mask[rr,jnp.clip(actions[:,1],0,cfg.pad_to-1),jnp.clip(actions[:,2],0,cfg.pad_to-1),actions[:,3]]
            ns,info=jax.vmap(lambda s,a:step(s,a,general_trade=True))(s,joint)
            natural=info.is_done;cap=(ns.time>=cfg.truncation)&~natural;done=natural|cap;live=~finished
            frozen=jax.tree.map(lambda n,v:jnp.where(live.reshape((count,)+(1,)*(n.ndim-1)),n,v),ns,s)
            return t+1,frozen,k,mem,finished|done,won|(live&natural&(info.winner==seats)),lost|(live&natural&(info.winner!=seats)),caps|(live&cap),illegal+jnp.sum(live&~valid)
        result=jax.lax.while_loop(condition,tick,(jnp.int32(0),states,key,memory,z,z,z,z,jnp.int32(0)))
        t,states,key,m,fin,won,lost,caps,illegal=result
        return {'wins':won.sum(),'losses':lost.sum(),'caps':caps.sum(),'finished':fin.sum(),'games':jnp.int32(count),'ticks_executed':t,'illegal_actions':illegal},key
    return evaluate
