"""CPU-only, source-bound first-stage map cache; no Modal allocation."""
from .bootstrap import activate,binding,sha
activate()
from pathlib import Path
import argparse,json,time,zipfile,io
from .framework import recipe,make_env
import jax,numpy as np
from generals.core.game import GameState

def save_arrays(path,arrays):
    # Low-cost lossless compression; never change states or their seed schedule.
    with zipfile.ZipFile(path,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as z:
        for k,a in arrays.items():
            with z.open(k+'.npy','w',force_zip64=True) as f:np.lib.format.write_array(f,np.asarray(a),allow_pickle=False)

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--pool-size',type=int);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);cfg=recipe();cfg.pool_size=a.pool_size or cfg.pool_size
    b=binding();env=make_env(cfg);started=time.monotonic()
    key,netkey=jax.random.split(jax.random.PRNGKey(cfg.seed));key,poolkey=jax.random.split(key)
    print('CPU map preparation, stage0 distance2..6, requested pool',cfg.pool_size,flush=True)
    pool,_=env.reset(poolkey);jax.block_until_ready(pool)
    save_arrays(a.output/'stage0-pool.npz',pool._asdict())
    # Initial rollout and evaluation start states follow upstream init_state:
    # 23x23 starts, then the mixed-size pool after natural resets.
    key,initkey=jax.random.split(key);initial=jax.jit(jax.vmap(env.init_state))(jax.random.split(initkey,cfg.num_envs))
    save_arrays(a.output/'stage0-start.npz',initial._asdict())
    evalkey=jax.random.PRNGKey(2026110944);evaluation=jax.jit(jax.vmap(env.init_state))(jax.random.split(evalkey,cfg.eval_games//2))
    save_arrays(a.output/'stage0-eval.npz',evaluation._asdict())
    meta={'binding':b,'config':cfg.to_dict(),'requested_pool_size':cfg.pool_size,'actual_pool_size':len(pool.time),'initial_key':np.asarray(key).tolist(),'stage':0,'stage_distance':[2,6],'cpu_preparation_seconds':time.monotonic()-started,'files':{n:sha(a.output/n) for n in ['stage0-pool.npz','stage0-start.npz','stage0-eval.npz']},'scope':'CPU preprocessing and upload separately recorded; GPU allocation still includes loading and transfers. Exact pinned map generator, no hidden evaluation schedules.'}
    (a.output/'maps.json').write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps({k:v for k,v in meta.items() if k not in ('binding','config')},indent=2),flush=True)
if __name__=='__main__':main()
