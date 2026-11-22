"""Unprivileged inference worker. Inputs are only visible features and legal masks."""
import os,sys,json,base64,io,importlib.util,resource
resource.setrlimit(resource.RLIMIT_AS,(32*1024**3,32*1024**3))
from pathlib import Path
os.environ['JAX_PLATFORMS']='cpu'
import numpy as np,jax,jax.numpy as jnp
from polytopia_bot.rl.observation import Features
from polytopia_bot.gpu.actions import Masks
from polytopia_bot.rl.infra import build_network,network_spec
from polytopia_bot.rl.runner import load_checkpoint
from polytopia_bot.rl.mode import SIZE
initialization_seed=int(json.loads(sys.stdin.readline())['initialization_seed'])
path=Path(sys.argv[1]);spec=importlib.util.spec_from_file_location('polytopia_bot.rl._submitted',path/'network.py');net=importlib.util.module_from_spec(spec);spec.loader.exec_module(net)
if (path/'final.npz').exists():
 arch=network_spec(path/'final.npz')
 # A network trained for another board cannot read this task's observations: refuse it by name, before any game.
 if arch.get('size')!=SIZE:raise SystemExit(f"This checkpoint was trained for a {arch.get('size')}x{arch.get('size')} board; the task plays {SIZE}x{SIZE}")
 params=load_checkpoint(path/'final.npz',build_network(net,arch))
else:
 arch={'size':SIZE,'width':32,'blocks':2,'architecture':'normalized-v2','obs_version':3};params=build_network(net,arch,jax.random.key(initialization_seed))
act=jax.jit(lambda x,m:net.sample(params,x,m,jax.random.key(0),size=SIZE,deterministic=True)[0])
for line in sys.stdin:
 with np.load(io.BytesIO(base64.b64decode(line)),allow_pickle=False) as z:
  feat=Features(jnp.asarray(z['f0']),jnp.asarray(z['f1']));masks=Masks(*(jnp.asarray(z['m'+str(i)]) for i in range(17)))
 choice=act(feat,masks)
 print(json.dumps([int(np.asarray(v)) for v in choice]),flush=True)
