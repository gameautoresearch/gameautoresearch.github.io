"""Fresh public Generals films from the immutable smoke bot; no grading maps.

The only executable submitted source accepted here is the audited residual CNN
below. Other bundles need an explicit trusted adapter rather than arbitrary import.
"""
from pathlib import Path
import argparse,base64,importlib.util,json,sys
_HERE=Path(__file__).resolve().parent
ROOT=_HERE.parents[2] if _HERE.parent.name=='generals' else _HERE.parents[1]
sys.path.insert(0,str(ROOT))
import jax
import jax.numpy as jnp
import numpy as np
from .bundle import verify,load_weights
from .contract import sha,write_json
from integrations.generals.environment import build_pool,map_dimensions,observations,encode,action_mask,encode_action,decode_action,transition,RULESET
from integrations.generals.runtime import sources,digest
from integrations.generals.replay import verify as verify_actions
from generals.agents import HunterAgent

POLICY_SHA='44a00a8fa708b424b91cc2f36c861a82bb674c4b54e22cb8f872645b76295343'
DEFAULT_BUNDLE=ROOT/'evidence/generals-autoresearch/smoke-20261108/deepseek-5/jobs/smoke/bundle'
SEEDS=[2026110801,2026110802,2026110803]

def packed(a):return base64.b64encode(np.packbits(np.asarray(a).reshape(-1),bitorder='little').tobytes()).decode()
def cells(s,h,w):
 own=np.asarray(s.ownership)[:,:h,:w];owner=np.where(own.any(0),own.argmax(0),-1)
 kind=np.where(s.mountains[:h,:w],1,np.where(s.generals[:h,:w],3,np.where(s.castles[:h,:w],2,0)))
 return np.stack((kind,owner,s.armies[:h,:w]),-1).reshape(-1,3).astype(int)

def main():
 p=argparse.ArgumentParser(__doc__);p.add_argument('--output',required=True);p.add_argument('--bundle',default=str(DEFAULT_BUNDLE));a=p.parse_args()
 out=Path(a.output)
 if out.exists():raise FileExistsError('Use a new output; recordings and receipts are immutable')
 bundle=Path(a.bundle);m=verify(bundle)
 if sha(bundle/'policy.py')!=POLICY_SHA:raise ValueError('This exporter accepts only the audited DeepSeek smoke policy source')
 spec=importlib.util.spec_from_file_location('generals_demo_policy',bundle/'policy.py');module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
 model=module.create(json.loads((bundle/'model.json').read_text()));params=load_weights(bundle/'weights.npz');hunter=HunterAgent()
 pool=build_pool(SEEDS);b=len(SEEDS);types=jnp.array([[0,1],[0,1],[0,0]])
 @jax.jit
 def run(states):
  def tick(c,_):
   s,active=c;o=observations(s);x=encode(o);mask=action_mask(o)
   logits=model.apply({'params':params},x.reshape((-1,24,24,14)),mask.reshape((-1,4609)))[0]
   neural=jnp.argmax(logits,-1).reshape((b,2)).astype(jnp.int32)
   hunting=encode_action(jax.vmap(jax.vmap(lambda v:hunter.act(v,jax.random.PRNGKey(0))))(o))
   ids=jnp.where(types==0,neural,hunting)
   invalid=active[:,None]&~jnp.take_along_axis(mask,ids[...,None],-1)[...,0]
   r=transition(s,decode_action(ids));current=jax.tree.map(lambda n,v:jnp.where(active.reshape((b,)+(1,)*(n.ndim-1)),n,v),r.final_state,s)
   return (current,active&~(r.terminated|r.truncated)),(current,ids,active,r.terminated,r.truncated,invalid)
  return jax.lax.scan(tick,(states,jnp.ones(b,bool)),None,length=2048)
 (final,live),(history,actions,active,natural,caps,invalid)=jax.device_get(run(pool))
 if live.any() or invalid.any():raise AssertionError('Replay generation left unfinished games or illegal moves')
 out.mkdir(parents=True)
 native=out/'actions.npz';np.savez_compressed(native,seeds=np.array(SEEDS,np.uint32),action_ids=actions,active=active,**{'final_'+k:np.asarray(v) for k,v in final._asdict().items()})
 write_json(str(native)+'.json',{'source':sources(),'sha256':digest(native.read_bytes())})
 receipt=verify_actions(native);write_json(out/'verification.json',receipt)
 @jax.jit
 def views(s):
  o=observations(s);return o.fog_cells,o.structures_in_fog
 all_states=jax.tree.map(lambda initial,rest:jnp.concatenate((initial[None],rest),0),pool,history)
 fog,shadows=jax.device_get(jax.jit(jax.vmap(views))(all_states))
 dimensions=map_dimensions(SEEDS);entries=[]
 for i,seed in enumerate(SEEDS):
  h,w=map(int,dimensions[i]);end=int(np.flatnonzero(active[:,i]&(natural[:,i]|caps[:,i]))[0])+1
  names=['DeepSeek · final','Hunter' if i<2 else 'DeepSeek · final'];identifier=f'generals-deepseek-{"hunter-"+str(i+1).zfill(2) if i<2 else "selfplay-01"}'
  frames=[];previous=None
  for t in range(end+1):
   state=jax.tree.map(lambda v:np.asarray(v[t,i]),all_states);board=cells(state,h,w)
   changed=np.arange(h*w) if previous is None else np.flatnonzero(np.any(board!=previous,axis=-1))
   changes=[[int(k),*board[k].tolist()] for k in changed]
   scores=[{'army':int((state.armies*state.ownership[p]).sum()),'land':int(state.ownership[p].sum())} for p in range(2)]
   frames.append({'tick':t,'changes':changes,'actions':None if t==0 else actions[t-1,i].tolist(),'fog':[packed(fog[t,i,p,:h,:w]) for p in range(2)],'shadows':[packed(shadows[t,i,p,:h,:w]) for p in range(2)],'scores':scores})
   previous=board
  truncated=bool(caps[end-1,i]);winner=None if truncated else int(np.asarray(final.winner)[i])
  film={'schemaVersion':1,'gameId':'generals','id':identifier,'title':f'DeepSeek vs {"Hunter · "+str(i+1).zfill(2) if i<2 else "DeepSeek"}',
   'subtitle':f'{h} × {w} · '+('General captured' if not truncated else '2048-step draw'),
   'policy':{'name':'DeepSeek smoke bot','checkpoint':m['bundle_sha256'],'description':'Final 19M-sample smoke checkpoint; these are new public demonstrations, not private grading games.'},
   'provenance':{'kind':'fresh-public-demo','demoSeed':seed,'ruleset':RULESET['id'] if 'id' in RULESET else 'generals-research-1v1-full-v2','engineCommit':sources()['commit'],'verification':'Every joint action replayed on the pinned native engine: exact final state, legality and episode boundary.'},
   'width':w,'height':h,'stepCap':2048,'players':names,'frames':frames,
   'outcome':{'winner':winner,'ticks':end,'terminated':not truncated,'truncated':truncated,'terminal':'step-cap' if truncated else 'general-captured'}}
  write_json(out/(identifier+'.json'),film,exclusive=True)
  entries.append({'id':identifier,'gameId':'generals','title':film['title'],'subtitle':film['subtitle'],'url':'replays/generals/'+identifier+'.json','moves':end})
 write_json(out/'index.json',entries,exclusive=True);print(json.dumps({'recordings':entries,'verification':receipt}),flush=True)
if __name__=='__main__':main()
