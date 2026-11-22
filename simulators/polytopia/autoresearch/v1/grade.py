"""Trusted Rust evaluator. Submitted Python runs under another UID, never in verifier globals."""
import os,sys,json,time,base64,io,subprocess,select,shutil
from pathlib import Path
os.environ['JAX_PLATFORMS']='cpu'
import numpy as np,jax
from polytopia_bot.rl.rust_engine import RustGames
from polytopia_bot.rl.actions import Choice
from polytopia_bot.maps.stream import MapSource
from polytopia_bot.maps.state import stack
from polytopia_bot.rl.mode import SIZE,PLAYERS,MAPS,ROUND_LIMIT
from polytopia_bot.rl.infra import game_outcomes

class Policy:
 def __init__(self,path,seed,uid=65534):
  self.uid=uid
  self.p=subprocess.Popen(['/opt/venv/bin/python','/opt/v1/policy.py',str(path)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=open('/private/policy-'+str(uid)+'.stderr','w'),text=True,user=uid,group=uid,extra_groups=[],env=dict(os.environ,PYTHONPATH='/workspace/src',HOME='/tmp'),cwd='/tmp')
  self.p.stdin.write(json.dumps({'initialization_seed':seed})+'\n');self.p.stdin.flush()
 @staticmethod
 def drop(uid=65534):
  import resource
  resource.setrlimit(resource.RLIMIT_AS,(24*1024**3,24*1024**3));os.setgroups([]);os.setgid(uid);os.setuid(uid)
 def choose(self,features,masks):
  b=io.BytesIO();np.savez(b,**{'f'+str(i):np.asarray(v[0]) for i,v in enumerate(features)},**{'m'+str(i):np.asarray(v[0]) for i,v in enumerate(masks)})
  self.p.stdin.write(base64.b64encode(b.getvalue()).decode()+'\n');self.p.stdin.flush()
  if not select.select([self.p.stdout],[],[],120)[0]:raise TimeoutError('policy inference timeout')
  line=self.p.stdout.readline(4097)
  if len(line)>4096 or not line:raise RuntimeError('policy inference failed')
  a=json.loads(line)
  if len(a)!=3 or any(type(v)!=int for v in a):raise ValueError('Invalid action protocol')
  return Choice(*(np.array([v],np.int32) for v in a))
 def close(self):self.p.kill();self.p.wait()
def summarize(rows,expected_seeds=None):
 if not rows:raise ValueError("Empty evaluation schedule")
 if expected_seeds is not None:
  actual=[(r["seed"],r["seat"]) for r in rows];expected=[(s,seat) for s in expected_seeds for seat in (1,2)]
  if sorted(actual)!=sorted(expected):raise ValueError("Incomplete or duplicated evaluation schedule")
 points=np.array([(r['reward']+1)/2 for r in rows]);by_seed={s:np.mean([(r['reward']+1)/2 for r in rows if r['seed']==s]) for s in {r['seed'] for r in rows}}
 vals=np.array(list(by_seed.values()));rng=np.random.default_rng(32);boots=np.mean(rng.choice(vals,(2000,len(vals)),replace=True),axis=1)
 return {'raw_score_0_100':float(points.mean()*100),'normalized_reference_score':None,'games':len(rows),'natural_games':sum(not r['capped'] for r in rows),'capped_games':sum(r['capped'] for r in rows),'illegal_actions':0,'simulator_faults':0,'ci95_map_cluster':[float(x*100) for x in np.quantile(boots,[.025,.975])],'by_seat':{str(s):float(np.mean([(r['reward']+1)/2 for r in rows if r['seat']==s])*100) for s in (1,2)},'by_opponent':{'independent_random_minimal':{'raw_score_0_100':float(points.mean()*100),'games':len(rows)}},'details':rows}
def main():
 req=json.loads(Path(sys.argv[1]).read_text());start=time.time();rows=[]
 os.chmod('/private',0o700)
 os.chown('/opponent',65533,65533);os.chmod('/opponent',0o700)
 candidate=Policy('/candidate',0);opponent=Policy('/opponent',req['opponent_seed'],65533)
 try:
  source=MapSource(MAPS,SIZE);engine=RustGames(SIZE,1,map_batch=True,max_turns=ROUND_LIMIT,players=PLAYERS,no_resign=True)
  for seed in req['seeds']:
   layout=source.layout(seed)[1];batch=stack([(seed,layout)])
   for seat in (1,2):
    state,prepared=engine.reset(batch);steps=0
    while True:
     mover=int(np.asarray(engine.batch.scalars()['current_player'])[0]);p=candidate if mover==seat else opponent
     choice=p.choose(prepared.features,prepared.masks);steps+=1
     status=engine.step(choice)
     if np.any(np.asarray(status)!=0):raise RuntimeError('illegal_action: no valid score')
     flags=engine.batch.scalars()
     if bool(flags['terminated'][0]) or bool(flags['truncated'][0]):
      # step() does not auto-reset: adjudicate the actual FINAL state, not the previous turn.
      final=engine.states();score=np.asarray(final.players.score)[0];turn=int(flags['turn'][0])
      # The game's result by the task's one outcome rule, which the starter trains on too (infra.game_outcomes):
      # the engine's win/loss at an elimination; at the round limit the higher in-game score wins, equal scores draw.
      capped=bool(flags['truncated'][0]);reward=float(game_outcomes([bool(flags['terminated'][0])],[capped],np.asarray(engine.batch.terminal_rewards())[0],score)[0,seat-1])
      rows.append({'seed':seed,'seat':seat,'reward':reward,'capped':capped,'turn':turn,'steps':steps,'scores':score.tolist()});break
     prepared=engine._prepare()
  result=summarize(rows,req['seeds']);result['by_opponent']={req.get('opponent_name','independent_random_minimal'):next(iter(result['by_opponent'].values()))};result['cpu_seconds']=time.time()-start
  Path('/private/result.json').write_text(json.dumps(result));print(json.dumps({k:v for k,v in result.items() if k!='details'}),flush=True)
 finally:candidate.close();opponent.close()
if __name__=='__main__':main()
