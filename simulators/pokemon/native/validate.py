"""Differential development validation; no full-task admission or speed claims."""
import argparse,ctypes as C,json,subprocess,hashlib
from pathlib import Path
import numpy as np
from .bridge import Native,Move,build,HERE

def oracle_identity():
 root=HERE.parents[2]/'candidates/pokemon-showdown'
 commit=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()
 if commit!='567c1cedddeaa2f35576005d3bfd526cf4f8a13e':raise ValueError('Native oracle pin changed')
 if subprocess.check_output(['git','-C',str(root),'diff','--name-only','HEAD'],text=True).strip():raise ValueError('Native oracle tracked source changed')
 return dict(commit=commit,runtime_sha256={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for directory in ('dist','node_modules') for p in sorted((root/directory).rglob('*')) if p.is_file() and p.suffix in ('.js','.cjs','.json','.node')})

def oracle(spec):
 return json.loads(subprocess.check_output(['node',str(HERE/'oracle.cjs')],input=json.dumps(spec),text=True))
def seed64(seed):
 if seed.startswith('gen5,'):return int(seed[5:],16)
 return sum(int(x)<<(48-16*i) for i,x in enumerate(seed.split(',')))
def fill(native,initial):
 s=native.records[0];s.rng=seed64(initial['seed']);s.turn=initial['turn'];s.fault=0
 for side in range(2):
  for i,p in enumerate(initial['mons'][side]):
   m=s.mons[side][i];m.species=1;m.type[0]=10;m.type[1]=-1;m.tera_type=10
   from .rules import TYPES
   m.hp=p['hp'];m.maxhp=p['maxhp'];m.level=p.get('level',100);m.tera_type=TYPES.index(p.get('tera','Psychic'))
   for j,k in enumerate(('atk','def','spa','spd','spe')):m.stats[j]=p['stats'][k]
   for j,x in enumerate(p['moves']):m.move_ids[j]=0;m.pp[j]=x['pp'];m.maxpp[j]=x['maxpp']

def run(out):
 out=Path(out);out.mkdir(parents=True,exist_ok=False);lib=build(out/'build')
 seeds=[f'{i},{i+1},{i+2},{i+3}' for i in range(1,129)]
 reference=oracle(dict(op='rng',seeds=seeds,frames=256));rng_states=np.asarray([seed64(s) for s in seeds],dtype=np.uint64);values=np.empty((len(seeds),256),dtype=np.uint32)
 with Native(lib) as n:
  n.lib.pkn_random(rng_states.ctypes.data,values.ctypes.data,len(seeds),256)
 assert np.array_equal(values,np.asarray([r['values'] for r in reference],dtype=np.uint32))
 assert rng_states.tolist()==[seed64(r['seed']) for r in reference]
 moves=(Move*1)();m=moves[0];m.attack_stat=-1;m.defense_stat=-1;m.id=0;m.power=40;m.accuracy=100;m.priority=0;m.type=0;m.category=0;m.maxpp=56
 chart=np.zeros((18,18),dtype=np.int8);matches=0;frames=0;failures=[]
 for seed in seeds[:32]:
  d=oracle(dict(op='duel',seed=seed,turns=30))
  with Native(lib) as n:
   fill(n,d['initial']);snapshot=n.snapshot()
   for i,frame in enumerate(d['frames']):
    n.step([[0,0]],moves,chart);s=n.records[0];expected=frame['state']
    actual=dict(hp=[s.mons[side][0].hp for side in range(2)],rng=s.rng,turn=s.turn,ended=bool(s.terminated),fault=s.fault)
    target=dict(hp=[x[0]['hp'] for x in expected['mons']],rng=seed64(expected['seed']),turn=expected['turn'],ended=expected['ended'],fault=0)
    frames+=1
    if actual!=target:
     failures.append(dict(seed=seed,frame=i,actual=actual,expected=target,oracle_calls=frame['calls']));break
   else:matches+=1
   first=n.snapshot();n.restore(snapshot)
   for i in range(len(d['frames'])):n.step([[0,0]],moves,chart)
   if not failures:assert n.snapshot()==first
 result=dict(full_gen9_admission=False,gen5_rng_frames_exact=len(seeds)*256,duels_exact=matches,duels_tested=32,transition_frames=frames,failures=failures,
  scope='Constructed Mew/No Ability/Tackle diagnostic only; not the frozen random-battle task',native_state_bytes=C.sizeof(__import__('integrations.pokemon.native.bridge',fromlist=['State']).State))
 (out/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='failures'},indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);run(p.parse_args().output)
