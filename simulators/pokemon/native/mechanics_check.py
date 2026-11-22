"""Native core component parity across damage/type/boost/Tera cases."""
import argparse,json,ctypes as C
from pathlib import Path
import numpy as np
from .bridge import build,Native,Move
from .validate import oracle,fill,seed64

def run(out):
 out=Path(out);out.mkdir(parents=True,exist_ok=False);lib=build(out/'build');rows=[]
 scenarios=[{},dict(tera=[True,False]),dict(tera=[False,True]),dict(tera=[True,True]),
  dict(species=['Snorlax','Blissey']),dict(species=['Gengar','Mew']),dict(species=['Mew','Gengar']),
  dict(levels=[50,80]),dict(boosts=[dict(atk=-2),{'def':3}]),dict(boosts=[dict(atk=2),{'def':-3}]),
  dict(move='watergun',species=['Blastoise','Charizard']),dict(move='watergun',species=['Blastoise','Venusaur']),
  dict(move='ember',species=['Charmander','Bulbasaur']),dict(move='quickattack',species=['Mew','Mew']),dict(move='pound'),dict(speed_evs=[0,0]),dict(statuses=['brn','']),dict(statuses=['','brn']),dict(statuses=['par','']),dict(statuses=['','par']),dict(statuses=['psn','']),dict(statuses=['tox','']),dict(statuses=['slp','']),dict(statuses=['frz',''])]
 for case,options in enumerate(scenarios):
  for master in range(1,9):
   spec=dict(op='duel',seed=f'{master},{master+1},{master+2},{master+3}',turns=32,**options);d=oracle(spec)
   moves=(Move*1)();m=moves[0];m.attack_stat=-1;m.defense_stat=-1;r=d['move'];m.power=r['basePower'];m.accuracy=101 if r['accuracy'] is True else r['accuracy'];m.priority=r['priority'];m.type=d['types'].index(r['type']);m.category=['Physical','Special','Status'].index(r['category'])
   # Secondary effects are deliberately a closed-admission gap; do not certify them here.
   if r.get('secondary'):continue
   with Native(lib) as n:
    fill(n,d['initial']);s=n.records[0]
    for side in range(2):
     p=d['initial']['mons'][side][0];mon=s.mons[side][0];mon.level=p['level'];mon.type[0]=d['types'].index(p['types'][0]);mon.type[1]=d['types'].index(p['types'][1]) if len(p['types'])>1 else -1;mon.tera_type=d['types'].index(p['tera']);mon.status=['','brn','par','slp','frz','psn','tox'].index(p['status']);mon.status_turns=p.get('statusState',{}).get('time',0);mon.toxic_turns=p.get('statusState',{}).get('stage',0)
     for j,k in enumerate(('atk','def','spa','spd','spe','accuracy','evasion')):mon.boosts[j]=p['boosts'][k]
    mismatch=None
    for i,frame in enumerate(d['frames']):
     actions=[[4 if i==0 and options.get('tera',[False,False])[side] else 0 for side in range(2)]]
     n.step(actions,moves,np.asarray(d['chart'],dtype=np.int8));s=n.records[0];x=frame['state']
     actual=dict(hp=[s.mons[k][0].hp for k in range(2)],rng=s.rng,turn=s.turn,ended=bool(s.terminated),fault=s.fault,status=[s.mons[k][0].status for k in range(2)],pp=[s.mons[k][0].pp[0] for k in range(2)])
     expected=dict(hp=[x['mons'][k][0]['hp'] for k in range(2)],rng=seed64(x['seed']),turn=x['turn'],ended=x['ended'],fault=0,status=[['','brn','par','slp','frz','psn','tox'].index(x['mons'][k][0]['status']) for k in range(2)],pp=[x['mons'][k][0]['moves'][0]['pp'] for k in range(2)])
     if actual!=expected:mismatch=dict(frame=i,actual=actual,expected=expected,calls=frame['calls']);break
    rows.append(dict(scenario=case,options=options,seed=master,exact=mismatch is None,mismatch=mismatch))
 result=dict(full_gen9_admission=False,cases=len(rows),exact=sum(r['exact'] for r in rows),rows=rows)
 (out/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);run(p.parse_args().output)
