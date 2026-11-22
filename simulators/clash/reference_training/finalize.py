"""Close selection before creating a fresh private assessment bank; no tuning after."""
from pathlib import Path
import argparse,json,secrets,time
from .launch import CAMPAIGN,LEDGER
from simulators.clash.autoresearch.common import dump,digest,sha
from simulators.clash.autoresearch.bundle import verify
from tools.overnight_budget import CampaignBudget

def close(candidates,required_seeds):
 root=CAMPAIGN/'clash/rl';target=root/'selection-close.json'
 if target.exists():raise FileExistsError('Selection already closed; use its exact roster')
 b=CampaignBudget('clash',path=LEDGER)
 if any(r['game']=='clash' and r['gpu_hold'] and r['status']!='settled' for r in b.snapshot()['operations']):raise ValueError('Active/unknown game GPU worker blocks selection closure')
 values=json.loads(Path(candidates).read_text())
 if not values or len({x['name'] for x in values})!=len(values):raise ValueError('Unique nonempty candidate roster required')
 for x in values:
  manifest=verify(x['bundle'],x['identity']);x['parameters']=manifest['parameters']
 if set(required_seeds)!=set(json.loads((root/'v3-precommit.json').read_text())['training_seeds']):raise ValueError('Required seeds differ from frozen training precommit')
 if set(required_seeds)!={x['training_seed'] for x in values}:raise ValueError('All precommitted training seeds must remain represented; no best-seed selection')
 families={x['family'] for x in values};base={'reference','control','untrained'};spatial={'spatial-reference','spatial-control','spatial-untrained'}
 if not base<=families or families-(base|spatial) or (families&spatial and not spatial<=families):raise ValueError('Complete matched family roster required')
 for family in families:
  records=[x for x in values if x['family']==family]
  expected=set(json.loads((root/'v4-precommit.json').read_text())['training_seeds']) if family in spatial else set(required_seeds)
  if len(records)!=len(expected) or {x['training_seed'] for x in records}!=expected:raise ValueError('Every matched family must preserve all of its precommitted seeds')
  if len({x['stage'] for x in records})!=1:raise ValueError('A family cannot select a different stage for each seed')
 teacher=Path(json.loads((root.parent/'coordination/evaluator.json').read_text())['public_teacher_source'])
 record={'closed_unix':time.time(),'candidates':values,'required_training_seeds':required_seeds,'candidate_roster_sha256':digest([{k:v for k,v in x.items() if k!='bundle'} for x in values]),'public_teacher_sha256':sha(teacher),'no_further_training_or_selection':True,'fresh_bank_created_after_selection':True}
 dump(target,record);target.chmod(0o444)
 private=root/'fresh-assessment';private.mkdir(mode=0o700)
 seeds=[]
 while len(seeds)<32:
  seed=3000000+secrets.randbelow(1900000000)
  if seed not in seeds and seed not in range(2026110901,2026110909):seeds.append(seed)
 bank={'engine_seeds':seeds,'policy_seed':secrets.randbelow(2**31),'purpose':'single frozen fresh assessment; no subsequent tuning','roster_sha256':record['candidate_roster_sha256']}
 dump(private/'bank.json',bank);(private/'bank.json').chmod(0o400)
 dump(private/'commitment.json',{'bank_sha256':sha(private/'bank.json'),'units':len(seeds),'selection_sha256':sha(target),'unblinded':False});return {'selection':str(target),'roster_sha256':record['candidate_roster_sha256'],'bank_commitment':sha(private/'bank.json'),'units':len(seeds)}

def main():
 p=argparse.ArgumentParser();p.add_argument('--candidates',required=True);p.add_argument('--required-seeds',type=int,nargs='+',required=True);a=p.parse_args();print(json.dumps(close(a.candidates,a.required_seeds),indent=2))
if __name__=='__main__':main()
