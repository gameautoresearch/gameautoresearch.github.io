"""Freeze a reviewable role roster; the evaluator alone creates private seeds."""
import argparse,json,time
from pathlib import Path
from .provider import OWN,LEDGER,CAMPAIGN
from tools.open_credit_campaign import OpenCreditBudget
from tools.overnight_budget import stamp
from simulators.clash.autoresearch.common import dump,digest,sha
from simulators.clash.autoresearch.bundle import verify
from simulators.clash.autoresearch.readiness_contract import admit

def checked(rows,primary):
 if not rows or len(rows)>8 or len({x['name'] for x in rows})!=len(rows) or primary not in {x['name'] for x in rows}:raise ValueError('One to eight unique named roles and an explicit primary required')
 output=[]
 for x in rows:
  x=dict(x);m=verify(x['bundle'],x['identity']);review=admit(x['bundle'])
  if not x.get('role') or not review['source_review']['greedy_entry_checked']:raise ValueError('Explicit role and admitted pure neural inference required')
  x.update(parameters=m['parameters'],bundle_manifest_sha256=sha(Path(x['bundle'])/'manifest.json'));output.append(x)
 return output,next(x['identity'] for x in output if x['name']==primary)

def prepare(roles,primary):
 if (OWN/'selection-close.json').exists():raise ValueError('Role roster is already frozen')
 budget=OpenCreditBudget('clash',LEDGER)
 if time.time()>=stamp(budget.auth['selection_close_utc']):raise ValueError('Selection deadline reached; do not introduce new choices')
 peer=json.loads((CAMPAIGN/'clash/coordination/evaluator.json').read_text())
 if peer.get('private_final_bank_opened') or (CAMPAIGN/'clash/evaluator/final-bank-unblinded.json').exists():raise ValueError('Private bank already opened; no new selection')
 rows,identity=checked(json.loads(Path(roles).read_text()),primary)
 base=OWN/'selection-proposals';index=1
 while (base/f'{index:03d}').exists():index+=1
 folder=base/f'{index:03d}';folder.mkdir(parents=True,exist_ok=False);record=folder/'proposal.json'
 value={'proposal_unix':time.time(),'proposal_record':str(record),'roles_input':str(Path(roles).resolve()),'candidates':rows,'primary_candidate':primary,'primary_identity':identity,'all_proposed_roles_will_be_reported':True,'higher_private_control_score_cannot_replace_primary':True,'private_feedback_used':False,'registry_commitment':digest(rows),'role_count':len(rows),'private_bank_created_by':'evaluator only, after absolute selection deadline','explicit_ledger':str(LEDGER)}
 dump(record,value);record.chmod(0o444);dump(OWN/'selection-proposal.json',value);return value

def freeze():
 target=OWN/'selection-close.json'
 if target.exists():raise FileExistsError('Preserve the existing immutable closure')
 proposal=OWN/'selection-proposal.json';value=json.loads(proposal.read_text());record=Path(value['proposal_record']);record.relative_to(OWN/'selection-proposals')
 if sha(record)!=sha(proposal):raise ValueError('Current proposal differs from its immutable versioned record')
 rows,identity=checked(value['candidates'],value['primary_candidate'])
 if identity!=value['primary_identity'] or digest(rows)!=value['registry_commitment']:raise ValueError('Role roster changed before freeze')
 budget=OpenCreditBudget('clash',LEDGER)
 if any(r['game']=='clash' and r['gpu_hold'] and r['status']!='settled' for r in budget.snapshot()['operations']):raise ValueError('Reconcile every Clash GPU worker before selection freeze')
 peer=json.loads((CAMPAIGN/'clash/coordination/evaluator.json').read_text())
 if peer.get('private_final_bank_opened') or (CAMPAIGN/'clash/evaluator/final-bank-unblinded.json').exists():raise ValueError('Private final bank was opened before this closure')
 value.update(selection_closed=True,closed_unix=time.time(),selection_proposal_sha256=sha(proposal),no_further_training_or_policy_choices=True,no_private_bank_created_here=True)
 dump(target,value);target.chmod(0o444);return {'selection':str(target),'selection_sha256':sha(target),'primary_identity':identity,'role_count':len(rows)}

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--roles');p.add_argument('--primary');p.add_argument('--freeze',action='store_true');a=p.parse_args();print(json.dumps(freeze() if a.freeze else prepare(a.roles,a.primary),indent=2))
