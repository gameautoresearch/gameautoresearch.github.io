"""One finite CPU-only, isolated-policy fresh assessment after selection closes."""
from pathlib import Path
import argparse,json,time,subprocess
from simulators.clash.autoresearch.common import dump,digest,sha,setup_native

def install_public_refusal_observer():
 # Observe public rejection fields without changing actions, masks or transitions.
 import cpu_batch,numpy as np
 original_batch=cpu_batch.NativeCPUBatch;recorders=[]
 class ObservedBatch(original_batch):
  def __init__(self,*args,**kwargs):
   super().__init__(*args,**kwargs);self.refusal_counts=np.zeros((len(self.infos),2,2),np.int64);recorders.append(self)
  def step(self,*args,**kwargs):
   before=self.infos[:,0].copy();value=super().step(*args,**kwargs);advanced=self.infos[:,0]>before
   self.refusal_counts[:,:,0]+=(self.observations[:,:,62]!=0)*advanced[:,None]
   self.refusal_counts[:,:,1]+=(self.observations[:,:,63]!=0)*advanced[:,None]
   return value
 cpu_batch.NativeCPUBatch=ObservedBatch
 return original_batch,recorders

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();root=Path(a.root);root.chmod(0o700);inputs=root/'inputs';inputs.chmod(0o700)
 selection=json.loads((inputs/'selection.json').read_text());bank=json.loads((inputs/'bank.json').read_text());metadata=json.loads((inputs/'metadata.json').read_text())
 if sha(inputs/'bank.json')!=metadata['bank_sha256'] or bank['roster_sha256']!=selection['candidate_roster_sha256']:raise ValueError('frozen bank or roster changed')
 from simulators.clash.autoresearch.bundle import verify
 if digest([{k:v for k,v in x.items() if k!='bundle'} for x in selection['candidates']])!=selection['candidate_roster_sha256']:raise ValueError('candidate roster metadata changed')
 for candidate in selection['candidates']:verify(inputs/candidate['bundle'],candidate['identity'])
 from simulators.clash.autoresearch.readiness_contract import INTERFACE
 if digest(INTERFACE)!=metadata['interface_sha256']:raise ValueError('assessment interface changed')
 setup_native();from simulators.clash.autoresearch.evaluation_v2 import screen
 original_batch,recorders=install_public_refusal_observer()
 opponents=json.loads((inputs/'opponents.json').read_text());results=[];started=time.time()
 for x in opponents:
  if x['kind']=='checkpoint':x['bundle']=str(inputs/x['bundle'])
 for candidate in selection['candidates']:
  spec={'name':candidate['name'],'kind':'checkpoint','bundle':str(inputs/candidate['bundle']),'identity':candidate['identity']}
  for opponent in opponents:
   out=root/'results'/candidate['name']/opponent['name'];out.parent.mkdir(parents=True,exist_ok=True)
   recorders.clear()
   result=screen([spec,opponent],out,seeds=bank['engine_seeds'],development=False,max_seconds=600)
   if result['complete']:
    if len(recorders)!=1:raise ValueError('unexpected assessment lane layout')
    for row in result['rows']:
     lane=2*row['public_seed_index']+row['left_seat'];counts=recorders[0].refusal_counts[lane]
     row['visible_request_refusal_nonzero_advanced_decisions_by_seat']=counts[:,0].tolist()
     row['visible_run_refusal_nonzero_advanced_decisions_by_seat']=counts[:,1].tolist()
    dump(out/'public-refusal-observation.json',{'fields':{'62':'last_refusal','63':'last_run_refusal'},'semantics':'nonzero-code decisions with actual tick advancement; not a count of unique rejection events','rows':result['rows']})
   if not result['complete']:raise ValueError('fresh assessment invalid; preserve failed pair, do not score partial games')
   results.append({'candidate':candidate['name'],'family':candidate['family'],'training_seed':candidate['training_seed'],'stage':candidate['stage'],'opponent':opponent['name'],'pair':result['pairs'][0],'rows':result['rows'],'entrant_probes':result['entrant_probes'],'latencies':result['latencies']})
   dump(root/'progress.json',{'completed_pairs':len(results),'elapsed_seconds':time.time()-started,'last_candidate':candidate['name'],'last_opponent':opponent['name']});subprocess.run(['sync',str(root)],check=True,timeout=20)
 value={'format':'clash-fresh-reference-assessment-v1','complete':True,'selection_roster_sha256':selection['candidate_roster_sha256'],'bank_sha256':metadata['bank_sha256'],'engine_units':len(bank['engine_seeds']),'results':results,'natural_games':sum(len(x['rows']) for x in results),'policy_isolation':'root supervisor, unprivileged no-new-privileges stateless candidates, public arrays only','elapsed_seconds':time.time()-started,'strong_panel_qualified':False,'superhuman_evidence':False,'reference_normalized_score':None}
 dump(root/'result.json',value);subprocess.run(['sync',str(root)],check=True,timeout=20)
if __name__=='__main__':main()
