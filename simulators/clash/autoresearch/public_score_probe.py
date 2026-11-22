"""Read-only raw-network/declared-decoder score diagnostics on public fixtures."""
import argparse,json,shutil
from pathlib import Path
import numpy as np
from .common import dump,load_module,sha,DATA
from .readiness_contract import admit

def run(specs,states,out):
 import torch
 torch.set_num_threads(1);out=Path(out);out.mkdir(parents=True,exist_ok=False);shutil.copyfile(__file__,out/'probe-source.py');vocab=json.loads(DATA.read_text())['vocabulary'];fixture=np.load(states,allow_pickle=False);rows=[]
 for spec in specs:
  admission=admit(spec['bundle']);bundle=Path(spec['bundle']);module=load_module(bundle/'policy.py','trusted_public_score_probe');model=module.create(json.loads((bundle/'config.json').read_text())).to('cpu');model.eval()
  with np.load(bundle/'weights.npz',allow_pickle=False) as weights:model.load_state_dict({k:torch.from_numpy(weights[k].copy()) for k in weights.files})
  for stage in ('initial_wallet','full_wallet'):
   obs=fixture[stage+'_observations'];masks=fixture[stage+'_masks'];x=torch.from_numpy(obs);mask=torch.from_numpy(masks)
   with torch.inference_mode():
    # These known audited policies expose the original weighted Policy class;
    # bypass only the declared GreedyHead for a labeled score diagnostic.
    raw,_=module.Policy.forward(model,x,mask);exported,_=model(x,mask);p=raw.softmax(-1);slots=raw[:,1:2305].reshape(-1,4,576).logsumexp(-1);choice=exported.argmax(-1)
   for i,o in enumerate(obs):
    hand=[(vocab[int(c)] if int(c)>=0 else 'Empty') for c in o[7:11]];giant=hand.index('Giant') if 'Giant' in hand else None
    if stage!='full_wallet' or giant is None or not masks[i,1+giant*576:1+(giant+1)*576].any():continue
    options=torch.cat((raw[i,:1],slots[i]));actual=int(choice[i]);rows.append({'model':spec['name'],'identity':admission['identity'],'stage':stage,'row':i,'hand':hand,'actual_declared_decoder_card':'wait' if actual==0 else hand[(actual-1)//576],'raw_PPO_wait_probability':float(p[i,0]),'raw_PPO_total_nonwait_probability':float(1-p[i,0]),'raw_PPO_card_probabilities':{hand[s]:float(p[i,1+s*576:1+(s+1)*576].sum()) for s in range(4)},'raw_option_argmax':'wait' if int(options.argmax())==0 else hand[int(options.argmax())-1],'Giant_best_card_marginal':int(slots[i].argmax())==giant})
 summary={}
 for spec in specs:
  selected=[r for r in rows if r['model']==spec['name']];summary[spec['name']]={'full_wallet_legal_Giant_rows':len(selected),'declared_decoder_waits':sum(r['actual_declared_decoder_card']=='wait' for r in selected),'raw_binary_nonwait_probability_over_half':sum(r['raw_PPO_total_nonwait_probability']>.5 for r in selected),'Giant_best_card_marginal_rows':sum(r['Giant_best_card_marginal'] for r in selected),'mean_raw_wait_probability':float(np.mean([r['raw_PPO_wait_probability'] for r in selected]))}
 dump(out/'result.json',{'source_sha256':sha(out/'probe-source.py'),'public_states_sha256':sha(states),'summary':summary,'rows':rows,'interpretation':'Raw weighted-network scores and actual declared decoder choices are diagnostic only. No policy/source/weights/actions were changed, no alternative policy was scored, and these sparse reachable contexts do not establish match strength or a causal defect.'});return summary

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--specs',required=True);p.add_argument('--states',required=True);p.add_argument('--out',required=True);a=p.parse_args();print(json.dumps(run(json.loads(Path(a.specs).read_text()),a.states,a.out),indent=2))
