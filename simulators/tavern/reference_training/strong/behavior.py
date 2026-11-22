"""Visible-only policy drift on preselected public incumbent trajectories.

This is a fixed-state diagnostic, not a rollout score or a counterfactual result.
"""
import argparse,collections,fcntl,hashlib,json,shutil
from pathlib import Path
import numpy as np
import torch
import taverngym as tg
from .common import ROOT,sha,write
from simulators.tavern.evaluation_readiness.policies import NeuralVisible

def audit(reference_screen,bundle,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    shutil.copy2(__file__,out/'helper-source.py')
    reference_screen=Path(reference_screen)
    commit=json.loads((reference_screen/'commitment.json').read_text())
    with open(ROOT/'evidence/local-performance.lock','a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);torch.set_num_threads(1)
        old=NeuralVisible(reference_screen/'registry/learned/candidate','incumbent')
        new=NeuralVisible(bundle,'challenger');rows=[];replays=[]
        # Every profile, the first published seed, candidate physical seat0.
        panel=json.loads((reference_screen/'registry/panel.json').read_text())
        for profile in commit['profiles']:
            trace=reference_screen/'screen/traces'/f"{profile}-{panel['seeds'][0]}-0"
            initial=(trace/'initial.json').read_text();result=json.loads((trace/'result.json').read_text())
            env=tg.BatchEnv(1,seed=result['seed'],controlled=range(8),config=json.loads(initial)['game']['cfg'],threads=1);env.load(0,initial)
            with np.load(trace/'actions.npz',allow_pickle=False) as z:actions=z['actions']
            for actions_frame in actions:
                obs=env.observe()
                if obs['ready'][0]:rows.append((obs['features'][0].copy(),obs['cards'][0].copy(),obs['action_mask'][0].copy(),int(actions_frame[0])))
                env.step(actions_frame)
            exact=hashlib.sha256(env.save(0).encode()).hexdigest()==result['final_state_sha256']
            if not exact:raise ValueError('Public replay state differs')
            replays.append(dict(profile=profile,trace=str(trace),exact_final_state=True,actions_sha256=sha(trace/'actions.npz')))
        f=torch.from_numpy(np.stack([r[0] for r in rows]));cards=torch.from_numpy(np.stack([r[1] for r in rows]).astype(np.int64));mask=torch.from_numpy(np.stack([r[2] for r in rows])).bool()
        with torch.no_grad():
            old_dist,_=old.model.distribution(f,cards,mask);new_dist,_=new.model.distribution(f,cards,mask)
            old_action=old_dist.logits.argmax(-1);new_action=new_dist.logits.argmax(-1)
            if not np.array_equal(old_action.numpy(),[r[3] for r in rows]):raise ValueError('Incumbent visible action differs from the committed public trace')
            kl=(old_dist.probs*(old_dist.logits-new_dist.logits)).sum(-1)
            groups={}
            for label,select in [('all',torch.ones(len(rows),dtype=torch.bool)),('turns1-5',f[:,0]*20<5.5),('turns6-10',(f[:,0]*20>=5.5)&(f[:,0]*20<10.5)),('turns11+',f[:,0]*20>=10.5)]:
                if not select.any():continue
                category=lambda a:dict(collections.Counter(tg.action_name(int(x)).split('(')[0].split(' {')[0] for x in a[select]))
                groups[label]=dict(rows=int(select.sum()),greedy_agreement=float((old_action[select]==new_action[select]).float().mean()),old_to_new_KL=float(kl[select].mean()),old_entropy=float(old_dist.entropy()[select].mean()),new_entropy=float(new_dist.entropy()[select].mean()),old_end_turn_probability=float(old_dist.probs[select,0].mean()),new_end_turn_probability=float(new_dist.probs[select,0].mean()),old_actions=category(old_action),new_actions=category(new_action))
    result=dict(reference=str(reference_screen.resolve()),challenger=str(Path(bundle).resolve()),reference_identity=sha(reference_screen/'registry/learned/candidate/identity.json'),challenger_identity=sha(Path(bundle)/'identity.json'),native=tg.build_info()['source_hash'],choice='All3profiles,first published seed,physical candidate seat0; no outcome-based trace selection',interpretation='Conditional fixed visible-state drift only; new actions are not executed and these are not performance estimates',replays=replays,groups=groups,helper_sha256=sha(__file__))
    write(out/'result.json',result);return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--reference-screen',type=Path,required=True);p.add_argument('--bundle',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();print(json.dumps(audit(a.reference_screen,a.bundle,a.out)['groups'],indent=2))
