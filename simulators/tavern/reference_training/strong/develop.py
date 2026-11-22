"""Fresh-campaign public diagnosis only; frozen opposition, no private-bank inputs."""
import argparse,json,shutil,time,fcntl
from pathlib import Path
from .common import ROOT,sha,write

def evaluate(bundle,out,extended=False):
 public=ROOT/'evidence/strong-bots-20261109-0516Z/tavern/evaluator/public-development-v1';out=Path(out);out.mkdir(parents=True,exist_ok=False)
 shutil.copy2(Path(__file__),out/'helper-source.py')
 expected=json.loads((public/'manifest.json').read_text())
 if any(sha(public/f)!=h for f,h in expected.items()):raise ValueError('Public development release mutated')
 evaluator=ROOT/'simulators/tavern/evaluation_readiness'
 if any(sha(evaluator/n)!=sha(public/n) for n in ('opponents.py','lobby.py')):raise ValueError('Executable lobby/script source differs from the frozen public release')
 executable={n:sha(evaluator/n) for n in ('opponents.py','lobby.py','policies.py','screen.py')}
 archive=out/'evaluator-source';archive.mkdir()
 for n in executable:shutil.copy2(evaluator/n,archive/n)
 from simulators.tavern.evaluation_readiness.policies import admit_bundle
 ident,summary=admit_bundle(bundle)
 registry=out/'registry';shutil.copytree(public,registry)
 for d in [registry,*[p for p in registry.rglob('*') if p.is_dir()]]:d.chmod(d.stat().st_mode|0o700)
 dest=registry/'learned/candidate';dest.mkdir()
 for f in ['policy.py','learner.py','train.py','identity.json','summary.json','weights.npz']:shutil.copy2(Path(bundle)/f,dest/f)
 config=json.loads((registry/'registry.json').read_text());config['opponents'].append(dict(policy_id='candidate',kind='learned',path='learned/candidate',files={f:sha(dest/f) for f in ['policy.py','learner.py','train.py','identity.json','summary.json','weights.npz']}));(registry/'registry.json').chmod(0o644);write(registry/'registry.json',config)
 write(out/'commitment.json',dict(candidate_identity_sha256=sha(dest/'identity.json'),candidate_weights_sha256=sha(dest/'weights.npz'),public_manifest_sha256=sha(public/'manifest.json'),panel_sha256=sha(public/('panel-extended.json' if extended else 'panel.json')),full_seats=8,profiles=config['profiles'],development_only=True,helper_source_sha256=sha(Path(__file__)),executed_evaluator_sources=executable))
 from simulators.tavern.evaluation_readiness.screen import screen
 with open(ROOT/'evidence/local-performance.lock','a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);result=screen(registry,registry/('panel-extended.json' if extended else 'panel.json'),'candidate',out/'screen',isolated=False,stage_seconds=1500 if extended else 500)
 if any(sha(evaluator/n)!=h for n,h in executable.items()):raise ValueError('Evaluator source changed during the public screen; retain raw diagnostic without promotion')
 write(out/'summary.json',dict(bundle=str(Path(bundle).resolve()),updates=summary.get('updates'),samples=summary.get('samples'),result={k:result[k] for k in ['raw_score','mean_placement','top4_rate','complete_valid_schedule','natural_complete_lobbies','failures','confidence_interval_95','per_profile','per_seat','per_opponent']}));return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--extended',action='store_true');a=p.parse_args();r=evaluate(a.bundle,a.out,a.extended);print(json.dumps({k:r[k] for k in ['raw_score','mean_placement','top4_rate','complete_valid_schedule','failures']}))
