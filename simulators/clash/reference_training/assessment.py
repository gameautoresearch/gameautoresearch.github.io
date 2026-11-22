"""Public development assessment for immutable neural bundles; no private bank."""
from pathlib import Path
import argparse,json
from simulators.clash.autoresearch.common import dump,sha

def prepare_artifacts(attempt):
 from simulators.clash.autoresearch.bundle import seal,verify
 from .export_adapter import derive
 root=Path(attempt);output=root/'collected/output';dest=root/'bundles';dest.mkdir(exist_ok=True)
 results={}
 for label in ('initial','post-teacher','final'):
  if not (output/label).is_dir():continue
  target=dest/label
  if target.exists():manifest=verify(target)
  else:
   derived=root/'derived-export'/label
   if not derived.exists():derive(output/label,derived)
   manifest=seal(derived,target,{'track':'reference_training','stage':label,'checkpoint_sha256':sha(output/'checkpoint.pt'),'lineage':json.loads((output/label/'lineage.json').read_text())})
  results[label]={'bundle':str(target.resolve()),'identity':manifest['identity'],'parameters':manifest['parameters']}
 dump(root/'admitted-bundles.json',results);return results

def run(attempt):
 from simulators.clash.autoresearch.arena import evaluate
 root=Path(attempt);admitted=prepare_artifacts(root)
 panels=[{'name':'standard','kind':'scripted'},{'name':'pressure','kind':'scripted','options':{'attack_elixir':40000,'support_elixir':30000,'back_row':8,'spell_targets':2,'spell_worth':1.0}},{'name':'untrained-control','kind':'checkpoint',**admitted['initial']}]
 # Public development seeds are fixed before observing the first learned scores.
 schedule={'engine_seeds':[1000101,1000102,1000103,1000104],'policy_seed':9191,'opponents':panels}
 dump(root/'development-schedule.json',schedule)
 scores={}
 for label in admitted:
  out=root/('development-'+label+'.json')
  if out.exists():result=json.loads(out.read_text())
  else:result=evaluate(admitted[label]['bundle'],root/'development-schedule.json',out,development=True,split='public-reference-development')
  scores[label]=result['summary'] if 'summary' in result else {k:v for k,v in result.items() if k!='rows'}
 dump(root/'development-comparison.json',scores);print(json.dumps(scores,indent=2));return scores

def main():
 p=argparse.ArgumentParser();p.add_argument('--attempt',required=True);a=p.parse_args();run(a.attempt)
if __name__=='__main__':main()
