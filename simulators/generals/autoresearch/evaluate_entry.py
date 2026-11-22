from pathlib import Path
import json
from .arena import evaluate,PolicyFailure
from .contract import write_json
from .grading import grade

def fault(error):
 """What kind of policy fault this was, for the researcher: a name and one sentence. The evaluator's own detail
 (paths, resource counters) stays private, except the end of the policy process's own error output when it crashed."""
 text=str(error)
 if 'not learned' in text or 'own initialization' in text:
  return 'not_learned','The policy is not learned: with its weights re-initialized it still chooses mostly the same moves, or its saved weights do not match its own initialization. Moves must come from trained weights.'
 if 'FLOPs' in text or 'flops' in text:
  return 'compute_cap','The policy is over the compute-per-move cap, or its forward pass could not be counted.'
 if 'did not finish loading' in text:
  return 'startup','The policy did not finish loading (imports, weights, the host checks) within the start-up allowance.'
 if 'deadline exceeded' in text:
  return 'deadline','The policy did not answer a batch within the per-answer deadline (the first, compiling answer included).'
 if 'Illegal policy response' in text:
  return 'illegal_action','The policy chose an action that is not legal on that board.'
 if 'Invalid policy logits' in text:
  return 'bad_scores','The policy returned scores of the wrong shape or non-finite scores for legal actions.'
 if 'Wrong policy response' in text or 'Oversized policy response' in text or 'start-up protocol' in text:
  return 'bad_response','The policy process answered in the wrong form.'
 tail=text.split('stderr=',1)[1].strip()[-700:] if 'stderr=' in text else ''
 return 'crashed','The policy process ended with an error'+(': ...'+tail if tail else '.')

def main():
 c=json.loads(Path('/private/evaluation.json').read_text())
 try:
  result=evaluate(c['candidate'],c['panel'],pool_path=c['pool_path'],map_indices=c.get('map_indices'),split=c['split'],development=False)
 except PolicyFailure as exc:
  write_json('/private/evaluation-result.json',{'fault_kind':'policy','error':str(exc),'games':0})
  kind,sentence=fault(exc)
  write_json('/job/evaluation-public.json',{'status':'invalid_policy','reward':0,'policy_fault':True,'fault':kind,
   'games':0,'reason':sentence+' No games were played, so nothing was scored as a loss.'})
  return
 write_json('/private/evaluation-result.json',result)
 # The submitted process never gets these root-owned files. Public feedback omits private map IDs/seeds and raw games.
 public=grade(result['rows'],c.get('anchors',{}),development=c.get('development',False),
  ratings={p['name']:p['elo'] for p in c['panel'] if p.get('elo') is not None})
 public['inference_platform']=result['inference_platform']
 write_json('/job/evaluation-public.json',public)

if __name__=='__main__':main()
