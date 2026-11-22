"""Finite CPU crossplay on the published development bank only."""
from pathlib import Path
import argparse,json,subprocess,time
from simulators.clash.autoresearch.common import dump,sha,digest,engine_binding

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();root=Path(a.root);root.chmod(0o700);inputs=root/'inputs';inputs.chmod(0o700)
 plan=json.loads((inputs/'plan.json').read_text())
 if plan['purpose']!='PUBLIC DEVELOPMENT ONLY' or plan['engine_sha256']!=digest(engine_binding()):raise ValueError('Public schedule/engine mismatch')
 from simulators.clash.autoresearch.bundle import verify
 from simulators.clash.autoresearch.evaluation_v2 import screen
 for item in plan['candidates']+plan['opponents']:
  if item['kind']=='checkpoint':verify(inputs/item['bundle'],item['identity']);item['bundle']=str(inputs/item['bundle'])
 started=time.time();results=[]
 for candidate in plan['candidates']:
  for opponent in plan['opponents']:
   out=root/'results'/candidate['name']/opponent['name']
   value=screen([candidate,opponent],out,seeds=plan['public_engine_seeds'],development=False,diagnostics=True,max_seconds=420)
   if not value['complete']:raise ValueError('Invalid public pair; retain raw failure and do not score a partial panel')
   results.append({'candidate':candidate['name'],'opponent':opponent['name'],'pair':value['pairs'][0],'execution_rows':value['rows'],'latencies':value['latencies'],'entrant_probes':value['entrant_probes']})
   dump(root/'progress.json',{'public_development_only':True,'completed_pairs':len(results),'candidate':candidate['name'],'opponent':opponent['name'],'elapsed_seconds':time.time()-started});subprocess.run(['sync',str(root)],check=True,timeout=20)
 summaries={}
 for candidate in plan['candidates']:
  rows=[x for x in results if x['candidate']==candidate['name']];summaries[candidate['name']]={'mean_points':sum(x['pair']['left_points_0_100'] for x in rows)/len(rows),'by_opponent':{x['opponent']:x['pair']['left_points_0_100'] for x in rows},'games':sum(x['pair']['games'] for x in rows),'complete':True,'identity':candidate['identity'],'public_development_only':True}
 dump(root/'result.json',{'complete':True,'public_development_only':True,'summaries':summaries,'results':results,'elapsed_seconds':time.time()-started,'plan_sha256':sha(inputs/'plan.json')});subprocess.run(['sync',str(root)],check=True,timeout=20)

if __name__=='__main__':main()
